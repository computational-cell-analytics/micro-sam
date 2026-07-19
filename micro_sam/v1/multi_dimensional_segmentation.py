"""Multi-dimensional segmentation with segment anything.
"""

import os
import multiprocessing as mp
import warnings
from concurrent import futures
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import xml.etree.ElementTree as ET

import imageio.v3 as imageio
import networkx as nx
import numpy as np
import torch
from scipy.ndimage import binary_closing
from skimage.measure import regionprops

from bioimage_cpp.segmentation import label, relabel_sequential
from bioimage_cpp.graph import UndirectedGraph
from bioimage_cpp.utils import segmentation_overlap

import elf.segmentation as seg_utils
import elf.tracking.tracking_utils as track_utils
from elf.tracking.motile_tracking import recolor_segmentation

from segment_anything.predictor import SamPredictor

try:
    from napari.utils import progress as tqdm
except ImportError:
    from tqdm import tqdm

try:
    from trackastra.model import Trackastra
    from trackastra.tracking import graph_to_ctc, graph_to_napari_tracks
except ImportError:
    Trackastra = None
    graph_to_ctc = None
    graph_to_napari_tracks = None


from .. import util
from .util import precompute_image_embeddings
from .prompt_based_segmentation import segment_from_mask
from .instance_segmentation import AutoSegBase


PROJECTION_MODES = ("box", "mask", "points", "points_and_mask", "single_point")


def _validate_projection(projection):
    use_single_point = False
    if isinstance(projection, str):
        if projection == "mask":
            use_box, use_mask, use_points = True, True, False
        elif projection == "points":
            use_box, use_mask, use_points = False, False, True
        elif projection == "box":
            use_box, use_mask, use_points = True, False, False
        elif projection == "points_and_mask":
            use_box, use_mask, use_points = False, True, True
        elif projection == "single_point":
            use_box, use_mask, use_points = False, False, True
            use_single_point = True
        else:
            raise ValueError(
                "Choose projection method from 'mask' / 'points' / 'box' / 'points_and_mask' / 'single_point'. "
                f"You have passed the invalid option {projection}."
            )
    elif isinstance(projection, dict):
        assert len(projection.keys()) == 3, "There should be three parameters assigned for the projection method."
        use_box, use_mask, use_points = projection["use_box"], projection["use_mask"], projection["use_points"]
    else:
        raise ValueError(f"{projection} is not a supported projection method.")
    return use_box, use_mask, use_points, use_single_point


# Advanced stopping criterions.
# In practice these did not make a big difference, so we do not use this at the moment.
# We still leave it here for reference.
def _advanced_stopping_criteria(
    z, seg_z, seg_prev, z_start, z_increment, segmentation, criterion_choice, score, increment
):
    def _compute_mean_iou_for_n_slices(z, increment, seg_z, n_slices):
        iou_list = [
            util.compute_iou(segmentation[z - increment * _slice], seg_z) for _slice in range(1, n_slices+1)
        ]
        return np.mean(iou_list)

    if criterion_choice == 1:
        # 1. current metric: iou of current segmentation and the previous slice
        iou = util.compute_iou(seg_prev, seg_z)
        criterion = iou

    elif criterion_choice == 2:
        # 2. combining SAM iou + iou: curr. slice & first segmented slice + iou: curr. slice vs prev. slice
        iou = util.compute_iou(seg_prev, seg_z)
        ff_iou = util.compute_iou(segmentation[z_start], seg_z)
        criterion = 0.5 * iou + 0.3 * score + 0.2 * ff_iou

    elif criterion_choice == 3:
        # 3. iou of current segmented slice w.r.t the previous n slices
        criterion = _compute_mean_iou_for_n_slices(z, increment, seg_z, min(5, abs(z - z_start)))

    return criterion


def segment_mask_in_volume(
    segmentation: np.ndarray,
    predictor: SamPredictor,
    image_embeddings: util.ImageEmbeddings,
    segmented_slices: np.ndarray,
    stop_lower: bool,
    stop_upper: bool,
    iou_threshold: float,
    projection: Union[str, dict],
    update_progress: Optional[callable] = None,
    box_extension: float = 0.0,
    verbose: bool = False,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Segment an object mask in in volumetric data.

    Args:
        segmentation: The initial segmentation for the object.
        predictor: The Segment Anything predictor.
        image_embeddings: The precomputed image embeddings for the volume.
        segmented_slices: List of slices for which this object has already been segmented.
        stop_lower: Whether to stop at the lowest segmented slice.
        stop_upper: Wheter to stop at the topmost segmented slice.
        iou_threshold: The IOU threshold for continuing segmentation across 3d.
        projection: The projection method to use. One of 'box', 'mask', 'points', 'points_and_mask' or 'single point'.
            Pass a dictionary to choose the excact combination of projection modes.
        update_progress: Callback to update an external progress bar.
        box_extension: Extension factor for increasing the box size after projection.
            By default, does not increase the projected box size.
        verbose: Whether to print details about the segmentation steps. By default, set to 'True'.

    Returns:
        Array with the volumetric segmentation.
        Tuple with the first and last segmented slice.
    """
    use_box, use_mask, use_points, use_single_point = _validate_projection(projection)

    if update_progress is None:
        def update_progress(*args):
            pass

    def segment_range(z_start, z_stop, increment, stopping_criterion, threshold=None, verbose=False):
        z = z_start + increment
        while True:
            if verbose:
                print(f"Segment {z_start} to {z_stop}: segmenting slice {z}")
            seg_prev = segmentation[z - increment]
            seg_z, score, _ = segment_from_mask(
                predictor, seg_prev, image_embeddings=image_embeddings, i=z, use_mask=use_mask,
                use_box=use_box, use_points=use_points, box_extension=box_extension, return_all=True,
                use_single_point=use_single_point,
            )
            if threshold is not None:
                iou = util.compute_iou(seg_prev, seg_z)
                if iou < threshold:
                    if verbose:
                        msg = f"Segmentation stopped at slice {z} due to IOU {iou} < {threshold}."
                        print(msg)
                    break

            segmentation[z] = seg_z
            z += increment
            if stopping_criterion(z, z_stop):
                if verbose:
                    print(f"Segment {z_start} to {z_stop}: stop at slice {z}")
                break
            update_progress(1)

        return z - increment

    z0, z1 = int(segmented_slices.min()), int(segmented_slices.max())

    # segment below the min slice
    if z0 > 0 and not stop_lower:
        z_min = segment_range(z0, 0, -1, np.less, iou_threshold, verbose=verbose)
    else:
        z_min = z0

    # segment above the max slice
    if z1 < segmentation.shape[0] - 1 and not stop_upper:
        z_max = segment_range(z1, segmentation.shape[0] - 1, 1, np.greater, iou_threshold, verbose=verbose)
    else:
        z_max = z1

    # segment in between min and max slice
    if z0 != z1:
        for z_start, z_stop in zip(segmented_slices[:-1], segmented_slices[1:]):
            slice_diff = z_stop - z_start
            z_mid = int((z_start + z_stop) // 2)

            if slice_diff == 1:  # the slices are adjacent -> we don't need to do anything
                pass

            elif z_start == z0 and stop_lower:  # the lower slice is stop: we just segment from upper
                segment_range(z_stop, z_start, -1, np.less_equal, verbose=verbose)

            elif z_stop == z1 and stop_upper:  # the upper slice is stop: we just segment from lower
                segment_range(z_start, z_stop, 1, np.greater_equal, verbose=verbose)

            elif slice_diff == 2:  # there is only one slice in between -> use combined mask
                z = z_start + 1
                seg_prompt = np.logical_or(segmentation[z_start] == 1, segmentation[z_stop] == 1)
                segmentation[z] = segment_from_mask(
                    predictor, seg_prompt, image_embeddings=image_embeddings, i=z,
                    use_mask=use_mask, use_box=use_box, use_points=use_points,
                    box_extension=box_extension
                )
                update_progress(1)

            else:  # there is a range of more than 2 slices in between -> segment ranges
                # segment from bottom
                segment_range(
                    z_start, z_mid, 1, np.greater_equal if slice_diff % 2 == 0 else np.greater, verbose=verbose
                )
                # segment from top
                segment_range(z_stop, z_mid, -1, np.less_equal, verbose=verbose)
                # if the difference between start and stop is even,
                # then we have a slice in the middle that is the same distance from top bottom
                # in this case the slice is not segmented in the ranges above, and we segment it
                # using the combined mask from the adjacent top and bottom slice as prompt
                if slice_diff % 2 == 0:
                    seg_prompt = np.logical_or(segmentation[z_mid - 1] == 1, segmentation[z_mid + 1] == 1)
                    segmentation[z_mid] = segment_from_mask(
                        predictor, seg_prompt, image_embeddings=image_embeddings, i=z_mid,
                        use_mask=use_mask, use_box=use_box, use_points=use_points,
                        box_extension=box_extension
                    )
                    update_progress(1)

    return segmentation, (z_min, z_max)


def _preprocess_closing(slice_segmentation, gap_closing, pbar_update):
    binarized = slice_segmentation > 0
    # Use a structuring element that only closes elements in z, to avoid merging objects in-plane.
    structuring_element = np.zeros((3, 1, 1))
    structuring_element[:, 0, 0] = 1
    closed_segmentation = binary_closing(binarized, iterations=gap_closing, structure=structuring_element)

    new_segmentation = np.zeros_like(slice_segmentation)
    n_slices = new_segmentation.shape[0]

    def process_slice(z, offset):
        seg_z = slice_segmentation[z]

        # Closing does not work for the first and last gap slices
        if z < gap_closing or z >= (n_slices - gap_closing):
            seg_z, _, _ = relabel_sequential(seg_z, offset=offset)
            offset = int(seg_z.max()) + 1
            return seg_z, offset

        # Apply connected components to the closed segmentation.
        closed_z = label(closed_segmentation[z])

        # Map objects in the closed and initial segmentation.
        # We take objects from the closed segmentation unless they
        # have overlap with more than one object from the initial segmentation.
        # This indicates wrong merging of closeby objects that we want to prevent.
        matches = segmentation_overlap(closed_z, seg_z)
        matches = {
            seg_id: matches.overlaps_for_label_a(seg_id)["label"] for seg_id in range(1, int(closed_z.max() + 1))
        }
        matches = {k: v[v != 0] for k, v in matches.items()}

        ids_initial, ids_closed = [], []
        for seg_id, matched in matches.items():
            if len(matched) > 1:
                ids_initial.extend(matched.tolist())
            else:
                ids_closed.append(seg_id)

        seg_new = np.zeros_like(seg_z)
        closed_mask = np.isin(closed_z, ids_closed)
        seg_new[closed_mask] = closed_z[closed_mask]

        if ids_initial:
            initial_mask = np.isin(seg_z, ids_initial)
            seg_new[initial_mask] = relabel_sequential(seg_z[initial_mask], offset=seg_new.max() + 1)[0]

        seg_new, _, _ = relabel_sequential(seg_new, offset=offset)
        max_z = seg_new.max()
        if max_z > 0:
            offset = int(max_z) + 1

        return seg_new, offset

    # Further optimization: parallelize
    offset = 1
    for z in range(n_slices):
        new_segmentation[z], offset = process_slice(z, offset)
        pbar_update(1)

    return new_segmentation


def _filter_z_extent(segmentation, min_z_extent):
    props = regionprops(segmentation)
    filter_ids = []
    for prop in props:
        box = prop.bbox
        z_extent = box[3] - box[0]
        if z_extent < min_z_extent:
            filter_ids.append(prop.label)
    if filter_ids:
        segmentation[np.isin(segmentation, filter_ids)] = 0
    return segmentation


def merge_instance_segmentation_3d(
    slice_segmentation: np.ndarray,
    beta: float = 0.5,
    with_background: bool = True,
    gap_closing: Optional[int] = None,
    min_z_extent: Optional[int] = None,
    verbose: bool = True,
    pbar_init: Optional[callable] = None,
    pbar_update: Optional[callable] = None,
) -> np.ndarray:
    """Merge stacked 2d instance segmentations into a consistent 3d segmentation.

    Solves a multicut problem based on the overlap of objects to merge across z.

    Args:
        slice_segmentation: The stacked segmentation across the slices.
            We assume that the segmentation is labeled consecutive across z.
        beta: The bias term for the multicut. Higher values lead to a larger
            degree of over-segmentation and vice versa. by default, set to '0.5'.
        with_background: Whether this is a segmentation problem with background.
            In that case all edges connecting to the background are set to be repulsive.
            By default, set to 'True'.
        gap_closing: If given, gaps in the segmentation are closed with a binary closing
            operation. The value is used to determine the number of iterations for the closing.
        min_z_extent: Require a minimal extent in z for the segmented objects.
            This can help to prevent segmentation artifacts.
        verbose: Verbosity flag. By default, set to 'True'.
        pbar_init: Callback to initialize an external progress bar. Must accept number of steps and description.
            Can be used together with pbar_update to handle napari progress bar in other thread.
            To enable using this function within a threadworker.
        pbar_update: Callback to update an external progress bar.

    Returns:
        The merged segmentation.
    """
    _, pbar_init, pbar_update, pbar_close = util.handle_pbar(verbose, pbar_init, pbar_update)

    if gap_closing is not None and gap_closing > 0:
        pbar_init(slice_segmentation.shape[0] + 1, "Merge segmentation")
        slice_segmentation = _preprocess_closing(slice_segmentation, gap_closing, pbar_update)
    else:
        pbar_init(1, "Merge segmentation")

    # Extract the overlap between slices.
    edges = track_utils.compute_edges_from_overlap(slice_segmentation, verbose=False)
    if len(edges) == 0:  # Nothing to merge.
        return slice_segmentation

    uv_ids = np.array([[edge["source"], edge["target"]] for edge in edges], dtype=np.uint64)
    overlaps = np.array([edge["score"] for edge in edges])

    n_nodes = int(slice_segmentation.max() + 1)
    graph = UndirectedGraph(n_nodes)
    graph.insert_edges(uv_ids)

    costs = seg_utils.multicut.compute_edge_costs(overlaps)
    # Set background weights to be maximally repulsive.
    if with_background:
        bg_edges = (uv_ids == 0).any(axis=1)
        costs[bg_edges] = -8.0

    node_labels = seg_utils.multicut.multicut_decomposition(graph, 1.0 - costs, beta=beta)

    segmentation = node_labels[slice_segmentation]
    if min_z_extent is not None and min_z_extent > 0:
        segmentation = _filter_z_extent(segmentation, min_z_extent)

    pbar_update(1)
    pbar_close()

    return segmentation


def _segment_slices(
    data, predictor, segmentor, embedding_path, verbose, tile_shape, halo, batch_size=1, **kwargs
):
    assert data.ndim == 3

    image_embeddings = precompute_image_embeddings(
        predictor=predictor,
        input_=data,
        save_path=embedding_path,
        ndim=3,
        tile_shape=tile_shape,
        halo=halo,
        verbose=verbose,
        batch_size=batch_size,
    )

    offset = 0
    segmentation = np.zeros(data.shape, dtype="uint32")

    for i in tqdm(range(segmentation.shape[0]), desc="Segment slices", disable=not verbose):
        segmentor.initialize(data[i], image_embeddings=image_embeddings, verbose=False, i=i)
        seg = segmentor.generate(**kwargs)

        # Set offset for instance per slice.
        max_z = int(seg.max())
        if max_z == 0:
            continue
        seg[seg != 0] += offset
        offset = max_z + offset
        segmentation[i] = seg

    return segmentation, image_embeddings


def automatic_3d_segmentation(
    volume: np.ndarray,
    predictor: SamPredictor,
    segmentor: AutoSegBase,
    embedding_path: Optional[Union[str, os.PathLike]] = None,
    with_background: bool = True,
    gap_closing: Optional[int] = None,
    min_z_extent: Optional[int] = None,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    verbose: bool = True,
    return_embeddings: bool = False,
    batch_size: int = 1,
    **kwargs,
) -> np.ndarray:
    """Automatically segment objects in a volume.

    First segments slices individually in 2d and then merges them across 3d
    based on overlap of objects between slices.

    Args:
        volume: The input volume.
        predictor: The Segment Anything predictor.
        segmentor: The instance segmentation class.
        embedding_path: The path to save pre-computed embeddings.
        with_background: Whether the segmentation has background. By default, set to 'True'.
        gap_closing: If given, gaps in the segmentation are closed with a binary closing
            operation. The value is used to determine the number of iterations for the closing.
        min_z_extent: Require a minimal extent in z for the segmented objects.
            This can help to prevent segmentation artifacts.
        tile_shape: Shape of the tiles for tiled prediction. By default prediction is run without tiling.
        halo: Overlap of the tiles for tiled prediction. By default prediction is run without tiling.
        verbose: Verbosity flag. By default, set to 'True'.
        return_embeddings: Whether to return the precomputed image embeddings. By default, set to 'False'.
        batch_size: The batch size to compute image embeddings over planes. By default, set to '1'.
        kwargs: Keyword arguments for the 'generate' method of the 'segmentor'.

    Returns:
        The segmentation.
    """
    segmentation, image_embeddings = _segment_slices(
        data=volume,
        predictor=predictor,
        segmentor=segmentor,
        embedding_path=embedding_path,
        verbose=verbose,
        tile_shape=tile_shape,
        halo=halo,
        batch_size=batch_size,
        **kwargs
    )
    segmentation = merge_instance_segmentation_3d(
        segmentation,
        beta=0.5,
        with_background=with_background,
        gap_closing=gap_closing,
        min_z_extent=min_z_extent,
        verbose=verbose,
    )
    if return_embeddings:
        return segmentation, image_embeddings
    else:
        return segmentation


def _filter_tracks(tracking_result, min_track_length):
    props = regionprops(tracking_result)
    discard_ids = []
    for prop in props:
        label_id = prop.label
        # The first axis of the tracking result is time, so the bbox extent along it is the track length.
        t_start, t_stop = prop.bbox[0], prop.bbox[3]
        if t_stop - t_start < min_track_length:
            discard_ids.append(label_id)
    tracking_result = tracking_result.copy()
    tracking_result[np.isin(tracking_result, discard_ids)] = 0
    # We deliberately do not relabel the result here, so that the remaining track ids stay consistent with
    # the lineage information. Non-consecutive track ids are filtered from the lineages via '_filter_lineages'.
    return tracking_result


def _extract_tracks_and_lineages(segmentations, track_data, parent_graph):
    # The track data has the following layout: n_tracks x 4
    # With the following columns:
    # track_id - id of the track (= result from trackastra)
    # timepoint
    # y coordinate
    # x coordinate

    # Use the last three columns to index the segmentation and get the segmentation id.
    index = np.round(track_data[:, 1:], 0).astype("int32")
    index = tuple(index[:, i] for i in range(index.shape[1]))
    segmentation_ids = segmentations[index]

    # Find the mapping of nodes (= segmented objects) to track-ids.
    track_ids = track_data[:, 0].astype("int32")
    assert len(segmentation_ids) == len(track_ids)
    node_to_track = {k: v for k, v in zip(segmentation_ids, track_ids)}

    # Find the lineages as connected components in the parent graph.
    # First, we build a proper graph.
    lineage_graph = nx.Graph()
    for k, v in parent_graph.items():
        lineage_graph.add_edge(k, v)

    # Then, find the connected components, and compute the lineage representation expected by micro-sam from it:
    # E.g. if we have three lineages, the first consisting of three tracks and the second and third of one track each:
    # [
    #   {1: [2, 3]},  lineage with a dividing cell
    #   {4: []}, lineage with just one cell
    #   {5: []}, lineage with just one cell
    # ]

    # Determine the first time point of each track, so that we can orient each lineage tree by time.
    # The parent_graph is undirected here, so we root each component at its temporally earliest track.
    # Otherwise the parent / child orientation would be arbitrary and could be temporally invalid
    # (a 'parent' appearing only after its 'child'), which breaks downstream consumers like CTC, GEFF
    # and TrackMate export as well as the napari lineage display.
    track_first_time = {}
    for track_id, t in zip(track_ids.tolist(), track_data[:, 1].astype("int32").tolist()):
        if track_id not in track_first_time or t < track_first_time[track_id]:
            track_first_time[track_id] = t

    # First, we fill the lineages which have one or more divisions, i.e. trees with more than one node.
    lineages = []
    for component in nx.connected_components(lineage_graph):
        root = min(component, key=lambda node: track_first_time.get(node, 0))
        lineage_dict = {}

        def dfs(node, parent):
            # Avoid revisiting the parent node
            children = [n for n in lineage_graph[node] if n != parent]
            lineage_dict[node] = children
            for child in children:
                dfs(child, node)

        dfs(root, None)
        lineages.append(lineage_dict)

    # Then add single node lineages, which are not reflected in the original graph.
    all_tracks = set(track_ids.tolist())
    lineage_tracks = []
    for lineage in lineages:
        for k, v in lineage.items():
            lineage_tracks.append(k)
            lineage_tracks.extend(v)
    singleton_tracks = list(all_tracks - set(lineage_tracks))
    lineages.extend([{track: []} for track in singleton_tracks])

    # Make sure node_to_track contains everything.
    all_seg_ids = np.unique(segmentations)
    missing_seg_ids = np.setdiff1d(all_seg_ids, list(node_to_track.keys()))
    node_to_track.update({seg_id: 0 for seg_id in missing_seg_ids})
    return node_to_track, lineages


def _filter_lineages(lineages, tracking_result):
    track_ids = set(np.unique(tracking_result)) - {0}
    filtered_lineages = []
    for lineage in lineages:
        # Drop nodes that are no longer present, both as parents (keys) and as children (values),
        # so that the lineage does not reference filtered-out tracks.
        filtered_lineage = {
            k: [c for c in v if c in track_ids] for k, v in lineage.items() if k in track_ids
        }
        if filtered_lineage:
            filtered_lineages.append(filtered_lineage)
    return filtered_lineages


def _tracking_impl(timeseries, segmentation, mode, min_time_extent, tracking_model="general_2d", output_folder=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Trackastra.from_pretrained(tracking_model, device=device)
    result = model.track(timeseries, segmentation, mode=mode)
    try:
        lineage_graph, _ = result
    except ValueError:
        lineage_graph = result

    track_data, parent_graph, _ = graph_to_napari_tracks(lineage_graph)
    if track_data.size == 0:
        warnings.warn("Tracking result is empty.")
        tracking_result = np.zeros_like(segmentation)
        lineages = []
        return tracking_result, lineages

    node_to_track, lineages = _extract_tracks_and_lineages(segmentation, track_data, parent_graph)
    tracking_result = recolor_segmentation(segmentation, node_to_track)

    if output_folder is not None:  # Store tracking results in CTC format.
        graph_to_ctc(lineage_graph, segmentation, outdir=output_folder)

    # Filter out short tracks. Trackastra has no native option for this, so we do it as a post-process.
    if min_time_extent is not None and min_time_extent > 0:
        tracking_result = _filter_tracks(tracking_result, min_time_extent)

    # Filter out pruned lineages.
    # May either be missing due to track filtering or non-consecutive track numbering in trackastra.
    lineages = _filter_lineages(lineages, tracking_result)

    return tracking_result, lineages


def track_across_frames(
    timeseries: np.ndarray,
    segmentation: np.ndarray,
    gap_closing: Optional[int] = None,
    min_time_extent: Optional[int] = None,
    mode: str = "greedy",
    tracking_model: str = "general_2d",
    verbose: bool = True,
    pbar_init: Optional[callable] = None,
    pbar_update: Optional[callable] = None,
    output_folder: Optional[Union[os.PathLike, str]] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """Track segmented objects over time.

    This function uses Trackastra: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf
    for tracking. Please cite it if you use the automated tracking functionality.

    Args:
        timeseries: The input timeseries of images.
        segmentation: The segmentation. Expect segmentation results per frame
            that are relabeled so that segmentation ids don't overlap.
        gap_closing: If given, gaps in the segmentation are closed with a binary closing
            operation. The value is used to determine the number of iterations for the closing.
        min_time_extent: Require a minimal extent in time for the tracked objects.
        mode: The trackastra linking solver. One of 'greedy_nodiv', 'greedy' or 'ilp'.
            'ilp' uses the motile solver. By default, set to 'greedy'.
        tracking_model: The pretrained trackastra model to use. By default, set to 'general_2d'.
        verbose: Verbosity flag. By default, set to 'True'.
        pbar_init: Function to initialize the progress bar.
        pbar_update: Function to update the progress bar.
        output_folder: The folder where the tracking results are stored in CTC format.

    Returns:
        The tracking result. Each object is colored by its track id.
        The lineages, which correspond to the cell divisions. Lineages are represented by a list of dicts,
            with each dict encoding a lineage, where keys correspond to parent track ids.
            Each key either maps to a list with two child track ids (cell division) or to an empty list (no division).
    """
    if Trackastra is None:
        raise RuntimeError(
            "Automatic tracking requires trackastra. You can install it via 'pip install trackastra'."
        )

    _, pbar_init, pbar_update, pbar_close = util.handle_pbar(verbose, pbar_init=pbar_init, pbar_update=pbar_update)

    if gap_closing is not None and gap_closing > 0:
        segmentation = _preprocess_closing(segmentation, gap_closing, pbar_update)

    segmentation, lineage = _tracking_impl(
        timeseries=np.asarray(timeseries),
        segmentation=segmentation,
        mode=mode,
        min_time_extent=min_time_extent,
        tracking_model=tracking_model,
        output_folder=output_folder,
    )
    return segmentation, lineage


def automatic_tracking_implementation(
    timeseries: np.ndarray,
    predictor: SamPredictor,
    segmentor: AutoSegBase,
    embedding_path: Optional[Union[str, os.PathLike]] = None,
    gap_closing: Optional[int] = None,
    min_time_extent: Optional[int] = None,
    mode: str = "greedy",
    tracking_model: str = "general_2d",
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    verbose: bool = True,
    return_embeddings: bool = False,
    batch_size: int = 1,
    output_folder: Optional[Union[os.PathLike, str]] = None,
    **kwargs,
) -> Tuple[np.ndarray, List[Dict]]:
    """Automatically track objects in a timesries based on per-frame automatic segmentation.

    This function uses Trackastra: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf
    for tracking. Please cite it if you use the automated tracking functionality.

    Args:
        timeseries: The input timeseries of images.
        predictor: The SAM model.
        segmentor: The instance segmentation class.
        embedding_path: The path to save pre-computed embeddings.
        gap_closing: If given, gaps in the segmentation are closed with a binary closing
            operation. The value is used to determine the number of iterations for the closing.
        min_time_extent: Require a minimal extent in time for the tracked objects.
        mode: The trackastra linking solver. One of 'greedy_nodiv', 'greedy' or 'ilp'.
            'ilp' uses the motile solver. By default, set to 'greedy'.
        tracking_model: The pretrained trackastra model to use. By default, set to 'general_2d'.
        tile_shape: Shape of the tiles for tiled prediction. By default prediction is run without tiling.
        halo: Overlap of the tiles for tiled prediction. By default prediction is run without tiling.
        verbose: Verbosity flag. By default, set to 'True'.
        return_embeddings: Whether to return the precomputed image embeddings. By default, set to 'False'.
        batch_size: The batch size to compute image embeddings over planes. By default, set to '1'.
        output_folder: The folder where the tracking results are stored in CTC format.
        kwargs: Keyword arguments for the 'generate' method of the 'segmentor'.

    Returns:
        The tracking result. Each object is colored by its track id.
        The lineages, which correspond to the cell divisions. Lineages are represented by a list of dicts,
            with each dict encoding a lineage, where keys correspond to parent track ids.
            Each key either maps to a list with two child track ids (cell division) or to an empty list (no division).
    """
    if Trackastra is None:
        raise RuntimeError(
            "Automatic tracking requires trackastra. You can install it via 'pip install trackastra'."
        )

    segmentation, image_embeddings = _segment_slices(
        timeseries, predictor, segmentor, embedding_path, verbose,
        tile_shape=tile_shape, halo=halo, batch_size=batch_size,
        **kwargs,
    )

    segmentation, lineage = track_across_frames(
        timeseries=timeseries,
        segmentation=segmentation,
        gap_closing=gap_closing,
        min_time_extent=min_time_extent,
        mode=mode,
        tracking_model=tracking_model,
        verbose=verbose,
        output_folder=output_folder,
    )

    if return_embeddings:
        return segmentation, lineage, image_embeddings
    else:
        return segmentation, lineage


def get_napari_track_data(
    segmentation: np.ndarray, lineages: List[Dict], n_threads: Optional[int] = None
) -> Tuple[np.ndarray, Dict[int, List]]:
    """Derive the inputs for the napari tracking layer from a tracking result.

    Args:
        segmentation: The segmentation, after relabeling with track ids.
        lineages: The lineage information.
        n_threads: Number of threads for extracting the track data from the segmentation.

    Returns:
        The array with the track data expected by napari.
        The parent dictionary for napari.
    """
    if n_threads is None:
        n_threads = mp.cpu_count()

    def compute_props(t):
        props = regionprops(segmentation[t])
        # Create the track data representation for napari, which expects:
        # track_id, timepoint, y, x
        track_data = np.array([[prop.label, t] + list(prop.centroid) for prop in props])
        return track_data

    with futures.ThreadPoolExecutor(n_threads) as tp:
        track_data = list(tp.map(compute_props, range(segmentation.shape[0])))
    track_data = [data for data in track_data if data.size > 0]
    # The segmentation may be empty, e.g. if all tracks were filtered out via 'min_time_extent'.
    track_data = np.concatenate(track_data) if track_data else np.zeros((0, 4), dtype="float64")

    # The graph representation of napari uses the children as keys and the parents as values,
    # whereas our representation uses parents as keys and children as values.
    # Hence, we need to translate the representation.
    parent_graph = {
        child: [parent] for lineage in lineages for parent, children in lineage.items() for child in children
    }

    return track_data, parent_graph


def export_tracking_result_to_ctc(
    segmentation: np.ndarray,
    lineages: List[Dict],
    output_folder: Union[os.PathLike, str],
) -> None:
    """Export a tracking result to the Cell Tracking Challenge (CTC) format.

    This writes the standard CTC folder layout into 'output_folder':
    - 'TRA/man_track.txt', the lineage file with one space-separated row 'L B E P' per track
      (track id, first frame, last frame, parent track id, where the parent is 0 if the track has no parent).
    - 'TRA/man_track<frame>.tif', the per-frame tracking masks, with each object labeled by its track id.
    - 'SEG/man_seg<frame>.tif', the per-frame segmentation masks.

    Args:
        segmentation: The tracking result of shape (T, Y, X), with each object labeled by its track id.
        lineages: The lineage information, a list of dicts mapping each parent track id to its child track ids.
        output_folder: The folder where the CTC results are written, with 'TRA' and 'SEG' subfolders.
    """
    if segmentation.ndim != 3:
        raise ValueError(f"Expected a 3d (T, Y, X) tracking result, got shape {segmentation.shape}.")

    tra_folder = os.path.join(output_folder, "TRA")
    seg_folder = os.path.join(output_folder, "SEG")
    os.makedirs(tra_folder, exist_ok=True)
    os.makedirs(seg_folder, exist_ok=True)

    # Map each child track to its parent track. Tracks without a parent get 0, as expected by the CTC format.
    child_to_parent = {}
    for lineage in lineages:
        for parent, children in lineage.items():
            for child in children:
                child_to_parent[child] = parent

    # Determine the first and last frame in which each track id is present.
    n_frames = segmentation.shape[0]
    first_frame, last_frame = {}, {}
    for t in range(n_frames):
        for label_id in np.unique(segmentation[t]):
            label_id = int(label_id)
            if label_id == 0:
                continue
            if label_id not in first_frame:
                first_frame[label_id] = t
            last_frame[label_id] = t

    # Write the lineage file into the tracking folder.
    with open(os.path.join(tra_folder, "man_track.txt"), "w") as f:
        for label_id in sorted(first_frame.keys()):
            parent = child_to_parent.get(label_id, 0)
            f.write(f"{label_id} {first_frame[label_id]} {last_frame[label_id]} {parent}\n")

    # Write the per-frame masks, using uint16 if the track ids fit and uint32 otherwise.
    # The tracking masks (TRA) and segmentation masks (SEG) share the same track-id labeling.
    max_id = int(segmentation.max())
    dtype = "uint16" if max_id < np.iinfo("uint16").max else "uint32"
    n_digits = max(4, len(str(n_frames - 1)))
    for t in range(n_frames):
        frame = segmentation[t].astype(dtype)
        imageio.imwrite(os.path.join(tra_folder, f"man_track{t:0{n_digits}d}.tif"), frame)
        imageio.imwrite(os.path.join(seg_folder, f"man_seg{t:0{n_digits}d}.tif"), frame)


def _validate_tracking_result(segmentation):
    if segmentation.ndim not in (3, 4):
        raise ValueError(
            f"Expected a tracking result with shape (T, Y, X) or (T, Z, Y, X), got shape {segmentation.shape}."
        )


def _child_to_parent(lineages):
    child_to_parent = {}
    for lineage in lineages:
        for parent, children in lineage.items():
            for child in children:
                child_to_parent[int(child)] = int(parent)
    return child_to_parent


def _equivalent_radius(area, ndim):
    if ndim == 2:
        return float(np.sqrt(area / np.pi))
    return float(((3.0 * area) / (4.0 * np.pi)) ** (1.0 / 3.0))


def _tracking_result_to_graph(segmentation, lineages):
    _validate_tracking_result(segmentation)

    graph = nx.DiGraph()
    records = []
    nodes_by_track = defaultdict(list)
    spatial_ndim = segmentation.ndim - 1

    for t, frame in enumerate(segmentation):
        for prop in regionprops(frame):
            node_id = len(records)
            track_id = int(prop.label)
            coords = tuple(float(coord) for coord in prop.centroid)
            area = float(prop.area)
            radius = _equivalent_radius(area, spatial_ndim)
            record = {
                "node_id": node_id,
                "track_id": track_id,
                "time": int(t),
                "coords": coords,
                "area": area,
                "radius": radius,
            }
            records.append(record)
            nodes_by_track[track_id].append(node_id)
            graph.add_node(
                node_id,
                label=track_id,
                track_id=track_id,
                time=int(t),
                coords=coords,
                area=area,
                radius=radius,
            )

    if not records:
        raise ValueError("Cannot export an empty tracking result.")

    records_by_node = {record["node_id"]: record for record in records}

    for track_nodes in nodes_by_track.values():
        track_nodes.sort(key=lambda node_id: records_by_node[node_id]["time"])
        for source, target in zip(track_nodes[:-1], track_nodes[1:]):
            graph.add_edge(source, target, weight=1.0)

    for child, parent in _child_to_parent(lineages).items():
        parent_nodes = nodes_by_track.get(parent, [])
        child_nodes = nodes_by_track.get(child, [])
        if not parent_nodes or not child_nodes:
            continue

        first_child = min(child_nodes, key=lambda node_id: records_by_node[node_id]["time"])
        child_start = records_by_node[first_child]["time"]
        parent_candidates = [
            node_id for node_id in parent_nodes if records_by_node[node_id]["time"] < child_start
        ]
        if not parent_candidates:
            warnings.warn(
                f"Could not add lineage edge from parent track {parent} to child track {child}: "
                "the parent has no detection before the child starts.",
                stacklevel=2,
            )
            continue

        last_parent = max(parent_candidates, key=lambda node_id: records_by_node[node_id]["time"])
        graph.add_edge(last_parent, first_child, weight=1.0, division=True)

    return graph, records, records_by_node


def _normalize_output_path(output_path, default_name, suffix):
    output_path = Path(output_path)
    if output_path.suffix.lower() != suffix:
        output_path = output_path / default_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def export_tracking_result_to_geff(
    segmentation: np.ndarray,
    lineages: List[Dict],
    output_path: Union[os.PathLike, str],
) -> Path:
    """Export a tracking result to GEFF.

    The export follows the Trackastra GEFF layout: a zarr group containing the tracking masks in
    'segmentation' and the graph in 'tracking_graph.geff'.

    Args:
        segmentation: The tracking result of shape (T, Y, X), with each object labeled by its track id.
        lineages: The lineage information, a list of dicts mapping each parent track id to its child track ids.
        output_path: The zarr path to write to. If a directory is passed, 'tracking_result.zarr' is written in it.

    Returns:
        The path of the written zarr group.
    """
    try:
        from trackastra.tracking import write_to_geff
    except ImportError as e:
        raise RuntimeError(
            "GEFF export requires trackastra with GEFF support. Install a recent trackastra/geff version."
        ) from e

    output_path = _normalize_output_path(output_path, "tracking_result.zarr", ".zarr")
    graph, _, _ = _tracking_result_to_graph(segmentation, lineages)
    write_to_geff(graph, segmentation, output_path)
    return output_path


def _format_float(value):
    return f"{float(value):.6f}"


def _trackmate_xyz(record):
    coords = record["coords"]
    if len(coords) == 2:
        y, x = coords
        z = 0.0
    elif len(coords) == 3:
        z, y, x = coords
    else:
        raise ValueError(f"Expected 2d or 3d coordinates, got {coords}.")
    return x, y, z


def _add_feature(parent, feature, name, shortname, dimension, isint=False):
    ET.SubElement(
        parent,
        "Feature",
        {
            "feature": feature,
            "name": name,
            "shortname": shortname,
            "dimension": dimension,
            "isint": str(isint).lower(),
        },
    )


def _add_trackmate_feature_declarations(model):
    declarations = ET.SubElement(model, "FeatureDeclarations")

    spot_features = ET.SubElement(declarations, "SpotFeatures")
    _add_feature(spot_features, "QUALITY", "Quality", "Quality", "QUALITY")
    _add_feature(spot_features, "POSITION_X", "X", "X", "POSITION")
    _add_feature(spot_features, "POSITION_Y", "Y", "Y", "POSITION")
    _add_feature(spot_features, "POSITION_Z", "Z", "Z", "POSITION")
    _add_feature(spot_features, "POSITION_T", "T", "T", "TIME")
    _add_feature(spot_features, "FRAME", "Frame", "Frame", "NONE", isint=True)
    _add_feature(spot_features, "RADIUS", "Radius", "R", "LENGTH")
    _add_feature(spot_features, "VISIBILITY", "Visibility", "Visibility", "NONE", isint=True)
    _add_feature(spot_features, "AREA", "Area", "Area", "AREA")
    _add_feature(spot_features, "TRACKLET_ID", "micro-sam tracklet id", "Tracklet id", "NONE", isint=True)

    edge_features = ET.SubElement(declarations, "EdgeFeatures")
    _add_feature(edge_features, "SPOT_SOURCE_ID", "Source spot ID", "Source", "NONE", isint=True)
    _add_feature(edge_features, "SPOT_TARGET_ID", "Target spot ID", "Target", "NONE", isint=True)
    _add_feature(edge_features, "LINK_COST", "Link cost", "Cost", "COST")
    _add_feature(edge_features, "EDGE_TIME", "Edge time", "T", "TIME")
    _add_feature(edge_features, "EDGE_X_LOCATION", "Edge X", "X", "POSITION")
    _add_feature(edge_features, "EDGE_Y_LOCATION", "Edge Y", "Y", "POSITION")
    _add_feature(edge_features, "EDGE_Z_LOCATION", "Edge Z", "Z", "POSITION")
    _add_feature(edge_features, "VELOCITY", "Velocity", "V", "VELOCITY")

    track_features = ET.SubElement(declarations, "TrackFeatures")
    _add_feature(track_features, "TRACK_ID", "Track ID", "ID", "NONE", isint=True)
    _add_feature(track_features, "TRACK_INDEX", "Track index", "Index", "NONE", isint=True)
    _add_feature(track_features, "NUMBER_SPOTS", "Number of spots", "N spots", "NONE", isint=True)
    _add_feature(track_features, "NUMBER_GAPS", "Number of gaps", "Gaps", "NONE", isint=True)
    _add_feature(track_features, "NUMBER_SPLITS", "Number of splits", "Splits", "NONE", isint=True)
    _add_feature(track_features, "NUMBER_MERGES", "Number of merges", "Merges", "NONE", isint=True)
    _add_feature(track_features, "TRACK_START", "Track start", "Start", "TIME")
    _add_feature(track_features, "TRACK_STOP", "Track stop", "Stop", "TIME")
    _add_feature(track_features, "TRACK_DURATION", "Track duration", "Duration", "TIME")
    _add_feature(track_features, "TRACK_X_LOCATION", "Track X", "X", "POSITION")
    _add_feature(track_features, "TRACK_Y_LOCATION", "Track Y", "Y", "POSITION")
    _add_feature(track_features, "TRACK_Z_LOCATION", "Track Z", "Z", "POSITION")


def _add_trackmate_spots(model, records):
    all_spots = ET.SubElement(model, "AllSpots", {"nspots": str(len(records))})
    records_by_frame = defaultdict(list)
    for record in records:
        records_by_frame[record["time"]].append(record)

    for frame in sorted(records_by_frame):
        spots_in_frame = ET.SubElement(all_spots, "SpotsInFrame", {"frame": str(frame)})
        for record in sorted(records_by_frame[frame], key=lambda rec: rec["node_id"]):
            x, y, z = _trackmate_xyz(record)
            node_id = record["node_id"]
            ET.SubElement(
                spots_in_frame,
                "Spot",
                {
                    "ID": str(node_id),
                    "name": f"ID{node_id}",
                    "QUALITY": "1.0",
                    "POSITION_X": _format_float(x),
                    "POSITION_Y": _format_float(y),
                    "POSITION_Z": _format_float(z),
                    "POSITION_T": _format_float(record["time"]),
                    "FRAME": str(record["time"]),
                    "RADIUS": _format_float(record["radius"]),
                    "VISIBILITY": "1",
                    "AREA": _format_float(record["area"]),
                    "TRACKLET_ID": str(record["track_id"]),
                },
            )


def _component_track_attributes(graph, component, track_id, track_index, records_by_node):
    times = [records_by_node[node_id]["time"] for node_id in component]
    xyz = np.array([_trackmate_xyz(records_by_node[node_id]) for node_id in component], dtype="float64")
    n_gaps = 0
    for source, target in graph.subgraph(component).edges:
        dt = records_by_node[target]["time"] - records_by_node[source]["time"]
        n_gaps += max(0, int(dt) - 1)

    return {
        "name": f"Track_{track_id}",
        "TRACK_ID": str(track_id),
        "TRACK_INDEX": str(track_index),
        "NUMBER_SPOTS": str(len(component)),
        "NUMBER_GAPS": str(n_gaps),
        "NUMBER_SPLITS": str(sum(1 for node_id in component if graph.out_degree(node_id) > 1)),
        "NUMBER_MERGES": str(sum(1 for node_id in component if graph.in_degree(node_id) > 1)),
        "TRACK_START": _format_float(min(times)),
        "TRACK_STOP": _format_float(max(times)),
        "TRACK_DURATION": _format_float(max(times) - min(times) + 1),
        "TRACK_X_LOCATION": _format_float(float(xyz[:, 0].mean())),
        "TRACK_Y_LOCATION": _format_float(float(xyz[:, 1].mean())),
        "TRACK_Z_LOCATION": _format_float(float(xyz[:, 2].mean())),
    }


def _edge_trackmate_attributes(source, target, records_by_node):
    source_record = records_by_node[source]
    target_record = records_by_node[target]
    sx, sy, sz = _trackmate_xyz(source_record)
    tx, ty, tz = _trackmate_xyz(target_record)
    dt = target_record["time"] - source_record["time"]
    distance = np.linalg.norm(np.array([tx - sx, ty - sy, tz - sz], dtype="float64"))
    velocity = 0.0 if dt <= 0 else distance / dt

    return {
        "SPOT_SOURCE_ID": str(source),
        "SPOT_TARGET_ID": str(target),
        "LINK_COST": "0.0",
        "EDGE_TIME": _format_float((source_record["time"] + target_record["time"]) / 2.0),
        "EDGE_X_LOCATION": _format_float((sx + tx) / 2.0),
        "EDGE_Y_LOCATION": _format_float((sy + ty) / 2.0),
        "EDGE_Z_LOCATION": _format_float((sz + tz) / 2.0),
        "VELOCITY": _format_float(velocity),
    }


def _add_trackmate_tracks(model, graph, records_by_node):
    all_tracks = ET.SubElement(model, "AllTracks")
    filtered_tracks = ET.SubElement(model, "FilteredTracks")

    def component_sort_key(component):
        return min(records_by_node[node_id]["time"] for node_id in component), min(component)

    def edge_sort_key(edge):
        source, target = edge
        return records_by_node[source]["time"], source, records_by_node[target]["time"], target

    components = list(nx.weakly_connected_components(graph))
    components.sort(key=component_sort_key)

    for track_index, component in enumerate(components):
        track_id = track_index
        track = ET.SubElement(
            all_tracks,
            "Track",
            _component_track_attributes(graph, component, track_id, track_index, records_by_node),
        )
        edges = list(graph.subgraph(component).edges)
        edges.sort(key=edge_sort_key)
        for source, target in edges:
            ET.SubElement(track, "Edge", _edge_trackmate_attributes(source, target, records_by_node))

        ET.SubElement(filtered_tracks, "TrackID", {"TRACK_ID": str(track_id)})


def _add_trackmate_settings(root, segmentation):
    settings = ET.SubElement(root, "Settings")
    spatial_shape = segmentation.shape[1:]
    if len(spatial_shape) == 2:
        height, width = spatial_shape
        nslices = 1
    else:
        nslices, height, width = spatial_shape

    ET.SubElement(
        settings,
        "ImageData",
        {
            "filename": "",
            "folder": "",
            "width": str(width),
            "height": str(height),
            "nslices": str(nslices),
            "nframes": str(segmentation.shape[0]),
            "pixelwidth": "1.0",
            "pixelheight": "1.0",
            "voxeldepth": "1.0",
            "timeinterval": "1.0",
        },
    )
    ET.SubElement(settings, "InitialSpotFilter", {"feature": "QUALITY", "value": "0.0", "isabove": "true"})
    ET.SubElement(settings, "SpotFilterCollection")
    ET.SubElement(settings, "TrackFilterCollection")
    ET.SubElement(settings, "AnalyzerCollection")


def export_tracking_result_to_trackmate_xml(
    segmentation: np.ndarray,
    lineages: List[Dict],
    output_path: Union[os.PathLike, str],
) -> Path:
    """Export a tracking result to TrackMate XML.

    The XML contains all per-frame detections as TrackMate spots and the committed tracking graph as TrackMate tracks.

    Args:
        segmentation: The tracking result of shape (T, Y, X), with each object labeled by its track id.
        lineages: The lineage information, a list of dicts mapping each parent track id to its child track ids.
        output_path: The XML path to write to. If a directory is passed, 'tracking_result.xml' is written in it.

    Returns:
        The path of the written XML file.
    """
    output_path = _normalize_output_path(output_path, "tracking_result.xml", ".xml")
    graph, records, records_by_node = _tracking_result_to_graph(segmentation, lineages)

    root = ET.Element("TrackMate", {"version": "micro-sam"})
    ET.SubElement(root, "Log").text = "Created by micro-sam."
    model = ET.SubElement(root, "Model", {"spatialunits": "pixel", "timeunits": "frame"})
    _add_trackmate_feature_declarations(model)
    _add_trackmate_spots(model, records)
    _add_trackmate_tracks(model, graph, records_by_node)
    _add_trackmate_settings(root, segmentation)
    ET.SubElement(root, "GUIState", {"state": "ConfigureViews"})
    ET.SubElement(root, "DisplaySettings")

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="UTF-8", xml_declaration=True)
    return output_path
