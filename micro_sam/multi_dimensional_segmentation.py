"""Multi-dimensional segmentation with segment anything.
"""

import os
from typing import Optional, Union, Tuple

import numpy as np
import nifty
import elf.tracking.tracking_utils as track_utils
import elf.segmentation as seg_utils

from segment_anything.predictor import SamPredictor
from scipy.ndimage import binary_closing
from skimage.measure import label, regionprops
from skimage.segmentation import relabel_sequential

try:
    from napari.utils import progress as tqdm
except ImportError:
    from tqdm import tqdm

from . import util
from .instance_segmentation import AMGBase, mask_data_to_segmentation
from .prompt_based_segmentation import segment_from_mask

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
        predictor: The segment anything predictor.
        image_embeddings: The precomputed image embeddings for the volume.
        segmented_slices: List of slices for which this object has already been segmented.
        stop_lower: Whether to stop at the lowest segmented slice.
        stop_upper: Wheter to stop at the topmost segmented slice.
        iou_threshold: The IOU threshold for continuing segmentation across 3d.
        projection: The projection method to use. One of 'box', 'mask', 'points', 'points_and_mask' or 'single point'.
            Pass a dictionary to choose the excact combination of projection modes.
        update_progress: Callback to update an external progress bar.
        box_extension: Extension factor for increasing the box size after projection.
        verbose: Whether to print details about the segmentation steps.

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
        matches = nifty.ground_truth.overlap(closed_z, seg_z)
        matches = {seg_id: matches.overlapArrays(seg_id, sorted=False)[0]
                   for seg_id in range(1, int(closed_z.max() + 1))}
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
            degree of over-segmentation and vice versa.
        with_background: Whether this is a segmentation problem with background.
            In that case all edges connecting to the background are set to be repulsive.
        gap_closing: If given, gaps in the segmentation are closed with a binary closing
            operation. The value is used to determine the number of iterations for the closing.
        min_z_extent: Require a minimal extent in z for the segmented objects.
            This can help to prevent segmentation artifacts.
        verbose: Verbosity flag.
        pbar_init: Callback to initialize an external progress bar. Must accept number of steps and description.
            Can be used together with pbar_update to handle napari progress bar in other thread.
            To enables using this function within a threadworker.
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

    uv_ids = np.array([[edge["source"], edge["target"]] for edge in edges])
    overlaps = np.array([edge["score"] for edge in edges])

    n_nodes = int(slice_segmentation[-1].max() + 1)
    graph = nifty.graph.undirectedGraph(n_nodes)
    graph.insertEdges(uv_ids)

    costs = seg_utils.multicut.compute_edge_costs(overlaps)
    # set background weights to be maximally repulsive
    if with_background:
        bg_edges = (uv_ids == 0).any(axis=1)
        costs[bg_edges] = -8.0

    node_labels = seg_utils.multicut.multicut_decomposition(graph, 1.0 - costs, beta=beta)

    segmentation = nifty.tools.take(node_labels, slice_segmentation)

    if min_z_extent is not None and min_z_extent > 0:
        props = regionprops(segmentation)
        filter_ids = []
        for prop in props:
            box = prop.bbox
            z_extent = box[3] - box[0]
            if z_extent < min_z_extent:
                filter_ids.append(prop.label)
        if filter_ids:
            segmentation[np.isin(segmentation, filter_ids)] = 0

    pbar_update(1)
    pbar_close()

    return segmentation


# TODO: Enable tiling
def automatic_3d_segmentation(
    volume: np.ndarray,
    predictor: SamPredictor,
    segmentor: AMGBase,
    embedding_path: Optional[Union[str, os.PathLike]] = None,
    with_background: bool = True,
    gap_closing: Optional[int] = None,
    min_z_extent: Optional[int] = None,
    verbose: bool = True,
    **kwargs,
) -> np.ndarray:
    """Segment volume in 3d.

    First segments slices individually in 2d and then merges them across 3d
    based on overlap of objects between slices.

    Args:
        volume: The input volume.
        predictor: The SAM model.
        segmentor: The instance segmentation class.
        embedding_path: The path to save pre-computed embeddings.
        with_background: Whether the segmentation has background.
        gap_closing: If given, gaps in the segmentation are closed with a binary closing
            operation. The value is used to determine the number of iterations for the closing.
        min_z_extent: Require a minimal extent in z for the segmented objects.
            This can help to prevent segmentation artifacts.
        verbose: Verbosity flag.
        kwargs: Keyword arguments for the 'generate' method of the 'segmentor'.

    Returns:
        The segmentation.
    """
    offset = 0
    segmentation = np.zeros(volume.shape, dtype="uint32")

    min_object_size = kwargs.pop("min_object_size", 0)
    image_embeddings = util.precompute_image_embeddings(predictor, volume, save_path=embedding_path, ndim=3)

    for i in tqdm(range(segmentation.shape[0]), desc="Segment slices", disable=not verbose):
        segmentor.initialize(volume[i], image_embeddings=image_embeddings, verbose=False, i=i)
        seg = segmentor.generate(**kwargs)
        if len(seg) == 0:
            continue
        else:
            seg = mask_data_to_segmentation(seg, with_background=with_background, min_object_size=min_object_size)
            max_z = seg.max()
            if max_z == 0:
                continue
            seg[seg != 0] += offset
            offset = max_z + offset
        segmentation[i] = seg

    segmentation = merge_instance_segmentation_3d(
        segmentation, beta=0.5, with_background=with_background, gap_closing=gap_closing, min_z_extent=min_z_extent
    )

    return segmentation
