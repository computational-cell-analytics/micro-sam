"""Multi-dimensional segmentation with segment anything.
"""

import os
from typing import Any, Optional, Union

import numpy as np
import nifty
import elf.tracking.tracking_utils as track_utils
import elf.segmentation as seg_utils

from segment_anything.predictor import SamPredictor
from tqdm import tqdm

from . import util
from .instance_segmentation import AutomaticMaskGenerator, mask_data_to_segmentation
from .precompute_state import cache_amg_state
from .prompt_based_segmentation import segment_from_mask


def segment_mask_in_volume(
    segmentation: np.ndarray,
    predictor: SamPredictor,
    image_embeddings: util.ImageEmbeddings,
    segmented_slices: np.ndarray,
    stop_lower: bool,
    stop_upper: bool,
    iou_threshold: float,
    projection: Union[str, dict],
    progress_bar: Optional[Any] = None,
    box_extension: float = 0.0,
    verbose: bool = False
) -> np.ndarray:
    """Segment an object mask in in volumetric data.

    Args:
        segmentation: The initial segmentation for the object.
        predictor: The segment anything predictor.
        image_embeddings: The precomputed image embeddings for the volume.
        segmented_slices: List of slices for which this object has already been segmented.
        stop_lower: Whether to stop at the lowest segmented slice.
        stop_upper: Wheter to stop at the topmost segmented slice.
        iou_threshold: The IOU threshold for continuing segmentation across 3d.
        projection: The projection method to use. One of 'mask', 'bounding_box' or 'points'.
            - (optional: you can also pass custom choices for the projection combination)
        progress_bar: Optional progress bar.
        box_extension: Extension factor for increasing the box size after projection.

    Returns:
        Array with the volumetric segmentation
    """
    if isinstance(projection, str):
        if projection == "mask":
            use_box, use_mask, use_points = True, True, False
        elif projection == "points":
            use_box, use_mask, use_points = True, True, True
        elif projection == "bounding_box":
            use_box, use_mask, use_points = True, False, False
        else:
            raise ValueError("Choose projection method from 'mask' / 'points' / 'bounding_box'.")
    elif isinstance(projection, dict):
        assert len(projection.keys()) == 3, "There should be three parameters assigned for the projection method."
        use_box, use_mask, use_points = projection["use_box"], projection["use_mask"], projection["use_points"]
    else:
        raise ValueError(f"{projection} is not a supported projection method.")

    def _update_progress():
        if progress_bar is not None:
            progress_bar.update(1)

    def _compute_mean_iou_for_n_slices(z, increment, seg_z, n_slices):
        iou_list = [
            util.compute_iou(segmentation[z - increment * _slice], seg_z) for _slice in range(1, n_slices+1)
        ]
        return np.mean(iou_list)

    def segment_range(z_start, z_stop, increment, stopping_criterion, threshold=None, verbose=False):
        z = z_start + increment
        while True:
            if verbose:
                print(f"Segment {z_start} to {z_stop}: segmenting slice {z}")
            seg_prev = segmentation[z - increment]
            seg_z, score, _ = segment_from_mask(
                predictor, seg_prev, image_embeddings=image_embeddings, i=z, use_mask=use_mask,
                use_box=use_box, use_points=use_points, box_extension=box_extension, return_all=True
            )
            if threshold is not None:

                criterion_choice = 1

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

                if criterion < threshold:
                    msg = f"Segmentation stopped at slice {z} due to IOU {criterion} < {threshold}."
                    print(msg)
                    break
            segmentation[z] = seg_z
            z += increment
            if stopping_criterion(z, z_stop):
                if verbose:
                    print(f"Segment {z_start} to {z_stop}: stop at slice {z}")
                break
            _update_progress()

    z0, z1 = int(segmented_slices.min()), int(segmented_slices.max())

    # segment below the min slice
    if z0 > 0 and not stop_lower:
        segment_range(z0, 0, -1, np.less, iou_threshold, verbose=verbose)

    # segment above the max slice
    if z1 < segmentation.shape[0] - 1 and not stop_upper:
        segment_range(z1, segmentation.shape[0] - 1, 1, np.greater, iou_threshold, verbose=verbose)

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
                _update_progress()

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
                    _update_progress()

    return segmentation


def segment_3d_from_slice(
    predictor: SamPredictor,
    raw: np.ndarray,
    z: Optional[int] = None,
    embedding_path: Optional[Union[str, os.PathLike]] = None,
    projection: str = "mask",
    box_extension: float = 0.0,
    verbose: bool = True,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    min_object_size_z: int = 50,
    max_object_size_z: Optional[int] = None,
    iou_threshold: float = 0.8,
):
    """Segment all objects in a volume intersecting with a specific slice.

    This function first segments the objects in the specified slice using the
    automatic instance segmentation functionality. Then it segments all objects that
    were found in that slice in the volume.

    Args:
        predictor: The segment anything predictor.
        raw: The volumetric image data.
        z: The slice from which to start segmentation.
            If none is given the central slice will be used.
        embedding_path: The path were embeddings will be cached.
            If none is given embeddings will not be cached.
        projection: The projection method to use. One of 'mask', 'bounding_box' or 'points'.
        box_extension: Extension factor for increasing the box size after projection.
        verbose: Whether to print progress bar and other status messages.
        pred_iou_thresh: The predicted iou value to filter objects in `AutomaticMaskGenerator.generate`.
        stability_score_thresh: The stability score to filter objects in `AutomaticMaskGenerator.generate`.
        min_object_size_z: Minimal object size in the segmented frame.
        max_object_size_z: Maximal object size in the segmented frame.
        iou_threshold: The IOU threshold for linking objects across slices.

    Returns:
        Segmentation volume.
    """
    # Compute the image embeddings.
    image_embeddings = util.precompute_image_embeddings(predictor, raw, save_path=embedding_path, ndim=3)

    # Select the middle slice if no slice is given.
    if z is None:
        z = raw.shape[0] // 2

    # Perform automatic instance segmentation.
    if embedding_path is not None:
        amg = cache_amg_state(predictor, raw[z], image_embeddings, embedding_path, verbose=verbose, i=z)
    else:
        amg = AutomaticMaskGenerator(predictor)
        amg.initialize(raw[z], image_embeddings, i=z, verbose=verbose)

    seg_z = amg.generate(pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh)
    seg_z = mask_data_to_segmentation(
        seg_z, with_background=True,
        min_object_size=min_object_size_z,
        max_object_size=max_object_size_z,
    )

    # Segment all objects that were found in 3d.
    seg_ids = np.unique(seg_z)[1:]
    segmentation = np.zeros(raw.shape, dtype=seg_z.dtype)
    for seg_id in tqdm(seg_ids, desc="Segment objects in 3d", disable=not verbose):
        this_seg = np.zeros_like(segmentation)
        this_seg[z][seg_z == seg_id] = 1
        this_seg = segment_mask_in_volume(
            this_seg, predictor, image_embeddings,
            segmented_slices=np.array([z]), stop_lower=False, stop_upper=False,
            iou_threshold=iou_threshold, projection=projection, box_extension=box_extension,
        )
        segmentation[this_seg > 0] = seg_id

    return segmentation


def merge_instance_segmentation_3d(
    slice_segmentation: np.ndarray,
    beta: float = 0.5,
    with_background: bool = True,
):
    """Merge stacked 2d instance segmentations into a consistent 3d segmentation.

    Solves a multicut problem based on the overlap of objects to merge across z.

    Args:
        slice_segmentation: The stacked segmentation across the slices.
            We assume that the segmentation is labeled consecutive across z.
        beta: The bias term for the multicut. Higher values lead to a larger
            degree of over-segmentation and vice versa.
        with_background: Whether this is a segmentation problem with background.
            In that case all edges connecting to the background are set to be repulsive.
    """

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
    return segmentation
