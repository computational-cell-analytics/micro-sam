import numpy as np
import torch
import vigra

from elf.evaluation.matching import label_overlap, intersection_over_union
from elf.segmentation import embeddings as embed
from elf.segmentation.stitching import stitch_segmentation
from nifty.tools import takeDict
from scipy.optimize import linear_sum_assignment

from segment_anything import SamAutomaticMaskGenerator
from segment_anything.utils.amg import (
    MaskData,
    batched_mask_to_box,
    calculate_stability_score,
    mask_to_rle_pytorch,
    rle_to_mask,
)
from skimage.transform import resize
from torchvision.ops.boxes import batched_nms

try:
    from napari.utils import progress as tqdm
except ImportError:
    from tqdm import tqdm

from . import util
from .segment_from_prompts import segment_from_mask

DEFAULT_OFFSETS = [[-1, 0], [0, -1], [-3, 0], [0, -3], [-9, 0], [0, -9]]


#
# Original SegmentAnything instance segmentation functionality
#


def _amg_to_seg(masks, shape, with_background):
    """Convert the output of the automatic mask generation to an instance segmentation."""

    masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
    segmentation = np.zeros(shape[:2], dtype="uint32")

    for seg_id, mask in enumerate(masks, 1):
        segmentation[mask["segmentation"]] = seg_id

    if with_background:
        seg_ids, sizes = np.unique(segmentation, return_counts=True)
        bg_id = seg_ids[np.argmax(sizes)]
        if bg_id != 0:
            segmentation[segmentation == bg_id] = 0
        vigra.analysis.relabelConsecutive(segmentation, out=segmentation)

    return segmentation


def segment_instances_sam(sam, image, with_background=False, **kwargs):
    segmentor = SamAutomaticMaskGenerator(sam, **kwargs)
    image_ = util._to_image(image)
    masks = segmentor.generate(image_)
    segmentation = _amg_to_seg(masks, image.shape, with_background)
    return segmentation


#
# Instance segmentation from embeddings
#


# adapted from:
# https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py#L266
def _refine_mask(
    predictor, mask, original_size,
    pred_iou_thresh, stability_score_offset,
    stability_score_thresh, seg_id, verbose,
):
    # Predict masks and store them as mask data
    masks, iou_preds, _ = segment_from_mask(
        predictor, mask, original_size=original_size,
        multimask_output=True, return_logits=True, return_all=True,
        use_box=True, use_mask=True, use_points=False,
        box_extension=4
    )
    data = MaskData(
        masks=torch.from_numpy(masks),
        iou_preds=torch.from_numpy(iou_preds),
        seg_id=torch.from_numpy(np.full(len(masks), seg_id, dtype="int64")),
    )
    del masks

    n_masks = len(data["masks"])
    # Filter by predicted IoU
    if pred_iou_thresh > 0.0:
        keep_mask = data["iou_preds"] > pred_iou_thresh
        data.filter(keep_mask)

        n_masks_filtered = len(data["masks"])
        if verbose > 2:
            print("Masks after IoU filter:", n_masks_filtered, "/", n_masks)
            print("IoU Threshold is:", pred_iou_thresh)

    # Calculate stability score
    data["stability_score"] = calculate_stability_score(
        data["masks"], predictor.model.mask_threshold, stability_score_offset
    )
    if stability_score_thresh > 0.0:
        keep_mask = data["stability_score"] >= stability_score_thresh
        data.filter(keep_mask)

        n_masks_filtered_stability = len(data["masks"])
        if verbose > 2:
            print("Masks after stability filter:", n_masks_filtered_stability, "/", n_masks_filtered)
            print("Stability Threshold is:", stability_score_thresh)

    # Threshold masks and calculate boxes
    data["masks"] = data["masks"] > predictor.model.mask_threshold
    data["boxes"] = batched_mask_to_box(data["masks"])

    # Compress to RLE
    data["rles"] = mask_to_rle_pytorch(data["masks"])
    del data["masks"]

    return data, n_masks - n_masks_filtered, n_masks_filtered - n_masks_filtered_stability


def _refine_initial_segmentation(
    predictor, initial_seg, original_size,
    box_nms_thresh, with_background,
    pred_iou_thresh, stability_score_offset,
    stability_score_thresh, verbose
):
    masks = MaskData()

    seg_ids = np.unique(initial_seg)
    n_filtered_total, n_filtered_stability_total = 0, 0
    for seg_id in tqdm(seg_ids, disable=not bool(verbose), desc="Refine masks for automatic instance segmentation"):

        # refine the segmentations via sam for this prediction
        mask = (initial_seg == seg_id)
        assert mask.shape == (256, 256)
        mask_data, n_filtered, n_filtered_stability = _refine_mask(
            predictor, mask, original_size,
            pred_iou_thresh, stability_score_offset,
            stability_score_thresh, seg_id, verbose,
        )
        n_filtered_total += n_filtered
        n_filtered_stability_total += n_filtered_stability

        # append to the mask data
        masks.cat(mask_data)

    # apply non-max-suppression to only keep the likely objects
    n_masks = len(masks["boxes"])
    keep_by_nms = batched_nms(
        masks["boxes"].float(),
        masks["iou_preds"],
        torch.zeros_like(masks["boxes"][:, 0]),  # categories
        iou_threshold=box_nms_thresh,
    )
    masks.filter(keep_by_nms)
    n_masks_filtered = len(masks["boxes"])

    if verbose > 1:
        print(n_filtered_total, "masks were filtered out due to the IOU threshold", pred_iou_thresh)
        print(
            n_filtered_stability_total, "masks were filtered out due to the stability threshold", stability_score_thresh
        )
        print(n_masks - n_masks_filtered, "masks were filtered out by nms with threshold", box_nms_thresh)

    # get the mask output (binary masks and area
    masks = [{"segmentation": rle_to_mask(rle), "area": len(rle), "seg_id": seg_id}
             for rle, seg_id in zip(masks["rles"], masks["seg_id"])]

    # convert to instance segmentation
    segmentation = _amg_to_seg(masks, original_size, with_background)
    return segmentation


def segment_instances_from_embeddings(
    predictor, image_embeddings, i=None,
    # mws settings
    offsets=DEFAULT_OFFSETS, distance_type="l2", bias=0.0,
    # sam settings
    box_nms_thresh=0.7, pred_iou_thresh=0.88,
    stability_score_thresh=0.95, stability_score_offset=1.0,
    # general settings
    min_initial_size=10, min_size=0, with_background=False,
    verbose=1, return_initial_segmentation=False,
):
    """
    """
    util.set_precomputed(predictor, image_embeddings, i)

    embeddings = predictor.get_image_embedding().squeeze().cpu().numpy()
    assert embeddings.shape == (256, 64, 64), f"{embeddings.shape}"

    initial_segmentation = embed.segment_embeddings_mws(
        embeddings, distance_type=distance_type, offsets=offsets, bias=bias,
    ).astype("uint32")
    assert initial_segmentation.shape == (64, 64), f"{initial_segmentation.shape}"

    # filter out small initial objects
    if min_initial_size > 0:
        seg_ids, sizes = np.unique(initial_segmentation, return_counts=True)
        initial_segmentation[np.isin(initial_segmentation, seg_ids[sizes < min_initial_size])] = 0

    # resize to 256 x 256, which is the mask input expected by SAM
    initial_segmentation = resize(
        initial_segmentation, (256, 256), order=0, preserve_range=True, anti_aliasing=False
    ).astype(initial_segmentation.dtype)

    original_size = image_embeddings["original_size"]
    segmentation = _refine_initial_segmentation(
        predictor, initial_segmentation, original_size,
        box_nms_thresh=box_nms_thresh, with_background=with_background,
        pred_iou_thresh=pred_iou_thresh, stability_score_offset=stability_score_offset,
        stability_score_thresh=stability_score_thresh, verbose=verbose,
    )

    if min_size > 0:
        segmentation_ids, counts = np.unique(segmentation, return_counts=True)
        filter_ids = segmentation_ids[counts < min_size]
        segmentation[np.isin(segmentation, filter_ids)] = 0
        vigra.analysis.relabelConsecutive(segmentation, out=segmentation)

    if return_initial_segmentation:
        initial_segmentation = resize(
            initial_segmentation, segmentation.shape, order=0, preserve_range=True, anti_aliasing=False
        ).astype(segmentation.dtype)
        return segmentation, initial_segmentation
    else:
        return segmentation


class FakeInput:
    def __init__(self, shape):
        self.shape = shape

    def __getitem__(self, index):
        block_shape = tuple(ind.stop - ind.start for ind in index)
        return np.zeros(block_shape, dtype="float32")


def segment_instances_from_embeddings_with_tiling(
    predictor, image_embeddings, i=None, verbose=True, with_background=True, **kwargs
 ):
    """
    """
    features = image_embeddings["features"]
    shape, tile_shape, halo = features.attrs["shape"], features.attrs["tile_shape"], features.attrs["halo"]

    def segment_tile(_, tile_id):
        tile_features = features[tile_id]
        tile_image_embeddings = {
            "features": tile_features,
            "input_size": tile_features.attrs["input_size"],
            "original_size": tile_features.attrs["original_size"]
        }
        seg = segment_instances_from_embeddings(
            predictor, image_embeddings=tile_image_embeddings, i=i,
            with_background=with_background, verbose=True, **kwargs,
        )
        return seg

    # fake input data
    input_ = FakeInput(shape)

    # run stitching based segmentation
    segmentation = stitch_segmentation(
        input_, segment_tile, tile_shape, halo, with_background=with_background, verbose=verbose
    )
    return segmentation


# this is still experimental and not yet ready to be integrated within the annotator_3d
# (will need to see how well it works with retrained models)
def segment_instances_from_embeddings_3d(predictor, image_embeddings, verbose=1, iou_threshold=0.50, **kwargs):
    """
    """
    if image_embeddings["original_size"] is None:  # tiled embeddings
        is_tiled = True
        image_shape = tuple(image_embeddings["features"].attrs["shape"])
        n_slices = len(image_embeddings["features"][0])

    else:  # normal embeddings (not tiled)
        is_tiled = False
        image_shape = tuple(image_embeddings["original_size"])
        n_slices = image_embeddings["features"].shape[0]

    shape = (n_slices,) + image_shape
    segmentation_function = segment_instances_from_embeddings_with_tiling if is_tiled else\
        segment_instances_from_embeddings

    segmentation = np.zeros(shape, dtype="uint32")

    def match_segments(seg, prev_seg):
        overlap, ignore_idx = label_overlap(seg, prev_seg, ignore_label=0)
        scores = intersection_over_union(overlap)
        # remove ignore_label (remapped to continuous object_ids)
        if ignore_idx[0] is not None:
            scores = np.delete(scores, ignore_idx[0], axis=0)
        if ignore_idx[1] is not None:
            scores = np.delete(scores, ignore_idx[1], axis=1)

        n_matched = min(scores.shape)
        no_match = n_matched == 0 or (not np.any(scores >= iou_threshold))

        max_id = segmentation.max()
        if no_match:
            seg[seg != 0] += max_id

        else:
            # compute optimal matching with scores as tie-breaker
            costs = -(scores >= iou_threshold).astype(float) - scores / (2*n_matched)
            seg_ind, prev_ind = linear_sum_assignment(costs)

            seg_ids, prev_ids = np.unique(seg)[1:], np.unique(prev_seg)[1:]
            match_ok = scores[seg_ind, prev_ind] >= iou_threshold

            id_updates = {0: 0}
            matched_ids, matched_prev = seg_ids[seg_ind[match_ok]], prev_ids[prev_ind[match_ok]]
            id_updates.update(
                {seg_id: prev_id for seg_id, prev_id in zip(matched_ids, matched_prev) if seg_id != 0}
            )

            unmatched_ids = np.setdiff1d(seg_ids, np.concatenate([np.zeros(1, dtype=matched_ids.dtype), matched_ids]))
            id_updates.update({seg_id: max_id + i for i, seg_id in enumerate(unmatched_ids, 1)})

            seg = takeDict(id_updates, seg)

        return seg

    ids_to_slices = {}
    # segment the objects starting from slice 0
    for z in tqdm(
        range(0, n_slices), total=n_slices, desc="Run instance segmentation in 3d", disable=not bool(verbose)
    ):
        # TODO set to non verbose once the fix is in new napari version
        seg = segmentation_function(predictor, image_embeddings, i=z, verbose=True, **kwargs)
        if z > 0:
            prev_seg = segmentation[z - 1]
            seg = match_segments(seg, prev_seg)

        # keep track of the slices per object id to get rid of unconnected objects in the post-processing
        this_ids = np.unique(seg)[1:]
        for id_ in this_ids:
            ids_to_slices[id_] = ids_to_slices.get(id_, []) + [z]

        segmentation[z] = seg

    # get rid of objects that are just in a single slice
    filter_objects = [seg_id for seg_id, slice_list in ids_to_slices.items() if len(slice_list) == 1]
    segmentation[np.isin(segmentation, filter_objects)] = 0
    vigra.analysis.relabelConsecutive(segmentation, out=segmentation)

    return segmentation
