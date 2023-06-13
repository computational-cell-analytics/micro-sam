import numpy as np
import torch
import vigra

from elf.segmentation import embeddings as embed
from elf.segmentation.stitching import stitch_segmentation

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
    pred_iou_thresh, stability_score_offset, stability_score_thresh,
):
    # Predict masks and store them as mask data
    masks, iou_preds, _ = segment_from_mask(
        predictor, mask, original_size=original_size,
        multimask_output=True, return_logits=True, return_all=True,
        box_extension=4
    )
    data = MaskData(
        masks=torch.from_numpy(masks),
        iou_preds=torch.from_numpy(iou_preds),
    )
    del masks

    # Filter by predicted IoU
    if pred_iou_thresh > 0.0:
        keep_mask = data["iou_preds"] > pred_iou_thresh
        data.filter(keep_mask)

    # Calculate stability score
    data["stability_score"] = calculate_stability_score(
        data["masks"], predictor.model.mask_threshold, stability_score_offset
    )
    if stability_score_thresh > 0.0:
        keep_mask = data["stability_score"] >= stability_score_thresh
        data.filter(keep_mask)

    # Threshold masks and calculate boxes
    data["masks"] = data["masks"] > predictor.model.mask_threshold
    data["boxes"] = batched_mask_to_box(data["masks"])

    # Compress to RLE
    data["rles"] = mask_to_rle_pytorch(data["masks"])
    del data["masks"]

    return data


def _refine_initial_segmentation(
    predictor, initial_seg, original_size,
    box_nms_thresh, with_background,
    pred_iou_thresh, stability_score_offset, stability_score_thresh, verbose
):
    masks = MaskData()

    seg_ids = np.unique(initial_seg)
    for seg_id in tqdm(seg_ids, disable=not verbose, desc="Refine masks for automatic instance segmentation"):

        # refine the segmentations via sam for this prediction
        mask = (initial_seg == seg_id)
        assert mask.shape == (256, 256)
        mask_data = _refine_mask(
            predictor, mask, original_size,
            pred_iou_thresh, stability_score_offset, stability_score_thresh
        )

        # append to the mask data
        masks.cat(mask_data)

    # apply non-max-suppression to only keep the likely objects
    keep_by_nms = batched_nms(
        masks["boxes"].float(),
        masks["iou_preds"],
        torch.zeros_like(masks["boxes"][:, 0]),  # categories
        iou_threshold=box_nms_thresh,
    )
    masks.filter(keep_by_nms)

    # get the mask output (binary masks and area
    masks = [{"segmentation": rle_to_mask(rle), "area": len(rle)} for rle in masks["rles"]]

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
    verbose=True, return_initial_seg=False,
):
    """
    """
    util.set_precomputed(predictor, image_embeddings, i)

    embeddings = predictor.get_image_embedding().squeeze().cpu().numpy()
    assert embeddings.shape == (256, 64, 64), f"{embeddings.shape}"
    initial_seg = embed.segment_embeddings_mws(
        embeddings, distance_type=distance_type, offsets=offsets, bias=bias
    ).astype("uint32")
    assert initial_seg.shape == (64, 64), f"{initial_seg.shape}"

    # filter out small initial objects
    if min_initial_size > 0:
        seg_ids, sizes = np.unique(initial_seg, return_counts=True)
        initial_seg[np.isin(initial_seg, seg_ids[sizes < min_initial_size])] = 0
        vigra.analysis.relabelConsecutive(initial_seg, out=initial_seg)

    # resize to 256 x 256, which is the mask input expected by SAM
    initial_seg = resize(
        initial_seg, (256, 256), order=0, preserve_range=True, anti_aliasing=False
    ).astype(initial_seg.dtype)

    original_size = image_embeddings["original_size"]
    seg = _refine_initial_segmentation(
        predictor, initial_seg, original_size,
        box_nms_thresh=box_nms_thresh, with_background=with_background,
        pred_iou_thresh=pred_iou_thresh, stability_score_offset=stability_score_offset,
        stability_score_thresh=stability_score_thresh, verbose=verbose,
    )

    if min_size > 0:
        seg_ids, counts = np.unique(seg, return_counts=True)
        filter_ids = seg_ids[counts < min_size]
        seg[np.isin(seg, filter_ids)] = 0
        vigra.analysis.relabelConsecutive(seg, out=seg)

    if return_initial_seg:
        initial_seg = resize(
            initial_seg, seg.shape, order=0, preserve_range=True, anti_aliasing=False
        ).astype(seg.dtype)
        return seg, initial_seg
    else:
        return seg


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
