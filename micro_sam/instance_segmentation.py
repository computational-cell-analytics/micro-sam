from abc import ABC
from typing import List, Optional

import numpy as np
import torch
import segment_anything.utils.amg as amg_utils
import vigra

from elf.evaluation.matching import label_overlap, intersection_over_union
from elf.segmentation import embeddings as embed
from elf.segmentation.stitching import stitch_segmentation
from nifty.tools import takeDict
from scipy.optimize import linear_sum_assignment

from segment_anything.modeling import Sam
from segment_anything.predictor import SamPredictor

from skimage.transform import resize
from torchvision.ops.boxes import batched_nms, box_area

try:
    from napari.utils import progress as tqdm
except ImportError:
    from tqdm import tqdm

from . import util
from .prompt_based_segmentation import segment_from_mask

DEFAULT_OFFSETS = [[-1, 0], [0, -1], [-3, 0], [0, -3], [-9, 0], [0, -9]]


class _AMGBase(ABC):
    """
    """

    def _postprocess_batch(
        self,
        data,
        crop_box,
        orig_size,
        pred_iou_thresh,
        stability_score_thresh,
        stability_score_offset,
        box_nms_thresh,
    ):
        orig_h, orig_w = orig_size

        # filter by predicted IoU
        if pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > pred_iou_thresh
            data.filter(keep_mask)

        # calculate stability score
        data["stability_score"] = amg_utils.calculate_stability_score(
            data["masks"], self.predictor.model.mask_threshold, stability_score_offset
        )
        if stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= stability_score_thresh
            data.filter(keep_mask)

        # threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.predictor.model.mask_threshold
        data["boxes"] = amg_utils.batched_mask_to_box(data["masks"])

        # filter boxes that touch crop boundaries
        keep_mask = ~amg_utils.is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # compress to RLE
        data["masks"] = amg_utils.uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = amg_utils.mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        # remove duplicates within this crop.
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = amg_utils.uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = amg_utils.uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

        return data

    def _postprocess_small_regions(self, mask_data, min_area, nms_thresh):

        if len(mask_data["rles"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = amg_utils.rle_to_mask(rle)

            mask, changed = amg_utils.remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = amg_utils.remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = amg_utils.batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = amg_utils.mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data

    def _postprocess_masks(self, mask_data, min_mask_region_area, box_nms_thresh, crop_nms_thresh, output_mode):
        # Filter small disconnected regions and holes in masks
        if min_mask_region_area > 0:
            mask_data = self._postprocess_small_regions(
                mask_data,
                min_mask_region_area,
                max(box_nms_thresh, crop_nms_thresh),
            )

        # Encode masks
        if output_mode == "coco_rle":
            mask_data["segmentations"] = [amg_utils.coco_encode_rle(rle) for rle in mask_data["rles"]]
        elif output_mode == "binary_mask":
            mask_data["segmentations"] = [amg_utils.rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # Write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": amg_utils.area_from_rle(mask_data["rles"][idx]),
                "bbox": amg_utils.box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": amg_utils.box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)

        return curr_anns


class AutomaticMaskGenerator(_AMGBase):
    """
    """
    def __init__(
        self, model: Sam,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        crop_n_layers: int = 0,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
    ):
        if points_per_side is not None:
            self.point_grids = amg_utils.build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None or not None.")

        self.predictor = SamPredictor(model)
        self.points_per_side = points_per_side
        self.points_per_batch = points_per_batch
        self.crop_n_layers = crop_n_layers
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor

        # the mask data that is computed by 'initialize'
        self._is_initialized = False
        self._crop_list = None
        self._crop_boxes = None
        self._orig_size = None

    @property
    def is_initialized(self):
        return self._is_initialized

    @property
    def crop_list(self):
        return self._crop_list

    @property
    def crop_boxes(self):
        return self._crop_boxes

    @property
    def orig_size(self):
        return self._orig_size

    def _process_batch(self, points, im_size):
        # run model on this batch
        transformed_points = self.predictor.transform.apply_coords(points, im_size)
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        masks, iou_preds, _ = self.predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=True,
            return_logits=True,
        )

        # serialize predictions and store in MaskData
        data = amg_utils.MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
        )
        del masks

        return data

    def _process_crop(self, image, crop_box, crop_layer_idx, verbose):
        # crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        self.predictor.set_image(cropped_im)

        # get the points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # generate masks for this crop in batches
        data = amg_utils.MaskData()
        n_batches = len(points_for_image) // self.points_per_batch +\
            int(len(points_for_image) % self.points_per_batch != 0)
        for (points,) in tqdm(
            amg_utils.batch_iterator(self.points_per_batch, points_for_image),
            disable=not verbose, total=n_batches,
            desc="Predict masks for point grid prompts",
        ):
            batch_data = self._process_batch(points, cropped_im_size)
            data.cat(batch_data)
            del batch_data

        self.predictor.reset_image()
        return data

    @torch.no_grad()
    def initialize(self, image: np.ndarray, verbose=False):
        """
        """
        image = util._to_image(image)
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = amg_utils.generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )
        crop_list = []
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, verbose=verbose)
            crop_list.append(crop_data)

        self._is_initialized = True
        self._crop_list = crop_list
        self._crop_boxes = crop_boxes
        self._orig_size = orig_size

    @torch.no_grad()
    def generate(
        self,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_nms_thresh: float = 0.7,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
    ):
        if not self.is_initialized:
            raise RuntimeError("AutomaticMaskGenerator has not been initialized. Call initialize first.")
        data = amg_utils.MaskData()
        for data_, crop_box in zip(self.crop_list, self.crop_boxes):
            crop_data = self._postprocess_batch(
                data=data_, crop_box=crop_box, orig_size=self.orig_size,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                stability_score_offset=stability_score_offset,
                box_nms_thresh=box_nms_thresh
            )
            data.cat(crop_data)

        if len(self.crop_boxes) > 1:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)

        data.to_numpy()
        masks = self._postprocess_masks(data, min_mask_region_area, box_nms_thresh, crop_nms_thresh, output_mode)
        return masks


class EmbeddingBasedMaskGenerator(_AMGBase):
    """
    """
    def __init__(self):
        pass

    @torch.no_grad()
    def initialize(self, image: np.ndarray):
        pass

    @torch.no_grad()
    def generate(
        self,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
    ):
        pass


#
# Original SegmentAnything instance segmentation functionality
#


def mask_data_to_segmentation(masks, shape, with_background):
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


#
# Instance segmentation from embeddings
#


# adapted from:
# https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py#L266
def _refine_mask(
    predictor, mask, original_size,
    pred_iou_thresh, stability_score_offset,
    stability_score_thresh, seg_id, verbose,
    box_extension, use_box, use_mask, use_points,
):
    # Predict masks and store them as mask data
    masks, iou_preds, _ = segment_from_mask(
        predictor, mask, original_size=original_size,
        multimask_output=True, return_logits=True, return_all=True,
        use_box=use_box, use_mask=use_mask, use_points=use_points,
        box_extension=box_extension,
    )
    data = amg_utils.MaskData(
        masks=torch.from_numpy(masks),
        iou_preds=torch.from_numpy(iou_preds),
        seg_id=torch.from_numpy(np.full(len(masks), seg_id, dtype="int64")),
    )
    del masks

    n_masks = len(data["masks"])
    iou_preds = data["iou_preds"].numpy()
    # Filter by predicted IoU
    if pred_iou_thresh > 0.0:
        keep_mask = data["iou_preds"] > pred_iou_thresh
        data.filter(keep_mask)

        n_masks_filtered = len(data["masks"])
        if verbose > 2:
            print("Masks after IoU filter:", n_masks_filtered, "/", n_masks)
            print("IoU Threshold is:", pred_iou_thresh)

    # Calculate stability score
    data["stability_score"] = amg_utils.calculate_stability_score(
        data["masks"], predictor.model.mask_threshold, stability_score_offset
    )
    stability_scores = data["stability_score"].numpy()
    if stability_score_thresh > 0.0:
        keep_mask = data["stability_score"] >= stability_score_thresh
        data.filter(keep_mask)

        n_masks_filtered_stability = len(data["masks"])
        if verbose > 2:
            print("Masks after stability filter:", n_masks_filtered_stability, "/", n_masks_filtered)
            print("Stability Threshold is:", stability_score_thresh)

    # Threshold masks and calculate boxes
    data["masks"] = data["masks"] > predictor.model.mask_threshold
    data["boxes"] = amg_utils.batched_mask_to_box(data["masks"])

    # Compress to RLE
    data["rles"] = amg_utils.mask_to_rle_pytorch(data["masks"])
    del data["masks"]

    return (
        data,
        n_masks - n_masks_filtered,
        n_masks_filtered - n_masks_filtered_stability,
        iou_preds, stability_scores
    )


def _refine_initial_segmentation(
    predictor, initial_seg, original_size,
    box_nms_thresh, with_background,
    pred_iou_thresh, stability_score_offset,
    stability_score_thresh, verbose,
    box_extension, return_mask_data,
    use_box, use_mask, use_points,
):
    masks = amg_utils.MaskData()

    seg_ids = np.unique(initial_seg)
    n_filtered_total, n_filtered_stability_total = 0, 0
    iou_preds, stability_scores = [], []
    for seg_id in tqdm(seg_ids, disable=not bool(verbose), desc="Refine masks for automatic instance segmentation"):

        # refine the segmentations via sam for this prediction
        mask = (initial_seg == seg_id)
        assert mask.shape == (256, 256)
        (mask_data, n_filtered, n_filtered_stability,
         this_iou_pred, this_stability_score) = _refine_mask(
            predictor, mask, original_size,
            pred_iou_thresh, stability_score_offset,
            stability_score_thresh, seg_id, verbose,
            box_extension=box_extension,
            use_box=use_box, use_mask=use_mask, use_points=use_points,
        )
        n_filtered_total += n_filtered
        n_filtered_stability_total += n_filtered_stability

        iou_preds.append(this_iou_pred)
        stability_scores.append(this_stability_score)

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

    iou_preds = np.concatenate(iou_preds)
    stability_scores = np.concatenate(stability_scores)
    stability_scores = stability_scores[~np.isnan(stability_scores)]
    if verbose > 1:
        print("IoU Predictions:", np.mean(iou_preds), "+-", np.std(iou_preds))
        print(n_filtered_total, "masks were filtered out due to the IOU threshold", pred_iou_thresh)
        print("Stability scores:", np.mean(stability_scores), "+-", np.std(stability_scores))
        print(
            n_filtered_stability_total, "masks were filtered out due to the stability threshold", stability_score_thresh
        )
        print(n_masks - n_masks_filtered, "masks were filtered out by nms with threshold", box_nms_thresh)

    # get the mask outputs (area, scores and run length encoding)
    masks = [
        {
            "area": len(rle["counts"]),
            "iou_pred": iou_pred,
            "segmentation": amg_utils.rle_to_mask(rle),
            "seg_id": seg_id,
            "stability_score": stability_score,
        } for rle, seg_id, iou_pred, stability_score in zip(
            masks["rles"], masks["seg_id"], masks["iou_preds"], masks["stability_score"]
        )
    ]

    if return_mask_data:
        return masks
    # convert to instance segmentation
    segmentation = mask_data_to_segmentation(masks, original_size, with_background)
    return segmentation


def _resize_segmentation(segmentation, shape):
    longest_size = max(shape)
    longest_shape = (longest_size, longest_size)
    segmentation = resize(
        segmentation, longest_shape, order=0, preserve_range=True, anti_aliasing=False
    ).astype(segmentation.dtype)
    crop = tuple(slice(0, sh) for sh in shape)
    segmentation = segmentation[crop]
    return segmentation


def segment_instances_from_embeddings(
    predictor, image_embeddings, i=None,
    # mws settings
    offsets=DEFAULT_OFFSETS, distance_type="l2", bias=0.0,
    # sam settings
    box_nms_thresh=0.7, pred_iou_thresh=0.88,
    stability_score_thresh=0.95, stability_score_offset=1.0,
    use_box=True, use_mask=True, use_points=False,
    # general settings
    min_initial_size=5, min_size=0, with_background=False,
    verbose=1, return_initial_segmentation=False, box_extension=0.1,
    return_mask_data=False,
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
        box_extension=box_extension, return_mask_data=return_mask_data,
        use_box=use_box, use_mask=use_mask, use_points=use_points,
    )

    if min_size > 0:
        segmentation_ids, counts = np.unique(segmentation, return_counts=True)
        filter_ids = segmentation_ids[counts < min_size]
        segmentation[np.isin(segmentation, filter_ids)] = 0
        vigra.analysis.relabelConsecutive(segmentation, out=segmentation)

    if return_initial_segmentation:
        initial_segmentation = _resize_segmentation(initial_segmentation, original_size)
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
    predictor, image_embeddings, i=None, verbose=True, with_background=True,
    return_initial_segmentation=False, **kwargs,
 ):
    """
    """
    features = image_embeddings["features"]
    shape, tile_shape, halo = features.attrs["shape"], features.attrs["tile_shape"], features.attrs["halo"]

    initial_segmentations = {}

    def segment_tile(_, tile_id):
        tile_features = features[tile_id]
        tile_image_embeddings = {
            "features": tile_features,
            "input_size": tile_features.attrs["input_size"],
            "original_size": tile_features.attrs["original_size"]
        }
        if return_initial_segmentation:
            seg, initial_seg = segment_instances_from_embeddings(
                predictor, image_embeddings=tile_image_embeddings, i=i,
                with_background=with_background, verbose=verbose,
                return_initial_segmentation=True, **kwargs,
            )
            initial_segmentations[tile_id] = initial_seg
        else:
            seg = segment_instances_from_embeddings(
                predictor, image_embeddings=tile_image_embeddings, i=i,
                with_background=with_background, verbose=verbose, **kwargs,
            )
        return seg

    # fake input data
    input_ = FakeInput(shape)

    # run stitching based segmentation
    segmentation = stitch_segmentation(
        input_, segment_tile, tile_shape, halo, with_background=with_background, verbose=verbose
    )

    if return_initial_segmentation:
        initial_segmentation = stitch_segmentation(
            input_, lambda _, tile_id: initial_segmentations[tile_id],
            tile_shape, halo,
            with_background=with_background, verbose=verbose
        )
        return segmentation, initial_segmentation

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
        # TODO set to non-verbose once new napari release is out
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
