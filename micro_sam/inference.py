import os
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from nifty.tools import blocking

import segment_anything.utils.amg as amg_utils
from segment_anything import SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

from . import util
from .instance_segmentation import mask_data_to_segmentation
from ._vendored import batched_mask_to_box


def _validate_inputs(
    boxes, points, point_labels, multimasking, return_instance_segmentation, segmentation_ids, logits_masks
):
    if multimasking and (segmentation_ids is not None) and (not return_instance_segmentation):
        raise NotImplementedError

    if (points is None) != (point_labels is None):
        raise ValueError(
            "If you have point prompts both `points` and `point_labels` have to be passed, "
            "but you passed only one of them."
        )

    have_points = points is not None
    have_boxes = boxes is not None
    have_logits = logits_masks is not None
    if (not have_points) and (not have_boxes):
        raise ValueError("Point and/or box prompts have to be passed, you passed neither.")

    if have_points and (len(point_labels) != len(points)):
        raise ValueError(
            f"The number of point coordinates and labels does not match: {len(point_labels)} != {len(points)}"
        )

    if (have_points and have_boxes) and (len(points) != len(boxes)):
        raise ValueError(
            f"The number of point and box prompts does not match: {len(points)} != {len(boxes)}"
        )

    if have_logits:
        if have_points and (len(logits_masks) != len(point_labels)):
            raise ValueError(
                f"The number of point and logits does not match: {len(points) != len(logits_masks)}"
            )
        elif have_boxes and (len(logits_masks) != len(boxes)):
            raise ValueError(
                f"The number of boxes and logits does not match: {len(boxes)} != {len(logits_masks)}"
            )

    n_prompts = boxes.shape[0] if have_boxes else points.shape[0]

    if (segmentation_ids is not None) and (len(segmentation_ids) != n_prompts):
        raise ValueError(
            f"The number of segmentation ids and prompts does not match: {len(segmentation_ids)} != {n_prompts}"
        )

    return n_prompts, have_boxes, have_points, have_logits


def _local_otsu_threshold(images: torch.Tensor, window_size: int = 31, num_bins: int = 64, eps: float = 1e-6):
    x = images
    B, _, H, W = x.shape
    device = x.device

    # Work in float32 for stability even if input is fp16
    x = x.to(torch.float32)

    # --- per-image min/max for normalization to [0, 1] ---
    x_flat = x.view(B, -1)
    x_min = x_flat.min(dim=1).values.view(B, 1, 1, 1)
    x_max = x_flat.max(dim=1).values.view(B, 1, 1, 1)
    x_range = (x_max - x_min).clamp_min(eps)

    x_norm = (x - x_min) / x_range  # (B,1,H,W), in [0,1]

    # --- extract local patches via unfold ---
    pad = window_size // 2
    patches = F.unfold(x_norm, kernel_size=window_size, padding=pad)  # (B, P, L)
    # P = window_size * window_size, L = H * W
    B_, P, L = patches.shape

    # --- quantize to bins ---
    bin_idx = (patches * (num_bins - 1)).long().clamp(0, num_bins - 1)  # (B, P, L)

    # --- build histograms per patch ---
    # one_hot: (B, L, num_bins)
    one_hot = torch.zeros(B, L, num_bins, device=device, dtype=torch.float32)
    idx = bin_idx.transpose(1, 2)  # (B, L, P)
    src = torch.ones_like(idx, dtype=one_hot.dtype)  # (B, L, P)
    one_hot.scatter_add_(2, idx, src)
    # hist: (B, num_bins, L)
    hist = one_hot.permute(0, 2, 1)

    # --- Otsu per patch (vectorized) ---
    # p: (B, bins, L)
    p = hist / hist.sum(dim=1, keepdim=True).clamp_min(eps)

    bins = torch.arange(num_bins, device=device, dtype=torch.float32).view(1, num_bins, 1)

    omega1 = torch.cumsum(p, dim=1)              # (B, bins, L)
    mu = torch.cumsum(p * bins, dim=1)           # (B, bins, L)
    mu_T = mu[:, -1:, :]                         # (B, 1, L)

    omega2 = 1.0 - omega1

    mu1 = mu / omega1.clamp_min(eps)
    mu2 = (mu_T - mu) / omega2.clamp_min(eps)

    sigma_b2 = omega1 * omega2 * (mu1 - mu2) ** 2  # (B, bins, L)

    # argmax over bins gives local threshold bin per patch
    t_bin = torch.argmax(sigma_b2, dim=1)        # (B, L)
    t_norm = t_bin.to(torch.float32) / (num_bins - 1)  # normalized [0,1]

    # --- map thresholds back to original intensity scale (per-image) ---
    # x_min, x_range: (B,1,1,1) -> flatten batch dims
    thr_vals = x_min.view(B, 1) + t_norm * x_range.view(B, 1)  # (B, L)
    # clamp to >= 0 because foreground is positive
    thr_vals = thr_vals.clamp_min(0.0)

    thresholds = thr_vals.view(B, H, W)
    # Take the spatial max over the thresholds.
    thresholds = torch.amax(thresholds, dim=(1, 2), keepdims=True)
    return thresholds


def _process_masks_for_batch(batch_masks, batch_ious, batch_logits, return_highres_logits, mask_threshold):
    batch_data = amg_utils.MaskData(masks=batch_masks.flatten(0, 1), iou_preds=batch_ious.flatten(0, 1))
    batch_data["logits"] = batch_masks.clone() if return_highres_logits else batch_logits
    if mask_threshold == "auto":
        thresholds = _local_otsu_threshold(batch_logits)
        batch_data["stability_scores"] = amg_utils.calculate_stability_score(batch_data["masks"], thresholds, 1.0)
        batch_data["masks"] = (batch_data["masks"] > thresholds).type(torch.bool)
    else:
        batch_data["stability_scores"] = amg_utils.calculate_stability_score(batch_data["masks"], mask_threshold, 1.0)
        batch_data["masks"] = (batch_data["masks"] > mask_threshold).type(torch.bool)
    batch_data["boxes"] = batched_mask_to_box(batch_data["masks"])
    return batch_data


@torch.no_grad()
def batched_inference(
    predictor: SamPredictor,
    image: Optional[np.ndarray],
    batch_size: int,
    boxes: Optional[np.ndarray] = None,
    points: Optional[np.ndarray] = None,
    point_labels: Optional[np.ndarray] = None,
    multimasking: bool = False,
    embedding_path: Optional[Union[str, os.PathLike]] = None,
    return_instance_segmentation: bool = True,
    segmentation_ids: Optional[list] = None,
    reduce_multimasking: bool = True,
    logits_masks: Optional[torch.Tensor] = None,
    verbose_embeddings: bool = True,
    mask_threshold: Optional[Union[float, str]] = None,
    return_highres_logits: bool = False,
) -> Union[List[List[Dict[str, Any]]], np.ndarray]:
    """Run batched inference for input prompts.

    Args:
        predictor: The Segment Anything predictor.
        image: The input image. If None, we assume that the image embeddings have already been computed.
        batch_size: The batch size to use for inference.
        boxes: The box prompts. Array of shape N_PROMPTS x 4.
            The bounding boxes are represented by [MIN_X, MIN_Y, MAX_X, MAX_Y].
        points: The point prompt coordinates. Array of shape N_PROMPTS x 1 x 2.
            The points are represented by their coordinates [X, Y], which are given in the last dimension.
        point_labels: The point prompt labels. Array of shape N_PROMPTS x 1.
            The labels are either 0 (negative prompt) or 1 (positive prompt).
        multimasking: Whether to predict with 3 or 1 mask. By default, set to 'False'.
        embedding_path: Cache path for the image embeddings. By default, computed on-the-fly.
        return_instance_segmentation: Whether to return a instance segmentation
            or the individual mask data. By default, set to 'True'.
        segmentation_ids: Fixed segmentation ids to assign to the masks
            derived from the prompts.
        reduce_multimasking: Whether to choose the most likely masks with
            highest ious from multimasking. By default, set to 'True'.
        logits_masks: The logits masks. Array of shape N_PROMPTS x 1 x 256 x 256.
            Whether to use the logits masks from previous segmentation.
        verbose_embeddings: Whether to show progress outputs of computing image embeddings.
            By default, set to 'True'.
        mask_threshold: The theshold for binarizing masks based on the predicted values.
            If None, the default threshold 0 is used. If "auto" is passed then the threshold is
            determined with a local otsu filter.
        return_highres_logits: Wheher to return high-resolution logits.

    Returns:
        The predicted segmentation masks.
    """
    n_prompts, have_boxes, have_points, have_logits = _validate_inputs(
        boxes, points, point_labels, multimasking, return_instance_segmentation, segmentation_ids, logits_masks
    )

    # Compute the image embeddings.
    if image is None:  # This means the image embeddings are computed already.
        # Call get image embeddings, this will throw an error if they have not yet been computed.
        predictor.get_image_embedding()
    else:
        image_embeddings = util.precompute_image_embeddings(
            predictor, image, embedding_path, ndim=2, verbose=verbose_embeddings
        )
        util.set_precomputed(predictor, image_embeddings)

    # Determine the number of batches.
    n_batches = int(np.ceil(float(n_prompts) / batch_size))

    # Preprocess the prompts.
    device = predictor.device
    transform_function = ResizeLongestSide(1024)
    image_shape = predictor.original_size
    if have_boxes:
        boxes = transform_function.apply_boxes(boxes, image_shape)
        boxes = torch.tensor(boxes, dtype=torch.float32).to(device)
    if have_points:
        points = transform_function.apply_coords(points, image_shape)
        points = torch.tensor(points, dtype=torch.float32).to(device)
        point_labels = torch.tensor(point_labels, dtype=torch.float32).to(device)

    masks = amg_utils.MaskData()
    mask_threshold = predictor.model.mask_threshold if mask_threshold is None else mask_threshold
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_stop = min((batch_idx + 1) * batch_size, n_prompts)

        batch_boxes = boxes[batch_start:batch_stop] if have_boxes else None
        batch_points = points[batch_start:batch_stop] if have_points else None
        batch_labels = point_labels[batch_start:batch_stop] if have_points else None
        batch_logits = logits_masks[batch_start:batch_stop] if have_logits else None

        batch_masks, batch_ious, batch_logits = predictor.predict_torch(
            point_coords=batch_points,
            point_labels=batch_labels,
            boxes=batch_boxes,
            mask_input=batch_logits,
            multimask_output=multimasking,
            return_logits=True,
        )

        # If we expect to reduce the masks from multimasking and use multi-masking,
        # then we need to select the most likely mask (according to the predicted IOU) here.
        if reduce_multimasking and multimasking:
            _, max_index = batch_ious.max(axis=1)
            batch_masks = torch.cat([batch_masks[i, max_id][None] for i, max_id in enumerate(max_index)]).unsqueeze(1)
            batch_ious = torch.cat([batch_ious[i, max_id][None] for i, max_id in enumerate(max_index)]).unsqueeze(1)
            batch_logits = torch.cat([batch_logits[i, max_id][None] for i, max_id in enumerate(max_index)]).unsqueeze(1)

        batch_data = _process_masks_for_batch(
            batch_masks, batch_ious, batch_logits, return_highres_logits, mask_threshold
        )
        masks.cat(batch_data)

    # Mask data to records.
    masks = [
        {
            "segmentation": masks["masks"][idx],
            "area": masks["masks"][idx].sum(),
            "bbox": amg_utils.box_xyxy_to_xywh(masks["boxes"][idx]).tolist(),
            "predicted_iou": masks["iou_preds"][idx].item(),
            "stability_score": masks["stability_scores"][idx].item(),
            "seg_id": idx + 1 if segmentation_ids is None else int(segmentation_ids[idx]),
            "logits": masks["logits"][idx]
        }
        for idx in range(len(masks["masks"]))
    ]

    if return_instance_segmentation:
        masks = mask_data_to_segmentation(masks, with_background=False, min_object_size=0)
    return masks


def _require_tiled_embeddings(
    predictor, image, image_embeddings, embedding_path, tile_shape, halo, verbose_embeddings
):
    if image_embeddings is None:
        assert image is not None
        assert (tile_shape is not None) and (halo is not None)
        shape = image.shape
        image_embeddings = util.precompute_image_embeddings(
            predictor, image, embedding_path, ndim=2, tile_shape=tile_shape, halo=halo, verbose=verbose_embeddings
        )
    else:  # This means the image embeddings are computed already.
        attrs = image_embeddings["features"].attrs
        tile_shape_, halo_ = attrs["tile_shape"], attrs["halo"]
        shape = attrs["shape"]
        if tile_shape is None:
            tile_shape = tile_shape_
        elif any(ts != ts_ for ts, ts_ in zip(tile_shape, tile_shape_)):
            raise ValueError(f"Incompatible tile shapes: {tile_shape} != {tile_shape_}")
        if halo is None:
            halo = halo_
        elif any(ts != ts_ for ts, ts_ in zip(halo, halo_)):
            raise ValueError(f"Incompatible tile shapes: {halo} != {halo_}")

    return image_embeddings, shape, tile_shape, halo


@torch.no_grad()
def batched_tiled_inference(
    predictor: SamPredictor,
    image: Optional[np.ndarray],
    batch_size: int,
    image_embeddings: Optional[util.ImageEmbeddings] = None,
    boxes: Optional[np.ndarray] = None,
    points: Optional[np.ndarray] = None,
    point_labels: Optional[np.ndarray] = None,
    multimasking: bool = False,
    embedding_path: Optional[Union[str, os.PathLike]] = None,
    return_instance_segmentation: bool = True,
    reduce_multimasking: bool = True,
    logits_masks: Optional[torch.Tensor] = None,
    verbose_embeddings: bool = True,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
) -> Union[List[List[Dict[str, Any]]], np.ndarray]:
    """
    """
    # Validate inputs and get input prompt summary.
    segmentation_ids = None
    n_prompts, have_boxes, have_points, have_logits = _validate_inputs(
        boxes, points, point_labels, multimasking, return_instance_segmentation, segmentation_ids, logits_masks
    )
    if have_logits:
        raise NotImplementedError

    # Get the tiling parameters and compute embeddings if needed.
    image_embeddings, shape, tile_shape, halo = _require_tiled_embeddings(
        predictor, image, image_embeddings, embedding_path, tile_shape, halo, verbose_embeddings
    )

    # Order the prompts by tile and then iterate over the tiles.
    tiling = blocking([0, 0], shape, tile_shape)
    box_to_tile, point_to_tile, label_to_tile, logits_to_tile = {}, {}, {}, {}
    tile_ids = []

    # Box prompts are in the format N x 4. Boxes are stored in order [MIN_X, MIN_Y, MAX_X, MAX_Y].
    # Points are in the Format N x 1 x 2 with coordinate order X, Y.
    # Point labels are in the format N x 1.
    # Mask prompts are in the format N x 1 x 256 x 256.
    for prompt_id in range(n_prompts):
        this_tile_id = None

        if have_boxes:
            box = boxes[prompt_id]
            center = np.array([(box[1] + box[3]) / 2, (box[0] + box[2]) / 2]).round().astype("int").tolist()
            this_tile_id = tiling.coordinatesToBlockId(center)
            tile = tiling.getBlockWithHalo(this_tile_id, list(halo)).outerBlock
            offset = tile.begin
            this_tile_shape = tile.shape
            box_in_tile = np.array(
                [
                    max(box[1] - offset[0], 0), max(box[0] - offset[1], 0),
                    min(box[3] - offset[0], this_tile_shape[0]), min(box[2] - offset[1], this_tile_shape[1])
                ]
            )[None]
            if this_tile_id in box_to_tile:
                box_to_tile[this_tile_id] = np.concatenate([box_to_tile[this_tile_id], box_in_tile])
            else:
                box_to_tile[this_tile_id] = box_in_tile

        if have_points:
            point = points[prompt_id, 0][::-1].round().astype("int").tolist()
            if this_tile_id is None:
                this_tile_id = tiling.coordinatesToBlockId(point)
            else:
                assert this_tile_id == tiling.coordinatesToBlockId(point)
            tile = tiling.getBlockWithHalo(this_tile_id, list(halo)).outerBlock
            offset = tile.begin
            point_in_tile = (points[prompt_id, 0] - np.array(offset)[::-1])[None, None]
            label_in_tile = point_labels[prompt_id][None]
            if this_tile_id in point_to_tile:
                point_to_tile[this_tile_id] = np.concatenate([point_to_tile[this_tile_id], point_in_tile])
                label_to_tile[this_tile_id] = np.concatenate([label_to_tile[this_tile_id], label_in_tile])
            else:
                point_to_tile[this_tile_id] = point_in_tile
                label_to_tile[this_tile_id] = label_in_tile

        # NOTE: logits are not yet supported.
        tile_ids.append(this_tile_id)

    # Find the tiles with prompts.
    tile_ids = sorted(list(set(tile_ids)))

    # Run batched inference for each tile.
    masks = []
    for tile_id in tile_ids:
        # Get the prompts for this tile.
        tile_boxes = box_to_tile.get(tile_id)
        tile_logits = logits_to_tile.get(tile_id)
        tile_points, tile_labels = point_to_tile.get(tile_id), label_to_tile.get(tile_id)

        # Set the correct embeddings, run inference.
        predictor = util.set_precomputed(predictor, image_embeddings, tile_id=tile_id)
        this_masks = batched_inference(
            predictor=predictor,
            image=None,
            batch_size=batch_size,
            boxes=tile_boxes,
            points=tile_points,
            point_labels=tile_labels,
            multimasking=multimasking,
            return_instance_segmentation=False,
            segmentation_ids=segmentation_ids,
            reduce_multimasking=reduce_multimasking,
            logits_masks=tile_logits,
        )

        # Take care of offsets for the current tile.
        tile = tiling.getBlockWithHalo(tile_id, list(halo)).outerBlock
        offset = tile.begin

        # TODO: this is an inefficient work-around. Instead of uncropping all the masks we should
        # store the offset of this tile and only fuse everything back to the full shape in the output
        # segmentation image. This should be updated for all the tiled segmentation functions.
        extended_masks = []
        # TODO
        for mask_data in this_masks:
            seg = mask_data.pop("segmentation")
            bbox = mask_data.pop("bbox")
            breakpoint()
            extended_mask = {
                "segmentation": seg,
                "bbox": bbox,
            }
            extended_mask.update(**mask_data)
            extended_masks.append(extended_mask)

        # "segmentation": masks["masks"][idx],
        # "bbox": amg_utils.box_xyxy_to_xywh(masks["boxes"][idx]).tolist(),
        masks.extend(extended_masks)

    if return_instance_segmentation:
        masks = mask_data_to_segmentation(masks, with_background=False, min_object_size=0)
    return masks
