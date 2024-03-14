import os
from typing import Optional, Union

import torch
import numpy as np

import segment_anything.utils.amg as amg_utils
from segment_anything import SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

from . import util
from .instance_segmentation import mask_data_to_segmentation
from ._vendored import batched_mask_to_box


@torch.no_grad()
def batched_inference(
    predictor: SamPredictor,
    image: np.ndarray,
    batch_size: int,
    boxes: Optional[np.ndarray] = None,
    points: Optional[np.ndarray] = None,
    point_labels: Optional[np.ndarray] = None,
    multimasking: bool = False,
    embedding_path: Optional[Union[str, os.PathLike]] = None,
    return_instance_segmentation: bool = True,
    segmentation_ids: Optional[list] = None,
    reduce_multimasking: bool = True
):
    """Run batched inference for input prompts.

    Args:
        predictor: The segment anything predictor.
        image: The input image.
        batch_size: The batch size to use for inference.
        boxes: The box prompts. Array of shape N_PROMPTS x 4.
            The bounding boxes are represented by [MIN_X, MIN_Y, MAX_X, MAX_Y].
        points: The point prompt coordinates. Array of shape N_PROMPTS x 1 x 2.
            The points are represented by their coordinates [X, Y], which are given
            in the last dimension.
        point_labels: The point prompt labels. Array of shape N_PROMPTS x 1.
            The labels are either 0 (negative prompt) or 1 (positive prompt).
        multimasking: Whether to predict with 3 or 1 mask.
        embedding_path: Cache path for the image embeddings.
        return_instance_segmentation: Whether to return a instance segmentation
            or the individual mask data.
        segmentation_ids: Fixed segmentation ids to assign to the masks
            derived from the prompts.
        reduce_multimasking: Whether to choose the most likely masks with
            highest ious from multimasking

    Returns:
        The predicted segmentation masks.
    """
    if multimasking and (segmentation_ids is not None) and (not return_instance_segmentation):
        raise NotImplementedError

    if (points is None) != (point_labels is None):
        raise ValueError(
            "If you have point prompts both `points` and `point_labels` have to be passed, "
            "but you passed only one of them."
        )

    have_points = points is not None
    have_boxes = boxes is not None
    if (not have_points) and (not have_boxes):
        raise ValueError("Point and/or box prompts have to be passed, you passed neither.")

    if have_points and (len(point_labels) != len(points)):
        raise ValueError(
            "The number of point coordinates and labels does not match: "
            f"{len(point_labels)} != {len(points)}"
        )

    if (have_points and have_boxes) and (len(points) != len(boxes)):
        raise ValueError(
            "The number of point and box prompts does not match: "
            f"{len(points)} != {len(boxes)}"
        )
    n_prompts = boxes.shape[0] if have_boxes else points.shape[0]

    if (segmentation_ids is not None) and (len(segmentation_ids) != n_prompts):
        raise ValueError(
            "The number of segmentation ids and prompts does not match: "
            f"{len(segmentation_ids)} != {n_prompts}"
        )

    # Compute the image embeddings.
    image_embeddings = util.precompute_image_embeddings(predictor, image, embedding_path, ndim=2)
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
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_stop = min((batch_idx + 1) * batch_size, n_prompts)

        batch_boxes = boxes[batch_start:batch_stop] if have_boxes else None
        batch_points = points[batch_start:batch_stop] if have_points else None
        batch_labels = point_labels[batch_start:batch_stop] if have_points else None

        batch_masks, batch_ious, _ = predictor.predict_torch(
            point_coords=batch_points, point_labels=batch_labels,
            boxes=batch_boxes, multimask_output=multimasking
        )

        # If we expect to reduce the masks from multimasking and use multi-masking,
        # then we need to select the most likely mask (according to the predicted IOU) here.
        if reduce_multimasking and multimasking:
            _, max_index = batch_ious.max(axis=1)
            batch_masks = torch.cat([batch_masks[i, max_id][None] for i, max_id in enumerate(max_index)]).unsqueeze(1)
            batch_ious = torch.cat([batch_ious[i, max_id][None] for i, max_id in enumerate(max_index)]).unsqueeze(1)

        batch_data = amg_utils.MaskData(masks=batch_masks.flatten(0, 1), iou_preds=batch_ious.flatten(0, 1))
        batch_data["masks"] = (batch_data["masks"] > predictor.model.mask_threshold).type(torch.bool)
        batch_data["boxes"] = batched_mask_to_box(batch_data["masks"])

        masks.cat(batch_data)

    # Mask data to records.
    masks = [
        {
            "segmentation": masks["masks"][idx],
            "area": masks["masks"][idx].sum(),
            "bbox": amg_utils.box_xyxy_to_xywh(masks["boxes"][idx]).tolist(),
            "predicted_iou": masks["iou_preds"][idx].item(),
            "seg_id": idx + 1 if segmentation_ids is None else int(segmentation_ids[idx]),
        }
        for idx in range(len(masks["masks"]))
    ]

    if return_instance_segmentation:
        masks = mask_data_to_segmentation(masks, with_background=False, min_object_size=0)

    return masks
