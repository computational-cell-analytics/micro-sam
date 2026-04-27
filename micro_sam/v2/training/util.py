import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from micro_sam.util import get_device
from micro_sam.prompt_generators import PointAndBoxPromptGenerator
from micro_sam.util import get_centers_and_bounding_boxes, segmentation_to_one_hot

from micro_sam.v2.util import get_sam2_model, CFG_PATHS, BACKBONE, _get_checkpoint
from micro_sam.v2.training.trainable_sam2 import TrainableSAM2


def get_trainable_sam2_model(
    model_type: str = "hvit_t",
    device: Optional[Union[str, torch.device]] = None,
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    freeze: Optional[List[str]] = None,
) -> TrainableSAM2:
    """Load a SAM2 model and wrap it for interactive segmentation training.

    Args:
        model_type: SAM2 variant — one of "hvit_t", "hvit_s", "hvit_b", "hvit_l".
        device: Target device. Defaults to best available.
        checkpoint_path: Path to a custom checkpoint. Uses the default SAM2 weights
            for the given model_type if None.
        freeze: Optional list of component name prefixes to freeze, e.g.
            ["image_encoder"] to freeze the backbone during finetuning.

    Returns:
        TrainableSAM2 instance on the target device.
    """
    device = get_device(device)
    sam2 = get_sam2_model(model_type=model_type, checkpoint_path=checkpoint_path)

    if freeze is not None:
        components = freeze if isinstance(freeze, list) else [freeze]
        for name, param in sam2.named_parameters():
            if any(name.startswith(c) for c in components):
                param.requires_grad = False

    return TrainableSAM2(sam2).to(device)


class ConvertToSam2Inputs:
    """Convert dataloader outputs to the batched input format expected by TrainableSAM2.

    Mirrors micro_sam.training.util.ConvertToSamInputs but uses SAM2Transform
    to scale coordinates into SAM2's 1024x1024 input space.

    Args:
        dilation_strength: Safety border (in pixels) from which prompts are not
            sampled to avoid ambiguous prompts near annotation borders.
        box_distortion_factor: If given, randomly enlarge/shrink bounding boxes
            by this fraction to improve robustness.
    """

    def __init__(
        self,
        dilation_strength: int = 10,
        box_distortion_factor: Optional[float] = None,
    ) -> None:
        self.dilation_strength = dilation_strength
        self.box_distortion_factor = box_distortion_factor

    def _distort_boxes(self, bbox_coordinates, shape):
        distorted = []
        for y0, x0, y1, x1 in bbox_coordinates:
            ly, lx = y1 - y0, x1 - x0
            f = self.box_distortion_factor
            y0 = int(round(max(0, y0 - np.random.uniform(0, f) * ly)))
            y1 = int(round(min(shape[0], y1 + np.random.uniform(0, f) * ly)))
            x0 = int(round(max(0, x0 - np.random.uniform(0, f) * lx)))
            x1 = int(round(min(shape[1], x1 + np.random.uniform(0, f) * lx)))
            distorted.append([y0, x0, y1, x1])
        return distorted

    def _get_prompt_lists(self, gt, n_samples, prompt_generator):
        _, bbox_coordinates = get_centers_and_bounding_boxes(gt, mode="p")
        cell_ids = np.unique(gt)[1:]

        if n_samples is None:
            sampled_ids = cell_ids
        else:
            sampled_ids = np.sort(
                np.random.choice(cell_ids, size=min(n_samples, len(cell_ids)), replace=False)
            )

        bbox_coordinates = [bbox_coordinates[sid] for sid in sampled_ids]
        if self.box_distortion_factor is not None:
            bbox_coordinates = self._distort_boxes(bbox_coordinates, gt.shape[-2:])

        object_masks = segmentation_to_one_hot(gt, None if n_samples is None else sampled_ids)
        box_prompts, point_prompts, point_label_prompts, _ = prompt_generator(
            object_masks, bbox_coordinates
        )
        return box_prompts, point_prompts, point_label_prompts, sampled_ids

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_pos: int,
        n_neg: int,
        get_boxes: bool = False,
        n_samples: Optional[int] = None,
    ) -> tuple:
        """Convert a batch to SAM2 batched inputs.

        Args:
            x: Raw image batch (B, C, H, W).
            y: Instance label batch (B, 1, H, W) with integer IDs.
            n_pos: Number of positive point prompts per object.
            n_neg: Number of negative point prompts per object.
            get_boxes: Whether to include bounding box prompts.
            n_samples: Max number of objects to sample per image (None = all).

        Returns:
            (batched_inputs, sampled_ids_list) where batched_inputs is a list of
            dicts compatible with TrainableSAM2.forward.
        """
        get_points = (n_pos > 0 or n_neg > 0)

        prompt_generator = PointAndBoxPromptGenerator(
            n_positive_points=n_pos,
            n_negative_points=n_neg,
            dilation_strength=self.dilation_strength,
            get_box_prompts=get_boxes,
            get_point_prompts=get_points,
        )

        batched_inputs, sampled_ids_list = [], []
        for image, gt in zip(x, y):
            gt_np = gt.squeeze().numpy().astype(np.int64)
            original_size = (gt_np.shape[-2], gt_np.shape[-1])
            image_size = 1024

            box_prompts, point_prompts, point_label_prompts, sampled_ids = self._get_prompt_lists(
                gt_np, n_samples, prompt_generator
            )
            sampled_ids_list.append(sampled_ids)

            inp: Dict = {"image": image, "original_size": original_size}

            if get_boxes:
                # boxes from _get_prompt_lists are (N, 4) in [y0, x0, y1, x1] format.
                # SAM2 prompt encoder expects [x0, y0, x1, y1] (xyxy).
                boxes = torch.as_tensor(box_prompts, dtype=torch.float32)
                # Reorder from [y0, x0, y1, x1] → [x0, y0, x1, y1]
                boxes = boxes[:, [1, 0, 3, 2]]
                # Scale from original image space to SAM2 input space (1024x1024)
                h, w = original_size
                scale = torch.tensor(
                    [image_size / w, image_size / h, image_size / w, image_size / h],
                    dtype=boxes.dtype,
                )
                inp["boxes"] = boxes * scale

            if get_points:
                # point_prompts: (N_obj, P, 2) in (x, y) format, original space
                points = torch.as_tensor(point_prompts, dtype=torch.float32)
                h, w = original_size
                scale = torch.tensor(
                    [image_size / w, image_size / h],
                    dtype=points.dtype,
                )
                inp["point_coords"] = points * scale
                inp["point_labels"] = torch.as_tensor(point_label_prompts, dtype=torch.int32)

            batched_inputs.append(inp)

        return batched_inputs, sampled_ids_list


def get_sam2_train_model(
    model_type: str = "hvit_t",
    device: Optional[Union[str, torch.device]] = None,
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    freeze: Optional[List[str]] = None,
    backbone: str = BACKBONE,
    prob_to_use_pt_input: float = 0.5,
    prob_to_use_box_input: float = 0.5,
    num_frames_to_correct: int = 1,
    rand_frames_to_correct: bool = True,
    prob_to_sample_from_gt: float = 0.1,
    add_all_frames_to_correct_as_cond: bool = True,
    num_correction_pt_per_frame: int = 7,
) -> torch.nn.Module:
    """Build a SAM2Train model for interactive segmentation training.

    SAM2Train uses SAM2's native prompting strategy (point/box/mask inputs sampled
    from GT, iterative correction on error regions) and supports both 2D (T=1) and
    3D (T=Z, video) batches in a single training run.

    Args:
        model_type: SAM2 variant — one of "hvit_t", "hvit_s", "hvit_b", "hvit_l".
        device: Target device.  Auto-selects if None.
        checkpoint_path: Path to a custom checkpoint.  Downloads default weights if None.
        freeze: Component name prefixes to freeze (e.g. ["image_encoder"]).
        backbone: SAM2 backbone version, "sam2.0" or "sam2.1".
        prob_to_use_pt_input: Probability of using point/box prompts (vs mask propagation).
        prob_to_use_box_input: Conditional probability of using a box instead of a click.
        num_frames_to_correct: Max number of frames per volume that receive iterative
            correction clicks.  Set to the number of z-slices to correct all frames.
        rand_frames_to_correct: If True, randomly sample 1..num_frames_to_correct frames
            to correct per step (more robust than always correcting the maximum).
        prob_to_sample_from_gt: Probability of sampling a correction click from the GT
            mask instead of the error region — reduces overfitting to error patterns.
        add_all_frames_to_correct_as_cond: If True, any frame that receives a correction
            click is also added as a conditioning frame for memory propagation.
        num_correction_pt_per_frame: Number of correction clicks sampled per frame per
            correction round. SAM2 default is 7.

    Returns:
        SAM2Train model on the target device in train mode.
    """
    from sam2.build_sam import build_sam2

    device = get_device(device)
    if checkpoint_path is None:
        checkpoint_path = _get_checkpoint(model_type=model_type, backbone=backbone)

    model_cfg = CFG_PATHS[backbone][model_type[:6]]

    model = build_sam2(
        config_file=model_cfg,
        ckpt_path=str(checkpoint_path),
        device=str(device),
        mode="train",
        hydra_overrides_extra=[
            "++model._target_=training.model.sam2.SAM2Train",
            f"++model.prob_to_use_pt_input_for_train={prob_to_use_pt_input}",
            f"++model.prob_to_use_box_input_for_train={prob_to_use_box_input}",
            f"++model.num_frames_to_correct_for_train={num_frames_to_correct}",
            f"++model.rand_frames_to_correct_for_train={rand_frames_to_correct}",
            f"++model.prob_to_sample_from_gt_for_train={prob_to_sample_from_gt}",
            f"++model.add_all_frames_to_correct_as_cond={add_all_frames_to_correct_as_cond}",
            f"++model.num_correction_pt_per_frame={num_correction_pt_per_frame}",
        ],
        apply_postprocessing=False,
    )

    if freeze is not None:
        components = [freeze] if isinstance(freeze, str) else freeze
        for name, param in model.named_parameters():
            if any(name.startswith(c) for c in components):
                param.requires_grad = False

    return model


class ConvertToSam2VideoBatch:
    """Convert torch-em (x, y) batches to BatchedVideoDatapoint for SAM2Train.

    2D inputs (x: B,C,H,W  /  y: B,1,H,W): each image becomes a 1-frame video (T=1).
    3D inputs (x: B,C,Z,H,W  /  y: B,1,Z,H,W): Z-slices become video frames (T=Z).

    Images are converted to SAM2 input format:
    - [0, 1] range → ImageNet-normalized → resized to 1024×1024
    - Single-channel inputs are broadcast to 3 channels.

    Masks are resized to 1024×1024 (required so that get_next_point returns
    coordinates in the model's input coordinate space).

    Args:
        max_num_objects: Maximum number of objects to sample per image/volume.
            Excess objects are randomly subsampled.
    """

    _PIXEL_MEAN = [0.485, 0.456, 0.406]
    _PIXEL_STD = [0.229, 0.224, 0.225]
    _SAM2_SIZE = 1024

    def __init__(self, max_num_objects: int = 20):
        self.max_num_objects = max_num_objects
        self.init_kwargs = {"max_num_objects": max_num_objects}

    def _to_sam2_image(self, x: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W) float [0,1] → (B,3,1024,1024) ImageNet-normalized."""
        x = x.float()
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        mean = torch.tensor(self._PIXEL_MEAN, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(self._PIXEL_STD, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        return F.interpolate(x, size=(self._SAM2_SIZE, self._SAM2_SIZE), mode="bilinear", align_corners=False)

    def _resize_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """(O,H,W) bool → (O,1024,1024) bool via nearest-neighbor resize."""
        m = masks.float().unsqueeze(1)  # (O,1,H,W)
        m = F.interpolate(m, size=(self._SAM2_SIZE, self._SAM2_SIZE), mode="nearest")
        return m.squeeze(1).bool()

    def _sample_obj_ids(self, label_2d: torch.Tensor) -> torch.Tensor:
        """Return up to max_num_objects non-zero unique IDs from a 2-D label map."""
        ids = torch.unique(label_2d)
        ids = ids[ids > 0]
        if len(ids) > self.max_num_objects:
            perm = torch.randperm(len(ids), device=ids.device)[:self.max_num_objects]
            ids = ids[perm]
        return ids

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        """
        Args:
            x: Images — (B,C,H,W) for 2D or (B,C,Z,H,W) for 3D, in [0, 1].
            y: Instance labels — (B,1,H,W) or (B,1,Z,H,W) with integer IDs.

        Returns:
            BatchedVideoDatapoint compatible with SAM2Train.forward().
        """
        from training.utils.data_utils import BatchedVideoDatapoint, BatchedVideoMetaData

        is_3d = (x.ndim == 5)
        B = x.shape[0]
        if is_3d:
            _, _, T, H, W = x.shape
        else:
            _, _, H, W = x.shape
            T = 1

        # img_batch: (T, B, 3, 1024, 1024)
        if is_3d:
            img_batch = torch.stack([self._to_sam2_image(x[:, :, t]) for t in range(T)])
        else:
            img_batch = self._to_sam2_image(x).unsqueeze(0)

        y = y.squeeze(1)  # (B,H,W) or (B,Z,H,W)

        # For 3D: sample from the union of IDs across all z-slices so that objects
        # present in any frame are included (frame 0 alone may be empty at patch boundaries).
        obj_ids_per_b = [
            self._sample_obj_ids(y[b].flatten() if is_3d else y[b]) for b in range(B)
        ]

        # Build per-time-step tensors (same structure as collate_fn in data_utils.py).
        step_masks, step_obj2frame, step_identifier, step_orig_size = [], [], [], []
        for t in range(T):
            masks_t, obj2frame_t, id_t, size_t = [], [], [], []
            for b in range(B):
                ids = obj_ids_per_b[b]
                if len(ids) == 0:
                    continue
                lbl = y[b, t] if is_3d else y[b]       # (H,W)
                raw = torch.stack([lbl == oid for oid in ids])  # (O_i,H,W)
                obj_masks = self._resize_masks(raw)             # (O_i,1024,1024)
                for o_i, oid in enumerate(ids):
                    masks_t.append(obj_masks[o_i])
                    obj2frame_t.append(torch.tensor([t, b], dtype=torch.int))
                    id_t.append(torch.tensor([b, int(oid.item()), t], dtype=torch.long))
                    size_t.append(torch.tensor([H, W], dtype=torch.long))

            if not masks_t:
                raise RuntimeError(
                    "ConvertToSam2VideoBatch: no objects found in batch at time step "
                    f"{t}. Use MinInstanceSampler to ensure each sample has objects."
                )
            step_masks.append(torch.stack(masks_t))
            step_obj2frame.append(torch.stack(obj2frame_t))
            step_identifier.append(torch.stack(id_t))
            step_orig_size.append(torch.stack(size_t))

        return BatchedVideoDatapoint(
            img_batch=img_batch,
            obj_to_frame_idx=torch.stack(step_obj2frame),   # (T,O,2)
            masks=torch.stack(step_masks),                  # (T,O,1024,1024)
            metadata=BatchedVideoMetaData(
                unique_objects_identifier=torch.stack(step_identifier),
                frame_orig_size=torch.stack(step_orig_size),
            ),
            dict_key="torch_em",
            batch_size=[T],
        )


class MixedLoader:
    """Round-robin DataLoader wrapper for joint 2D + 3D training.

    Each iteration yields one batch from the first loader, then one from the
    second, cycling until the shorter one is exhausted.  This ensures that every
    training step sees both 2D and 3D data.

    Args:
        loader_2d: DataLoader yielding (B,C,H,W) batches.
        loader_3d: DataLoader yielding (B,C,Z,H,W) batches.
    """

    def __init__(self, loader_2d, loader_3d):
        self.loader_2d = loader_2d
        self.loader_3d = loader_3d
        self.shuffle = getattr(loader_2d, "shuffle", True)

    def __len__(self):
        return len(self.loader_2d) + len(self.loader_3d)

    def __iter__(self):
        iter_2d = iter(self.loader_2d)
        iter_3d = iter(self.loader_3d)
        done_2d = done_3d = False
        while not (done_2d and done_3d):
            if not done_2d:
                try:
                    yield next(iter_2d)
                except StopIteration:
                    done_2d = True
            if not done_3d:
                try:
                    yield next(iter_3d)
                except StopIteration:
                    done_3d = True
