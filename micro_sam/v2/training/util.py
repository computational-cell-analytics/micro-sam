import os
import functools
from typing import Callable, Dict, List, Optional, Union

import torch

from micro_sam.util import get_device
from micro_sam.v2.util import CFG_PATHS, _get_checkpoint


@functools.lru_cache(maxsize=1)
def sam2_train_class():
    """Build the SAM2Train subclass that micro_sam uses for training.

    The function builds the class once. It imports the SAM2 training package, which exists only in
    training environments.
    """
    from training.model.sam2 import SAM2Train

    class MicroSAM2Train(SAM2Train):
        """SAM2Train with optional bidirectional 3D propagation.

        With ``bidirectional`` set, each 3D training step samples a random start frame. The model
        propagates forward from the start frame to the last frame. Then it propagates backward from the
        frame before the start to the first frame, with ``track_in_reverse=True``, and reads the memory
        of the forward pass. It samples correction clicks for both directions. 2D inputs (T=1) and eval
        use the standard SAM2Train pass.
        """

        bidirectional = False

        def forward(self, input):
            if not (self.bidirectional and self.training and input.num_frames > 1):
                return super().forward(input)
            backbone_out = self.forward_image(input.flat_img_batch)
            # Clamp so there are always enough forward frames for num_init_cond_frames.
            # prepare_prompt_inputs samples (num_init_cond_frames - 1) additional cond
            # frames from range(start + 1, T), which is empty if start >= T - 1 when
            # num_init_cond_frames > 1.
            max_start = max(0, input.num_frames - self.num_init_cond_frames_for_train)
            start_frame_idx = int(self.rng.integers(0, max_start + 1))
            backbone_out = self.prepare_prompt_inputs(backbone_out, input, start_frame_idx)
            return self._forward_bidirectional(backbone_out, input, start_frame_idx)

        def _forward_bidirectional(self, backbone_out, input, start_frame_idx):
            img_feats_already_computed = backbone_out["backbone_fpn"] is not None
            feat_sizes = None
            if img_feats_already_computed:
                _, vision_feats, vision_pos_embeds, feat_sizes = self._prepare_backbone_features(backbone_out)

            num_frames = backbone_out["num_frames"]
            init_cond_frames = backbone_out["init_cond_frames"]
            forward_order = init_cond_frames + backbone_out["frames_not_in_init_cond"]
            backward_order = list(range(start_frame_idx - 1, -1, -1))

            # Extend correction frames symmetrically to backward frames.
            # Only when use_pt_input=True: when False, prepare_prompt_inputs sets
            # frames_to_add_correction_pt=[] (no corrections in mask-input mode),
            # so backward frames must follow the same rule.
            frames_to_add_correction_pt = list(backbone_out["frames_to_add_correction_pt"])
            if backbone_out["use_pt_input"] and backward_order and self.num_frames_to_correct_for_train > 0:
                n_back = min(self.num_frames_to_correct_for_train, len(backward_order))
                if self.rand_frames_to_correct_for_train and n_back > 1:
                    n_back = int(self.rng.integers(1, n_back, endpoint=True))
                back_correct = list(self.rng.choice(backward_order, n_back, replace=False))
                frames_to_add_correction_pt = frames_to_add_correction_pt + back_correct

            output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}

            for stage_id in forward_order:
                img_ids = input.flat_obj_to_img_idx[stage_id]
                if img_feats_already_computed:
                    cvf = [x[:, img_ids] for x in vision_feats]
                    cvpe = [x[:, img_ids] for x in vision_pos_embeds]
                else:
                    _, cvf, cvpe, feat_sizes = self._prepare_backbone_features_per_frame(
                        input.flat_img_batch, img_ids
                    )
                current_out = self.track_step(
                    frame_idx=stage_id,
                    is_init_cond_frame=stage_id in init_cond_frames,
                    current_vision_feats=cvf,
                    current_vision_pos_embeds=cvpe,
                    feat_sizes=feat_sizes,
                    point_inputs=backbone_out["point_inputs_per_frame"].get(stage_id, None),
                    mask_inputs=backbone_out["mask_inputs_per_frame"].get(stage_id, None),
                    gt_masks=backbone_out["gt_masks_per_frame"].get(stage_id, None),
                    frames_to_add_correction_pt=frames_to_add_correction_pt,
                    output_dict=output_dict,
                    num_frames=num_frames,
                    track_in_reverse=False,
                )
                add_as_cond = stage_id in init_cond_frames or (
                    self.add_all_frames_to_correct_as_cond and stage_id in frames_to_add_correction_pt
                )
                if add_as_cond:
                    output_dict["cond_frame_outputs"][stage_id] = current_out
                else:
                    output_dict["non_cond_frame_outputs"][stage_id] = current_out

            for stage_id in backward_order:
                img_ids = input.flat_obj_to_img_idx[stage_id]
                if img_feats_already_computed:
                    cvf = [x[:, img_ids] for x in vision_feats]
                    cvpe = [x[:, img_ids] for x in vision_pos_embeds]
                else:
                    _, cvf, cvpe, feat_sizes = self._prepare_backbone_features_per_frame(
                        input.flat_img_batch, img_ids
                    )
                current_out = self.track_step(
                    frame_idx=stage_id,
                    is_init_cond_frame=False,
                    current_vision_feats=cvf,
                    current_vision_pos_embeds=cvpe,
                    feat_sizes=feat_sizes,
                    point_inputs=None,
                    mask_inputs=None,
                    gt_masks=backbone_out["gt_masks_per_frame"].get(stage_id, None),
                    frames_to_add_correction_pt=frames_to_add_correction_pt,
                    output_dict=output_dict,
                    num_frames=num_frames,
                    track_in_reverse=True,
                )
                add_as_cond = (
                    self.add_all_frames_to_correct_as_cond and stage_id in frames_to_add_correction_pt
                )
                if add_as_cond:
                    output_dict["cond_frame_outputs"][stage_id] = current_out
                else:
                    output_dict["non_cond_frame_outputs"][stage_id] = current_out

            all_frame_outputs = {}
            all_frame_outputs.update(output_dict["cond_frame_outputs"])
            all_frame_outputs.update(output_dict["non_cond_frame_outputs"])
            all_frame_outputs = [all_frame_outputs[t] for t in range(num_frames)]
            all_frame_outputs = [{k: v for k, v in d.items() if k != "obj_ptr"} for d in all_frame_outputs]
            return all_frame_outputs

    return MicroSAM2Train


def get_sam2_train_model(
    model_type: str = "hvit_t",
    device: Optional[Union[str, torch.device]] = None,
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    freeze: Optional[List[str]] = None,
    peft_kwargs: Optional[Dict] = None,
    prob_to_use_pt_input: float = 0.5,
    prob_to_use_box_input: float = 0.5,
    num_frames_to_correct: int = 1,
    rand_frames_to_correct: bool = True,
    prob_to_sample_from_gt: float = 0.1,
    add_all_frames_to_correct_as_cond: bool = True,
    num_correction_pt_per_frame: int = 7,
    num_init_cond_frames_for_train: int = 1,
    prob_to_use_pt_input_for_eval: Optional[float] = None,
    prob_to_use_box_input_for_eval: Optional[float] = None,
    num_frames_to_correct_for_eval: Optional[int] = None,
    num_init_cond_frames_for_eval: Optional[int] = None,
    bidirectional: bool = False,
) -> torch.nn.Module:
    """Build a SAM2Train model for interactive segmentation training.

    SAM2Train uses SAM2's native prompting strategy (point/box/mask inputs sampled
    from GT, iterative correction on error regions) and supports both 2D (T=1) and
    3D (T=Z, video) batches in a single training run.

    Args:
        model_type: SAM2 variant - one of "hvit_t", "hvit_s", "hvit_b", "hvit_l".
        device: Target device. Auto-selects if None.
        checkpoint_path: Path to a custom checkpoint. Downloads default weights if None.
        freeze: Component name prefixes to freeze (e.g. ["image_encoder"]).
        peft_kwargs: The arguments for `PEFT_Sam2`. These arguments freeze the encoder and apply the PEFT method.
        prob_to_use_pt_input: Probability of using point/box prompts (vs mask propagation).
        prob_to_use_box_input: Conditional probability of using a box instead of a click.
        num_frames_to_correct: Max number of frames per volume that receive iterative
            correction clicks. Set to the number of z-slices to correct all frames.
        rand_frames_to_correct: If True, randomly sample 1..num_frames_to_correct frames
            to correct per step (more robust than always correcting the maximum).
        prob_to_sample_from_gt: Probability of sampling a correction click from the GT
            mask instead of the error region - reduces overfitting to error patterns.
        add_all_frames_to_correct_as_cond: If True, any frame that receives a correction
            click is also added as a conditioning frame for memory propagation.
        num_correction_pt_per_frame: Number of correction clicks sampled per frame per
            correction round. SAM2 default is 7.
        num_init_cond_frames_for_train: Number of initial conditioning frames (frames that
            receive the first prompt before any correction round). SAM2 default is 1;
            the MOSE finetune config uses 2.
        prob_to_use_pt_input_for_eval: Probability of using point/box prompts during validation.
            Defaults to prob_to_use_pt_input, so validation measures the task that was trained.
            SAM2's own default is 0.0, which validates GT-mask propagation instead.
        prob_to_use_box_input_for_eval: Conditional probability of a box during validation.
            Defaults to prob_to_use_box_input.
        num_frames_to_correct_for_eval: Frames receiving correction clicks during validation.
            Defaults to num_frames_to_correct.
        num_init_cond_frames_for_eval: Initial conditioning frames during validation.
            Defaults to num_init_cond_frames_for_train.
        bidirectional: If True, replace the forward pass with bidirectional propagation:
            a random z-slice is chosen as the start frame each step, memory is propagated
            both forward (to higher z) and backward (to lower z, with track_in_reverse=True),
            and correction clicks are sampled for frames in both directions.

    Returns:
        SAM2Train model on the target device in train mode.
    """
    from sam2.build_sam import build_sam2

    device = get_device(device)
    if checkpoint_path is None:
        checkpoint_path = _get_checkpoint(model_type=model_type)

    model_cfg = CFG_PATHS[model_type[:6]]

    # SAM2's eval defaults propagate a GT mask instead of prompting, so mirror the training
    # configuration and validation scores the task actually being trained.
    if prob_to_use_pt_input_for_eval is None:
        prob_to_use_pt_input_for_eval = prob_to_use_pt_input
    if prob_to_use_box_input_for_eval is None:
        prob_to_use_box_input_for_eval = prob_to_use_box_input
    if num_frames_to_correct_for_eval is None:
        num_frames_to_correct_for_eval = num_frames_to_correct
    if num_init_cond_frames_for_eval is None:
        num_init_cond_frames_for_eval = num_init_cond_frames_for_train

    model = build_sam2(
        config_file=model_cfg,
        ckpt_path=str(checkpoint_path),
        device=str(device),
        mode="train",
        hydra_overrides_extra=[
            "++model._target_=training.model.sam2.SAM2Train",
            f"++model.prob_to_use_pt_input_for_train={prob_to_use_pt_input}",
            f"++model.prob_to_use_box_input_for_train={prob_to_use_box_input}",
            f"++model.num_frames_to_correct_for_train={max(num_frames_to_correct, num_init_cond_frames_for_train)}",
            f"++model.rand_frames_to_correct_for_train={rand_frames_to_correct}",
            f"++model.prob_to_sample_from_gt_for_train={prob_to_sample_from_gt}",
            f"++model.add_all_frames_to_correct_as_cond={add_all_frames_to_correct_as_cond}",
            f"++model.num_correction_pt_per_frame={num_correction_pt_per_frame}",
            f"++model.num_init_cond_frames_for_train={num_init_cond_frames_for_train}",
            f"++model.prob_to_use_pt_input_for_eval={prob_to_use_pt_input_for_eval}",
            f"++model.prob_to_use_box_input_for_eval={prob_to_use_box_input_for_eval}",
            "++model.num_frames_to_correct_for_eval="
            f"{max(num_frames_to_correct_for_eval, num_init_cond_frames_for_eval)}",
            f"++model.num_init_cond_frames_for_eval={num_init_cond_frames_for_eval}",
            # Deterministic eval sampling, so the metric is comparable across epochs.
            "++model.rand_frames_to_correct_for_eval=False",
            "++model.rand_init_cond_frames_for_eval=False",
        ],
        apply_postprocessing=False,
    )

    if peft_kwargs:
        from micro_sam.v2.models.peft_sam2 import PEFT_Sam2
        from micro_sam.models.peft import serialize_peft_kwargs

        model = PEFT_Sam2(model, **peft_kwargs).sam.to(device)
        model.peft_config = serialize_peft_kwargs(peft_kwargs)

    if freeze is not None:
        components = [freeze] if isinstance(freeze, str) else freeze
        if peft_kwargs and "image_encoder" in components:
            raise ValueError("You cannot use PEFT and freeze the image encoder at the same time.")
        for name, param in model.named_parameters():
            if any(name.startswith(c) for c in components):
                param.requires_grad = False

    model.__class__ = sam2_train_class()
    model.bidirectional = bidirectional

    return model


class ConvertToSam2VideoBatch:
    """Convert torch-em (x, y) batches to BatchedVideoDatapoint for SAM2Train.

    2D inputs (x: B,C,H,W  /  y: B,1,H,W): each image becomes a 1-frame video (T=1).
    3D inputs (x: B,C,Z,H,W  /  y: B,1,Z,H,W): Z-slices become video frames (T=Z).

    Images are converted to SAM2 input format:
    - [0, 1] range -> ImageNet-normalized -> resized to 1024x1024
    - Single-channel inputs are broadcast to 3 channels.

    Masks are resized to 1024×1024 (required so that get_next_point returns
    coordinates in the model's input coordinate space).

    Args:
        max_num_objects: Maximum number of objects to sample per image/volume.
            Excess objects are randomly subsampled.
        largest_first: Fill half the object slots with the largest instances and the rest
            at random, instead of sampling all of them at random.
        augmentor: Optional augmentation, called as ``augmentor(x, y)`` and returning the
            augmented pair. Must apply spatial transforms to both, or the masks stop matching
            the image. See :class:`~micro_sam.v2.transforms.raw.VideoAugment`.
    """

    _PIXEL_MEAN = [0.485, 0.456, 0.406]
    _PIXEL_STD = [0.229, 0.224, 0.225]
    _SAM2_SIZE = 1024

    def __init__(self, max_num_objects: int = 20, largest_first: bool = False, augmentor: Optional[Callable] = None):
        self.max_num_objects = max_num_objects
        self.largest_first = largest_first
        self.augmentor = augmentor
        self.init_kwargs = {"max_num_objects": max_num_objects, "largest_first": largest_first}
        # Built once instead of per frame.
        self.pixel_mean = torch.tensor(self._PIXEL_MEAN).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor(self._PIXEL_STD).view(1, 3, 1, 1)

    def _to_sam2_size(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """Resize longest side to SAM2_SIZE then zero-pad to square.

        Padding is applied in image space to the right and bottom edges.
        """
        from micro_sam.v2.transforms.resize import resize_longest_side_and_pad_tensor
        return resize_longest_side_and_pad_tensor(x, self._SAM2_SIZE, mode=mode)[0]

    def _to_sam2_image(self, x: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W) float [0,1] -> (B,3,1024,1024) ImageNet-normalized."""
        x = x.float()
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        # Pad before normalization to keep padding black in image space.
        x = self._to_sam2_size(x, mode="bilinear")
        mean = self.pixel_mean.to(device=x.device, dtype=x.dtype)
        std = self.pixel_std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def _resize_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """(O,H,W) bool -> (O,1024,1024) bool, aspect-ratio preserving + padded."""
        m = masks.float().unsqueeze(1)  # (O,1,H,W)
        m = self._to_sam2_size(m, mode="nearest")
        return m.squeeze(1).bool()

    def _sample_obj_ids(self, label_2d: torch.Tensor) -> torch.Tensor:
        """Return up to max_num_objects non-zero unique IDs from a 2-D label map."""
        ids = torch.unique(label_2d)
        ids = ids[ids > 0]
        if len(ids) <= self.max_num_objects:
            return ids
        if not self.largest_first:
            perm = torch.randperm(len(ids), device=ids.device)[:self.max_num_objects]
            return ids[perm]
        # Mixed: first n//2 by descending size, remaining n - n//2 at random.
        n_largest = self.max_num_objects // 2
        n_random = self.max_num_objects - n_largest
        counts = torch.bincount(label_2d.flatten().long(), minlength=int(ids.max().item()) + 1)
        sorted_idx = torch.argsort(counts[ids.long()], descending=True)
        largest_ids = ids[sorted_idx[:n_largest]]
        perm = torch.randperm(len(ids) - n_largest, device=ids.device)[:n_random]
        random_ids = ids[sorted_idx[n_largest:]][perm]
        return torch.cat([largest_ids, random_ids])

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        """
        Args:
            x: Images - (B,C,H,W) for 2D or (B,C,Z,H,W) for 3D, in [0, 1].
            y: Instance labels - (B,1,H,W) or (B,1,Z,H,W) with integer IDs.

        Returns:
            BatchedVideoDatapoint compatible with SAM2Train.forward().
        """
        from training.utils.data_utils import BatchedVideoDatapoint, BatchedVideoMetaData

        if self.augmentor is not None:
            x, y = self.augmentor(x, y)

        is_3d = (x.ndim == 5)
        B = x.shape[0]
        if is_3d:
            _, _, T, H, W = x.shape
        else:
            _, _, H, W = x.shape
            T = 1

        # Resized per frame: x[:, :, t] is a view, whereas flattening to one frame batch needs
        # a permute+copy for no gain, since the resize cost is the same either way.
        if is_3d:
            img_batch = torch.stack([self._to_sam2_image(x[:, :, t]) for t in range(T)])
        else:
            img_batch = self._to_sam2_image(x).unsqueeze(0)

        y = y.squeeze(1)  # (B,H,W) or (B,Z,H,W)

        # For 3D: sample from the union of IDs across all z-slices so that it includes objects
        # present in any frame (frame 0 alone can be empty at patch boundaries).
        obj_ids_per_b = [
            self._sample_obj_ids(y[b].flatten() if is_3d else y[b]) for b in range(B)
        ]
        # The sampled IDs do not depend on the time step, so emptiness is the same for all of them.
        n_objects = sum(len(ids) for ids in obj_ids_per_b)
        if n_objects == 0:
            raise RuntimeError(
                "ConvertToSam2VideoBatch: no objects found in batch. "
                "Use MinInstanceSampler to ensure each sample has objects."
            )

        # One broadcast comparison per frame instead of stacking objects one at a time. The
        # 1024x1024 resize dominates regardless of how the masks are batched into it.
        step_masks = []
        for t in range(T):
            raw = [
                (y[b, t] if is_3d else y[b]).unsqueeze(0) == obj_ids_per_b[b].view(-1, 1, 1)
                for b in range(B) if len(obj_ids_per_b[b]) > 0
            ]
            step_masks.append(self._resize_masks(torch.cat(raw)))  # (O,1024,1024)

        # Per-time-step metadata (same structure as collate_fn in data_utils.py). The object
        # slots are identical for every frame, so the flat lists are built once.
        batch_index = [b for b in range(B) for _ in range(len(obj_ids_per_b[b]))]
        object_ids = [int(oid) for b in range(B) for oid in obj_ids_per_b[b].tolist()]
        step_obj2frame, step_identifier = [], []
        for t in range(T):
            step_obj2frame.append(torch.tensor([[t, b] for b in batch_index], dtype=torch.int))
            step_identifier.append(
                torch.tensor([[b, oid, t] for b, oid in zip(batch_index, object_ids)], dtype=torch.long)
            )
        orig_size = torch.tensor([H, W], dtype=torch.long).expand(T, n_objects, 2).contiguous()

        return BatchedVideoDatapoint(
            img_batch=img_batch,
            obj_to_frame_idx=torch.stack(step_obj2frame),  # (T,O,2)
            masks=torch.stack(step_masks),  # (T,O,1024,1024)
            metadata=BatchedVideoMetaData(
                unique_objects_identifier=torch.stack(step_identifier),
                frame_orig_size=orig_size,
            ),
            dict_key="torch_em",
            batch_size=[T],
        )


class MixedLoader:
    """Round-robin DataLoader wrapper for joint 2D + 3D training.

    Each iteration yields one batch from the first loader, then one from the
    second, cycling until the shorter one is exhausted. This ensures that every
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
