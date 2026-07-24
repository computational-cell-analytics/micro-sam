import os
import re
import time
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from micro_sam.util import get_device
from micro_sam.v2.loss.directed_distance_based import DirectedDistanceLoss
from micro_sam.v2.loss.custom_sam2_loss import CustomSAM2Loss

from micro_sam.v2.transforms.raw import VideoAugment
from .util import get_sam2_train_model, ConvertToSam2VideoBatch
from .sam2_trainer import Sam2Trainer, Sam2Logger, UniSAM2Trainer, UniSAM2Logger
from .joint_sam2_trainer import JointSam2Trainer, JointSam2Logger


def _no_wd_names(model):
    """Return the set of parameter names that must have weight_decay=0.

    Matches the MOSE finetune config: bias params and all LayerNorm params.
    """
    names = set()
    for mn, m in model.named_modules():
        if isinstance(m, torch.nn.LayerNorm):
            for pn, _ in m.named_parameters(recurse=False):
                names.add(f"{mn}.{pn}" if mn else pn)
    for pn, _ in model.named_parameters():
        if "bias" in pn:
            names.add(pn)
    return names


def _param_lr_map(model, lr, vision_lr, layer_decay):
    """Return {param_name: learning_rate} for layer-decayed vision encoder params.

    Per the MOSE finetune YAML:
    - pos_embed gets vision_lr (no decay, override to scale=1.0 in YAML)
    - trunk block k gets vision_lr * layer_decay^(n_blocks - k)
      (block 0 is shallowest, gets the most decay)
    - other trunk params (patch_embed, stem norm, etc.) get vision_lr * layer_decay^n_blocks
    - non-trunk encoder params (neck, fpn, etc.) get vision_lr unchanged
    - non-encoder params get lr
    """
    n_blocks = 0
    for name, _ in model.named_parameters():
        m = re.search(r"image_encoder\.trunk\.blocks\.(\d+)\.", name)
        if m:
            n_blocks = max(n_blocks, int(m.group(1)) + 1)

    param_lr = {}
    for name, _ in model.named_parameters():
        if not name.startswith("image_encoder"):
            param_lr[name] = lr
        elif "pos_embed" in name:
            param_lr[name] = vision_lr
        else:
            m = re.search(r"image_encoder\.trunk\.blocks\.(\d+)\.", name)
            if m:
                k = int(m.group(1))
                param_lr[name] = vision_lr * (layer_decay ** (n_blocks - k))
            elif name.startswith("image_encoder.trunk"):
                param_lr[name] = vision_lr * (layer_decay ** n_blocks)
            else:
                param_lr[name] = vision_lr
    return param_lr


def _build_optimizer(model, lr, vision_lr=None, layer_decay=1.0):
    no_wd = _no_wd_names(model)

    if vision_lr is None:
        decay = [p for n, p in model.named_parameters() if n not in no_wd and p.requires_grad]
        no_decay = [p for n, p in model.named_parameters() if n in no_wd and p.requires_grad]
        return torch.optim.AdamW([
            {"params": decay, "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0},
        ], lr=lr)

    p_lr_map = _param_lr_map(model, lr=lr, vision_lr=vision_lr, layer_decay=layer_decay)
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        p_lr = p_lr_map[name]
        p_wd = 0.0 if name in no_wd else 0.1
        key = (f"{p_lr:.12e}", p_wd)
        if key not in groups:
            groups[key] = {"params": [], "lr": p_lr, "weight_decay": p_wd}
        groups[key]["params"].append(param)
    return torch.optim.AdamW(list(groups.values()))


def _build_scheduler(optimizer, patience=5, factor=0.9, mode="min"):
    """ReduceLROnPlateau scheduler keyed on the validation metric.

    torch-em's DefaultTrainer calls lr_scheduler.step(current_metric) after each
    validation, so the LR is multiplied by `factor` after `patience` epochs without
    improvement in the validation metric (lower is better).
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=mode, factor=factor, patience=patience,
    )


def train_sam2(
    name: str,
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 100,
    n_iterations: Optional[int] = None,
    early_stopping: Optional[int] = 10,
    max_num_objects: int = 20,
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    device: Optional[Union[str, torch.device]] = None,
    lr: float = 1e-5,
    vision_lr: Optional[float] = None,
    save_root: Optional[Union[str, os.PathLike]] = None,
    save_every_kth_epoch: Optional[int] = None,
    overwrite_training: bool = True,
    prob_to_use_pt_input: float = 0.5,
    prob_to_use_box_input: float = 0.5,
    num_frames_to_correct: int = 1,
    rand_frames_to_correct: bool = True,
    prob_to_sample_from_gt: float = 0.1,
    add_all_frames_to_correct_as_cond: bool = True,
    num_correction_pt_per_frame: int = 7,
    num_init_cond_frames: int = 2,
    clip_grad_norm: Optional[float] = 0.1,
    layer_decay: float = 1.0,
    largest_first: bool = False,
    augment: bool = False,
    bidirectional: bool = False,
    use_focal_loss: bool = False,
    focal_weight: float = 20.0,
    use_object_score_loss: bool = False,
    average_over_frames: bool = False,
) -> None:
    """Train SAM2 for interactive segmentation with SAM2's native prompting strategy.

    Uses SAM2Train (full model with video memory) which handles both 2D (T=1) and
    3D/video (T>1) batches. All prompting logic - initial point/box/mask selection
    and iterative correction from error regions - is embedded in the model forward
    pass.  Pass a MixedLoader(loader_2d, loader_3d) as train_loader for joint 2D+3D
    training.

    Args:
        name: Checkpoint/log folder name.
        model_type: SAM2 variant - one of "hvit_t", "hvit_s", "hvit_b", "hvit_l".
        train_loader: DataLoader yielding (x, y) tuples with integer instance labels.
            x: (B,C,H,W) for 2D or (B,C,Z,H,W) for 3D in [0, 255].
            y: (B,1,H,W) or (B,1,Z,H,W) with integer instance IDs.
        val_loader: Same format as train_loader, used for validation.
        n_epochs: Training epochs.
        n_iterations: Override n_epochs with a fixed iteration budget.
        early_stopping: Stop after this many epochs without improvement (None = off).
        max_num_objects: Max objects sampled per image/volume per step.
        checkpoint_path: Custom checkpoint path.  Downloads default weights if None.
        device: Training device.  Auto-selects if None.
        lr: Learning rate. SAM2 OG fine-tuning uses 1e-5 (tiny) or 5e-6 (b+).
        vision_lr: Separate LR for the image encoder. If None, uses lr for all parameters.
            SAM2 OG fine-tuning uses 6e-6 (tiny) or 3e-6 (b+), i.e. ~0.6x the base lr.
        save_root: Root directory for checkpoints and logs.
        save_every_kth_epoch: Save a separate checkpoint every k-th epoch.
        overwrite_training: Overwrite an existing checkpoint at the same path.
        prob_to_use_pt_input: Probability of using point/box prompts (vs mask propagation).
        prob_to_use_box_input: Conditional probability of using a box instead of a click.
        num_frames_to_correct: Max frames per volume that receive iterative correction
            clicks.  Set equal to the number of z-slices to correct every frame.
        rand_frames_to_correct: Randomly sample 1..num_frames_to_correct frames to
            correct per step rather than always correcting the maximum.
        prob_to_sample_from_gt: Probability of sampling a correction click from GT
            instead of the error region - reduces overfitting to error patterns.
        add_all_frames_to_correct_as_cond: Add frames that receive correction clicks
            as conditioning frames for memory propagation.
        num_correction_pt_per_frame: Number of correction clicks per frame per correction
            round. SAM2 default is 7.
        num_init_cond_frames: Number of initial conditioning frames (frames that receive
            the first prompt before any correction round). The MOSE finetune config uses 2.
        clip_grad_norm: Max gradient norm for clipping (None = disabled). Matches
            SAM2's own finetuning default of 0.1.
        layer_decay: Multiplicative LR decay applied per trunk block toward shallower
            layers (block 0 gets vision_lr * layer_decay^n_blocks). The MOSE finetune
            config uses 0.9. Only active when vision_lr is set. Default 1.0 = no decay.
        augment: Apply SAM2 MOSE-style data augmentations (RandomHorizontalFlip,
            RandomAffine, ColorJitter, RandomGrayscale) before each forward pass.
            Augmentations are applied consistently across frames for 3D volumes.
        bidirectional: If True, use bidirectional propagation during training: a random
            z-frame is chosen as the start frame each step, with propagation and correction
            clicks applied in both directions. No effect for 2D (T=1) batches.
        use_focal_loss: Add SAM2's sigmoid focal mask loss on top of the Dice loss in
            CustomSAM2Loss. Off by default for a simpler micro_sam-style loss.
        focal_weight: Weight of the focal mask loss when enabled. SAM2 uses 20; set to 1
            to keep it on equal footing with the Dice loss.
        use_object_score_loss: Supervise SAM2's object-presence score with a BCE loss in
            CustomSAM2Loss. Off by default for a simpler micro_sam-style loss.
        average_over_frames: Average the interactive loss over frames instead of summing,
            so 2D and 3D batches share a scale when training on mixed 2D + 3D data.
    """
    t_start = time.time()

    device = get_device(device)
    model = get_sam2_train_model(
        model_type=model_type,
        device=device,
        checkpoint_path=checkpoint_path,
        prob_to_use_pt_input=prob_to_use_pt_input,
        prob_to_use_box_input=prob_to_use_box_input,
        num_frames_to_correct=num_frames_to_correct,
        rand_frames_to_correct=rand_frames_to_correct,
        prob_to_sample_from_gt=prob_to_sample_from_gt,
        add_all_frames_to_correct_as_cond=add_all_frames_to_correct_as_cond,
        num_correction_pt_per_frame=num_correction_pt_per_frame,
        num_init_cond_frames_for_train=num_init_cond_frames,
        bidirectional=bidirectional,
    )

    interactive_loss = CustomSAM2Loss(
        use_focal_loss=use_focal_loss, focal_weight=focal_weight,
        use_object_score_loss=use_object_score_loss, average_over_frames=average_over_frames,
    )

    augmentor = VideoAugment() if augment else None
    convert_inputs = ConvertToSam2VideoBatch(
        max_num_objects=max_num_objects, largest_first=largest_first, augmentor=augmentor,
    )

    optimizer = _build_optimizer(model, lr=lr, vision_lr=vision_lr, layer_decay=layer_decay)
    scheduler = _build_scheduler(optimizer)

    trainer = Sam2Trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        lr_scheduler=scheduler,
        logger=Sam2Logger,
        log_image_interval=100,
        compile_model=False,
        early_stopping=early_stopping,
        save_root=save_root,
        convert_inputs=convert_inputs,
        loss=interactive_loss,
        clip_grad_norm=clip_grad_norm,
    )

    fit_kwargs = {"epochs": n_epochs} if n_iterations is None else {"iterations": n_iterations}
    if save_every_kth_epoch is not None:
        fit_kwargs["save_every_kth_epoch"] = save_every_kth_epoch
    fit_kwargs["overwrite_training"] = overwrite_training
    trainer.fit(**fit_kwargs)

    elapsed = time.time() - t_start
    h, m, s = int(elapsed // 3600), int(elapsed % 3600 // 60), int(elapsed % 60)
    print(f"Training took {elapsed:.1f}s (= {h:02}:{m:02}:{s:02})")


def _train_sam2_rank(
    rank: int,
    local_rank: int,
    name: str,
    model_type: str,
    input_path: str,
    n_epochs: int,
    n_iterations: Optional[int],
    early_stopping: Optional[int],
    max_num_objects: int,
    checkpoint_path,
    lr: float,
    save_root,
    save_every_kth_epoch: Optional[int],
    overwrite_training: bool,
    prob_to_use_pt_input: float,
    prob_to_use_box_input: float,
    num_frames_to_correct: int,
    rand_frames_to_correct: bool,
    prob_to_sample_from_gt: float,
    add_all_frames_to_correct_as_cond: bool,
    num_correction_pt_per_frame: int,
    num_init_cond_frames: int,
    clip_grad_norm: Optional[float],
    layer_decay: float,
    vision_lr: Optional[float],
    batch_size: int,
    batch_size_2d: int,
    z_slices: List[int],
    dataset_choice: str,
    n_workers: int,
    find_unused_parameters: bool,
    largest_first: bool,
    augment: bool,
    bidirectional: bool = False,
    use_focal_loss: bool = False,
    focal_weight: float = 20.0,
    use_object_score_loss: bool = False,
    average_over_frames: bool = False,
):
    """Single-rank torchrun worker for train_sam2_multi_gpu."""
    from torch_em.multi_gpu_training import DDP

    from micro_sam.v2.datasets.generalist_loader import _build_interactive_datasets, seed_worker
    from micro_sam.v2.datasets.sampler import DistributedUniBatchSampler, _build_group_map

    dist.init_process_group("nccl")
    world_size = dist.get_world_size()

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Each rank builds its own copy of the datasets.
    train_ds, val_ds = _build_interactive_datasets(input_path, z_slices, dataset_choice)

    batch_size_per_group = {2: batch_size_2d} if batch_size_2d != batch_size else None

    train_sampler = DistributedUniBatchSampler(
        group_per_index=_build_group_map(train_ds),
        batch_size=batch_size,
        batch_size_per_group=batch_size_per_group,
        shuffle=True,
        rank=rank,
        world_size=world_size,
    )
    val_sampler = DistributedUniBatchSampler(
        group_per_index=_build_group_map(val_ds),
        batch_size=batch_size,
        batch_size_per_group=batch_size_per_group,
        shuffle=False,
        rank=rank,
        world_size=world_size,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_sampler=train_sampler, num_workers=n_workers,
        pin_memory=True, persistent_workers=True,
    )
    train_loader.shuffle = True
    # Non-persistent workers + a deterministic worker_init_fn re-seed each epoch, so the
    # validation crops are identical across epochs (see seed_worker / Sam2Trainer._validate_impl).
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_sampler=val_sampler, num_workers=n_workers,
        pin_memory=True, persistent_workers=False, worker_init_fn=seed_worker,
    )
    val_loader.shuffle = False

    model = get_sam2_train_model(
        model_type=model_type,
        device=device,
        checkpoint_path=checkpoint_path,
        prob_to_use_pt_input=prob_to_use_pt_input,
        prob_to_use_box_input=prob_to_use_box_input,
        num_frames_to_correct=num_frames_to_correct,
        rand_frames_to_correct=rand_frames_to_correct,
        prob_to_sample_from_gt=prob_to_sample_from_gt,
        add_all_frames_to_correct_as_cond=add_all_frames_to_correct_as_cond,
        num_correction_pt_per_frame=num_correction_pt_per_frame,
        num_init_cond_frames_for_train=num_init_cond_frames,
        bidirectional=bidirectional,
    )
    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=find_unused_parameters)

    interactive_loss = CustomSAM2Loss(
        use_focal_loss=use_focal_loss, focal_weight=focal_weight,
        use_object_score_loss=use_object_score_loss, average_over_frames=average_over_frames,
    )

    augmentor = VideoAugment() if augment else None
    convert_inputs = ConvertToSam2VideoBatch(
        max_num_objects=max_num_objects, largest_first=largest_first, augmentor=augmentor,
    )

    optimizer = _build_optimizer(model, lr=lr, vision_lr=vision_lr, layer_decay=layer_decay)
    scheduler = _build_scheduler(optimizer)

    trainer = Sam2Trainer(
        name=name,
        model=ddp_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        lr_scheduler=scheduler,
        logger=Sam2Logger,
        log_image_interval=100,
        compile_model=False,
        early_stopping=early_stopping,
        save_root=save_root,
        convert_inputs=convert_inputs,
        loss=interactive_loss,
        clip_grad_norm=clip_grad_norm,
        rank=rank,
    )

    fit_kwargs = {"epochs": n_epochs} if n_iterations is None else {"iterations": n_iterations}
    if save_every_kth_epoch is not None:
        fit_kwargs["save_every_kth_epoch"] = save_every_kth_epoch
    fit_kwargs["overwrite_training"] = overwrite_training
    try:
        trainer.fit(**fit_kwargs)
    finally:
        dist.destroy_process_group()


def train_sam2_multi_gpu(
    name: str,
    model_type: str,
    input_path: str,
    batch_size: int = 2,
    batch_size_2d: Optional[int] = None,
    z_slices: Optional[List[int]] = None,
    dataset_choice: str = "both",
    n_workers: int = 16,
    n_epochs: int = 100,
    n_iterations: Optional[int] = None,
    early_stopping: Optional[int] = 10,
    max_num_objects: int = 20,
    checkpoint_path=None,
    lr: float = 1e-5,
    vision_lr: Optional[float] = None,
    save_root=None,
    save_every_kth_epoch: Optional[int] = None,
    overwrite_training: bool = True,
    prob_to_use_pt_input: float = 0.5,
    prob_to_use_box_input: float = 0.5,
    num_frames_to_correct: int = 1,
    rand_frames_to_correct: bool = True,
    prob_to_sample_from_gt: float = 0.1,
    add_all_frames_to_correct_as_cond: bool = True,
    num_correction_pt_per_frame: int = 7,
    num_init_cond_frames: int = 2,
    clip_grad_norm: Optional[float] = 0.1,
    layer_decay: float = 1.0,
    find_unused_parameters: bool = True,
    largest_first: bool = False,
    augment: bool = False,
    bidirectional: bool = False,
    use_focal_loss: bool = False,
    focal_weight: float = 20.0,
    use_object_score_loss: bool = False,
    average_over_frames: bool = False,
) -> None:
    """Train SAM2 for interactive segmentation across multiple GPUs with DDP.

    Must be launched via ``torchrun`` - one process per GPU, on one or more nodes.
    Example (single node): ``torchrun --nproc_per_node=4 train_sam2.py``
    Example (multi-node):  ``srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=4 train_sam2.py``

    Each rank reads its identity from the ``RANK``, ``LOCAL_RANK``, and
    ``WORLD_SIZE`` environment variables set by torchrun and builds its own
    dataset independently.

    Args:
        name: Checkpoint/log folder name.
        model_type: SAM2 variant - one of "hvit_t", "hvit_s", "hvit_b", "hvit_l".
        input_path: Root path to the generalist training data.
        batch_size: Batch size per GPU for 3D groups.
        batch_size_2d: Batch size per GPU for 2D groups (falls back to batch_size).
        z_slices: Z-slice counts for 3D data groups (default [8]).
        dataset_choice: ``"lm"``, ``"em"``, or ``"both"``.
        n_workers: DataLoader workers per GPU.
        n_epochs: Training epochs.
        n_iterations: Override n_epochs with a fixed iteration budget.
        early_stopping: Stop after this many epochs without improvement.
        max_num_objects: Max objects sampled per image/volume per step.
        checkpoint_path: SAM2 checkpoint path. Downloads default if None.
        lr: Learning rate. SAM2 OG fine-tuning uses 1e-5 (tiny) or 5e-6 (b+).
        vision_lr: Separate LR for the image encoder. If None, uses lr for all parameters.
            SAM2 OG fine-tuning uses 6e-6 (tiny) or 3e-6 (b+), i.e. ~0.6x the base lr.
        save_root: Root directory for checkpoints and logs.
        save_every_kth_epoch: Save a checkpoint every k-th epoch.
        overwrite_training: Overwrite an existing checkpoint at the same path.
        prob_to_use_pt_input: P(point/box prompt) vs P(mask propagation).
        prob_to_use_box_input: Conditional P(box) given point/box mode.
        num_frames_to_correct: Max frames per volume receiving correction clicks.
        rand_frames_to_correct: Randomly sample how many frames to correct.
        prob_to_sample_from_gt: P(correction click from GT vs error region).
        add_all_frames_to_correct_as_cond: Add corrected frames as cond frames.
        clip_grad_norm: Max gradient norm for clipping (None = disabled).
        find_unused_parameters: Passed to DistributedDataParallel.
    """
    if z_slices is None:
        z_slices = [8]
    if batch_size_2d is None:
        batch_size_2d = batch_size

    if "RANK" not in os.environ:
        raise RuntimeError(
            "train_sam2_multi_gpu must be launched via torchrun, not called directly. "
            "Example: torchrun --nproc_per_node=4 train_sam2.py"
        )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    _train_sam2_rank(
        rank=rank,
        local_rank=local_rank,
        name=name,
        model_type=model_type,
        input_path=input_path,
        n_epochs=n_epochs,
        n_iterations=n_iterations,
        early_stopping=early_stopping,
        max_num_objects=max_num_objects,
        checkpoint_path=checkpoint_path,
        lr=lr,
        save_root=save_root,
        save_every_kth_epoch=save_every_kth_epoch,
        overwrite_training=overwrite_training,
        prob_to_use_pt_input=prob_to_use_pt_input,
        prob_to_use_box_input=prob_to_use_box_input,
        num_frames_to_correct=num_frames_to_correct,
        rand_frames_to_correct=rand_frames_to_correct,
        prob_to_sample_from_gt=prob_to_sample_from_gt,
        add_all_frames_to_correct_as_cond=add_all_frames_to_correct_as_cond,
        num_correction_pt_per_frame=num_correction_pt_per_frame,
        num_init_cond_frames=num_init_cond_frames,
        clip_grad_norm=clip_grad_norm,
        layer_decay=layer_decay,
        vision_lr=vision_lr,
        batch_size=batch_size,
        batch_size_2d=batch_size_2d,
        z_slices=z_slices,
        dataset_choice=dataset_choice,
        n_workers=n_workers,
        find_unused_parameters=find_unused_parameters,
        largest_first=largest_first,
        augment=augment,
        bidirectional=bidirectional,
        use_focal_loss=use_focal_loss,
        focal_weight=focal_weight,
        use_object_score_loss=use_object_score_loss,
        average_over_frames=average_over_frames,
    )


def train_automatic(
    name: str,
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 100,
    n_iterations: Optional[int] = None,
    early_stopping: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    lr: float = 1e-5,
    save_root: Optional[Union[str, os.PathLike]] = None,
    save_every_kth_epoch: Optional[int] = None,
    overwrite_training: bool = True,
) -> None:
    """Train UniSAM2 for automatic instance segmentation with directed distance targets.

    Trains the UNETR3D-based UniSAM2 model using
    :class:`~micro_sam.v2.loss.directed_distance_based.DirectedDistanceLoss` on
    combined 2D + 3D data.  Pass a loader built by
    :func:`~micro_sam.v2.datasets.generalist_loader.get_dataloaders` with
    ``label_trafo=DirectedPerObjectBoundaryDistanceTransform``.

    Args:
        name: Checkpoint / log folder name.
        model_type: SAM2 encoder variant - one of "hvit_t", "hvit_s", "hvit_b", "hvit_l".
        train_loader: DataLoader yielding (x, y) tuples with directed-distance targets.
        val_loader: Same format, used for validation.
        n_epochs: Number of training epochs (default 100). Ignored if n_iterations is set.
        n_iterations: If set, train for this many iterations instead of epochs.
        early_stopping: Stop after this many epochs without improvement (None = off).
        device: Training device. Auto-selects if None.
        lr: Learning rate.
        save_root: Root directory for checkpoints and logs.
        save_every_kth_epoch: Save a separate checkpoint every k-th epoch.
        overwrite_training: Overwrite an existing checkpoint at the same path.
    """
    import torch_em
    from micro_sam.v2.models.util import UniSAM2

    device = get_device(device)
    model = UniSAM2(encoder=model_type, output_channels=4).to(device)

    scheduler_kwargs = {"mode": "min", "factor": 0.9, "patience": 10}
    loss = DirectedDistanceLoss(mask_distances_in_bg=True)

    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=lr,
        loss=loss,
        metric=loss,
        logger=UniSAM2Logger,
        log_image_interval=50,
        save_root=save_root,
        compile_model=False,
        scheduler_kwargs=scheduler_kwargs,
        optimizer_kwargs={"weight_decay": 0.1},
        mixed_precision=True,
        device=device,
        early_stopping=early_stopping,
        trainer_class=UniSAM2Trainer,
    )

    fit_kwargs = {"epochs": n_epochs} if n_iterations is None else {"iterations": n_iterations}
    fit_kwargs["overwrite_training"] = overwrite_training
    if save_every_kth_epoch is not None:
        fit_kwargs["save_every_kth_epoch"] = save_every_kth_epoch
    trainer.fit(**fit_kwargs)


def _train_automatic_rank(
    rank: int,
    local_rank: int,
    name: str,
    model_type: str,
    input_path: str,
    n_epochs: int,
    n_iterations: Optional[int],
    early_stopping: Optional[int],
    lr: float,
    save_root,
    save_every_kth_epoch: Optional[int],
    overwrite_training: bool,
    batch_size: int,
    batch_size_2d: int,
    z_slices: List[int],
    dataset_choice: str,
    n_workers: int,
    find_unused_parameters: bool,
):
    """Single-rank torchrun worker for train_automatic_multi_gpu."""
    import torch_em
    from torch_em.multi_gpu_training import DDP

    from micro_sam.v2.datasets.generalist_loader import _build_automatic_datasets
    from micro_sam.v2.datasets.sampler import DistributedUniBatchSampler, _build_group_map
    from micro_sam.v2.models.util import UniSAM2

    dist.init_process_group("nccl")
    world_size = dist.get_world_size()

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    train_ds, val_ds = _build_automatic_datasets(input_path, z_slices, dataset_choice)

    batch_size_per_group = {2: batch_size_2d} if batch_size_2d != batch_size else None

    train_sampler = DistributedUniBatchSampler(
        group_per_index=_build_group_map(train_ds),
        batch_size=batch_size,
        batch_size_per_group=batch_size_per_group,
        shuffle=True,
        rank=rank,
        world_size=world_size,
    )
    val_sampler = DistributedUniBatchSampler(
        group_per_index=_build_group_map(val_ds),
        batch_size=batch_size,
        batch_size_per_group=batch_size_per_group,
        shuffle=False,
        rank=rank,
        world_size=world_size,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_sampler=train_sampler, num_workers=n_workers,
        pin_memory=True, persistent_workers=True,
    )
    train_loader.shuffle = True
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_sampler=val_sampler, num_workers=n_workers,
        pin_memory=True, persistent_workers=True,
    )
    val_loader.shuffle = False

    model = UniSAM2(encoder=model_type, output_channels=4).to(device)
    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=find_unused_parameters)

    scheduler_kwargs = {"mode": "min", "factor": 0.9, "patience": 10}
    loss = DirectedDistanceLoss(mask_distances_in_bg=True)

    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=ddp_model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=lr,
        loss=loss,
        metric=loss,
        logger=UniSAM2Logger,
        log_image_interval=50,
        save_root=save_root,
        compile_model=False,
        scheduler_kwargs=scheduler_kwargs,
        optimizer_kwargs={"weight_decay": 0.1},
        mixed_precision=True,
        device=device,
        early_stopping=early_stopping,
        trainer_class=UniSAM2Trainer,
        rank=rank,
    )

    fit_kwargs = {"epochs": n_epochs} if n_iterations is None else {"iterations": n_iterations}
    fit_kwargs["overwrite_training"] = overwrite_training
    if save_every_kth_epoch is not None:
        fit_kwargs["save_every_kth_epoch"] = save_every_kth_epoch
    try:
        trainer.fit(**fit_kwargs)
    finally:
        dist.destroy_process_group()


def train_automatic_multi_gpu(
    name: str,
    model_type: str,
    input_path: str,
    batch_size: int = 2,
    batch_size_2d: Optional[int] = None,
    z_slices: Optional[List[int]] = None,
    dataset_choice: str = "both",
    n_workers: int = 16,
    n_epochs: int = 100,
    n_iterations: Optional[int] = None,
    early_stopping: Optional[int] = None,
    lr: float = 1e-5,
    save_root=None,
    save_every_kth_epoch: Optional[int] = None,
    overwrite_training: bool = True,
    find_unused_parameters: bool = True,
) -> None:
    """Train UniSAM2 for automatic segmentation across multiple GPUs with DDP.

    Mirrors :func:`train_automatic` but must be launched via ``torchrun`` - one
    process per GPU. Takes ``input_path`` instead of pre-built dataloaders because
    each rank must build its own dataset objects.

    Args:
        name: Checkpoint / log folder name.
        model_type: SAM2 encoder variant - one of "hvit_t", "hvit_s", "hvit_b", "hvit_l".
        input_path: Root path to the generalist training data.
        batch_size: Batch size per GPU for 3D groups.
        batch_size_2d: Batch size per GPU for 2D groups (falls back to batch_size).
        z_slices: Z-slice counts for 3D data groups (default [8]).
        dataset_choice: ``"lm"``, ``"em"``, or ``"both"``.
        n_workers: DataLoader workers per GPU.
        n_epochs: Number of training epochs (default 100). Ignored if n_iterations is set.
        n_iterations: If set, train for this many iterations instead of epochs.
        early_stopping: Stop after this many epochs without improvement.
        lr: Learning rate.
        save_root: Root directory for checkpoints and logs.
        save_every_kth_epoch: Save a checkpoint every k-th epoch.
        overwrite_training: Overwrite an existing checkpoint at the same path.
        find_unused_parameters: Passed to DistributedDataParallel.
    """
    if z_slices is None:
        z_slices = [8]
    if batch_size_2d is None:
        batch_size_2d = batch_size

    if "RANK" not in os.environ:
        raise RuntimeError(
            "train_automatic_multi_gpu must be launched via torchrun, not called directly. "
            "Example: torchrun --nproc_per_node=4 train_automatic.py"
        )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    _train_automatic_rank(
        rank=rank,
        local_rank=local_rank,
        name=name,
        model_type=model_type,
        input_path=input_path,
        n_epochs=n_epochs,
        n_iterations=n_iterations,
        early_stopping=early_stopping,
        lr=lr,
        save_root=save_root,
        save_every_kth_epoch=save_every_kth_epoch,
        overwrite_training=overwrite_training,
        batch_size=batch_size,
        batch_size_2d=batch_size_2d,
        z_slices=z_slices,
        dataset_choice=dataset_choice,
        n_workers=n_workers,
        find_unused_parameters=find_unused_parameters,
    )


def train_joint_sam2(
    name: str,
    model_type: str,
    input_path: str,
    batch_size: int = 2,
    batch_size_2d: Optional[int] = None,
    z_slices: Optional[List[int]] = None,
    dataset_choice: str = "both",
    n_workers: int = 16,
    n_epochs: int = 100,
    n_iterations: Optional[int] = None,
    early_stopping: Optional[int] = None,
    max_num_objects: int = 20,
    checkpoint_path=None,
    freeze: Optional[List[str]] = None,
    device: Optional[Union[str, torch.device]] = None,
    lr: float = 1e-5,
    save_root: Optional[Union[str, os.PathLike]] = None,
    scheduler_kwargs: Optional[Dict[str, Any]] = None,
    save_every_kth_epoch: Optional[int] = None,
    overwrite_training: bool = True,
    prob_to_use_pt_input: float = 0.5,
    prob_to_use_box_input: float = 0.5,
    num_frames_to_correct: int = 1,
    rand_frames_to_correct: bool = True,
    prob_to_sample_from_gt: float = 0.1,
    add_all_frames_to_correct_as_cond: bool = True,
    num_correction_pt_per_frame: int = 7,
    num_init_cond_frames: int = 2,
    clip_grad_norm: Optional[float] = 0.1,
    largest_first: bool = False,
    bidirectional: bool = False,
    use_focal_loss: bool = False,
    focal_weight: float = 20.0,
    use_object_score_loss: bool = False,
    average_over_frames: bool = False,
) -> None:
    """Train SAM2Train and UniSAM2 jointly with a shared image encoder (single GPU).

    Builds both the interactive (SAM2Train) and automatic (UniSAM2) models from
    ``model_type``, wires the shared image encoder, then interleaves the two
    losses in each training step.  The interactive loader uses integer instance
    labels; the automatic loader uses directed-distance targets.

    Args:
        name: Checkpoint / log folder name.
        model_type: SAM2 encoder variant - one of "hvit_t", "hvit_s", "hvit_b", "hvit_l".
        input_path: Root path to the generalist training data.
        batch_size: Batch size for 3D groups (both branches).
        batch_size_2d: Batch size for 2D groups (both branches). Defaults to ``batch_size``.
        z_slices: Z-slice counts for 3D groups (default [8]).
        dataset_choice: ``"lm"``, ``"em"``, or ``"both"``.
        n_workers: DataLoader workers.
        n_epochs: Number of training epochs. Ignored if n_iterations is set.
        n_iterations: If set, train for this many iterations instead of epochs.
        early_stopping: Stop after this many epochs without improvement (None = off).
        max_num_objects: Max objects per interactive step.
        checkpoint_path: SAM2 checkpoint path. Downloads default if None.
        freeze: Component name prefixes to freeze (e.g. ["image_encoder"]).
        device: Training device. Auto-selects if None.
        lr: Learning rate.
        save_root: Root directory for checkpoints and logs.
        scheduler_kwargs: ReduceLROnPlateau kwargs. Defaults to patience=10.
        save_every_kth_epoch: Save a checkpoint every k-th epoch.
        overwrite_training: Overwrite an existing checkpoint at the same path.
        prob_to_use_pt_input: P(point/box prompt) vs P(mask propagation).
        prob_to_use_box_input: Conditional P(box) given point/box mode.
        num_frames_to_correct: Max frames per volume receiving correction clicks.
        rand_frames_to_correct: Randomly sample how many frames to correct.
        prob_to_sample_from_gt: P(correction click from GT vs error region).
        add_all_frames_to_correct_as_cond: Add corrected frames as cond frames.
        num_correction_pt_per_frame: Correction clicks per frame per round (SAM2 default 7).
        num_init_cond_frames: Initial conditioning frames before the first correction round.
        clip_grad_norm: Max gradient norm for clipping (None = disabled).
        bidirectional: Bidirectional propagation during training for 3D z-stacks.
        use_focal_loss: Add SAM2's focal mask loss on top of Dice in CustomSAM2Loss.
        focal_weight: Weight of the focal mask loss when enabled.
        use_object_score_loss: Supervise SAM2's object-presence score with a BCE loss.
        average_over_frames: Average the interactive loss over frames instead of summing.
    """
    from micro_sam.v2.datasets.generalist_loader import _build_joint_datasets, _prepare_data_loader
    from micro_sam.v2.models.util import UniSAM2

    if z_slices is None:
        z_slices = [8]
    if batch_size_2d is None:
        batch_size_2d = batch_size

    device = get_device(device)

    train_ds, val_ds = _build_joint_datasets(input_path, z_slices, dataset_choice)
    bpg = {2: batch_size_2d} if batch_size_2d != batch_size else None
    train_loader = _prepare_data_loader(
        train_ds, batch_size=batch_size, shuffle=True,
        batch_size_per_group=bpg, num_workers=n_workers,
    )
    val_loader = _prepare_data_loader(
        val_ds, batch_size=batch_size, shuffle=False,
        batch_size_per_group=bpg, num_workers=n_workers,
    )

    sam2_model = get_sam2_train_model(
        model_type=model_type, device=device,
        checkpoint_path=checkpoint_path, freeze=freeze,
        prob_to_use_pt_input=prob_to_use_pt_input,
        prob_to_use_box_input=prob_to_use_box_input,
        num_frames_to_correct=num_frames_to_correct,
        rand_frames_to_correct=rand_frames_to_correct,
        prob_to_sample_from_gt=prob_to_sample_from_gt,
        add_all_frames_to_correct_as_cond=add_all_frames_to_correct_as_cond,
        num_correction_pt_per_frame=num_correction_pt_per_frame,
        num_init_cond_frames_for_train=num_init_cond_frames,
        bidirectional=bidirectional,
    )
    unetr = UniSAM2(encoder=sam2_model.image_encoder, output_channels=4).to(device)

    interactive_loss = CustomSAM2Loss(
        use_focal_loss=use_focal_loss, focal_weight=focal_weight,
        use_object_score_loss=use_object_score_loss, average_over_frames=average_over_frames,
    )
    automatic_loss = DirectedDistanceLoss(mask_distances_in_bg=True)
    convert_inputs = ConvertToSam2VideoBatch(max_num_objects=max_num_objects, largest_first=largest_first)

    if scheduler_kwargs is None:
        scheduler_kwargs = {"mode": "min", "factor": 0.9, "patience": 10}
    sam2_params = list(sam2_model.parameters())
    unetr_decoder_params = [p for n, p in unetr.named_parameters() if not n.startswith("encoder")]
    optimizer = torch.optim.AdamW(sam2_params + unetr_decoder_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, **scheduler_kwargs)

    trainer = JointSam2Trainer(
        name=name,
        model=sam2_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        lr_scheduler=scheduler,
        logger=JointSam2Logger,
        log_image_interval=100,
        mixed_precision=True,
        compile_model=False,
        early_stopping=early_stopping,
        save_root=save_root,
        unetr=unetr,
        convert_inputs=convert_inputs,
        interactive_loss=interactive_loss,
        automatic_loss=automatic_loss,
        clip_grad_norm=clip_grad_norm,
    )

    fit_kwargs = {"epochs": n_epochs} if n_iterations is None else {"iterations": n_iterations}
    fit_kwargs["overwrite_training"] = overwrite_training
    if save_every_kth_epoch is not None:
        fit_kwargs["save_every_kth_epoch"] = save_every_kth_epoch
    trainer.fit(**fit_kwargs)


def _train_joint_rank(
    rank: int,
    local_rank: int,
    name: str,
    model_type: str,
    input_path: str,
    n_epochs: int,
    n_iterations: Optional[int],
    early_stopping: Optional[int],
    max_num_objects: int,
    checkpoint_path,
    freeze,
    lr: float,
    save_root,
    scheduler_kwargs: Optional[Dict],
    save_every_kth_epoch: Optional[int],
    overwrite_training: bool,
    prob_to_use_pt_input: float,
    prob_to_use_box_input: float,
    num_frames_to_correct: int,
    rand_frames_to_correct: bool,
    prob_to_sample_from_gt: float,
    add_all_frames_to_correct_as_cond: bool,
    num_correction_pt_per_frame: int,
    num_init_cond_frames: int,
    clip_grad_norm: Optional[float],
    batch_size: int,
    batch_size_2d: int,
    z_slices: List[int],
    dataset_choice: str,
    n_workers: int,
    find_unused_parameters: bool,
    largest_first: bool,
    bidirectional: bool = False,
    use_focal_loss: bool = False,
    focal_weight: float = 20.0,
    use_object_score_loss: bool = False,
    average_over_frames: bool = False,
):
    """Single-rank torchrun worker for train_joint_sam2_multi_gpu."""
    from torch_em.multi_gpu_training import DDP

    from micro_sam.v2.datasets.generalist_loader import _build_joint_datasets
    from micro_sam.v2.datasets.sampler import DistributedUniBatchSampler, _build_group_map
    from micro_sam.v2.models.util import UniSAM2

    dist.init_process_group("nccl")
    world_size = dist.get_world_size()

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    batch_size_per_group = {2: batch_size_2d} if batch_size_2d != batch_size else None

    train_ds, val_ds = _build_joint_datasets(input_path, z_slices, dataset_choice)

    train_sampler = DistributedUniBatchSampler(
        group_per_index=_build_group_map(train_ds),
        batch_size=batch_size,
        batch_size_per_group=batch_size_per_group,
        shuffle=True, rank=rank, world_size=world_size,
    )
    val_sampler = DistributedUniBatchSampler(
        group_per_index=_build_group_map(val_ds),
        batch_size=batch_size,
        batch_size_per_group=batch_size_per_group,
        shuffle=False, rank=rank, world_size=world_size,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_sampler=train_sampler, num_workers=n_workers,
        pin_memory=True, persistent_workers=True,
    )
    train_loader.shuffle = True
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_sampler=val_sampler, num_workers=n_workers,
        pin_memory=True, persistent_workers=True,
    )
    val_loader.shuffle = False

    sam2_model = get_sam2_train_model(
        model_type=model_type, device=device,
        checkpoint_path=checkpoint_path, freeze=freeze,
        prob_to_use_pt_input=prob_to_use_pt_input,
        prob_to_use_box_input=prob_to_use_box_input,
        num_frames_to_correct=num_frames_to_correct,
        rand_frames_to_correct=rand_frames_to_correct,
        prob_to_sample_from_gt=prob_to_sample_from_gt,
        add_all_frames_to_correct_as_cond=add_all_frames_to_correct_as_cond,
        num_correction_pt_per_frame=num_correction_pt_per_frame,
        num_init_cond_frames_for_train=num_init_cond_frames,
        bidirectional=bidirectional,
    )
    unetr = UniSAM2(encoder=sam2_model.image_encoder, output_channels=4).to(device)

    # Only DDP-wrap sam2_model. We sync the unetr decoder grads manually.
    ddp_model = DDP(sam2_model, device_ids=[local_rank], find_unused_parameters=find_unused_parameters)

    interactive_loss = CustomSAM2Loss(
        use_focal_loss=use_focal_loss, focal_weight=focal_weight,
        use_object_score_loss=use_object_score_loss, average_over_frames=average_over_frames,
    )
    automatic_loss = DirectedDistanceLoss(mask_distances_in_bg=True)
    convert_inputs = ConvertToSam2VideoBatch(max_num_objects=max_num_objects, largest_first=largest_first)

    if scheduler_kwargs is None:
        scheduler_kwargs = {"mode": "min", "factor": 0.9, "patience": 10}
    sam2_params = list(sam2_model.parameters())
    unetr_decoder_params = [p for n, p in unetr.named_parameters() if not n.startswith("encoder")]
    optimizer = torch.optim.AdamW(sam2_params + unetr_decoder_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, **scheduler_kwargs)

    trainer = JointSam2Trainer(
        name=name,
        model=ddp_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        lr_scheduler=scheduler,
        logger=JointSam2Logger,
        log_image_interval=100,
        mixed_precision=True,
        compile_model=False,
        early_stopping=early_stopping,
        save_root=save_root,
        rank=rank,
        unetr=unetr,
        convert_inputs=convert_inputs,
        interactive_loss=interactive_loss,
        automatic_loss=automatic_loss,
        clip_grad_norm=clip_grad_norm,
    )

    fit_kwargs = {"epochs": n_epochs} if n_iterations is None else {"iterations": n_iterations}
    fit_kwargs["overwrite_training"] = overwrite_training
    if save_every_kth_epoch is not None:
        fit_kwargs["save_every_kth_epoch"] = save_every_kth_epoch
    try:
        trainer.fit(**fit_kwargs)
    finally:
        dist.destroy_process_group()


def train_joint_sam2_multi_gpu(
    name: str,
    model_type: str,
    input_path: str,
    batch_size: int = 2,
    batch_size_2d: Optional[int] = None,
    z_slices: Optional[List[int]] = None,
    dataset_choice: str = "both",
    n_workers: int = 16,
    n_epochs: int = 100,
    n_iterations: Optional[int] = None,
    early_stopping: Optional[int] = None,
    max_num_objects: int = 20,
    checkpoint_path=None,
    freeze: Optional[List[str]] = None,
    lr: float = 1e-5,
    save_root=None,
    scheduler_kwargs: Optional[Dict[str, Any]] = None,
    save_every_kth_epoch: Optional[int] = None,
    overwrite_training: bool = True,
    prob_to_use_pt_input: float = 0.5,
    prob_to_use_box_input: float = 0.5,
    num_frames_to_correct: int = 1,
    rand_frames_to_correct: bool = True,
    prob_to_sample_from_gt: float = 0.1,
    add_all_frames_to_correct_as_cond: bool = True,
    num_correction_pt_per_frame: int = 7,
    num_init_cond_frames: int = 2,
    clip_grad_norm: Optional[float] = 0.1,
    find_unused_parameters: bool = True,
    largest_first: bool = False,
    bidirectional: bool = False,
    use_focal_loss: bool = False,
    focal_weight: float = 20.0,
    use_object_score_loss: bool = False,
    average_over_frames: bool = False,
) -> None:
    """Train SAM2Train and UniSAM2 jointly across multiple GPUs with DDP.

    Must be launched via ``torchrun`` - one process per GPU, on one or more nodes.
    Example (single node): ``torchrun --nproc_per_node=4 train_joint.py``
    Example (multi-node):  ``srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=4 train_joint.py``

    Both the interactive and automatic datasets are constructed independently in
    each rank.  Only the SAM2 model is DDP-wrapped; UniSAM2 decoder gradients are
    manually all_reduced after each backward so the shared encoder is not double-reduced.

    Args:
        name: Checkpoint / log folder name.
        model_type: SAM2 encoder variant - one of "hvit_t", "hvit_s", "hvit_b", "hvit_l".
        input_path: Root path to the generalist training data.
        batch_size: Batch size per GPU for 3D groups (both branches).
        batch_size_2d: Batch size per GPU for 2D groups. Defaults to ``batch_size``.
        z_slices: Z-slice counts for 3D groups (default [8]).
        dataset_choice: ``"lm"``, ``"em"``, or ``"both"``.
        n_workers: DataLoader workers per GPU.
        n_epochs: Number of training epochs. Ignored if n_iterations is set.
        n_iterations: If set, train for this many iterations instead of epochs.
        early_stopping: Stop after this many epochs without improvement.
        max_num_objects: Max objects per interactive step.
        checkpoint_path: SAM2 checkpoint path. Downloads default if None.
        freeze: Component name prefixes to freeze (e.g. ["image_encoder"]).
        lr: Learning rate.
        save_root: Root directory for checkpoints and logs.
        scheduler_kwargs: ReduceLROnPlateau kwargs. Defaults to patience=10.
        save_every_kth_epoch: Save a checkpoint every k-th epoch.
        overwrite_training: Overwrite an existing checkpoint at the same path.
        prob_to_use_pt_input: P(point/box prompt) vs P(mask propagation).
        prob_to_use_box_input: Conditional P(box) given point/box mode.
        num_frames_to_correct: Max frames per volume receiving correction clicks.
        rand_frames_to_correct: Randomly sample how many frames to correct.
        prob_to_sample_from_gt: P(correction click from GT vs error region).
        add_all_frames_to_correct_as_cond: Add corrected frames as cond frames.
        num_correction_pt_per_frame: Correction clicks per frame per round (SAM2 default 7).
        num_init_cond_frames: Initial conditioning frames before the first correction round.
        clip_grad_norm: Max gradient norm for clipping (None = disabled).
        find_unused_parameters: Passed to DistributedDataParallel.
        bidirectional: Bidirectional propagation during training for 3D z-stacks.
        use_focal_loss: Add SAM2's focal mask loss on top of Dice in CustomSAM2Loss.
        focal_weight: Weight of the focal mask loss when enabled.
        use_object_score_loss: Supervise SAM2's object-presence score with a BCE loss.
        average_over_frames: Average the interactive loss over frames instead of summing.
    """
    if z_slices is None:
        z_slices = [8]
    if batch_size_2d is None:
        batch_size_2d = batch_size

    if "RANK" not in os.environ:
        raise RuntimeError(
            "train_joint_sam2_multi_gpu must be launched via torchrun, not called directly. "
            "Example: torchrun --nproc_per_node=4 train_joint.py"
        )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    _train_joint_rank(
        rank=rank,
        local_rank=local_rank,
        name=name,
        model_type=model_type,
        input_path=input_path,
        n_epochs=n_epochs,
        n_iterations=n_iterations,
        early_stopping=early_stopping,
        max_num_objects=max_num_objects,
        checkpoint_path=checkpoint_path,
        freeze=freeze,
        lr=lr,
        save_root=save_root,
        scheduler_kwargs=scheduler_kwargs,
        save_every_kth_epoch=save_every_kth_epoch,
        overwrite_training=overwrite_training,
        prob_to_use_pt_input=prob_to_use_pt_input,
        prob_to_use_box_input=prob_to_use_box_input,
        num_frames_to_correct=num_frames_to_correct,
        rand_frames_to_correct=rand_frames_to_correct,
        prob_to_sample_from_gt=prob_to_sample_from_gt,
        add_all_frames_to_correct_as_cond=add_all_frames_to_correct_as_cond,
        num_correction_pt_per_frame=num_correction_pt_per_frame,
        num_init_cond_frames=num_init_cond_frames,
        clip_grad_norm=clip_grad_norm,
        batch_size=batch_size,
        batch_size_2d=batch_size_2d,
        z_slices=z_slices,
        dataset_choice=dataset_choice,
        n_workers=n_workers,
        find_unused_parameters=find_unused_parameters,
        largest_first=largest_first,
        bidirectional=bidirectional,
        use_focal_loss=use_focal_loss,
        focal_weight=focal_weight,
        use_object_score_loss=use_object_score_loss,
        average_over_frames=average_over_frames,
    )
