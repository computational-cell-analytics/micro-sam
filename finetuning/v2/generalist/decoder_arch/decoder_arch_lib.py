"""Building blocks of the decoder architecture evaluation (plan: ~/.claude/plans/decoder-architecture-evaluation.md).

UniSAM2 models with different decoders are trained end to end (SAM2.1 hvit_t encoder, trainable) on fixed subsets
of three 2D and three 3D datasets with the automatic segmentation targets of production v6. The subsets are
written once by `build_manifest.py` into a manifest file, and every arm reads the same manifest.

Everything the trainer pickles into its checkpoints (datasets, the LR schedule, the trainer class) lives in this
importable module, so a checkpoint can be re-opened from another process after `import decoder_arch_lib`.
"""

import json
import math
import os
import random
import time
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch_em
from torch_em.data import ConcatDataset, MinInstanceSampler

from micro_sam.v2.datasets.generalist_loader import _configure_training_normalization, _prepare_data_loader
from micro_sam.v2.datasets.wrapper import UniDataWrapper
from micro_sam.v2.loss import DirectedDistanceLoss
from micro_sam.v2.models.util import UniSAM2
from micro_sam.v2.training.sam2_trainer import UniSAM2Trainer
from micro_sam.v2.transforms.labels import GeodesicHybridDistanceTransform, _plantseg_label_trafo
from micro_sam.v2.transforms.raw import _cellpose_raw_trafo, _identity, _normalize_percentile, _to_8bit

DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"
EXPERIMENT_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/decoder_arch"
MANIFEST_PATH = os.path.join(EXPERIMENT_ROOT, "manifests", "manifest_v1.json")

MODEL_TYPE = "hvit_t"
OUTPUT_CHANNELS = 5  # foreground, three directed distances, boundary (production v6)
INITIAL_FEATURES = 32  # production width of the default decoder
PATCH_SHAPE = (512, 512)
Z_3D = 8
BATCH_SIZE_2D = 8  # 8 images per 2D step
BATCH_SIZE_3D = 1  # one crop with Z = 8 per 3D step
GROUP_2D, GROUP_3D = 2, (3, Z_3D)

ENCODER_LR = 1e-5
DECODER_LR = 1e-4
WEIGHT_DECAY = 0.1
WARMUP_ITERATIONS = 500

# The arms. Step 1: A, B, C. Step 2: D, E. 'aux_weights' are the weights of the deep supervision losses
# (MedCAGD's 0.4 / 0.3 / 0.2 for the stages at strides 16, 8, 4; the main output has weight 1).
ARMS: Dict[str, Dict] = {
    "A": {"name": "current", "decoder_type": "unetr"},
    "B": {"name": "current_skips", "decoder_type": "unetr_skip"},
    "C": {"name": "hiera_u", "decoder_type": "hiera_u"},
    "D": {"name": "cagd_3d", "decoder_type": "cagd_3d", "aux_weights": (0.4, 0.3, 0.2)},
    "E": {"name": "c3f2", "decoder_type": "c3f2"},
}
STEP1_ARMS = ("A", "B", "C")

# Anisotropy (z / xy) of the 3D datasets for the geodesic targets, as in the generalist loader.
EMBEDSEG_SAMPLING = {
    "Mouse-Organoid-Cells-CBG": (6, 1, 1),
    "Mouse-Skull-Nuclei-CBG": (5, 1, 1),
    "Platynereis-ISH-Nuclei-CBG": None,
    "Platynereis-Nuclei-CBG": (5, 1, 1),
}


def run_name(arm: str, seed: int) -> str:
    return f"arm{arm}_{ARMS[arm]['name']}_seed{seed}"


def load_manifest(path: str = MANIFEST_PATH) -> Dict:
    with open(path) as f:
        return json.load(f)


#
# Data
#


def _label_transform(sampling=None):
    return GeodesicHybridDistanceTransform(with_boundaries=True, sampling=sampling)


def _rois(unit) -> Optional[List]:
    if unit.get("rois") is None:
        return None
    # torch-em needs a tuple of slices for every file of a multi-file dataset; a whole volume is slice(None).
    return [(slice(None),) if roi is None else (slice(roi[0], roi[1]),) for roi in unit["rois"]]


def build_unit_dataset(unit: Dict):
    """The torch-em dataset of one manifest unit, wrapped for the UniSAM2 batch sampler.

    The per-dataset settings (raw transform, keys, sampler, label transform) follow the generalist loader.
    """
    source, ndim = unit["source"], unit["ndim"]
    n_samples = unit.get("n_samples")
    kwargs = {
        "label_dtype": torch.float32,
        "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0]),
        "label_transform2": _label_transform(),
        "raw_transform": _identity,
        "n_samples": n_samples,
    }
    if ndim == 2:
        kwargs.update({"patch_shape": PATCH_SHAPE, "ndim": 2})
    else:
        kwargs.update({"patch_shape": (Z_3D, *PATCH_SHAPE), "is_seg_dataset": True})
    raw_key = label_key = None

    if source == "livecell":
        kwargs.update({"is_seg_dataset": False, "sampler": MinInstanceSampler(min_num_instances=6, exclude_ids=[0])})
    elif source == "cellpose":
        kwargs.update({"is_seg_dataset": False, "raw_transform": _cellpose_raw_trafo})
    elif source == "tissuenet":
        raw_key, label_key = "raw/rgb", "labels/cell"
        kwargs.update({
            "is_seg_dataset": True, "with_channels": True, "raw_transform": partial(_normalize_percentile, axis=(1, 2)),
        })
    elif source == "nis3d":
        kwargs.update({"raw_transform": _to_8bit})
    elif source == "plantseg_root":
        raw_key, label_key = "raw", "label"
        kwargs.update({
            "label_transform2": partial(_plantseg_label_trafo, data="root", label_trafo=_label_transform()),
            "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0, 1]),
        })
    elif source.startswith("embedseg:"):
        collection = source.split(":", 1)[1]
        kwargs.update({"raw_transform": _to_8bit, "label_transform2": _label_transform(EMBEDSEG_SAMPLING[collection])})
    else:
        raise ValueError(f"Unknown source {source}")

    rois = _rois(unit)
    if rois is not None:
        kwargs["rois"] = rois if len(rois) > 1 else rois[0]
    dataset = torch_em.default_segmentation_dataset(
        raw_paths=unit["raw_paths"] if len(unit["raw_paths"]) > 1 or ndim == 2 else unit["raw_paths"][0],
        raw_key=raw_key,
        label_paths=unit["label_paths"] if len(unit["label_paths"]) > 1 or ndim == 2 else unit["label_paths"][0],
        label_key=label_key,
        **kwargs,
    )
    if ndim == 2:
        return UniDataWrapper(dataset, source_ndim=2, group_key=GROUP_2D)
    return UniDataWrapper(dataset, source_ndim=3, group_key=GROUP_3D)


def build_datasets(manifest: Dict) -> Tuple[List, List]:
    train = [build_unit_dataset(unit) for unit in manifest["units"] if unit["split"] == "train"]
    val = [build_unit_dataset(unit) for unit in manifest["units"] if unit["split"] == "val"]
    _configure_training_normalization(train, val)
    return train, val


def build_loaders(manifest: Dict, n_workers: int, val_workers: int):
    """The train and validation loaders: 8 images per 2D batch, one Z = 8 crop per 3D batch."""
    train, val = build_datasets(manifest)
    per_group = {GROUP_2D: BATCH_SIZE_2D, GROUP_3D: BATCH_SIZE_3D}
    train_loader = _prepare_data_loader(
        ConcatDataset(*train), batch_size=1, shuffle=True, batch_size_per_group=per_group, num_workers=n_workers,
    )
    val_loader = _prepare_data_loader(
        ConcatDataset(*val), batch_size=1, shuffle=False, batch_size_per_group=per_group, num_workers=val_workers,
        deterministic=True,
    )
    return train_loader, val_loader


#
# Model, optimizer, schedule, trainer
#


def build_model(arm: str, device) -> UniSAM2:
    """UniSAM2 with the arm's decoder on the original SAM2.1 encoder; the decoder is initialised from scratch."""
    return UniSAM2(
        encoder=MODEL_TYPE, output_channels=OUTPUT_CHANNELS, initial_features=INITIAL_FEATURES,
        decoder_type=ARMS[arm]["decoder_type"], decoder_kwargs=ARMS[arm].get("decoder_kwargs"), device=device,
    )


def parameter_groups(model: torch.nn.Module):
    """The encoder (with the SAM2 stride-4 / 8 projections) at the encoder LR, everything else at the decoder LR."""
    encoder = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    decoder = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    return [{"params": encoder, "lr": ENCODER_LR}, {"params": decoder, "lr": DECODER_LR}]


class WarmupCosine:
    """Linear warm-up, then cosine decay to zero at the last iteration (a picklable LR multiplier)."""

    def __init__(self, total_iterations: int, warmup_iterations: int = WARMUP_ITERATIONS):
        self.total_iterations = int(total_iterations)
        self.warmup_iterations = int(warmup_iterations)

    def __call__(self, iteration: int) -> float:
        if iteration < self.warmup_iterations:
            return (iteration + 1) / self.warmup_iterations
        progress = (iteration - self.warmup_iterations) / max(1, self.total_iterations - self.warmup_iterations)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


class IterationLR(torch.optim.lr_scheduler.LambdaLR):
    """A per-iteration LambdaLR. torch-em steps the scheduler with the validation metric after every epoch;
    that call is ignored, the trainer below steps it after every optimizer step."""

    def step(self, metric=None):
        if metric is None:
            super().step()


class DecoderArchTrainer(UniSAM2Trainer):
    """UniSAM2Trainer that steps the LR schedule per iteration, records the decoder in its checkpoints and appends
    every validation to '<checkpoint folder>/validation.csv'. Set 'step_timestamps' to a list to record the time
    after every optimizer step (calibration).

    With 'aux_weights' (arm D), the training loss adds the weighted losses of the model's deep supervision outputs
    (`UniSAM2.auxiliary_outputs`, only produced in training mode). The validation loss is the loss of the main
    output alone, as for every other arm."""

    step_timestamps = None
    aux_weights = None

    def _forward_and_loss(self, x, y):
        pred, loss = super()._forward_and_loss(x, y)
        model = self.model.module if hasattr(self.model, "module") else self.model
        aux = getattr(model, "auxiliary_outputs", None)
        if self.aux_weights and model.training:
            if aux is None or len(aux) != len(self.aux_weights):
                raise RuntimeError(f"Expected {len(self.aux_weights)} auxiliary outputs, got "
                                   f"{None if aux is None else len(aux)}.")
            for weight, output in zip(self.aux_weights, aux):
                loss = loss + weight * self.loss(output, y)
        model.auxiliary_outputs = None
        return pred, loss

    def _after_step(self):
        self.lr_scheduler.step()
        if self.step_timestamps is not None:
            torch.cuda.synchronize()
            self.step_timestamps.append(time.perf_counter())

    def _backprop(self, loss):
        super()._backprop(loss)
        self._after_step()

    def _backprop_mixed(self, loss):
        super()._backprop_mixed(loss)
        self._after_step()

    def _validate_impl(self, forward_context):
        started = time.perf_counter()
        metric = super()._validate_impl(forward_context)
        path = os.path.join(self.checkpoint_folder, "validation.csv")
        new = not os.path.exists(path)
        with open(path, "a") as f:
            if new:
                f.write("time,iteration,epoch,val_loss,encoder_lr,decoder_lr,val_seconds\n")
            lrs = [group["lr"] for group in self.optimizer.param_groups]
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{self._iteration},{self._epoch},{metric:.6f},"
                    f"{lrs[0]:.3e},{lrs[-1]:.3e},{time.perf_counter() - started:.1f}\n")
        return metric

    def save_checkpoint(self, name, current_metric, best_metric, **extra_save_dict):
        model = self.model.module if hasattr(self.model, "module") else self.model
        extra_save_dict.setdefault("decoder_config", model.decoder_config)
        super().save_checkpoint(name, current_metric, best_metric, **extra_save_dict)


def build_loss():
    return DirectedDistanceLoss(mask_distances_in_bg=True, with_boundaries=True, boundary_dice_weight=0.5)


def build_trainer(name, model, train_loader, val_loader, total_iterations, device, save_root, log_image_interval=600,
                  aux_weights=None):
    optimizer = torch.optim.AdamW(
        parameter_groups(model), weight_decay=WEIGHT_DECAY, fused=torch.device(device).type == "cuda",
    )
    scheduler = IterationLR(optimizer, WarmupCosine(total_iterations))
    loss = build_loss()
    trainer = DecoderArchTrainer(
        name=name, model=model, train_loader=train_loader, val_loader=val_loader, loss=loss, metric=loss,
        optimizer=optimizer, device=device, lr_scheduler=scheduler, mixed_precision=True,
        mixed_precision_dtype="bfloat16", early_stopping=None, log_image_interval=log_image_interval,
        logger=_logger_class(), logger_kwargs=None, id_=None, save_root=save_root, compile_model=False, rank=None,
    )
    trainer.aux_weights = tuple(aux_weights) if aux_weights else None
    return trainer


def _logger_class():
    from micro_sam.v2.training.sam2_trainer import UniSAM2Logger
    return UniSAM2Logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
