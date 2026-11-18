"""Finetune the joint micro-sam2 model 'hvit_l_cells' on the Gruen lab crops.

The three labeled crops are (16, 512, 512) stacks with three channels and 3d instance labels. The crop with the
fewest objects validates, the other two train. Each crop feeds a 2d task (single slices) and a 3d task (z-stacks).
Training reads the raw uint16 intensities, which the training pipeline percentile-normalizes per channel.
"""

import os
import re
import argparse
from glob import glob
from functools import partial

import h5py
import tifffile
import numpy as np

import torch

import torch_em
from torch_em.data import MinInstanceSampler, ConcatDataset

from micro_sam.util import get_device
from micro_sam.v2.models.util import UniSAM2
from micro_sam.v2.datasets.wrapper import UniDataWrapper
from micro_sam.v2.transforms.raw import _normalize_percentile
from micro_sam.v2.loss.custom_sam2_loss import CustomSAM2Loss
from micro_sam.v2.transforms.labels import _JointGeodesicLabelTransform
from micro_sam.v2.loss.directed_distance_based import DirectedDistanceLoss
from micro_sam.v2.training.training import _init_sam2_decoder, _configure_joint_speed, _check_compile
from micro_sam.v2.datasets.generalist_loader import _configure_training_normalization, _prepare_data_loader
from micro_sam.v2.training import get_sam2_train_model, ConvertToSam2VideoBatch, JointSam2Trainer, JointSam2Logger


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data/gruen_lab_data"
RAW_STACK = "70EtOH_10um_EDTA_WGA_CD45_RSP9_14539_RGB.tif"  # (C, Y, X, Z) = (3, 4370, 4352, 30), uint16

# The annotated crops cover raw z-slices 13 to 28, found by correlating the normalized crops with the raw stack.
Z_START, Z_STOP = 13, 29
CROP_SIZE = 512

# The stack stores red (membrane), green (cell body) and blue (nuclei). The v6 generalist saw multi-channel cell
# data as membrane, nucleus, extra stain, so 'raw' is stored as membrane, nuclei, cell body.
CHANNEL_ORDER = (0, 2, 1)
CHANNEL_NAMES = ("membrane", "nuclei", "cell_body")

MODEL_TYPE = "hvit_l_cells"

# The batch size of the 2d group and the z-slices of the 3d group follow the v6 'hvit_l' config. The objects per
# interactive step are 3 instead of 5: with 5, a 3d batch of these dense crops runs out of memory on an 80GB A100.
BATCH_SIZE_2D, Z_SLICES, MAX_NUM_OBJECTS = 8, 8, 3


def prepare_data(out_dir):
    """Cut the annotated crops out of the raw stack and store them as channel-first h5 files.

    Each file holds 'raw' (the uint16 raw intensities in the order of CHANNEL_ORDER), 'raw_normalized'
    (the uint8 crop as normalized by the Gruen lab, in the original RGB order, kept for reference) and 'labels'
    (the instance labels). The raw arrays are (C, Z, Y, X), as torch-em expects for multi-channel data.

    Args:
        out_dir: The folder for the h5 files.

    Returns:
        The paths to the h5 files, one per crop.
    """
    os.makedirs(out_dir, exist_ok=True)
    label_paths = sorted(glob(os.path.join(DATA_ROOT, "masks", "*_mask.tif")))
    assert len(label_paths) == 3, label_paths
    raw_stack = tifffile.memmap(os.path.join(DATA_ROOT, RAW_STACK), mode="r")

    paths = []
    for label_path in label_paths:
        crop, y, x = map(int, re.search(r"_crop(\d+)_y(\d+)_x(\d+)_mask", label_path).groups())
        out_path = os.path.join(out_dir, f"crop{crop}.h5")
        paths.append(out_path)
        if os.path.exists(out_path):
            continue

        labels = tifffile.imread(label_path)  # (Z, Y, X)
        raw = raw_stack[list(CHANNEL_ORDER), y:y + CROP_SIZE, x:x + CROP_SIZE, Z_START:Z_STOP].transpose(0, 3, 1, 2)
        normalized_path = label_path.replace("/masks/", "/RGB/").replace("_mask.tif", ".tif")
        raw_normalized = tifffile.imread(normalized_path).transpose(3, 0, 1, 2)  # (Z, Y, X, C) -> (C, Z, Y, X)
        assert raw.shape == raw_normalized.shape == (3, *labels.shape), (raw.shape, raw_normalized.shape)

        with h5py.File(out_path, "w") as f:
            f.attrs.update({"source": RAW_STACK, "y": y, "x": x, "z_start": Z_START, "z_stop": Z_STOP})
            ds = f.create_dataset("raw", data=np.ascontiguousarray(raw), compression="gzip")
            ds.attrs.update({"channel_names": list(CHANNEL_NAMES), "source_channels": list(CHANNEL_ORDER)})
            f.create_dataset("raw_normalized", data=raw_normalized, compression="gzip")
            f.create_dataset("labels", data=labels.astype("uint32"), compression="gzip")
    return paths


def split_crops(paths):
    """Validate on the crop with the fewest objects and train on the others."""
    n_objects = []
    for path in paths:
        with h5py.File(path, "r") as f:
            n_objects.append(len(np.unique(f["labels"][:])) - 1)
    val_idx = int(np.argmin(n_objects))
    for path, n in zip(paths, n_objects):
        print(f"{os.path.basename(path)}: {n} objects ({'val' if path == paths[val_idx] else 'train'})")
    return [p for i, p in enumerate(paths) if i != val_idx], [paths[val_idx]]


def get_dataset(paths, ndim, n_samples):
    """Get the 2d (single slices) or the 3d (z-stacks) dataset for the crops.

    The raw transform normalizes each channel over the patch: over the slice in 2d, over the stack in 3d.
    The training normalization later replaces the fixed percentiles with random ones.
    """
    if ndim == 2:
        patch_shape, axis, group_key = (1, 512, 512), (1, 2), 2
    else:
        patch_shape, axis, group_key = (Z_SLICES, 512, 512), (1, 2, 3), (3, Z_SLICES)

    ds = torch_em.default_segmentation_dataset(
        raw_paths=paths, raw_key="raw", label_paths=paths, label_key="labels",
        patch_shape=patch_shape, ndim=ndim, with_channels=True, is_seg_dataset=True, n_samples=n_samples,
        raw_transform=partial(_normalize_percentile, axis=axis),
        label_transform2=_JointGeodesicLabelTransform(with_boundaries=True),
        sampler=MinInstanceSampler(min_num_instances=3, exclude_ids=[0]),
        label_dtype=torch.float32,
    )
    return UniDataWrapper(ds, source_ndim=ndim, group_key=group_key)


def get_loaders(data_dir, n_samples_2d, n_samples_3d, n_val_samples_2d, n_val_samples_3d, n_workers):
    """Build the train and val loaders with a 2d and a 3d dataset each."""
    train_paths, val_paths = split_crops(prepare_data(data_dir))

    train_ds = [get_dataset(train_paths, 2, n_samples_2d), get_dataset(train_paths, 3, n_samples_3d)]
    val_ds = [get_dataset(val_paths, 2, n_val_samples_2d), get_dataset(val_paths, 3, n_val_samples_3d)]
    _configure_training_normalization(train_ds, val_ds)

    batch_size_per_group = {2: BATCH_SIZE_2D}
    train_loader = _prepare_data_loader(
        ConcatDataset(*train_ds), batch_size=1, shuffle=True,
        batch_size_per_group=batch_size_per_group, num_workers=n_workers,
    )
    val_loader = _prepare_data_loader(
        ConcatDataset(*val_ds), batch_size=1, shuffle=False,
        batch_size_per_group=batch_size_per_group, num_workers=n_workers, deterministic=True,
    )
    return train_loader, val_loader


def get_models(device, compile):
    """Build the interactive SAM2Train and the automatic UniSAM2 model with a shared image encoder.

    Both start from the registered 'hvit_l_cells' weights and decoder. The prompt settings match the v6 recipe.
    """
    sam2_model = get_sam2_train_model(
        model_type=MODEL_TYPE, device=device,
        prob_to_use_pt_input=1.0,
        prob_to_use_box_input=0.5,
        num_frames_to_correct=2,
        rand_frames_to_correct=True,
        prob_to_sample_from_gt=0.1,
        add_all_frames_to_correct_as_cond=True,
        num_correction_pt_per_frame=7,
        num_init_cond_frames_for_train=2,
        bidirectional=True,
    )
    unetr = UniSAM2(
        encoder=sam2_model.image_encoder, output_channels=5, initial_features=32, device=device,
        perform_range_checks=False,
    )
    _init_sam2_decoder(unetr, MODEL_TYPE)
    _configure_joint_speed(sam2_model, unetr, _check_compile(compile))
    return sam2_model, unetr


def train(args):
    device = get_device(None)
    train_loader, val_loader = get_loaders(
        args.data_dir, args.n_samples_2d, args.n_samples_3d, args.n_val_samples_2d, args.n_val_samples_3d,
        args.n_workers,
    )
    compile = ["encoder", "decoder", "loss"] if args.compile else None
    sam2_model, unetr = get_models(device, compile)

    sam2_params = list(sam2_model.parameters())
    unetr_decoder_params = [p for n, p in unetr.named_parameters() if not n.startswith("encoder")]
    optimizer = torch.optim.AdamW(
        sam2_params + unetr_decoder_params, lr=args.lr, fused=torch.device(device).type == "cuda"
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=args.patience)

    trainer = JointSam2Trainer(
        name=args.name or f"gruen_lab_{MODEL_TYPE}",
        model=sam2_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        lr_scheduler=scheduler,
        logger=JointSam2Logger,
        log_image_interval=50,
        compile_model=False,
        mixed_precision=True,
        mixed_precision_dtype="bfloat16",
        early_stopping=args.early_stopping,
        save_root=args.save_root,
        unetr=unetr,
        convert_inputs=ConvertToSam2VideoBatch(max_num_objects=MAX_NUM_OBJECTS, largest_first=False),
        interactive_loss=CustomSAM2Loss(
            use_focal_loss=True, focal_weight=1.0, use_object_score_loss=True, average_over_frames=False,
            compile_kernels=args.compile,
        ),
        automatic_loss=DirectedDistanceLoss(mask_distances_in_bg=True, with_boundaries=True, boundary_dice_weight=1.0),
        automatic_metric_weight=0.25,
    )
    fit_kwargs = {"epochs": args.n_epochs} if args.n_iterations is None else {"iterations": args.n_iterations}
    trainer.fit(**fit_kwargs, overwrite_training=True)


def main():
    parser = argparse.ArgumentParser(description="Finetune 'hvit_l_cells' jointly on the Gruen lab crops.")
    parser.add_argument("--data_dir", default=os.path.join(os.getcwd(), "data"), help="Folder for the h5 crops.")
    parser.add_argument("--save_root", default=os.getcwd(), help="Folder for the checkpoints and logs.")
    parser.add_argument("--name", default=None, help="The checkpoint name.")
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--n_iterations", type=int, default=None, help="Train for iterations instead of epochs.")
    parser.add_argument("--n_samples_2d", type=int, default=640, help="2d slices per training epoch (80 batches).")
    parser.add_argument("--n_samples_3d", type=int, default=160, help="3d stacks per training epoch (160 batches).")
    parser.add_argument("--n_val_samples_2d", type=int, default=16, help="2d slices per validation epoch.")
    parser.add_argument("--n_val_samples_3d", type=int, default=9, help="3d stacks per validation epoch.")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=3, help="Epochs without improvement before LR decay.")
    parser.add_argument("--early_stopping", type=int, default=None, help="Epochs without improvement before stopping.")
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True,
        help="Compile the encoder, the decoder and the loss.",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
