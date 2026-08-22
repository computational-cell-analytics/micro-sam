"""Train the UniSAM2 automatic branch on GoNuclear with the euclidean or the geodesic hybrid target.

The two runs differ only in ``label_transform2``, so the comparison isolates the distance
representation. GoNuclear has 5 volumes and no dataset-level train/val/test split anywhere in the
codebase, so this script holds out one whole volume for validation and final evaluation; the rest
train. Trains for a fixed number of epochs (10 by default) rather than iterations.

The oracle gap (ground truth fields fed straight into the AIS v2 post-processing) is mSA 0.94
euclidean vs. 0.997 hybrid. This script tests whether that ceiling survives training.

Run both arms and then evaluate them:

    python train_geodesic_gonuclear.py train --target euclidean --save_root /path/to/runs
    python train_geodesic_gonuclear.py train --target hybrid --save_root /path/to/runs
    python train_geodesic_gonuclear.py evaluate --target euclidean --save_root /path/to/runs
    python train_geodesic_gonuclear.py evaluate --target hybrid --save_root /path/to/runs

``--dry_run`` builds the loaders and pulls a single batch, which checks the target shapes without
touching a GPU.
"""

import os
import json
import time
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from elf.evaluation import mean_segmentation_accuracy

import torch_em
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_gonuclear_dataset

from micro_sam.v2.transforms.raw import _identity
from micro_sam.v2.datasets.wrapper import UniDataWrapper
from micro_sam.v2.training.training import train_automatic
from micro_sam.v2.postprocessing import DEFAULT_POSTPROCESSING
from micro_sam.v2.automatic_segmentation import get_predictor_and_segmenter
from micro_sam.v2.transforms.labels import (
    DirectedPerObjectBoundaryDistanceTransform, GeodesicHybridDistanceTransform
)

from common import ITER_GRID_3D, DENSITY_GRID, SIGMA_GRID, load_gonuclear

TARGETS = {
    "euclidean": DirectedPerObjectBoundaryDistanceTransform,
    "hybrid": GeodesicHybridDistanceTransform,
}

# The post-processing preset each target is tuned for.
PRESETS = {"euclidean": "sparse", "hybrid": "sparse_hybrid"}

# All valid GoNuclear sample ids; one is held out for validation and evaluation.
ALL_SAMPLES = (1135, 1136, 1137, 1139, 1170)


def train_val_samples(args):
    """@private"""
    train_samples = tuple(s for s in ALL_SAMPLES if s != args.val_sample)
    return train_samples, (args.val_sample,)


def get_loaders(args):
    """Build GoNuclear loaders whose only difference between the arms is the label transform."""
    train_samples, val_samples = train_val_samples(args)
    kwargs = {
        "path": os.path.join(args.input_path, "gonuclear"),
        "patch_shape": tuple(args.patch_shape),
        "raw_transform": _identity,
        "label_transform2": TARGETS[args.target](),
        "sampler": MinInstanceSampler(min_num_instances=args.min_num_instances, exclude_ids=[0]),
        "label_dtype": torch.float32,
        "download": args.download,
    }
    train_ds = UniDataWrapper(
        get_gonuclear_dataset(sample_ids=train_samples, n_samples=args.n_train, **kwargs), source_ndim=3,
    )
    val_ds = UniDataWrapper(
        get_gonuclear_dataset(sample_ids=val_samples, n_samples=args.n_val, **kwargs), source_ndim=3,
    )

    train_loader = torch_em.get_data_loader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers
    )
    val_loader = torch_em.get_data_loader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers
    )
    return train_loader, val_loader


def run_name(args):
    """@private"""
    name = f"gonuclear_{args.target}"
    if args.initial_features != 64:
        name = f"{name}_if{args.initial_features}"
    return name


def run_dry_run(args):
    """@private"""
    train_loader, val_loader = get_loaders(args)
    x, y = next(iter(train_loader))
    print(f"target={args.target}  raw {tuple(x.shape)} {x.dtype}  labels {tuple(y.shape)} {y.dtype}")
    print(f"foreground channel unique: {torch.unique(y[:, 0])[:4].tolist()}")
    for channel, axis in enumerate("zyx", start=1):
        values = y[:, channel]
        print(f"d{axis}: min {values.min():.3f} max {values.max():.3f} nonzero {int((values != 0).sum())}")
    print(f"train batches {len(train_loader)}, val batches {len(val_loader)}")

    # The label transform runs in the dataloader workers, so its cost competes with the GPU step.
    start, n_batches = time.perf_counter(), 0
    for _ in train_loader:
        n_batches += 1
    elapsed = time.perf_counter() - start
    print(
        f"{elapsed / max(n_batches, 1):.3f} s per batch of {args.batch_size} "
        f"over {n_batches} batches with {args.n_workers} workers"
    )


def run_training(args):
    """@private"""
    train_loader, val_loader = get_loaders(args)
    train_automatic(
        name=run_name(args),
        model_type=args.model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.n_epochs,
        n_iterations=args.n_iterations,
        lr=args.lr,
        save_root=args.save_root,
        early_stopping=args.early_stopping,
        initial_features=args.initial_features,
    )


def settings_grid(n_settings):
    """The same 3d post-processing grid the oracle experiment swept, optionally subsampled."""
    grid = [
        {"n_iter": n_iter, "density_threshold": threshold, "sigma": sigma}
        for n_iter in ITER_GRID_3D["sparse"] for threshold in DENSITY_GRID["sparse"] for sigma in SIGMA_GRID["sparse"]
    ]
    if n_settings is not None and n_settings < len(grid):
        step = len(grid) / n_settings
        grid = [grid[int(i * step)] for i in range(n_settings)]
    return grid


def run_evaluation(args):
    """@private"""
    checkpoint = os.path.join(args.save_root or ".", "checkpoints", run_name(args), "best.pt")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"No checkpoint at {checkpoint}. Train this arm first.")

    predictor, segmenter = get_predictor_and_segmenter(
        model_type=args.model_type, checkpoint=checkpoint, segmentation_mode="ais", ndim=3,
    )
    _, val_samples = train_val_samples(args)
    samples = load_gonuclear(
        os.path.join(args.input_path, "gonuclear"), val_samples, args.eval_shape, args.min_size, args.sampling
    )
    grid = settings_grid(args.n_settings)
    print(f"Evaluating {len(samples)} volumes over {len(grid)} post-processing settings.")

    rows = []
    for sample in tqdm(samples, desc="Segmenting"):
        volume = np.stack([sample["image"]] * 3, axis=-1)
        segmenter.initialize(volume, ndim=3)
        for setting in grid:
            segmentation = segmenter.generate(mode="sparse", **setting)
            score = mean_segmentation_accuracy(segmentation.astype("uint32"), sample["labels"])
            rows.append({"sample": sample["name"], "target": args.target, **setting, "mSA": score})

    table = pd.DataFrame(rows)
    keys = ["n_iter", "density_threshold", "sigma"]
    per_setting = table.groupby(keys)["mSA"].mean()
    best = per_setting.idxmax()
    print(f"\n{args.target}: best mSA {per_setting.max():.4f} at {dict(zip(keys, best))}")

    preset = DEFAULT_POSTPROCESSING[PRESETS[args.target]]
    at_preset = table[
        (table.n_iter == preset["n_iter"]) & (table.density_threshold == preset["density_threshold"])
        & (table.sigma == preset["sigma"])
    ]["mSA"]
    if len(at_preset):
        print(f"{args.target}: mSA {at_preset.mean():.4f} at the '{PRESETS[args.target]}' preset")

    if args.result_path is not None:
        with open(args.result_path, "w") as f:
            json.dump({"target": args.target, "checkpoint": checkpoint, "rows": rows}, f, indent=2)
        print(f"Wrote the per sample results to {args.result_path}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "evaluate", "dry_run"], help="What to run.")
    parser.add_argument("--target", choices=list(TARGETS), required=True, help="Which distance target to use.")
    parser.add_argument("--input_path", default="./data", help="Root folder holding the 'gonuclear' folder.")
    parser.add_argument("--save_root", default="./runs", help="Where checkpoints and logs are written.")
    parser.add_argument("--model_type", default="hvit_t", help="The SAM2 backbone.")
    parser.add_argument(
        "--val_sample", type=int, default=1170, choices=ALL_SAMPLES,
        help="GoNuclear sample id held out of training for validation and final evaluation."
    )
    parser.add_argument(
        "--patch_shape", type=int, nargs=3, default=[6, 512, 512], help="The training patch shape (z, y, x)."
    )
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size.")
    parser.add_argument(
        "--n_train", type=int, default=None, help="Samples drawn per epoch. None uses all training volumes."
    )
    parser.add_argument("--n_val", type=int, default=25, help="Validation crops drawn per epoch.")
    parser.add_argument("--n_epochs", type=int, default=10, help="Training epochs. Ignored if --n_iterations is set.")
    parser.add_argument("--n_iterations", type=int, default=None, help="Training iterations, overrides --n_epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate.")
    parser.add_argument("--early_stopping", type=int, default=None, help="Epochs without improvement to stop after.")
    parser.add_argument("--initial_features", type=int, default=64, help="Width of the convolutional decoder.")
    parser.add_argument("--min_num_instances", type=int, default=4, help="Minimum instances a training crop needs.")
    parser.add_argument("--n_workers", type=int, default=8, help="DataLoader workers.")
    parser.add_argument("--download", action="store_true", help="Download GoNuclear if it is missing.")
    parser.add_argument(
        "--eval_shape", type=int, nargs=3, default=[64, 256, 256], help="Centered crop shape used for evaluation."
    )
    parser.add_argument("--sampling", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="Voxel size for evaluation.")
    parser.add_argument("--n_settings", type=int, default=None, help="Subsample the post-processing grid.")
    parser.add_argument("--min_size", type=int, default=50, help="Objects below this size are discarded in the GT.")
    parser.add_argument("--result_path", default=None, help="Write the per sample evaluation to this json.")
    args = parser.parse_args()

    if args.mode == "dry_run":
        run_dry_run(args)
    elif args.mode == "train":
        run_training(args)
    else:
        run_evaluation(args)


if __name__ == "__main__":
    main()
