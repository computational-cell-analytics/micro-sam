"""Train the UniSAM2 automatic branch on LIVECell with the euclidean or the geodesic hybrid target.

The two runs differ only in ``label_transform2``, so the comparison isolates the distance
representation. SHSY5Y is the default cell type because that is where the oracle gap is largest:
feeding ground truth fields into the AIS v2 post-processing reaches mSA 0.62 with the euclidean
target and 0.91 with the hybrid one. This script tests whether that ceiling survives training.

Run both arms and then evaluate them:

    python train_geodesic_livecell.py train --target euclidean --save_root /path/to/runs
    python train_geodesic_livecell.py train --target hybrid    --save_root /path/to/runs
    python train_geodesic_livecell.py evaluate --target euclidean --save_root /path/to/runs
    python train_geodesic_livecell.py evaluate --target hybrid    --save_root /path/to/runs

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
from torch_em.data import MinInstanceSampler, ConcatDataset
from torch_em.data.datasets import get_livecell_dataset

from micro_sam.v2.datasets.wrapper import UniDataWrapper
from micro_sam.v2.transforms.raw import _identity
from micro_sam.v2.postprocessing import DEFAULT_POSTPROCESSING
from micro_sam.v2.transforms.labels import (
    DirectedPerObjectBoundaryDistanceTransform, GeodesicHybridDistanceTransform
)
from micro_sam.v2.training.training import train_automatic
from micro_sam.v2.automatic_segmentation import get_predictor_and_segmenter

from common import DENSITY_GRID, ITER_GRID, SIGMA_GRID, load_livecell

TARGETS = {
    "euclidean": DirectedPerObjectBoundaryDistanceTransform,
    "hybrid": GeodesicHybridDistanceTransform,
}

# The post-processing preset each target is tuned for.
PRESETS = {"euclidean": "sparse", "hybrid": "sparse_hybrid"}


def get_loaders(args):
    """Build LIVECell loaders whose only difference between the arms is the label transform."""
    kwargs = {
        "path": os.path.join(args.input_path, "livecell"),
        "patch_shape": tuple(args.patch_shape),
        "raw_transform": _identity,
        "label_transform2": TARGETS[args.target](),
        "sampler": MinInstanceSampler(min_num_instances=6, exclude_ids=[0]),
        "label_dtype": torch.float32,
        "download": args.download,
    }
    train_ds = ConcatDataset(*[
        UniDataWrapper(
            get_livecell_dataset(split="train", cell_types=[cell_type], n_samples=args.n_train, **kwargs),
            source_ndim=2,
        ) for cell_type in args.cell_types
    ])
    val_ds = ConcatDataset(*[
        UniDataWrapper(
            get_livecell_dataset(split="val", cell_types=[cell_type], n_samples=args.n_val, **kwargs),
            source_ndim=2,
        ) for cell_type in args.cell_types
    ])

    train_loader = torch_em.get_data_loader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers
    )
    val_loader = torch_em.get_data_loader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers
    )
    return train_loader, val_loader


def run_name(args):
    """@private"""
    return f"livecell_{'_'.join(args.cell_types).lower()}_{args.target}"


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
    print(f"{elapsed / max(n_batches, 1):.3f} s per batch of {args.batch_size} "
          f"over {n_batches} batches with {args.n_workers} workers")


def run_training(args):
    """@private"""
    train_loader, val_loader = get_loaders(args)
    train_automatic(
        name=run_name(args),
        model_type=args.model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_iterations=args.n_iterations,
        lr=args.lr,
        save_root=args.save_root,
        early_stopping=args.early_stopping,
        initial_features=args.initial_features,
    )


def settings_grid(n_settings):
    """The same post-processing grid the oracle experiment swept, optionally subsampled."""
    grid = [
        {"n_iter": n_iter, "density_threshold": threshold, "sigma": sigma}
        for n_iter in ITER_GRID["sparse"] for threshold in DENSITY_GRID["sparse"] for sigma in SIGMA_GRID["sparse"]
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
        model_type=args.model_type, checkpoint=checkpoint, segmentation_mode="ais", ndim=2,
    )
    samples = load_livecell(
        os.path.join(args.input_path, "livecell"), args.cell_types, args.n_eval_images, args.min_size
    )
    grid = settings_grid(args.n_settings)
    print(f"Evaluating {len(samples)} images over {len(grid)} post-processing settings.")

    rows = []
    for sample in tqdm(samples, desc="Segmenting"):
        image = np.stack([sample["image"]] * 3, axis=-1) if sample["image"].ndim == 2 else sample["image"]
        segmenter.initialize(image, ndim=2)
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
    parser.add_argument("--input_path", default="./data", help="Root folder holding the 'livecell' folder.")
    parser.add_argument("--save_root", default="./runs", help="Where checkpoints and logs are written.")
    parser.add_argument("--model_type", default="hvit_t", help="The SAM2 backbone.")
    parser.add_argument("--cell_types", nargs="+", default=["SHSY5Y"], help="The LIVECell cell types to train on.")
    parser.add_argument("--patch_shape", type=int, nargs=2, default=[512, 512], help="The training patch shape.")
    parser.add_argument("--batch_size", type=int, default=2, help="The batch size.")
    parser.add_argument("--n_train", type=int, default=400, help="Samples drawn per cell type per epoch.")
    parser.add_argument("--n_val", type=int, default=100, help="Validation samples per cell type.")
    parser.add_argument("--n_iterations", type=int, default=25000, help="Training iterations.")
    parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate.")
    parser.add_argument("--early_stopping", type=int, default=None, help="Epochs without improvement to stop after.")
    parser.add_argument("--initial_features", type=int, default=64, help="Width of the convolutional decoder.")
    parser.add_argument("--n_workers", type=int, default=8, help="DataLoader workers.")
    parser.add_argument("--download", action="store_true", help="Download LIVECell if it is missing.")
    parser.add_argument("--n_eval_images", type=int, default=10, help="Validation images per cell type to score.")
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
