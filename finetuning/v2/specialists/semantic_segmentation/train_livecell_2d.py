"""Train a SAM2 based model for 2d semantic segmentation on LIVECell.

The model predicts three classes: background, cell boundary and cell interior.
"""

import os
import argparse

import torch

import torch_em
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_livecell_dataset

from micro_sam.v2.training import train_semantic
from micro_sam.v2.transforms.raw import _identity
from micro_sam.v2.datasets.wrapper import UniDataWrapper
from micro_sam.v2.transforms.labels import semantic_labels


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"
NUM_CLASSES = 3


def get_dataloaders(input_path, patch_shape, batch_size, n_workers, n_val_samples):
    """Return the LIVECell train and val loaders with three class semantic maps.

    The loaders return 'x, y' tensors of shape (B, 3, 1, Y, X) and (B, 1, 1, Y, X). The label values are
    0 for the background, 1 for the cell boundary and 2 for the cell interior.

    Args:
        input_path: The folder that holds the LIVECell data.
        patch_shape: The patch shape for training.
        batch_size: The number of patches per batch.
        n_workers: The number of workers per loader.
        n_val_samples: The number of patches per validation epoch.

    Returns:
        The train loader and the val loader.
    """
    kwargs = {
        "path": os.path.join(input_path, "livecell"),
        "patch_shape": patch_shape,
        "raw_transform": _identity,
        "label_transform2": semantic_labels,
        "label_dtype": torch.int64,
        "sampler": MinInstanceSampler(min_num_instances=6, exclude_ids=[0]),
        "download": True,
    }
    train_ds = UniDataWrapper(get_livecell_dataset(split="train", **kwargs), source_ndim=2)
    val_ds = UniDataWrapper(get_livecell_dataset(split="val", n_samples=n_val_samples, **kwargs), source_ndim=2)

    train_loader = torch_em.get_data_loader(train_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    val_loader = torch_em.get_data_loader(val_ds, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    return train_loader, val_loader


def train_livecell(args):
    """Train a SAM2 based model for 2d semantic segmentation on LIVECell."""
    patch_shape = (512, 512)
    train_loader, val_loader = get_dataloaders(
        args.input_path, patch_shape, args.batch_size, args.n_workers, args.n_val_samples
    )

    train_semantic(
        name=f"livecell_semantic_2d_{args.model_type}",
        model_type=args.model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=NUM_CLASSES,
        n_epochs=args.n_epochs,
        n_iterations=args.n_iterations,
        save_root=args.save_root,
        lr=args.lr,
        initial_features=args.initial_features,
        dice_weight=args.dice_weight,
    )


def main():
    parser = argparse.ArgumentParser(description="Train SAM2 for 2d semantic segmentation on LIVECell.")
    parser.add_argument("-i", "--input_path", default=DATA_ROOT, help="The folder that holds the LIVECell data.")
    parser.add_argument("-m", "--model_type", default="hvit_t", choices=["hvit_t", "hvit_s", "hvit_b", "hvit_l"])
    parser.add_argument("-s", "--save_root", default=None, help="Where to save the checkpoints and the logs.")
    parser.add_argument("-e", "--n_epochs", type=int, default=100, help="The number of training epochs.")
    parser.add_argument("--n_iterations", type=int, default=None, help="A fixed iteration budget, instead of epochs.")
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="The number of patches per batch.")
    parser.add_argument("--lr", type=float, default=1e-5, help="The learning rate.")
    parser.add_argument("--dice_weight", type=float, default=0.5, help="The weight of the dice loss.")
    parser.add_argument("--initial_features", type=int, default=32, help="The width of the convolutional decoder.")
    parser.add_argument("--n_val_samples", type=int, default=100, help="The number of patches per validation epoch.")
    parser.add_argument("--n_workers", type=int, default=16, help="The number of workers per loader.")
    args = parser.parse_args()
    train_livecell(args)


if __name__ == "__main__":
    main()
