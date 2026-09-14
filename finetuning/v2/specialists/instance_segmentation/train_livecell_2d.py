"""Train UniSAM2 for 2d automatic instance segmentation on LIVECell.

The SAM2 image encoder feeds a UNETR decoder that regresses a foreground mask and directed distances.
The postprocessing turns those targets into instances.
"""

import os
import argparse

import torch

import torch_em
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_livecell_dataset

from micro_sam.v2.training import train_automatic
from micro_sam.v2.transforms.raw import _identity
from micro_sam.v2.datasets.wrapper import UniDataWrapper
from micro_sam.v2.transforms.labels import GeodesicHybridDistanceTransform


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def get_dataloaders(input_path, patch_shape, batch_size, n_workers, n_val_samples):
    """Return the LIVECell train and val loaders with directed distance targets.

    The loaders return 'x, y' tensors of shape (B, 3, 1, Y, X) and (B, 4, 1, Y, X). The four label channels
    hold the foreground mask and the three directed distances.

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
        "label_transform2": GeodesicHybridDistanceTransform(),
        "label_dtype": torch.float32,
        "sampler": MinInstanceSampler(min_num_instances=6, exclude_ids=[0]),
        "download": True,
    }
    train_ds = UniDataWrapper(get_livecell_dataset(split="train", **kwargs), source_ndim=2)
    val_ds = UniDataWrapper(get_livecell_dataset(split="val", n_samples=n_val_samples, **kwargs), source_ndim=2)

    train_loader = torch_em.get_data_loader(train_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    val_loader = torch_em.get_data_loader(val_ds, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    return train_loader, val_loader


def train_livecell(args):
    """Train UniSAM2 for 2d automatic instance segmentation on LIVECell."""
    patch_shape = (512, 512)
    train_loader, val_loader = get_dataloaders(
        args.input_path, patch_shape, args.batch_size, args.n_workers, args.n_val_samples
    )

    train_automatic(
        name=f"livecell_instance_2d_{args.model_type}",
        model_type=args.model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.n_epochs,
        n_iterations=args.n_iterations,
        save_root=args.save_root,
        lr=1e-5,
        initial_features=args.initial_features,
    )


def main():
    parser = argparse.ArgumentParser(description="Train UniSAM2 for 2d instance segmentation on LIVECell.")
    parser.add_argument("-i", "--input_path", default=DATA_ROOT, help="The folder that holds the LIVECell data.")
    parser.add_argument("-m", "--model_type", default="hvit_t", choices=["hvit_t", "hvit_s", "hvit_b", "hvit_l"])
    parser.add_argument("-s", "--save_root", default=None, help="Where to save the checkpoints and the logs.")
    parser.add_argument("-e", "--n_epochs", type=int, default=100, help="The number of training epochs.")
    parser.add_argument("--n_iterations", type=int, default=None, help="A fixed iteration budget, instead of epochs.")
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="The number of patches per batch.")
    parser.add_argument("--initial_features", type=int, default=32, help="The width of the convolutional decoder.")
    parser.add_argument("--n_val_samples", type=int, default=100, help="The number of patches per validation epoch.")
    parser.add_argument("--n_workers", type=int, default=16, help="The number of workers per loader.")
    args = parser.parse_args()
    train_livecell(args)


if __name__ == "__main__":
    main()
