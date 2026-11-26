"""Train UniSAM2 for 3d automatic instance segmentation on Lucchi.

The SAM2 image encoder runs on every z slice and a 3d UNETR decoder regresses a foreground mask and
directed distances over the whole volume.
"""

import os
import argparse

import torch

import torch_em
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_lucchi_dataset
from torch_em.transform.label import connected_components

from micro_sam.v2.training import train_automatic
from micro_sam.v2.transforms.raw import _identity
from micro_sam.v2.datasets.wrapper import UniDataWrapper
from micro_sam.v2.transforms.labels import GeodesicHybridDistanceTransform


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def get_dataloaders(input_path, patch_shape, batch_size, n_workers, n_train_samples, n_val_samples):
    """Return the Lucchi train and val loaders with directed distance targets.

    The loaders return 'x, y' tensors of shape (B, 3, Z, Y, X) and (B, 4, Z, Y, X). Lucchi ships a train
    and a test volume, so the test volume validates.

    Args:
        input_path: The folder that holds the Lucchi data.
        patch_shape: The patch shape for training.
        batch_size: The number of patches per batch.
        n_workers: The number of workers per loader.
        n_train_samples: The number of patches per training epoch.
        n_val_samples: The number of patches per validation epoch.

    Returns:
        The train loader and the val loader.
    """
    kwargs = {
        "path": os.path.join(input_path, "lucchi"),
        "patch_shape": patch_shape,
        "raw_transform": _identity,
        "label_transform2": GeodesicHybridDistanceTransform(),
        "label_dtype": torch.float32,
        # The labels are a binary mask, so the sampler needs the instances before it counts them.
        "pre_label_transform": connected_components,
        "sampler": MinInstanceSampler(min_num_instances=2, exclude_ids=[0]),
        "download": True,
    }
    train_ds = UniDataWrapper(get_lucchi_dataset(split="train", n_samples=n_train_samples, **kwargs), source_ndim=3)
    val_ds = UniDataWrapper(get_lucchi_dataset(split="test", n_samples=n_val_samples, **kwargs), source_ndim=3)

    train_loader = torch_em.get_data_loader(train_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    val_loader = torch_em.get_data_loader(val_ds, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    return train_loader, val_loader


def train_lucchi(args):
    """Train UniSAM2 for 3d automatic instance segmentation on Lucchi."""
    patch_shape = (args.z_slices, 512, 512)
    train_loader, val_loader = get_dataloaders(
        args.input_path, patch_shape, args.batch_size, args.n_workers, args.n_train_samples, args.n_val_samples
    )

    train_automatic(
        name=f"lucchi_instance_3d_{args.model_type}",
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
    parser = argparse.ArgumentParser(description="Train UniSAM2 for 3d instance segmentation on Lucchi.")
    parser.add_argument("-i", "--input_path", default=DATA_ROOT, help="The folder that holds the Lucchi data.")
    parser.add_argument("-m", "--model_type", default="hvit_t", choices=["hvit_t", "hvit_s", "hvit_b", "hvit_l"])
    parser.add_argument("-s", "--save_root", default=None, help="Where to save the checkpoints and the logs.")
    parser.add_argument("-e", "--n_epochs", type=int, default=100, help="The number of training epochs.")
    parser.add_argument("--n_iterations", type=int, default=None, help="A fixed iteration budget, instead of epochs.")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="The number of patches per batch.")
    parser.add_argument("-z", "--z_slices", type=int, default=8, help="The number of z slices per patch.")
    parser.add_argument("--initial_features", type=int, default=32, help="The width of the convolutional decoder.")
    parser.add_argument("--n_train_samples", type=int, default=200, help="The number of patches per training epoch.")
    parser.add_argument("--n_val_samples", type=int, default=25, help="The number of patches per validation epoch.")
    parser.add_argument("--n_workers", type=int, default=8, help="The number of workers per loader.")
    args = parser.parse_args()
    train_lucchi(args)


if __name__ == "__main__":
    main()
