"""Finetune SAM2 for 2d interactive segmentation on LIVECell.

The model learns SAM2's native prompting: point and box prompts with iterative correction clicks.
"""

import os
import argparse

import torch

import torch_em
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_livecell_dataset
from torch_em.transform.label import connected_components

from micro_sam.v2.training import train_sam2
from micro_sam.v2.transforms.raw import _identity
from micro_sam.v2.datasets.wrapper import UniDataWrapper


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def get_dataloaders(input_path, patch_shape, batch_size, n_workers, n_val_samples):
    """Return the LIVECell train and val loaders with instance labels.

    The loaders return 'x, y' tensors of shape (B, 3, 1, Y, X) and (B, 1, 1, Y, X). The singleton z axis
    turns every image into a one frame video, which is how SAM2 reads a 2d input.

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
        "label_transform2": connected_components,
        "label_dtype": torch.int64,
        "sampler": MinInstanceSampler(min_num_instances=6, exclude_ids=[0]),
        "download": True,
    }
    train_ds = UniDataWrapper(get_livecell_dataset(split="train", **kwargs), source_ndim=2)
    val_ds = UniDataWrapper(get_livecell_dataset(split="val", n_samples=n_val_samples, **kwargs), source_ndim=2)

    train_loader = torch_em.get_data_loader(train_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    val_loader = torch_em.get_data_loader(val_ds, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    return train_loader, val_loader


def finetune_livecell(args):
    """Finetune SAM2 for 2d interactive segmentation on LIVECell."""
    patch_shape = (512, 512)
    train_loader, val_loader = get_dataloaders(
        args.input_path, patch_shape, args.batch_size, args.n_workers, args.n_val_samples
    )

    train_sam2(
        name=f"livecell_interactive_2d_{args.model_type}",
        model_type=args.model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.n_epochs,
        n_iterations=args.n_iterations,
        max_num_objects=args.max_num_objects,
        checkpoint_path=args.checkpoint_path,
        save_root=args.save_root,
        lr=1e-5,
        prob_to_use_pt_input=1.0,  # always point or box prompts, never the ground-truth mask
        prob_to_use_box_input=0.5,  # conditional probability of a box instead of a click
        num_correction_pt_per_frame=7,  # correction clicks per round
        num_frames_to_correct=1,  # a 2d image is a single frame
        prob_to_sample_from_gt=0.1,
        largest_first=True,
        use_focal_loss=True,
        focal_weight=1.0,  # keep the focal loss on equal footing with dice, where SAM2 uses 20
        use_object_score_loss=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Finetune SAM2 for 2d interactive segmentation on LIVECell.")
    parser.add_argument("-i", "--input_path", default=DATA_ROOT, help="The folder that holds the LIVECell data.")
    parser.add_argument("-m", "--model_type", default="hvit_t", choices=["hvit_t", "hvit_s", "hvit_b", "hvit_l"])
    parser.add_argument("-s", "--save_root", default=None, help="Where to save the checkpoints and the logs.")
    parser.add_argument("-c", "--checkpoint_path", default=None, help="A custom checkpoint to start the training from.")
    parser.add_argument("-e", "--n_epochs", type=int, default=100, help="The number of training epochs.")
    parser.add_argument("--n_iterations", type=int, default=None, help="A fixed iteration budget, instead of epochs.")
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="The number of patches per batch.")
    parser.add_argument("--max_num_objects", type=int, default=8, help="The number of objects prompted per patch.")
    parser.add_argument("--n_val_samples", type=int, default=100, help="The number of patches per validation epoch.")
    parser.add_argument("--n_workers", type=int, default=16, help="The number of workers per loader.")
    args = parser.parse_args()
    finetune_livecell(args)


if __name__ == "__main__":
    main()
