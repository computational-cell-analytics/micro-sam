"""Finetune SAM2 for 3d interactive segmentation on Lucchi.

The z slices of a volume become the frames of a video. SAM2 propagates a prompt through them with its
memory attention, and correction clicks refine the propagated masks.
"""

import os
import argparse

import torch

import torch_em
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_lucchi_dataset
from torch_em.transform.label import connected_components

from micro_sam.v2.training import train_sam2
from micro_sam.v2.transforms.raw import _identity
from micro_sam.v2.datasets.wrapper import UniDataWrapper


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def get_dataloaders(input_path, patch_shape, batch_size, n_workers, n_train_samples, n_val_samples):
    """Return the Lucchi train and val loaders with mitochondria instances.

    The loaders return 'x, y' tensors of shape (B, 3, Z, Y, X) and (B, 1, Z, Y, X). Lucchi ships a train
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
        "label_transform2": connected_components,
        "label_dtype": torch.int64,
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


def finetune_lucchi(args):
    """Finetune SAM2 for 3d interactive segmentation on Lucchi."""
    patch_shape = (args.z_slices, 512, 512)
    train_loader, val_loader = get_dataloaders(
        args.input_path, patch_shape, args.batch_size, args.n_workers, args.n_train_samples, args.n_val_samples
    )

    train_sam2(
        name=f"lucchi_interactive_3d_{args.model_type}",
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
        num_correction_pt_per_frame=7,  # correction clicks per frame per round
        num_frames_to_correct=2,  # the number of frames that receive correction clicks
        rand_frames_to_correct=True,
        prob_to_sample_from_gt=0.1,
        add_all_frames_to_correct_as_cond=True,
        num_init_cond_frames=2,  # the frames that receive the first prompt
        largest_first=True,
        bidirectional=True,  # propagate from a random start slice in both z directions
        use_focal_loss=True,
        focal_weight=1.0,  # keep the focal loss on equal footing with dice, where SAM2 uses 20
        use_object_score_loss=True,
        average_over_frames=False,  # sum over the frames, so a volume keeps its per-slice weight
    )


def main():
    parser = argparse.ArgumentParser(description="Finetune SAM2 for 3d interactive segmentation on Lucchi.")
    parser.add_argument("-i", "--input_path", default=DATA_ROOT, help="The folder that holds the Lucchi data.")
    parser.add_argument("-m", "--model_type", default="hvit_t", choices=["hvit_t", "hvit_s", "hvit_b", "hvit_l"])
    parser.add_argument("-s", "--save_root", default=None, help="Where to save the checkpoints and the logs.")
    parser.add_argument("-c", "--checkpoint_path", default=None, help="A custom checkpoint to start the training from.")
    parser.add_argument("-e", "--n_epochs", type=int, default=100, help="The number of training epochs.")
    parser.add_argument("--n_iterations", type=int, default=None, help="A fixed iteration budget, instead of epochs.")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="The number of patches per batch.")
    parser.add_argument("-z", "--z_slices", type=int, default=8, help="The number of z slices per patch.")
    parser.add_argument("--max_num_objects", type=int, default=8, help="The number of objects prompted per patch.")
    parser.add_argument("--n_train_samples", type=int, default=200, help="The number of patches per training epoch.")
    parser.add_argument("--n_val_samples", type=int, default=25, help="The number of patches per validation epoch.")
    parser.add_argument("--n_workers", type=int, default=8, help="The number of workers per loader.")
    args = parser.parse_args()
    finetune_lucchi(args)


if __name__ == "__main__":
    main()
