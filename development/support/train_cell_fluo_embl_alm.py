import os
from glob import glob
from natsort import natsorted

import torch

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model


def train_embl_alm_data(checkpoint_name):
    """Training a MicroSAM model for https://github.com/computational-cell-analytics/micro-sam/issues/1084.
    """
    # All hyperparameters for training.
    batch_size = 1
    patch_shape = (512, 512)
    n_objects_per_batch = 25
    device = torch.device("cuda")

    # Get the filepaths to images and corresponding labels.
    image_paths = natsorted(glob(os.path.join(os.getcwd(), "data_same_size", "*.tif")))
    label_paths = natsorted(glob(os.path.join(os.getcwd(), "masks_same_size", "*.tif")))

    # Next, prepare the dataloaders.
    kwargs = {
        "batch_size": batch_size,
        "patch_shape": patch_shape,
        "with_segmentation_decoder": True,
        "num_workers": 16,
        "shuffle": True,
    }

    train_loader = sam_training.default_sam_loader(
        raw_paths=image_paths[:-5], raw_key=None, label_paths=label_paths[:-5], label_key=None, **kwargs,
    )
    val_loader = sam_training.default_sam_loader(
        raw_paths=image_paths[-5:], raw_key=None, label_paths=label_paths[-5:], label_key=None, **kwargs,
    )

    # Run training.
    sam_training.train_sam(
        name=checkpoint_name,
        model_type="vit_b_lm",
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=10,
        n_objects_per_batch=n_objects_per_batch,
        with_segmentation_decoder=True,
        device=device,
    )


def main():
    checkpoint_name = "sam_embl_alm_fluo"  # Name of the checkpoint, stored at "./checkpoints/<CHECKPOINT_NAME>"

    train_embl_alm_data(checkpoint_name)

    # Export the trained model.
    export_custom_sam_model(
        checkpoint_path=os.path.join("checkpoints", checkpoint_name, "best.pt"),
        model_type="vit_b",
        save_path="./finetuned_embl_alm_fluo_model.pth",
    )


if __name__ == "__main__":
    main()
