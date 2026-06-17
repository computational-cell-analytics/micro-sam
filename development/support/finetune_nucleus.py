import os
from glob import glob

import torch

from torch_em.data import MinInstanceSampler

import micro_sam.training as sam_training


def get_loaders(data_root, patch_shape, batch_size, train_with_decoder):
    image_dir = os.path.join(data_root, "image")
    mask_dir = os.path.join(data_root, "mask")

    image_paths = sorted(glob(os.path.join(image_dir, "*.tif")))
    mask_paths = sorted(glob(os.path.join(mask_dir, "*.tif")))
    assert len(image_paths) == len(mask_paths) > 0, "No data found - check data_root."

    # Use the last embryo for validation, the rest for training.
    # Splitting by embryo (rather than by frame) avoids the val set being too similar to train.
    train_image_paths = image_paths[:-1]
    train_mask_paths = mask_paths[:-1]
    val_image_paths = image_paths[-1:]
    val_mask_paths = mask_paths[-1:]

    print(f"Train embryos ({len(train_image_paths)}): {[os.path.basename(p) for p in train_image_paths]}")
    print(f"Val embryos ({len(val_image_paths)}): {[os.path.basename(p) for p in val_image_paths]}")

    sampler = MinInstanceSampler(min_size=25)

    # raw_key=None / label_key=None: each tif path is opened as a plain image stack,
    # avoiding the broadcast error from the directory + glob approach.
    train_loader = sam_training.default_sam_loader(
        raw_paths=train_image_paths,
        raw_key=None,
        label_paths=train_mask_paths,
        label_key=None,
        with_segmentation_decoder=train_with_decoder,
        patch_shape=patch_shape,
        batch_size=batch_size,
        shuffle=True,
        raw_transform=sam_training.identity,
        sampler=sampler,
        is_train=True,
    )
    val_loader = sam_training.default_sam_loader(
        raw_paths=val_image_paths,
        raw_key=None,
        label_paths=val_mask_paths,
        label_key=None,
        with_segmentation_decoder=train_with_decoder,
        patch_shape=patch_shape,
        batch_size=batch_size,
        shuffle=True,
        raw_transform=sam_training.identity,
        sampler=sampler,
        is_train=False,
    )

    print(f"Train loader: {len(train_loader)} batches/epoch")
    print(f"Val loader: {len(val_loader)} batches/epoch")
    return train_loader, val_loader


def main():
    data_root = "/mnt/vast-nhr/home/archit/u12090/micro-sam/development/support/nucleus_seg/data"
    save_root = "/mnt/vast-nhr/home/archit/u12090/micro-sam/development/support/nucleus_seg/models"

    checkpoint_name = "sam_embryo_nucleus"
    model_type = "vit_b_lm"
    n_epochs = 100
    n_objects_per_batch = 5
    batch_size = 1
    patch_shape = (1, 512, 512)  # micro-sam is 2D; the leading 1 selects a single frame
    train_with_decoder = True

    train_loader, val_loader = get_loaders(data_root, patch_shape, batch_size, train_with_decoder)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    sam_training.train_sam(
        name=checkpoint_name,
        save_root=save_root,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        n_objects_per_batch=n_objects_per_batch,
        with_segmentation_decoder=train_with_decoder,
        device=device,
    )


if __name__ == "__main__":
    main()
