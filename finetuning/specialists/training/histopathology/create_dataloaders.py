import torch

from torch_em.util.debug import check_loader
from torch_em.data.datasets import get_pannuke_loader
from torch_em.transform.label import PerObjectDistanceTransform

import micro_sam.training as sam_training


def get_dataloaders(patch_shape, data_path):
    """This returns the pannuke data loaders implemented in torch_em:
    https://github.com/constantinpape/torch-em/blob/main/torch_em/data/datasets/histopathology/pannuke.py
    It will automatically download the pannuke data.

    Note: to replace this with another data loader you need to return a torch data loader
    that retuns `x, y` tensors, where `x` is the image data and `y` are the labels.
    The labels have to be in a label mask instance segmentation format.
    I.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID.
    Important: the ID 0 is reseved for background, and the IDs must be consecutive
    """
    label_transform = PerObjectDistanceTransform(
        distances=True, boundary_distances=True, directed_distances=False, foreground=True, instances=True, min_size=25
    )
    raw_transform = sam_training.identity  # the current workflow avoids rescaling the inputs to [-1, 1]

    # TODO: update the dataloaders to pannuke's expected inputs
    train_loader = get_pannuke_loader(
        path=data_path, patch_shape=patch_shape, split="train", batch_size=2, num_workers=16,
        download=True, shuffle=True, label_transform=label_transform,
        raw_transform=raw_transform, label_dtype=torch.float32
    )
    val_loader = get_pannuke_loader(
        path=data_path, patch_shape=patch_shape, split="val", batch_size=1, num_workers=16,
        download=True, shuffle=True, label_transform=label_transform,
        raw_transform=raw_transform, label_dtype=torch.float32
    )

    return train_loader, val_loader


def visualize_images(data_path):
    train_loader, val_loader = get_dataloaders(patch_shape=(1, 512, 512), data_path=data_path)

    # let's visualize train loader first
    check_loader(train_loader, 8)


if __name__ == "__main__":
    visualize_images()
