import os

import numpy as np
import torch_em

from torch_em.transform.label import PerObjectDistanceTransform
from micro_sam.training import train_sam, identity
from micro_sam.util import export_custom_sam_model


def get_loader(split):
    raw_path = "/scratch-emmy/projects/nim00007/fruit-fly-data/nucleus_segmentation/micro_sam/data_block_58.h5"
    label_path = "/scratch-emmy/projects/nim00007/fruit-fly-data/nucleus_segmentation/micro_sam/data_block_58_segmentation.tif"

    batch_size = 2
    if split == "train":
        roi = np.s_[:50, :, :]
        n_samples = batch_size * 100
    else:
        roi = np.s_[50:, :, :]
        n_samples = batch_size * 4

    label_transform = PerObjectDistanceTransform(
        distances=True, boundary_distances=True, directed_distances=False,
        foreground=True, instances=True, min_size=25
    )

    patch_shape = (1, 512, 512)
    loader = torch_em.default_segmentation_loader(
        raw_paths=raw_path, raw_key="raw",
        label_paths=label_path, label_key=None,
        patch_shape=patch_shape, batch_size=batch_size,
        ndim=2, is_seg_dataset=True, rois=roi,
        label_transform=label_transform,
        num_workers=8, shuffle=True, raw_transform=identity,
        n_samples=n_samples
    )
    return loader


def run_training(name):
    train_loader = get_loader("train")
    val_loader = get_loader("val")

    train_sam(
        name=name,
        model_type="vit_b",
        train_loader=train_loader,
        val_loader=val_loader,
    )


def export_model(name):
    export_path = f"./{name}.pth"
    checkpoint_path = os.path.join("checkpoints", name, "best.pt")
    export_custom_sam_model(
        checkpoint_path=checkpoint_path,
        model_type="vit_b",
        save_path=export_path,
    )


def main():
    name = "finetuned-vitb-em-nuclei"
    # run_training(name)
    export_model(name)


if __name__ == "__main__":
    main()
