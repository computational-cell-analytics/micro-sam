import os
from glob import glob

import torch_em

from torch_em.transform.label import PerObjectDistanceTransform
from micro_sam.training import train_sam, identity


def get_loader(split):
    data_root = "/scratch-grete/projects/nim00007/data/pdo/user_study_v2/user_study_data"
    label_root = "/scratch-grete/projects/nim00007/data/pdo/user_study_v2/result-sam"
    images = sorted(glob(os.path.join(data_root, "*.tif")))
    labels = sorted(glob(os.path.join(label_root, "*.tif")))

    batch_size = 1
    if split == "train":
        n_samples = batch_size * 50
        images = images[:-1]
        labels = labels[:-1]
    else:
        n_samples = batch_size * 4
        images = images[-1:]
        labels = labels[-1:]

    label_transform = PerObjectDistanceTransform(
        distances=True, boundary_distances=True, directed_distances=False,
        foreground=True, instances=True, min_size=25
    )

    patch_shape = (1024, 1024)
    loader = torch_em.default_segmentation_loader(
        raw_paths=images, raw_key=None,
        label_paths=labels, label_key=None,
        patch_shape=patch_shape, batch_size=batch_size,
        ndim=2, is_seg_dataset=True,
        label_transform=label_transform,
        num_workers=8, shuffle=True,
        raw_transform=identity,
        n_samples=n_samples
    )
    return loader


def run_training(name):
    train_loader = get_loader("train")
    val_loader = get_loader("val")

    checkpoint = "/scratch-grete/projects/nim00007/sam/models/new_models/v2/lm/generalist/vit_b/best.pt"

    train_sam(
        name=name,
        model_type="vit_b",
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_path=checkpoint,
    )


def main():
    name = "finetuned-vitb-organoids"
    run_training(name)


if __name__ == "__main__":
    main()
