import os
from glob import glob
from pathlib import Path
from natsort import natsorted

import numpy as np
import imageio.v3 as imageio
from skimage.measure import label as connected_components

import torch_em
from torch_em.transform.raw import normalize

import micro_sam.training as sam_training


def _normalize_images(raw):
    raw = normalize(raw) * 255
    return raw


def _get_data_loaders(path):
    label_paths = natsorted(glob(os.path.join(path, "images_and_masks", "*_seg.npy")))
    image_paths = natsorted(glob(os.path.join(path, "images_and_masks", "*.tif")))
    image_paths = [ipath for ipath in image_paths if not ipath.endswith("_seg.tif")]

    # Store labels as tif.
    target_paths = []
    for lpath in label_paths:
        target_path = str(Path(lpath).with_suffix(".tif"))
        target_paths.append(target_path)
        if os.path.exists(target_path):
            continue

        label = np.load(lpath, allow_pickle=True)
        # Extract masks.
        label = label.item()["masks"]
        label = connected_components(label)
        imageio.imwrite(target_path, label, compression="zlib")

    # Divide paths into splits.
    train_image_paths, train_target_paths = image_paths[:100], target_paths[:100]
    val_image_paths, val_target_paths = image_paths[100:], target_paths[100:]

    # Get the dataset.
    label_transform = torch_em.transform.label.PerObjectDistanceTransform(
        distances=True, boundary_distances=True, directed_distances=False, foreground=True, instances=True, min_size=25,
    )
    ds_kwargs = {
        "is_seg_dataset": False, "patch_shape": (1024, 1024), "with_channels": True,
        "raw_transform": _normalize_images, "label_transform": label_transform,
    }
    train_ds = torch_em.default_segmentation_dataset(
        raw_paths=train_image_paths, raw_key=None, label_paths=train_target_paths, label_key=None, **ds_kwargs
    )
    val_ds = torch_em.default_segmentation_dataset(
        raw_paths=val_image_paths, raw_key=None, label_paths=val_target_paths, label_key=None, **ds_kwargs
    )

    # Get the dataloader
    loader_kwargs = {"batch_size": 1, "shuffle": True, "num_workers": 16}
    return torch_em.get_data_loader(train_ds, **loader_kwargs), torch_em.get_data_loader(val_ds, **loader_kwargs)


def main():
    # Assuming the same top-level data structure:
    # - images_and_masks
    #     - img<ID>.tif, img<ID>.npy
    #     - ...
    # - models (...)
    # img<ID>.tif
    path = "/media/anwai/ANWAI/data/image_sc/stellate_cells"

    train_loader, val_loader = _get_data_loaders(path)

    # Verify loaders.
    # from torch_em.util.debug import check_loader
    # check_loader(train_loader, 8)
    # check_loader(val_loader, 8)

    sam_training.train_sam(
        name="stellate_cells",
        model_type="vit_b_lm",
        train_loader=train_loader,
        val_loader=val_loader,
    )

    breakpoint()


if __name__ == "__main__":
    main()
