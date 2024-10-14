import os
from glob import glob
from pathlib import Path
from natsort import natsorted

import numpy as np
import imageio.v3 as imageio
from skimage.measure import label as connected_components

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

    # Get the dataloaders
    ds_kwargs = {
        "raw_key": None, "label_key": None, "is_seg_dataset": False, "patch_shape": (512, 512),
        "with_channels": True, "raw_transform": _normalize_images, "with_segmentation_decoder": True,
    }
    loader_kwargs = {"batch_size": 1, "shuffle": True, "num_workers": 16}
    train_loader = sam_training.default_sam_loader(
        raw_paths=train_image_paths, label_paths=train_target_paths, **ds_kwargs, **loader_kwargs
    )
    val_loader = sam_training.default_sam_loader(
        raw_paths=val_image_paths, label_paths=val_target_paths, **ds_kwargs, **loader_kwargs
    )
    return train_loader, val_loader


def main():
    # Assuming the same top-level data structure:
    # - images_and_masks
    #     - img<ID>.tif, img<ID>.npy
    #     - ...
    # - models (...)
    # img<ID>.tif
    path = "/scratch/share/cidas/cca/data/image_sc/stellate_cells"

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


if __name__ == "__main__":
    main()
