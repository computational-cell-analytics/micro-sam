"""Shared data-loading utilities for evaluate_automatic_baselines and evaluate_interactive_baselines."""

import numpy as np
import imageio.v3 as imageio
from skimage.measure import label as connected_components

from elf.io import open_file
from torch_em.transform.raw import normalize

from common import get_data_paths, load_volume, _center_crop_roi

CROP_SHAPE_2D = (512, 512)
CROP_SHAPE_3D = (8, 512, 512)


def _read_2d(path, key):
    """Read a 2D array from an image file or from an H5/zarr file using key."""
    if key is not None:
        arr = open_file(path, mode="r")[key][:]
    else:
        arr = np.asarray(imageio.imread(path))
    # Transpose channel-first (C, H, W) to channel-last (H, W, C).
    if arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[1] > arr.shape[0] and arr.shape[2] > arr.shape[0]:
        arr = arr.transpose(1, 2, 0)
    return arr


def _load_data(dataset_name, data_root, ndim):
    """Yield (image_or_volume, labels) pairs for the given dataset."""
    if ndim == 3:
        raw_paths, label_paths, raw_key, label_key = get_data_paths(dataset_name, data_root)
        for raw_path, label_path in zip(raw_paths, label_paths):
            raw, labels = load_volume(raw_path, label_path, raw_key, label_key, dataset_name, CROP_SHAPE_3D)
            yield raw, labels
    else:
        image_paths, gt_paths, raw_key, label_key = get_data_paths(dataset_name, data_root)
        for img_path, gt_path in zip(image_paths, gt_paths):
            image = _read_2d(img_path, raw_key)
            if image.max() > 255:
                image = normalize(image) * 255
            roi = _center_crop_roi(image.shape[:2], CROP_SHAPE_2D)
            image = image[roi].astype("float32")
            gt = _read_2d(gt_path, label_key)
            gt = connected_components(gt[roi]).astype("uint32")
            yield image, gt
