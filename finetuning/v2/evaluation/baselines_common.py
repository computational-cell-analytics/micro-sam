"""Shared data-loading utilities for evaluate_automatic_baselines and evaluate_interactive_baselines."""

import os

import numpy as np
import imageio.v3 as imageio
from skimage.measure import label as connected_components

from elf.io import open_file
from torch_em.transform.raw import normalize

from common import get_data_paths, load_volume, _center_crop_roi

CROP_SHAPE_2D = (512, 512)
CROP_SHAPE_3D = (8, 512, 512)
MAX_EVALUATION_SAMPLES = int(os.environ.get("MICRO_SAM_EVAL_MAX_SAMPLES", "200"))


def _ensure_8bit_range(raw):
    raw = raw.astype("float32", copy=False)
    if raw.size == 0:
        return raw
    if raw.max() <= 1:
        raw = raw * 255
    elif raw.max() > 255 or raw.min() < 0:
        raw = normalize(raw) * 255
    return np.clip(raw, 0, 255).astype("float32")


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


def _sorted_path_pairs(raw_paths, label_paths):
    return sorted(zip(raw_paths, label_paths), key=lambda pair: (str(pair[0]), str(pair[1])))


def _load_data(dataset_name, data_root, ndim):
    """Yield (image_or_volume, labels, valid_roi) triples for the given dataset.

    valid_roi is a boolean mask (True = annotated) for partially annotated datasets
    (platynereis_nuclei), or None for all others.
    """
    if ndim == 3:
        raw_paths, label_paths, raw_key, label_key = get_data_paths(dataset_name, data_root)
        path_pairs = _sorted_path_pairs(raw_paths, label_paths)[:MAX_EVALUATION_SAMPLES]
        for raw_path, label_path in path_pairs:
            raw, labels, valid_roi = load_volume(raw_path, label_path, raw_key, label_key, dataset_name, CROP_SHAPE_3D)
            raw = _ensure_8bit_range(raw)
            yield raw, labels, valid_roi
    else:
        image_paths, gt_paths, raw_key, label_key = get_data_paths(dataset_name, data_root)
        path_pairs = _sorted_path_pairs(image_paths, gt_paths)[:MAX_EVALUATION_SAMPLES]
        for img_path, gt_path in path_pairs:
            image = _read_2d(img_path, raw_key)
            image = _ensure_8bit_range(image)
            roi = _center_crop_roi(image.shape[:2], CROP_SHAPE_2D)
            image = image[roi]
            gt = _read_2d(gt_path, label_key)
            gt = connected_components(gt[roi]).astype("uint32")
            yield image, gt, None
