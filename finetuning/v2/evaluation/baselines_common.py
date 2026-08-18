"""Shared data-loading utilities for evaluate_automatic_baselines and evaluate_interactive_baselines."""

import warnings

import numpy as np
import imageio.v3 as imageio
from skimage.measure import label as connected_components

from elf.io import open_file

from torch_em.util.segmentation import size_filter

from micro_sam.v2.normalization import normalize_raw

from common import get_data_paths, load_volume, GT_MIN_SIZE_2D, _center_crop_roi

CROP_SHAPE_2D = (512, 512)
CROP_SHAPE_3D = (8, 512, 512)


def _ensure_8bit_range(raw):
    if raw.size == 0:
        return raw.astype("float32", copy=False)
    return normalize_raw(raw) * 255.0


def _read_2d(path, key):
    """Read a 2D array from an image file or from an H5/zarr file using key."""
    if key is not None:
        arr = open_file(path, mode="r")[key][:]
    else:
        arr = np.asarray(imageio.imread(path))
    # Transpose channel-first (C, H, W) to channel-last (H, W, C).
    if arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[1] > arr.shape[0] and arr.shape[2] > arr.shape[0]:
        arr = arr.transpose(1, 2, 0)
    # Some 2d datasets mix in multi-frame stacks, e.g. yeaz. Evaluate their first frame.
    if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
        arr = arr[0]
    return arr


def _sorted_path_pairs(raw_paths, label_paths):
    return sorted(zip(raw_paths, label_paths), key=lambda pair: (str(pair[0]), str(pair[1])))


def interactive_result_name(
    dataset_name, method, model_type, prompt, iteration,
    ndim=2, use_masks=True, mask_threshold=0.0, min_size=0,
):
    """Name of the result CSV for one iteration of an interactive run.

    The single definition of this convention. The evaluation writes these names, the status check and
    the aggregation read them, so all three must import this rather than rebuild the name. Every
    setting that changes the numbers is encoded, otherwise a run would silently reuse another's
    results. The 3d path has no mask tag, since its logits masks always go through the video
    predictor, and no threshold tag, since only the 2d path binarizes mask logits itself.
    """
    dim_suffix = "" if ndim == 2 else "_3d"
    tag = interactive_run_tag(ndim, use_masks, mask_threshold, min_size)
    return f"{dataset_name}_{method}_{model_type}{dim_suffix}_{prompt}{tag}_iter{iteration:02d}.csv"


def interactive_run_tag(ndim=2, use_masks=True, mask_threshold=0.0, min_size=0):
    """The settings suffix shared by an interactive run's result names and prediction directory.

    Keeping both on one tag means a run can never read back another run's cached predictions.
    """
    tag = "" if ndim == 3 else ("_with_masks" if use_masks else "_without_masks")
    if ndim == 2 and mask_threshold != 0.0:
        tag += f"_t{mask_threshold:g}"
    if min_size:
        tag += f"_min{min_size}"
    return tag


def _apply_min_size(labels, min_size, dataset_name):
    """Drop ground-truth objects below 'min_size' pixels, warning if that guts the dataset.

    A single threshold does not suit every dataset: gonuclear nuclei have a median of ~3200 pixels
    per object, while cremi neurite cross-sections in a thin crop have a median of ~6, where the
    same threshold would remove most of the genuine annotations.
    """
    if not min_size:
        return labels
    before = len(np.unique(labels)) - 1
    filtered = size_filter(seg=labels, min_size=min_size)
    after = len(np.unique(filtered)) - 1
    if before and (before - after) / before > 0.25:
        warnings.warn(
            f"min_size={min_size} removes {before - after} of {before} ground-truth objects in "
            f"'{dataset_name}'. That is more than a quarter, so the threshold is likely too large "
            f"for this dataset and is discarding real annotations."
        )
    return filtered


def drop_severed_objects(labels, min_size):
    """Drop the objects that a crop face cut down to a sliver, in a ground truth or a prediction alike.

    An object touching a face is only partly in the image, but one that is still large is perfectly
    segmentable, so only the small remnants go. Both sides are filtered the same way, so removing a
    remnant from the ground truth does not turn the prediction of it into a false positive.

    Both conditions are needed. A bare size threshold also deletes whole interior objects that
    simply are small: half of the livecell objects below 50 pixels are those, while on
    dynamicnuclearnet every one of them does touch a face. Contact alone would instead delete large,
    perfectly usable cells that merely reach the edge.
    """
    if not min_size:
        return labels
    if labels.ndim == 2:
        edges = (labels[0], labels[-1], labels[:, 0], labels[:, -1])
    else:
        # In-plane faces only. A thin z-crop touches nearly every object on its top and bottom slice.
        edges = (labels[:, 0], labels[:, -1], labels[:, :, 0], labels[:, :, -1])
    border_ids = np.unique(np.concatenate([np.unique(edge) for edge in edges]))
    border_ids = border_ids[border_ids != 0]
    if border_ids.size == 0:
        return labels

    ids, sizes = np.unique(labels[labels != 0], return_counts=True)
    severed = np.intersect1d(border_ids, ids[sizes < min_size], assume_unique=True)
    if severed.size == 0:
        return labels
    return np.where(np.isin(labels, severed), 0, labels).astype(labels.dtype)


def load_evaluation_sample_2d(raw_path, label_path, raw_key, label_key, dataset_name):
    """Load one 2d sample the way the evaluation scores it.

    The single source of truth for what an image and its ground truth are, so that a parameter
    search selects on exactly the data the evaluation measures.
    """
    # Normalized before cropping: the percentiles are taken over the whole image.
    image = _ensure_8bit_range(_read_2d(raw_path, raw_key))
    roi = _center_crop_roi(image.shape[:2], CROP_SHAPE_2D)
    gt = connected_components(_read_2d(label_path, label_key)[roi]).astype("uint32")
    return image[roi], drop_severed_objects(gt, GT_MIN_SIZE_2D.get(dataset_name, 0))


def load_evaluation_sample_3d(
    raw_path, label_path, raw_key, label_key, dataset_name,
    crop_shape=CROP_SHAPE_3D, z_range=None, min_size=0,
):
    """Load one volumetric sample the way the evaluation scores it."""
    raw, labels, valid_roi = load_volume(
        raw_path, label_path, raw_key, label_key, dataset_name, crop_shape, z_range=z_range
    )
    return _ensure_8bit_range(raw), _apply_min_size(labels, min_size, dataset_name), valid_roi


def _load_data(dataset_name, data_root, ndim, min_size=0):
    """Yield (image_or_volume, labels, valid_roi) triples for the given dataset.

    valid_roi is a boolean mask (True = annotated) for partially annotated datasets
    (platynereis_nuclei), or None for all others.

    A 2d ground truth is filtered by `drop_border_objects`, which is what cropping makes necessary.
    `min_size` additionally drops volumetric objects below that many voxels, and defaults to keeping
    all of them. Filtering must happen here, the single source of the labels used for both prompting
    and scoring: filtering only the prompting copy would leave the dropped objects in the scored
    ground truth as unmatched ones.
    """
    if ndim == 3:
        raw_paths, label_paths, raw_key, label_key = get_data_paths(dataset_name, data_root)
        for raw_path, label_path in _sorted_path_pairs(raw_paths, label_paths):
            yield load_evaluation_sample_3d(
                raw_path, label_path, raw_key, label_key, dataset_name, min_size=min_size
            )
    else:
        image_paths, gt_paths, raw_key, label_key = get_data_paths(dataset_name, data_root)
        for img_path, gt_path in _sorted_path_pairs(image_paths, gt_paths):
            image, gt = load_evaluation_sample_2d(img_path, gt_path, raw_key, label_key, dataset_name)
            yield image, gt, None
