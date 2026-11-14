"""Data loaders for training micro-sam2 models on your own data.

The images and labels can be regular image files (e.g. tif or png), given as a list of files or as a folder
with a glob pattern, or container files (e.g. h5, zarr, ome-zarr or n5), given with the internal keys of the
image and label data. `default_sam2_loader` returns a loader in the format that the trainer of each training
mode expects:

- 'interactive': `micro_sam.v2.training.train_sam2`, for interactive segmentation.
- 'automatic': `micro_sam.v2.training.train_automatic`, for automatic instance segmentation (AIS and APG).
- 'joint': `micro_sam.v2.training.train_joint_sam2`, for interactive and automatic segmentation.
- 'semantic': `micro_sam.v2.training.train_semantic`, for semantic segmentation.
"""

import os
from glob import glob
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import imageio.v3 as imageio

import torch
from torch.utils.data import DataLoader, Dataset

from elf.io import open_file

FilePath = Union[str, os.PathLike]

TRAINING_MODES = ("interactive", "automatic", "joint", "semantic")
# The smallest number of samples per epoch, so that an epoch of a small dataset is not just a few steps.
MIN_TRAIN_SAMPLES = 100
MIN_VAL_SAMPLES = 5
GLOB_CHARACTERS = ("*", "?", "[")


def _is_pattern(key: Optional[str]) -> bool:
    return key is not None and any(character in key for character in GLOB_CHARACTERS)


def _ome_zarr_scale_key(path, key):
    """The key of the first (full resolution) scale, if the key points to an ome-zarr image or label group."""
    import zarr

    if not os.path.isdir(path) or not str(path).rstrip("/").endswith(".zarr"):
        return key
    try:
        node = zarr.open(path, mode="r") if not key else zarr.open(path, mode="r", path=key)
    except Exception:
        return key
    if not isinstance(node, zarr.Group):
        return key
    attrs = node.attrs.asdict()
    multiscales = attrs.get("ome", attrs).get("multiscales")
    if not multiscales:
        return key
    scale = multiscales[0]["datasets"][0]["path"]
    return scale if not key else f"{key.rstrip('/')}/{scale}"


def resolve_data_paths(
    paths: Union[FilePath, Sequence[FilePath], Sequence[np.ndarray]], key: Optional[str],
) -> Tuple[list, Optional[str]]:
    """Resolve the image or label paths and key that a user passes to a list of inputs and a key.

    A folder with a glob pattern as key, e.g. ('images/', '*.tif'), is expanded to the sorted list of matching
    files, and the key becomes None. Each file is then one image (2d) or one volume (3d). A container file
    (e.g. h5, zarr or n5) keeps its key, which selects the data inside the file. For an ome-zarr file, the
    key can also be the image or label group, e.g. None for the image and 'labels/cells' for the labels:
    it then selects the full resolution scale of that group.

    Args:
        paths: A file path, a folder, or a list of either. A list of numpy arrays is returned as it is.
        key: The key of the data inside a container file, a glob pattern for a folder, or None for image files.

    Returns:
        The list of inputs, and the key for them.

    Raises:
        ValueError: If a folder with a glob pattern contains no matching file.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    paths = list(paths)
    if len(paths) > 0 and isinstance(paths[0], (np.ndarray, torch.Tensor)):
        return paths, None
    if not _is_pattern(key):
        paths = [str(path) for path in paths]
        return paths, _ome_zarr_scale_key(paths[0], key) if paths else key

    files = []
    for folder in paths:
        matches = sorted(glob(os.path.join(folder, key)))
        if len(matches) == 0:
            raise ValueError(f"There are no files matching '{key}' in '{folder}'.")
        files.extend(matches)
    return files, None


def _data_shape(path, key):
    if isinstance(path, (np.ndarray, torch.Tensor)):
        return tuple(path.shape)
    if key is None and str(path).lower().endswith((".tif", ".tiff")):
        # The image properties of a tif file describe its first page only, not a stack of pages.
        import tifffile
        with tifffile.TiffFile(path) as f:
            return tuple(f.series[0].shape)
    if key is None:
        return tuple(imageio.improps(path).shape)
    with open_file(path, mode="r") as f:
        return tuple(f[key].shape)


def _is_image_with_channels(shape):
    """Whether the shape is a 2d image with a channel axis, either (C, Y, X) or (Y, X, C)."""
    return len(shape) == 3 and (shape[-1] <= 4 or shape[0] <= 4)


def _label_settings(mode, with_boundaries, num_classes):
    """The label transform, the label dtype and the default sampler for a training mode."""
    from torch_em.data.sampler import MinInstanceSampler
    from ..transforms.labels import _instance_labels, _JointGeodesicLabelTransform, GeodesicHybridDistanceTransform

    if mode == "interactive":
        # Each connected region becomes one object, for prompting with points, boxes and masks.
        return _instance_labels, torch.int64, MinInstanceSampler(min_num_instances=2)
    if mode == "automatic":
        transform = GeodesicHybridDistanceTransform(with_boundaries=with_boundaries)
        return transform, torch.float32, MinInstanceSampler(min_num_instances=2)
    if mode == "joint":
        # The instance ids for the interactive branch, followed by the targets of the automatic branch.
        transform = _JointGeodesicLabelTransform(with_boundaries=with_boundaries)
        return transform, torch.float32, MinInstanceSampler(min_num_instances=2)
    if num_classes is None:
        raise ValueError("Pass 'num_classes' for the 'semantic' training mode.")
    # The labels are the class ids. The sampler rejects patches with the background class only.
    return None, torch.int64, MinInstanceSampler(min_num_instances=2)


def default_sam2_dataset(
    raw_paths: Union[FilePath, Sequence[FilePath], Sequence[np.ndarray]],
    raw_key: Optional[str],
    label_paths: Union[FilePath, Sequence[FilePath], Sequence[np.ndarray]],
    label_key: Optional[str],
    patch_shape: Tuple[int, ...],
    mode: str,
    ndim: Optional[int] = None,
    with_boundaries: bool = True,
    num_classes: Optional[int] = None,
    n_samples: Optional[int] = None,
    is_train: bool = True,
    sampler: Optional[Callable] = None,
    label_transform: Optional[Callable] = None,
    rois: Optional[Union[Tuple[slice, ...], List[Tuple[slice, ...]]]] = None,
    with_channels: Optional[bool] = None,
    **kwargs,
) -> Dataset:
    """Create a dataset for training a micro-sam2 model on images and their labels.

    The images are normalized to [0, 1] with `micro_sam.v2.normalization.normalize_raw`. Grayscale images
    are copied to three channels. The labels are converted to the targets of the training mode.

    Args:
        raw_paths: The images. Either a file path, a folder (with a glob pattern in `raw_key`), a list of file
            paths, or a list of numpy arrays. Supported are regular image files (e.g. tif or png) and container
            files (e.g. h5, zarr, ome-zarr or n5).
        raw_key: The key of the images: the internal dataset in a container file, e.g. 'raw', a glob pattern
            for a folder, e.g. '*.tif', or None for image files. For an ome-zarr file, None selects the full
            resolution image, and the path of a scale, e.g. 's0', selects that scale.
        label_paths: The labels, in the same form as `raw_paths`. For 'interactive', 'automatic' and 'joint'
            training, these are instance segmentations, with 0 for the background. For 'semantic' training,
            these are the class ids, with 0 for the background.
        label_key: The key of the labels, in the same form as `raw_key`. For the labels of an ome-zarr file,
            this is the label group, e.g. 'labels/cells', which selects its full resolution scale.
        patch_shape: The shape of the training patches. Use (y, x) or (1, y, x) to train on 2d images or on
            the 2d slices of volumes, and (z, y, x) to train on 3d patches of volumes.
        mode: The training mode: 'interactive', 'automatic', 'joint' or 'semantic'.
        ndim: The number of spatial dimensions to train in (2 or 3). By default, it is derived from `patch_shape`.
        with_boundaries: Whether the targets of the automatic decoder include the object boundaries.
            The decoders of the micro-sam2 models, e.g. 'hvit_t_cells', have them, so this must be True to
            finetune them. Only used for the 'automatic' and the 'joint' training mode.
        num_classes: The number of classes, the background included. Only used for 'semantic' training.
        n_samples: The number of samples per epoch. By default, the number of images, or of patches that fit
            into the volumes, and at least 100 for training and 5 for validation.
        is_train: Whether this dataset is used for training or for validation. Only affects the default `n_samples`.
        sampler: A sampler that rejects patches, e.g. patches without enough objects. By default, a patch needs
            one object and background (or one class and background for 'semantic' training).
        label_transform: An optional transform applied to the labels before the targets of the training mode
            are computed. For example, `micro_sam.v2.transforms.labels.semantic_labels` converts an instance
            segmentation into background, boundary and interior classes for 'semantic' training.
            It must be importable from a module, because the checkpoint stores a reference to it.
        rois: The region(s) of interest of the data, e.g. to train and validate on different parts of one volume.
        with_channels: Whether the images of a container file have a channel axis. By default, image files are
            checked for a channel axis, and container data is expected without one.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The dataset. Each sample is a pair of tensors: the image, with shape (3, z, y, x), and the targets,
        with shape (channels, z, y, x). A 2d image has a single z slice.

    Raises:
        ValueError: If the training mode, the patch shape or the data is not valid.
    """
    import torch_em
    from ..normalization import normalize_raw
    from ..datasets.wrapper import UniDataWrapper

    if mode not in TRAINING_MODES:
        raise ValueError(f"Invalid training mode '{mode}'. Choose one of {TRAINING_MODES}.")
    patch_shape = tuple(int(size) for size in patch_shape)
    if len(patch_shape) not in (2, 3):
        raise ValueError(f"The patch shape must have two or three entries, got {patch_shape}.")
    if ndim is None:
        ndim = 3 if len(patch_shape) == 3 and patch_shape[0] > 1 else 2
    if ndim not in (2, 3):
        raise ValueError(f"The number of dimensions must be 2 or 3, got {ndim}.")
    if ndim == 3 and len(patch_shape) != 3:
        raise ValueError(f"Training in 3d needs a (z, y, x) patch shape, got {patch_shape}.")

    raw_paths, raw_key = resolve_data_paths(raw_paths, raw_key)
    label_paths, label_key = resolve_data_paths(label_paths, label_key)
    if len(raw_paths) != len(label_paths):
        raise ValueError(f"There are {len(raw_paths)} images, but {len(label_paths)} label images.")
    if len(raw_paths) == 0:
        raise ValueError("There are no images.")

    # A list of 2d images (e.g. tif or png files, with or without channels) becomes an image collection,
    # in which the images can have different shapes. Volumes and container data become a segmentation dataset.
    shape = _data_shape(raw_paths[0], raw_key)
    if ndim == 3 and len(shape) < 3:
        raise ValueError(f"Training in 3d needs volumes, but the images have the shape {shape}.")
    is_image = raw_key is None and (len(shape) == 2 or (ndim == 2 and _is_image_with_channels(shape)))

    if is_image:
        is_seg_dataset, data_patch_shape = False, patch_shape[-2:]
        with_channels = len(shape) == 3 if with_channels is None else with_channels
    else:
        is_seg_dataset = True
        with_channels = False if with_channels is None else with_channels
        spatial_ndim = len(shape) - int(with_channels)
        if ndim == 3:
            data_patch_shape = patch_shape
        elif spatial_ndim == 2:
            data_patch_shape = patch_shape[-2:]
        else:  # The 2d slices of a volume.
            data_patch_shape = (1,) + patch_shape[-2:]

    label_transform2, label_dtype, default_sampler = _label_settings(mode, with_boundaries, num_classes)
    dataset_kwargs = dict(
        raw_paths=raw_paths, raw_key=raw_key, label_paths=label_paths, label_key=label_key,
        patch_shape=data_patch_shape, raw_transform=normalize_raw, label_transform=label_transform,
        label_transform2=label_transform2, label_dtype=label_dtype, rois=rois,
        sampler=default_sampler if sampler is None else sampler,
        ndim=ndim, is_seg_dataset=is_seg_dataset, with_channels=with_channels, **kwargs,
    )
    dataset = torch_em.default_segmentation_dataset(n_samples=n_samples, **dataset_kwargs)
    min_samples = MIN_TRAIN_SAMPLES if is_train else MIN_VAL_SAMPLES
    if n_samples is None and len(dataset) < min_samples:
        dataset = torch_em.default_segmentation_dataset(n_samples=min_samples, **dataset_kwargs)

    # The micro-sam2 trainers expect every sample as (channels, z, y, x).
    return UniDataWrapper(dataset, source_ndim=ndim)


def default_sam2_loader(
    raw_paths: Union[FilePath, Sequence[FilePath], Sequence[np.ndarray]],
    raw_key: Optional[str],
    label_paths: Union[FilePath, Sequence[FilePath], Sequence[np.ndarray]],
    label_key: Optional[str],
    patch_shape: Tuple[int, ...],
    mode: str,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
    **dataset_kwargs,
) -> DataLoader:
    """Create a data loader for training a micro-sam2 model on images and their labels.

    Pass the loader to the trainer of the training mode: `train_sam2` ('interactive'), `train_automatic`
    ('automatic'), `train_joint_sam2` ('joint') or `train_semantic` ('semantic'), all in `micro_sam.v2.training`.

    Example:
        >>> train_loader = default_sam2_loader(
        ...     "data/train.h5", "raw", "data/train.h5", "labels/cells", patch_shape=(512, 512), mode="joint",
        ... )
        >>> val_loader = default_sam2_loader(
        ...     "data/val.h5", "raw", "data/val.h5", "labels/cells", patch_shape=(512, 512), mode="joint",
        ...     is_train=False,
        ... )
        >>> train_joint_sam2("my_model", "hvit_t_cells", train_loader=train_loader, val_loader=val_loader)

    Args:
        raw_paths: The images, see `default_sam2_dataset`.
        raw_key: The key of the images, see `default_sam2_dataset`.
        label_paths: The labels, see `default_sam2_dataset`.
        label_key: The key of the labels, see `default_sam2_dataset`.
        patch_shape: The shape of the training patches, see `default_sam2_dataset`.
        mode: The training mode: 'interactive', 'automatic', 'joint' or 'semantic'.
        batch_size: The batch size.
        num_workers: The number of loader workers.
        shuffle: Whether to shuffle the samples. By default, they are shuffled for training only.
        dataset_kwargs: Additional keyword arguments for `default_sam2_dataset`.

    Returns:
        The data loader.
    """
    import torch_em

    dataset = default_sam2_dataset(raw_paths, raw_key, label_paths, label_key, patch_shape, mode, **dataset_kwargs)
    if shuffle is None:
        shuffle = dataset_kwargs.get("is_train", True)
    return torch_em.get_data_loader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def split_train_val_files(
    paths: List, label_paths: List, val_fraction: float = 0.1, seed: int = 42,
) -> Tuple[List, List, List, List]:
    """Split the images and labels at the file level, into a training and a validation set.

    Args:
        paths: The image files, as returned by `resolve_data_paths`.
        label_paths: The label files, in the same order.
        val_fraction: The fraction of the files that are used for validation. At least one file is used.
        seed: The seed for the random split.

    Returns:
        The training images, the training labels, the validation images and the validation labels.

    Raises:
        ValueError: If there are fewer than two files.
    """
    if len(paths) < 2:
        raise ValueError(
            "The data has only one file, so it cannot be split into a training and a validation set. "
            "Pass separate validation data."
        )
    n_val = max(1, int(round(val_fraction * len(paths))))
    order = np.random.default_rng(seed).permutation(len(paths))
    val_ids, train_ids = sorted(order[:n_val]), sorted(order[n_val:])
    return (
        [paths[i] for i in train_ids], [label_paths[i] for i in train_ids],
        [paths[i] for i in val_ids], [label_paths[i] for i in val_ids],
    )
