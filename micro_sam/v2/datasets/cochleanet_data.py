"""CochleaNet holds annotations for 3d instance segmentation in light-sheet volumes of the mouse cochlea:
inner hair cells (IHC, Vglut3 stain) and spiral ganglion neurons (SGN, PV stain).

This is an internal dataset of the CochleaNet project. It is not public and cannot be downloaded; the functions
expect the project's export folder, which holds the 'IHC_v11_2026-07' and 'SGN_v3_2026-07' subfolders, each split
into 'train' and 'val', with every volume a tif next to a '<name>_annotations.tif' instance segmentation. The
volumes are converted once to h5 files under 'preprocessed'.
"""

import os
from glob import glob
from typing import List, Tuple, Union, Literal

import h5py
import tifffile
import numpy as np
from natsort import natsorted

from torch.utils.data import Dataset, DataLoader

import torch_em
from torch_em.data.datasets import util

SOURCES = {"sgn": "SGN_v3_2026-07", "ihc": "IHC_v11_2026-07"}

# The 'empty' volumes hold no object. The SGN volumes named 'resized' were interpolated from a coarser acquisition
# and are only partly annotated. The IHC 'G-LR' crops label a few inner hair cells next to unlabelled rows of cells.
EXCLUDED = {"sgn": ("_empty", "resized"), "ihc": ("_empty", "G-LR")}

# These IHC blocks are near-isotropic crops stored with their shortest axis last; that axis serves as z.
Z_LAST_SHAPE = (512, 512, 256)


def _preprocess_split(source_dir, data_dir, excluded):
    os.makedirs(data_dir, exist_ok=True)
    for raw_path in natsorted(glob(os.path.join(source_dir, "*.tif"))):
        name = os.path.basename(raw_path)[:-len(".tif")]
        if name.endswith("_annotations") or any(token in name for token in excluded):
            continue
        h5_path = os.path.join(data_dir, f"{name}.h5")
        if os.path.exists(h5_path):
            continue
        labels = tifffile.imread(os.path.join(source_dir, f"{name}_annotations.tif"))
        if labels.max() == 0:
            continue
        raw = tifffile.imread(raw_path)
        assert raw.shape == labels.shape, f"Shape mismatch for {name}: raw={raw.shape}, labels={labels.shape}"
        if raw.shape == Z_LAST_SHAPE:
            raw, labels = np.moveaxis(raw, 2, 0), np.moveaxis(labels, 2, 0)
        # Some blocks number their objects with 64 bit ids; relabel to 1..n.
        labels = np.unique(labels, return_inverse=True)[1].reshape(labels.shape).astype("uint32")
        chunks = (1, min(512, raw.shape[1]), min(512, raw.shape[2]))
        # Several ranks may convert at once: each writes its own file and the rename is atomic.
        tmp_path = f"{h5_path}.tmp{os.getpid()}"
        with h5py.File(tmp_path, "w") as f:
            f.create_dataset("raw", data=np.ascontiguousarray(raw), chunks=chunks, compression="gzip")
            f.create_dataset("labels", data=np.ascontiguousarray(labels), chunks=chunks, compression="gzip")
        os.replace(tmp_path, h5_path)


def get_cochleanet_data(
    path: Union[os.PathLike, str], name: Literal["sgn", "ihc"], split: Literal["train", "val"]
) -> str:
    """Convert one split of a CochleaNet subset to h5 volumes with 'raw' and 'labels' datasets.

    Args:
        path: The project's export folder.
        name: The subset, 'sgn' or 'ihc'.
        split: The data split, 'train' or 'val'.

    Returns:
        The folder with the converted volumes.
    """
    if name not in SOURCES:
        raise ValueError(f"'{name}' is not a valid subset. Choose one of {list(SOURCES)}.")
    if split not in ("train", "val"):
        raise ValueError(f"'{split}' is not a valid split. Choose 'train' or 'val'.")
    data_dir = os.path.join(path, "preprocessed", name, split)
    _preprocess_split(os.path.join(path, SOURCES[name], split), data_dir, EXCLUDED[name])
    return data_dir


def get_cochleanet_sgn_data(path: Union[os.PathLike, str], split: Literal["train", "val"]) -> str:
    """Convert one split of the spiral ganglion neuron volumes, see `get_cochleanet_data`."""
    return get_cochleanet_data(path, "sgn", split)


def get_cochleanet_ihc_data(path: Union[os.PathLike, str], split: Literal["train", "val"]) -> str:
    """Convert one split of the inner hair cell volumes, see `get_cochleanet_data`."""
    return get_cochleanet_data(path, "ihc", split)


def get_cochleanet_paths(
    path: Union[os.PathLike, str], name: Literal["sgn", "ihc"], split: Literal["train", "val"]
) -> List[str]:
    """Get the paths to the CochleaNet volumes of one subset and split.

    Args:
        path: The project's export folder.
        name: The subset, 'sgn' or 'ihc'.
        split: The data split, 'train' or 'val'.

    Returns:
        The filepaths of the h5 volumes, which hold the 'raw' and 'labels' datasets.
    """
    data_dir = get_cochleanet_data(path, name, split)
    paths = natsorted(glob(os.path.join(data_dir, "*.h5")))
    assert len(paths) > 0
    return paths


def get_cochleanet_dataset(
    path: Union[os.PathLike, str],
    patch_shape: Tuple[int, int, int],
    name: Literal["sgn", "ihc"],
    split: Literal["train", "val"],
    **kwargs
) -> Dataset:
    """Get the CochleaNet dataset for 3d instance segmentation of inner hair cells or spiral ganglion neurons.

    Args:
        path: The project's export folder.
        patch_shape: The patch shape to use for training.
        name: The subset, 'sgn' or 'ihc'.
        split: The data split, 'train' or 'val'.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """
    paths = get_cochleanet_paths(path, name, split)
    kwargs = util.ensure_transforms(ndim=3, **kwargs)
    return torch_em.default_segmentation_dataset(
        raw_paths=paths, raw_key="raw", label_paths=paths, label_key="labels", patch_shape=patch_shape, ndim=3, **kwargs
    )


def get_cochleanet_loader(
    path: Union[os.PathLike, str],
    batch_size: int,
    patch_shape: Tuple[int, int, int],
    name: Literal["sgn", "ihc"],
    split: Literal["train", "val"],
    **kwargs
) -> DataLoader:
    """Get the CochleaNet dataloader for 3d instance segmentation of inner hair cells or spiral ganglion neurons.

    Args:
        path: The project's export folder.
        batch_size: The batch size for training.
        patch_shape: The patch shape to use for training.
        name: The subset, 'sgn' or 'ihc'.
        split: The data split, 'train' or 'val'.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    ds_kwargs, loader_kwargs = util.split_kwargs(torch_em.default_segmentation_dataset, **kwargs)
    dataset = get_cochleanet_dataset(path, patch_shape, name, split, **ds_kwargs)
    return torch_em.get_data_loader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
