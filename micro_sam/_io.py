"""Resolve and open input data, including folder containers (zarr, n5 and ome-zarr).

A folder container is a directory that holds zarr or n5 data. Without a key, these rules select the array:
- ome-zarr: the highest resolution of the image at the root. For a segmentation, the single label image in 'labels/'.
- zarr and n5: the single array in the container.
In all other cases, the caller must give a key.
"""

import os
import json
from glob import glob
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import z5py
import zarr
import imageio.v3 as imageio
from elf.io import open_file

CONTAINER_EXTENSIONS = (".ome.zarr", ".zarr", ".zr", ".n5")
ZARR_METADATA_FILES = ("zarr.json", ".zgroup", ".zarray")
# Root attributes of ome-zarr collections (bioformats2raw, HCS) that have no image at the root.
OME_COLLECTION_ATTRIBUTES = ("bioformats2raw.layout", "plate", "well")
MAX_LISTED_KEYS = 10

PathLike = Union[str, os.PathLike]


def _is_n5(path: PathLike) -> bool:
    if Path(path).suffix.lower() == ".n5":
        return True
    attributes_file = os.path.join(path, "attributes.json")
    if not os.path.isfile(attributes_file):
        return False
    try:
        with open(attributes_file) as f:
            return "n5" in json.load(f)
    except (OSError, ValueError):
        return False


def is_container(path) -> bool:
    """Check if a path is a folder container (zarr, ome-zarr or n5).

    Args:
        path: The path to check.

    Returns:
        Whether the path is a folder container.
    """
    if not isinstance(path, (str, os.PathLike)) or not os.path.isdir(path):
        return False
    if str(path).rstrip("/").lower().endswith(CONTAINER_EXTENSIONS):
        return True
    if any(os.path.exists(os.path.join(path, name)) for name in ZARR_METADATA_FILES):
        return True
    return _is_n5(path)


def open_container(path: PathLike):
    """Open a folder container in read mode.

    Args:
        path: The path to the container.

    Returns:
        The root group, or the array if the path points to a zarr array.
    """
    if _is_n5(path):
        return z5py.N5File(str(path), mode="r")
    return zarr.open(str(path), mode="r")


def _is_array(node) -> bool:
    return isinstance(node, (zarr.Array, z5py.Dataset))


def _join(prefix: str, name: str) -> str:
    return f"{prefix}/{name}" if prefix else name


def _ome_attrs(node) -> dict:
    """The ome-zarr metadata of a group: under 'ome' for version 0.5, at the top level for version 0.4."""
    attrs = dict(node.attrs)
    return attrs.get("ome", attrs)


def _is_ome(node) -> bool:
    attrs = dict(node.attrs)
    return "ome" in attrs or any(name in attrs for name in ("multiscales", "labels") + OME_COLLECTION_ATTRIBUTES)


def _list_arrays(group, prefix: str) -> List[str]:
    keys = []
    for name in sorted(group.keys()):
        child = group[name]
        if _is_array(child):
            keys.append(_join(prefix, name))
        else:
            keys.extend(_list_arrays(child, _join(prefix, name)))
    return keys


def _format_keys(keys: Sequence[str]) -> str:
    listed = ", ".join(f"'{key}'" for key in keys[:MAX_LISTED_KEYS])
    return listed + (", ..." if len(keys) > MAX_LISTED_KEYS else "")


def _resolve_group(path, group, prefix: str, resolve_labels: bool) -> str:
    attrs = _ome_attrs(group)
    location = f"'{path}'" if not prefix else f"'{prefix}' in '{path}'"

    if "labels" in attrs:  # A 'labels' group, which lists its label images.
        names = attrs["labels"]
        if len(names) != 1:
            raise ValueError(
                f"{location} contains {len(names)} label images ({_format_keys(names)}). "
                "Pass the key of the label image you want to load, e.g. 'labels/<name>'."
            )
        return _resolve_group(path, group[names[0]], _join(prefix, names[0]), resolve_labels=False)

    if resolve_labels and _is_ome(group):
        if "labels" not in group:
            raise ValueError(f"{location} has no 'labels' group. Pass the key of the segmentation you want to load.")
        return _resolve_group(path, group["labels"], _join(prefix, "labels"), resolve_labels=False)

    multiscales = attrs.get("multiscales")
    if multiscales:
        if len(multiscales) != 1:
            names = [ms.get("name", str(i)) for i, ms in enumerate(multiscales)]
            raise ValueError(
                f"{location} contains {len(multiscales)} multiscale images ({_format_keys(names)}). "
                "Pass the key of the scale you want to load."
            )
        # The ome-zarr specification orders the datasets from the highest to the lowest resolution.
        return _join(prefix, multiscales[0]["datasets"][0]["path"])

    if _is_ome(group):
        children = sorted(group.keys())
        raise ValueError(
            f"{location} is an ome-zarr file without an image at this location "
            f"(it contains: {_format_keys(children)}). Pass the key of the image you want to load."
        )

    keys = _list_arrays(group, prefix)
    if len(keys) != 1:
        found = f"{len(keys)} arrays ({_format_keys(keys)})" if keys else "no arrays"
        raise ValueError(f"{location} contains {found}. Pass the key of the array you want to load.")
    return keys[0]


def resolve_container_key(path: PathLike, key: Optional[str] = None, is_segmentation: bool = False) -> Optional[str]:
    """Resolve the key of the array to load from a folder container.

    Args:
        path: The path to the container.
        key: The internal key. If it points to an array, it is used as given. If it points to an ome-zarr image
            or label group, the highest resolution is selected. If it points to another group, the group must
            contain a single array. If None, the rules of this module select the array.
        is_segmentation: Whether to load a segmentation. Without a key, this selects the single label image of an
            ome-zarr file instead of the image.

    Returns:
        The key of the array, or None if the path points to an array.
    """
    root = open_container(path)
    key = key.strip("/") if key else None
    if key is None and _is_array(root):
        return None

    if key is not None:
        if key not in root:
            raise KeyError(f"The key '{key}' does not exist in '{path}'.")
        node = root[key]
        return key if _is_array(node) else _resolve_group(path, node, key, resolve_labels=False)

    return _resolve_group(path, root, "", resolve_labels=is_segmentation)


def load_container_data(
    path: PathLike, key: Optional[str] = None, lazy_loading: bool = False, is_segmentation: bool = False,
):
    """Load an array from a folder container.

    Args:
        path: The path to the container.
        key: The internal key, see `resolve_container_key`.
        lazy_loading: Whether to return the array without loading it into memory.
        is_segmentation: Whether to load a segmentation, see `resolve_container_key`.

    Returns:
        The array data.
    """
    key = resolve_container_key(path, key, is_segmentation=is_segmentation)
    root = open_container(path)
    data = root if key is None else root[key]
    return data if lazy_loading else data[:]


def get_image_shape(path: PathLike, key: Optional[str] = None) -> Tuple[int, ...]:
    """Get the shape of image data without loading it.

    Args:
        path: The path to the image file or container.
        key: The internal key for containers or a wildcard for an image stack, as for `util.load_image_data`.

    Returns:
        The shape of the image data.
    """
    if is_container(path):
        return tuple(load_container_data(path, key, lazy_loading=True).shape)
    if key is None:
        return tuple(imageio.improps(path).shape)
    with open_file(path, mode="r") as f:
        return tuple(f[key].shape)


def get_input_paths(paths: Union[PathLike, Sequence[PathLike]], pattern: Optional[str] = None) -> List[str]:
    """Get the input paths from files, containers and folders.

    Args:
        paths: The input path(s). A file or a container is one input. Any other folder is searched with `pattern`.
        pattern: The glob pattern for searching folders. By default all files and containers are selected.

    Returns:
        The input paths. The matches of a folder are sorted.
    """
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]

    input_paths = []
    for path in paths:
        path = str(path)
        if os.path.isfile(path) or is_container(path):
            input_paths.append(path)
        else:
            hits = glob(os.path.join(path, "*" if pattern is None else pattern))
            input_paths.extend(sorted(hit for hit in hits if os.path.isfile(hit) or is_container(hit)))
    return input_paths


def get_input_name(path: PathLike) -> str:
    """Get the name of an input without its extension, e.g. 'image' for 'image.tif' or 'image.ome.zarr'.

    Args:
        path: The input path.

    Returns:
        The input name.
    """
    name = Path(str(path).rstrip("/")).name
    for extension in CONTAINER_EXTENSIONS:
        if name.lower().endswith(extension):
            return name[:-len(extension)]
    return Path(name).stem
