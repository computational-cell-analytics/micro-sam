"""Resolve, open and write data, including folder containers (zarr, n5 and ome-zarr).

A folder container is a directory that holds zarr or n5 data. Without a key, these rules select the array:
- ome-zarr: the highest resolution of the image at the root. For a segmentation, the single label image in 'labels/'.
- zarr and n5: the single array in the container.
In all other cases, the caller must give a key.

An output path that ends with '.ome.zarr' stores a result as the label image 'labels/<name>' in that container.
Any other output path stores a result as an image file, by default a tif.
"""

import os
import re
import json
from glob import glob
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import z5py
import zarr
import imageio.v3 as imageio
from elf.io import open_file

CONTAINER_EXTENSIONS = (".ome.zarr", ".zarr", ".zr", ".n5")
ZARR_METADATA_FILES = ("zarr.json", ".zgroup", ".zarray")
# Root attributes of ome-zarr collections (bioformats2raw, HCS) that have no image at the root.
OME_COLLECTION_ATTRIBUTES = ("bioformats2raw.layout", "plate", "well")
MAX_LISTED_KEYS = 10
DEFAULT_LABEL_NAME = "segmentation"
# The array of the single resolution level of a label image that we write.
LABEL_SCALE_PATH = "0"

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


def is_ome_zarr_path(path: PathLike) -> bool:
    """Check if a path has the ome-zarr extension '.ome.zarr'.

    Args:
        path: The path to check. It does not have to exist.

    Returns:
        Whether the path has the ome-zarr extension.
    """
    return str(path).rstrip("/").lower().endswith(".ome.zarr")


def get_output_path(path: PathLike) -> str:
    """Get the path to store a result: add the extension '.tif' to a path without an extension.

    Args:
        path: The output path.

    Returns:
        The output path with an extension.
    """
    path = str(path)
    if is_ome_zarr_path(path) or os.path.splitext(path)[1]:
        return path
    return f"{path}.tif"


def _normalize_name(name: Optional[str]) -> str:
    return re.sub(r"[^\w.-]+", "_", name or "").strip("_")


def _get_image_name(path: PathLike, key: Optional[str]) -> Optional[str]:
    root = open_container(path)
    if not _is_ome(root):
        return None
    array_key = resolve_container_key(path, key)
    if array_key is None:
        return None
    group_key = array_key.rpartition("/")[0]
    multiscales = _ome_attrs(root[group_key] if group_key else root).get("multiscales")
    return multiscales[0].get("name") if multiscales else None


def get_label_name(
    input_path: Union[PathLike, np.ndarray], key: Optional[str] = None, label_name: Optional[str] = None,
) -> str:
    """Get the name of the label image for an ome-zarr output.

    Args:
        input_path: The input path or array.
        key: The key of the input data, see `resolve_container_key`.
        label_name: The name given by the user. If None, the name is derived from the input:
            the name of the image for an ome-zarr input, otherwise the normalized name of the input file.

    Returns:
        The name of the label image.
    """
    if label_name is not None:
        if not label_name or "/" in label_name:
            raise ValueError(f"Invalid label name '{label_name}': it must not be empty or contain a '/'.")
        return label_name
    if not isinstance(input_path, (str, os.PathLike)):
        return DEFAULT_LABEL_NAME

    image_name = _get_image_name(input_path, key) if is_container(input_path) else None
    for name in (image_name, get_input_name(input_path)):
        name = _normalize_name(name)
        if name:
            return name
    return DEFAULT_LABEL_NAME


def describe_output(output_path: PathLike, label_name: str) -> str:
    """Describe where a result is stored, for messages to the user.

    Args:
        output_path: The output path.
        label_name: The name of the label image for an ome-zarr output.

    Returns:
        The description.
    """
    path = os.path.abspath(str(output_path))
    return f"'labels/{label_name}' in '{path}'" if is_ome_zarr_path(output_path) else f"'{path}'"


def output_exists(output_path: PathLike, label_name: str) -> bool:
    """Check if a result is already stored.

    Args:
        output_path: The output path.
        label_name: The name of the label image for an ome-zarr output.

    Returns:
        Whether the output file exists, or the label image exists in the ome-zarr output.
    """
    if not is_ome_zarr_path(output_path):
        return os.path.exists(output_path)
    if not os.path.isdir(output_path):
        return False
    return f"labels/{label_name}" in zarr.open_group(str(output_path), mode="r")


def skip_existing_output(output_path: PathLike, label_name: str, overwrite: bool) -> bool:
    """Check if a result is already stored and must not be replaced. Print an info in this case.

    Args:
        output_path: The output path.
        label_name: The name of the label image for an ome-zarr output.
        overwrite: Whether to replace a stored result.

    Returns:
        Whether to skip the computation of this result.
    """
    if overwrite or not output_exists(output_path, label_name):
        return False
    print(
        f"The result is already stored at {describe_output(output_path, label_name)}. Skipping it. "
        "Pass 'overwrite=True' (python) or '--overwrite' (CLI) to replace it."
    )
    return True


def load_output(output_path: PathLike, label_name: str) -> np.ndarray:
    """Load a stored result.

    Args:
        output_path: The output path.
        label_name: The name of the label image for an ome-zarr output.

    Returns:
        The result.
    """
    if is_ome_zarr_path(output_path):
        return load_container_data(output_path, key=f"labels/{label_name}")
    return imageio.imread(output_path)


def _select_axes(transform: dict, axes: Sequence[int]) -> Optional[dict]:
    values = transform.get(transform.get("type"))
    if not isinstance(values, list):
        return None
    return {**transform, transform["type"]: [values[axis] for axis in axes]}


def _label_multiscale(root, shape: Tuple[int, ...]) -> dict:
    """The axes and the scale of a label image, taken from the image of the container if they match."""
    ndim = len(shape)
    names = "tzyx"[-ndim:] if ndim <= 4 else [f"dim_{axis}" for axis in range(ndim)]
    axes = [{"name": name, "type": "time" if name == "t" else "space"} for name in names]
    transforms = [{"type": "scale", "scale": [1.0] * ndim}]

    multiscales = _ome_attrs(root).get("multiscales")
    if not multiscales:
        return {"axes": axes, "transforms": transforms}

    image_axes = multiscales[0].get("axes", [])
    if not all(isinstance(axis, dict) for axis in image_axes):  # Versions before 0.4 give the axes as strings.
        return {"axes": axes, "transforms": transforms}
    keep = list(range(len(image_axes)))
    if len(keep) != ndim:
        keep = [i for i, axis in enumerate(image_axes) if axis.get("type") != "channel"]
    if len(keep) != ndim:
        return {"axes": axes, "transforms": transforms}
    axes = [image_axes[i] for i in keep]

    # The segmentation can be computed for any scale of the image, so we use the scale with the same shape.
    for dataset in multiscales[0]["datasets"]:
        if dataset["path"] not in root:
            continue
        image_shape = root[dataset["path"]].shape
        if len(image_shape) == len(image_axes) and tuple(image_shape[i] for i in keep) == tuple(shape):
            selected = [_select_axes(transform, keep) for transform in dataset.get("coordinateTransformations", [])]
            if selected and all(transform is not None for transform in selected):
                transforms = selected
            break
    return {"axes": axes, "transforms": transforms}


def _write_ome_zarr_labels(path: str, segmentation: np.ndarray, label_name: str) -> None:
    # A new file uses ome-zarr 0.5 (zarr v3). An existing file keeps its version.
    is_new = not os.path.exists(path)
    root = zarr.open_group(path, mode="a", zarr_format=3 if is_new else None)
    version = "0.5" if root.metadata.zarr_format == 3 else "0.4"

    def _update_ome_attrs(group, attrs):
        if version == "0.5":
            group.attrs["ome"] = {**group.attrs.get("ome", {}), "version": version, **attrs}
        else:
            group.attrs.update(attrs)

    if is_new:
        _update_ome_attrs(root, {})

    labels = root.require_group("labels")
    if label_name in labels:
        del labels[label_name]

    group = labels.create_group(label_name)
    group.create_array(LABEL_SCALE_PATH, data=segmentation)

    metadata = _label_multiscale(root, segmentation.shape)
    multiscale = {
        "name": label_name,
        "axes": metadata["axes"],
        "datasets": [{"path": LABEL_SCALE_PATH, "coordinateTransformations": metadata["transforms"]}],
    }
    if version == "0.4":
        multiscale["version"] = version
    _update_ome_attrs(group, {"multiscales": [multiscale], "image-label": {"version": version}})

    names = list(_ome_attrs(labels).get("labels", []))
    if label_name not in names:
        _update_ome_attrs(labels, {"labels": names + [label_name]})


def write_segmentation(
    output_path: PathLike,
    segmentation: np.ndarray,
    label_name: str = DEFAULT_LABEL_NAME,
    overwrite: bool = False,
) -> None:
    """Store a segmentation as an image file or as a label image in an ome-zarr file.

    An output path that ends with '.ome.zarr' stores the segmentation as 'labels/<label_name>'.
    The ome-zarr file can exist already, for example the input image. A new file uses ome-zarr version 0.5.
    Any other output path stores the segmentation with imageio, a tif with zlib compression.

    Args:
        output_path: The output path.
        segmentation: The segmentation.
        label_name: The name of the label image for an ome-zarr output, see `get_label_name`.
        overwrite: Whether to replace a stored result.
    """
    output_path = str(output_path)
    if output_exists(output_path, label_name) and not overwrite:
        raise FileExistsError(f"{describe_output(output_path, label_name)} exists already. Pass overwrite=True.")

    output_dir = os.path.dirname(output_path.rstrip("/"))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if is_ome_zarr_path(output_path):
        _write_ome_zarr_labels(output_path, np.asarray(segmentation), label_name)
    else:
        imageio.imwrite(output_path, segmentation, compression="zlib")
