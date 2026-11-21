"""Resolve, open and write data, including folder containers (zarr, n5 and ome-zarr).

A folder container is a directory that holds zarr or n5 data. Without a key, these rules select the array:
- ome-zarr: the highest resolution of the image at the root. For a segmentation, the single label image in 'labels/',
  or the root image if it is a label image.
- zarr and n5: the single array in the container.
In all other cases, the caller must give a key.

An output path that ends with '.ome.zarr' stores a result as an ome-zarr label image:
- In an existing ome-zarr image, as 'labels/<name>', with the same scale levels as the image.
- In a new file, as the root image, with a single scale level.
Any other output path stores a result as an image file, by default a tif.

clearscale reads and writes the ome-zarr metadata. z5py writes the data.
"""

import os
import re
import json
from glob import glob
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import clearscale as cs
import numpy as np
import z5py
import zarr
import imageio.v3 as imageio
from elf.io import open_file

CONTAINER_EXTENSIONS = (".ome.zarr", ".zarr", ".zr", ".n5")
ZARR_METADATA_FILES = ("zarr.json", ".zgroup", ".zarray")
MAX_LISTED_KEYS = 10
DEFAULT_LABEL_NAME = "segmentation"

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
    location = f"'{path}'" if not prefix else f"'{prefix}' in '{path}'"
    ome = cs.OmeZarrGroup.from_group(group)

    if ome.kind is None:  # A zarr or n5 group without ome-zarr metadata.
        keys = _list_arrays(group, prefix)
        if len(keys) != 1:
            found = f"{len(keys)} arrays ({_format_keys(keys)})" if keys else "no arrays"
            raise ValueError(f"{location} contains {found}. Pass the key of the array you want to load.")
        return keys[0]

    if ome.kind is cs.GroupKind.INVALID:
        reasons = "; ".join(obj.error for obj in ome.invalid_objects)
        raise ValueError(f"{location} has invalid ome-zarr metadata: {reasons}")

    if ome.kind is cs.GroupKind.LABELS:
        names = [child.path for child in ome.children]
        if len(names) != 1:
            raise ValueError(
                f"{location} contains {len(names)} label images ({_format_keys(names)}). "
                "Pass the key of the label image you want to load, e.g. 'labels/<name>'."
            )
        return _resolve_group(path, group[names[0]], _join(prefix, names[0]), resolve_labels=False)

    if ome.kind is cs.GroupKind.MULTISCALE:
        multiscale = ome.multiscales[0]
        if resolve_labels and multiscale.ome.image_label is None:
            if "labels" not in group:
                raise ValueError(
                    f"{location} has no 'labels' group. Pass the key of the segmentation you want to load."
                )
            return _resolve_group(path, group["labels"], _join(prefix, "labels"), resolve_labels=False)
        # The ome-zarr specification orders the scale levels from the highest to the lowest resolution.
        return _join(prefix, next(iter(multiscale)))

    if len(ome.multiscales) > 1:
        names = [ms.ome.name or str(i) for i, ms in enumerate(ome.multiscales)]
        raise ValueError(
            f"{location} contains {len(ome.multiscales)} multiscale images ({_format_keys(names)}). "
            "Pass the key of the scale you want to load."
        )

    children = sorted(group.keys())
    raise ValueError(
        f"{location} is an ome-zarr file without an image at this location "
        f"(it contains: {_format_keys(children)}). Pass the key of the image you want to load."
    )


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
    if _is_array(root) or cs.OmeZarrGroup.from_group(root).kind is None:
        return None
    array_key = resolve_container_key(path, key)
    group_key = array_key.rpartition("/")[0]
    multiscales = cs.OmeZarrGroup.from_group(root[group_key] if group_key else root).multiscales
    return multiscales[0].ome.name if multiscales else None


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
    return f"the label image '{label_name}' in '{path}'" if is_ome_zarr_path(output_path) else f"'{path}'"


def _get_ome_zarr_target(output_path: str, label_name: str) -> Tuple[str, Optional[cs.OmeZarrGroup]]:
    """Find where to store a label image in an ome-zarr output.

    Returns:
        The target and the ome-zarr metadata of the root. The target is 'new' for a file that does not exist,
        'labels' for 'labels/<label_name>' in an image, or 'root' for a root label image with this name.
    """
    if not os.path.exists(output_path):
        return "new", None
    try:
        root = cs.OmeZarrGroup.from_group(open_container(output_path))
    except zarr.errors.BaseZarrError as e:
        raise ValueError(f"Cannot store the result in '{output_path}': it is not a zarr file ({e}).")

    if root.kind is not cs.GroupKind.MULTISCALE:
        raise ValueError(
            f"Cannot store the result in '{output_path}': it exists, but it does not contain an ome-zarr image "
            "at the root. Use an ome-zarr image or a new file as output."
        )
    if root.version not in cs.ome_zarr.SUPPORTED_VERSIONS_WRITE:
        raise ValueError(
            f"Cannot store the result in '{output_path}': ome-zarr version {root.version} is not supported."
        )

    multiscale = root.multiscales[0]
    if multiscale.ome.image_label is None:
        return "labels", root
    if multiscale.ome.name == label_name:
        return "root", root
    raise ValueError(
        f"Cannot store the label image '{label_name}' in '{output_path}': its root image is the label image "
        f"'{multiscale.ome.name}'. Use another output path, or the label name '{multiscale.ome.name}'."
    )


def output_exists(output_path: PathLike, label_name: str) -> bool:
    """Check if a result is already stored.

    Args:
        output_path: The output path.
        label_name: The name of the label image for an ome-zarr output.

    Returns:
        Whether the output file exists, or the label image exists in the ome-zarr output.

    Raises:
        ValueError: If the ome-zarr output exists, but cannot store the label image.
    """
    if not is_ome_zarr_path(output_path):
        return os.path.exists(output_path)
    target, _ = _get_ome_zarr_target(str(output_path), label_name)
    if target == "labels":
        return f"labels/{label_name}" in open_container(output_path)
    return target == "root"


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
        f"The result is already stored in {describe_output(output_path, label_name)}. Skipping it. "
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
    if not is_ome_zarr_path(output_path):
        return imageio.imread(output_path)
    target, _ = _get_ome_zarr_target(str(output_path), label_name)
    return load_container_data(output_path, key=f"labels/{label_name}" if target == "labels" else None)


def _default_chunks(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    ndim = len(shape)
    return tuple(min(size, 256 if axis >= ndim - 2 else 32) for axis, size in enumerate(shape))


def _spatial_axes(multiscale: cs.Multiscale, ndim: int) -> List[str]:
    """The axes of an image that a segmentation with `ndim` dimensions keeps: no channel axis and no singleton axes."""
    scale = multiscale.first_value
    axes = [axis for axis in multiscale.axes if scale.ome_zarr_axes[axis].type != "channel"]
    if len(axes) > ndim:
        axes = [axis for axis in axes if scale.shape[axis] != 1]
    if len(axes) != ndim:
        raise ValueError(
            f"The segmentation has {ndim} dimensions, but the image has the axes {list(multiscale.axes)}."
        )
    return axes


def _write_root_label_image(
    path: str, root: Optional[cs.OmeZarrGroup], segmentation: np.ndarray, label_name: str
) -> None:
    # A new file uses ome-zarr 0.5 (zarr v3). An existing file keeps its version.
    if root is None:
        f, version = z5py.File(path, mode="a", zarr_format=3), "0.5"
    else:
        f, version = z5py.File(path, mode="a"), root.version
        for key in root.multiscales[0]:
            del f[key]

    if segmentation.ndim not in (2, 3, 4):
        raise ValueError(f"Cannot store a segmentation with {segmentation.ndim} dimensions in ome-zarr.")
    axes = "tzyx"[-segmentation.ndim:]
    multiscale = cs.Multiscale.from_single(cs.Scale(dict(zip(axes, segmentation.shape))), scale_key="s0")
    multiscale.ome.name = label_name
    multiscale.ome.image_label = cs.ome_zarr.ImageLabel()

    f.create_dataset("s0", data=segmentation, chunks=_default_chunks(segmentation.shape))
    cs.OmeZarrGroup.from_single(multiscale).update_attrs(f, version=version)


def _write_labels_image(
    path: str, image: cs.OmeZarrGroup, segmentation: np.ndarray, label_name: str, num_workers: int
) -> None:
    from bioimage_py import copy
    from bioimage_py.wrapper import ResizedSource

    multiscale = image.multiscales[0]
    axes = _spatial_axes(multiscale, segmentation.ndim)
    shapes = {key: tuple(scale.shape[axis] for axis in axes) for key, scale in multiscale.items()}
    if segmentation.shape not in shapes.values():
        raise ValueError(
            f"The segmentation has the shape {segmentation.shape}, but the scale levels of the image in '{path}' "
            f"have the shapes {list(shapes.values())} for the axes {axes}."
        )

    # The spec requires the same scale levels for a label image as for its image.
    label_multiscale = cs.Multiscale({key: scale.with_axes(axes) for key, scale in multiscale.items()})
    label_multiscale.ome.name = label_name
    label_multiscale.ome.image_label = cs.ome_zarr.ImageLabel(source="../../")

    f = z5py.File(path, mode="a")
    labels = f.require_group("labels")
    if label_name in labels:
        del labels[label_name]
    group = labels.create_group(label_name)

    axis_indices = [multiscale.axes.index(axis) for axis in axes]
    for key, shape in shapes.items():
        image_chunks = f[key].chunks
        chunks = tuple(min(image_chunks[index], size) for index, size in zip(axis_indices, shape))
        dataset = group.create_dataset(key, shape=shape, chunks=chunks, dtype=segmentation.dtype)
        # Nearest neighbor resizing keeps the label ids.
        source = segmentation if shape == segmentation.shape else ResizedSource(segmentation, shape, order=0)
        copy(source, output=dataset, block_shape=chunks, num_workers=num_workers)
    cs.OmeZarrGroup.from_single(label_multiscale).update_attrs(group, version=image.version)

    # Add the label image to the list after its data is written.
    children = cs.OmeZarrGroup.from_group(labels).children
    if label_name not in [child.path for child in children]:
        children += (cs.ChildRef("label_multiscale", label_name),)
    cs.OmeZarrGroup(children=children).update_attrs(labels, version=image.version)


def write_segmentation(
    output_path: PathLike,
    segmentation: np.ndarray,
    label_name: str = DEFAULT_LABEL_NAME,
    overwrite: bool = False,
    num_workers: Optional[int] = None,
) -> None:
    """Store a segmentation as an image file or as a label image in an ome-zarr file.

    An output path that ends with '.ome.zarr' stores the segmentation as an ome-zarr label image:
    - In an existing ome-zarr image, for example the input image, as 'labels/<label_name>'. The label image has the
      same scale levels as the image: the segmentation must have the shape of one level without the channel axis,
      and the other levels are computed with nearest neighbor resizing.
    - In a new file, as the root image, with a single scale level and ome-zarr version 0.5.
    Any other output path stores the segmentation with imageio, a tif with zlib compression.

    Args:
        output_path: The output path.
        segmentation: The segmentation.
        label_name: The name of the label image for an ome-zarr output, see `get_label_name`.
        overwrite: Whether to replace a stored result.
        num_workers: The number of threads to compute the scale levels of an ome-zarr label image.
            By default the number of CPUs.
    """
    output_path = str(output_path)
    if output_exists(output_path, label_name) and not overwrite:
        raise FileExistsError(f"{describe_output(output_path, label_name)} exists already. Pass overwrite=True.")

    output_dir = os.path.dirname(output_path.rstrip("/"))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if not is_ome_zarr_path(output_path):
        imageio.imwrite(output_path, segmentation, compression="zlib")
        return

    segmentation = np.asarray(segmentation)
    target, root = _get_ome_zarr_target(output_path, label_name)
    if target == "labels":
        _write_labels_image(output_path, root, segmentation, label_name, num_workers or os.cpu_count() or 1)
    else:
        _write_root_label_image(output_path, root, segmentation, label_name)
