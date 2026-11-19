import os

import h5py
import numpy as np
import pytest
import z5py
import zarr
import imageio.v3 as imageio

from micro_sam._io import get_image_shape, get_input_name, get_input_paths, is_container, resolve_container_key
from micro_sam._test_util import write_ome_zarr
from micro_sam.util import load_image_data

SHAPE = (32, 32)


def _data(seed, shape=SHAPE):
    return np.random.default_rng(seed).integers(0, 255, size=shape).astype("uint8")


@pytest.fixture
def raw():
    return _data(0)


@pytest.fixture
def labels():
    return _data(1)


# Image files and folders keep the behavior they had before containers were supported.


def test_image_files_and_folders(tmp_path, raw):
    folder = tmp_path / "images"
    folder.mkdir()
    for i in range(3):
        imageio.imwrite(folder / f"im{i}.tif", _data(i))
        imageio.imwrite(folder / f"im{i}.png", _data(i))

    path = str(folder / "im0.tif")
    np.testing.assert_array_equal(load_image_data(path), _data(0))
    assert get_image_shape(path) == SHAPE

    stack = load_image_data(str(folder), key="*.tif")
    assert stack.shape == (3,) + SHAPE
    assert get_image_shape(str(folder), key="*.tif") == (3,) + SHAPE
    assert not is_container(str(folder))

    assert get_input_paths(path) == [path]
    assert get_input_paths(str(folder), "*.png") == [str(folder / f"im{i}.png") for i in range(3)]
    assert len(get_input_paths(str(folder))) == 6

    h5_path = str(tmp_path / "data.h5")
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("raw", data=raw)
    np.testing.assert_array_equal(load_image_data(h5_path, key="raw"), raw)


# zarr and n5 files.


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_zarr(tmp_path, raw, labels, zarr_format):
    path = str(tmp_path / "single.zarr")
    zarr.open_group(path, mode="w", zarr_format=zarr_format).create_group("a").create_array("b", data=raw)
    assert is_container(path)
    assert resolve_container_key(path) == "a/b"
    np.testing.assert_array_equal(load_image_data(path), raw)
    np.testing.assert_array_equal(load_image_data(path, key="a"), raw)
    assert isinstance(load_image_data(path, lazy_loading=True), zarr.Array)

    # A path to the array itself, and a zarr folder without a container extension.
    np.testing.assert_array_equal(load_image_data(os.path.join(path, "a", "b")), raw)
    path = str(tmp_path / "no_extension")
    zarr.open_group(path, mode="w", zarr_format=zarr_format).create_array("raw", data=raw)
    assert is_container(path)
    np.testing.assert_array_equal(load_image_data(path), raw)

    path = str(tmp_path / "multiple.zarr")
    group = zarr.open_group(path, mode="w", zarr_format=zarr_format)
    group.create_array("raw", data=raw)
    group.create_array("labels", data=labels)
    with pytest.raises(ValueError, match="2 arrays"):
        load_image_data(path)
    with pytest.raises(ValueError, match="2 arrays"):
        load_image_data(path, is_segmentation=True)
    np.testing.assert_array_equal(load_image_data(path, key="labels"), labels)
    with pytest.raises(KeyError):
        load_image_data(path, key="missing")


def test_n5(tmp_path, raw, labels):
    path = str(tmp_path / "single.n5")
    with z5py.File(path, "w") as f:
        f.create_group("a").create_dataset("raw", data=raw, chunks=SHAPE)
    assert is_container(path)
    np.testing.assert_array_equal(load_image_data(path), raw)
    assert get_image_shape(path) == SHAPE

    path = str(tmp_path / "multiple.n5")
    with z5py.File(path, "w") as f:
        f.create_dataset("raw", data=raw, chunks=SHAPE)
        f.create_dataset("labels", data=labels, chunks=SHAPE)
    with pytest.raises(ValueError, match="2 arrays"):
        load_image_data(path)
    np.testing.assert_array_equal(load_image_data(path, key="labels"), labels)


# ome-zarr files.


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_ome_zarr(tmp_path, raw, labels, version):
    path = str(tmp_path / "image.ome.zarr")
    write_ome_zarr(path, raw, labels, version=version, n_scales=2)
    low_res = "s1" if version == "0.5" else "1"

    np.testing.assert_array_equal(load_image_data(path), raw)
    np.testing.assert_array_equal(load_image_data(path, is_segmentation=True), labels)
    np.testing.assert_array_equal(load_image_data(path, key="labels/cells"), labels)
    np.testing.assert_array_equal(load_image_data(path, key="labels"), labels)
    np.testing.assert_array_equal(load_image_data(path, key=low_res), raw[::2, ::2])
    assert get_image_shape(path) == SHAPE

    path = str(tmp_path / "two_labels.ome.zarr")
    write_ome_zarr(path, raw, labels, version=version, label_names=("cells", "nuclei"))
    with pytest.raises(ValueError, match="2 label images"):
        load_image_data(path, is_segmentation=True)
    np.testing.assert_array_equal(load_image_data(path, key="labels/nuclei"), labels)
    np.testing.assert_array_equal(load_image_data(path), raw)

    path = str(tmp_path / "no_labels.ome.zarr")
    write_ome_zarr(path, raw, version=version)
    with pytest.raises(ValueError, match="no 'labels' group"):
        load_image_data(path, is_segmentation=True)


def test_ome_zarr_collection(tmp_path, raw):
    # A bioformats2raw layout has no image at the root, only one image per series.
    path = str(tmp_path / "collection.ome.zarr")
    root = zarr.open_group(path, mode="w")
    root.attrs["ome"] = {"version": "0.5", "bioformats2raw.layout": 3}
    for series in ("0", "1"):
        write_ome_zarr(os.path.join(path, series), raw)

    with pytest.raises(ValueError, match="without an image"):
        load_image_data(path)
    np.testing.assert_array_equal(load_image_data(path, key="1"), raw)


# Input paths and names.


def test_input_paths_and_names(tmp_path, raw):
    folder = tmp_path / "inputs"
    folder.mkdir()
    write_ome_zarr(str(folder / "b.ome.zarr"), raw)
    zarr.open_group(str(folder / "a.zarr"), mode="w").create_array("raw", data=raw)
    imageio.imwrite(folder / "c.tif", raw)
    (folder / "not_a_container").mkdir()

    assert get_input_paths(str(folder)) == [str(folder / name) for name in ("a.zarr", "b.ome.zarr", "c.tif")]
    assert get_input_paths(str(folder), "*.zarr") == [str(folder / name) for name in ("a.zarr", "b.ome.zarr")]
    container = str(folder / "b.ome.zarr")
    assert get_input_paths(container, "*.tif") == [container]
    assert get_input_paths([container, str(folder / "c.tif")]) == [container, str(folder / "c.tif")]

    assert get_input_name(container) == "b"
    assert get_input_name(container + "/") == "b"
    assert get_input_name(str(folder / "a.zarr")) == "a"
    assert get_input_name("data/image.n5") == "image"
    assert get_input_name("data/image.tif") == "image"
    assert get_input_name("data/image.ome.tif") == "image.ome"
