import os

import h5py
import numpy as np
import pytest
import z5py
import zarr
import imageio.v3 as imageio

from micro_sam._io import (
    get_image_shape, get_input_name, get_input_paths, get_label_name, get_output_path, is_container, load_output,
    resolve_container_key, skip_existing_output, write_segmentation,
)
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


# Outputs.


def _label_attrs(path, name, version="0.5"):
    group = zarr.open_group(path, mode="r")
    if version == "0.5":
        return group["labels"].attrs["ome"], group[f"labels/{name}"].attrs["ome"]
    return dict(group["labels"].attrs), dict(group[f"labels/{name}"].attrs)


def test_output_path():
    assert get_output_path("out") == "out.tif"
    assert get_output_path("out.tif") == "out.tif"
    assert get_output_path("out.png") == "out.png"
    assert get_output_path("out.ome.zarr") == "out.ome.zarr"


def test_write_new_ome_zarr(tmp_path, labels):
    path = str(tmp_path / "new" / "out.ome.zarr")
    write_segmentation(path, labels, label_name="cells")

    assert zarr.open_group(path, mode="r").metadata.zarr_format == 3
    labels_attrs, label_attrs = _label_attrs(path, "cells")
    assert labels_attrs["labels"] == ["cells"]
    assert label_attrs["image-label"] == {"version": "0.5"}
    multiscale = label_attrs["multiscales"][0]
    assert [axis["name"] for axis in multiscale["axes"]] == ["y", "x"]
    assert multiscale["datasets"][0]["path"] == "0"
    np.testing.assert_array_equal(load_image_data(path, is_segmentation=True), labels)
    np.testing.assert_array_equal(load_output(path, "cells"), labels)


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_write_into_input_ome_zarr(tmp_path, raw, labels, version):
    path = str(tmp_path / "image.ome.zarr")
    write_ome_zarr(path, raw, labels, version=version, n_scales=2)
    segmentation = labels[::2, ::2] + 1
    write_segmentation(path, segmentation, label_name="result")

    assert zarr.open_group(path, mode="r").metadata.zarr_format == (3 if version == "0.5" else 2)
    labels_attrs, label_attrs = _label_attrs(path, "result", version)
    assert labels_attrs["labels"] == ["cells", "result"]
    assert label_attrs["image-label"] == {"version": version}
    # The result has the shape of the second scale, so it gets the scale of this level.
    transforms = label_attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"]
    assert transforms == [{"type": "scale", "scale": [2.0, 2.0]}]
    np.testing.assert_array_equal(load_image_data(path, key="labels/cells"), labels)
    np.testing.assert_array_equal(load_image_data(path, key="labels/result"), segmentation)
    np.testing.assert_array_equal(load_image_data(path), raw)


def test_write_ome_zarr_drops_the_channel_axis(tmp_path, labels):
    path = str(tmp_path / "image.ome.zarr")
    root = zarr.open_group(path, mode="w")
    root.create_array("0", data=np.zeros((3,) + SHAPE, dtype="uint8"))
    axes = [{"name": "c", "type": "channel"}] + [{"name": name, "type": "space", "unit": "micrometer"} for name in "yx"]
    transforms = [{"type": "scale", "scale": [1.0, 0.5, 0.25]}, {"type": "translation", "translation": [0, 1, 2]}]
    datasets = [{"path": "0", "coordinateTransformations": transforms}]
    root.attrs["ome"] = {"version": "0.5", "multiscales": [{"axes": axes, "datasets": datasets}]}

    write_segmentation(path, labels, label_name="cells")
    multiscale = _label_attrs(path, "cells")[1]["multiscales"][0]
    assert multiscale["axes"] == axes[1:]
    assert multiscale["datasets"][0]["coordinateTransformations"] == [
        {"type": "scale", "scale": [0.5, 0.25]}, {"type": "translation", "translation": [1, 2]}
    ]


def test_label_name(tmp_path, raw):
    path = str(tmp_path / "my image.ome.zarr")
    write_ome_zarr(path, raw)
    assert get_label_name(path) == "my_image"

    root = zarr.open_group(path, mode="a")
    ome = root.attrs["ome"]
    root.attrs["ome"] = {**ome, "multiscales": [{**ome["multiscales"][0], "name": "nuclei"}]}
    assert get_label_name(path) == "nuclei"

    assert get_label_name(str(tmp_path / "a (1).tif")) == "a_1"
    assert get_label_name(raw) == "segmentation"
    assert get_label_name(path, label_name="cells") == "cells"
    with pytest.raises(ValueError):
        get_label_name(path, label_name="a/b")


@pytest.mark.parametrize("extension", [".tif", ".ome.zarr"])
def test_skip_and_overwrite_output(tmp_path, labels, extension, capsys):
    path = str(tmp_path / f"out{extension}")
    assert not skip_existing_output(path, "cells", overwrite=False)
    write_segmentation(path, labels, label_name="cells")

    assert skip_existing_output(path, "cells", overwrite=False)
    assert "already stored" in capsys.readouterr().out
    assert not skip_existing_output(path, "cells", overwrite=True)
    if extension == ".ome.zarr":
        assert not skip_existing_output(path, "nuclei", overwrite=False)

    with pytest.raises(FileExistsError):
        write_segmentation(path, labels, label_name="cells")
    write_segmentation(path, labels + 1, label_name="cells", overwrite=True)
    np.testing.assert_array_equal(load_output(path, "cells"), labels + 1)
    if extension == ".ome.zarr":
        assert _label_attrs(path, "cells")[0]["labels"] == ["cells"]
