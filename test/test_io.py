import os

import h5py
import numpy as np
import pytest
import z5py
import zarr
import imageio.v3 as imageio

from micro_sam._io import (
    check_unique_input_names, get_image_shape, get_input_name, get_input_paths, get_label_name, get_output_path,
    is_container, load_output, resolve_container_key, skip_existing_output, write_segmentation,
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

    # Without a 'labels' group, the image is the segmentation.
    path = str(tmp_path / "no_labels.ome.zarr")
    write_ome_zarr(path, labels, version=version, n_scales=2)
    np.testing.assert_array_equal(load_image_data(path, is_segmentation=True), labels)


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


def test_ome_zarr_invalid_metadata(tmp_path, raw):
    path = str(tmp_path / "invalid.ome.zarr")
    root = zarr.open_group(path, mode="w")
    root.create_array("0", data=raw)
    root.attrs["ome"] = {"version": "0.5", "multiscales": [{"axes": [{"name": "y", "type": "space"}], "datasets": []}]}
    with pytest.raises(ValueError, match="invalid ome-zarr metadata"):
        load_image_data(path)


# Input paths and names.


def test_input_paths_and_names(tmp_path, raw):
    folder = tmp_path / "inputs"
    folder.mkdir()
    write_ome_zarr(str(folder / "b.ome.zarr"), raw)
    zarr.open_group(str(folder / "a.zarr"), mode="w").create_array("raw", data=raw)
    imageio.imwrite(folder / "c.tif", raw)
    (folder / "not_a_container").mkdir()

    # Without a pattern, a folder gives its files, but not its containers.
    assert get_input_paths(str(folder)) == [str(folder / "c.tif")]
    assert get_input_paths(str(folder), "*") == [str(folder / name) for name in ("a.zarr", "b.ome.zarr", "c.tif")]
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

    check_unique_input_names(["data/a.tif", "data/b.tif", "other/c.ome.zarr"])
    with pytest.raises(ValueError, match="same name"):
        check_unique_input_names(["data/a.tif", "data/b.tif", "data/a.ome.zarr"])


# Outputs.


def _ome_attrs(path, key=""):
    group = zarr.open_group(path, mode="r")
    attrs = dict((group[key] if key else group).attrs)
    return attrs.get("ome", attrs)


def test_output_path():
    assert get_output_path("out") == "out.tif"
    assert get_output_path("out.tif") == "out.tif"
    assert get_output_path("out.png") == "out.png"
    assert get_output_path("out.ome.zarr") == "out.ome.zarr"


def test_write_new_ome_zarr(tmp_path, labels):
    # A new file stores the segmentation as the root image, marked as a label image.
    path = str(tmp_path / "new" / "out.ome.zarr")
    write_segmentation(path, labels, label_name="cells")

    assert zarr.open_group(path, mode="r").metadata.zarr_format == 3
    attrs = _ome_attrs(path)
    assert attrs["version"] == "0.5"
    assert attrs["image-label"] == {"version": "0.5"}
    multiscale = attrs["multiscales"][0]
    assert multiscale["name"] == "cells"
    assert [axis["name"] for axis in multiscale["axes"]] == ["y", "x"]
    assert [dataset["path"] for dataset in multiscale["datasets"]] == ["s0"]
    np.testing.assert_array_equal(load_image_data(path), labels)
    np.testing.assert_array_equal(load_image_data(path, is_segmentation=True), labels)
    np.testing.assert_array_equal(load_output(path, "cells"), labels)


def _attrs(path, key=""):
    group = zarr.open_group(path, mode="r")
    return dict((group[key] if key else group).attrs)


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_write_provenance(tmp_path, raw, labels, version):
    from micro_sam._provenance import CONVENTION, make_step

    steps = [make_step("Segment", "Segmentation", "micro_sam.test", {"tile_shape": (16, 16)}, [])]

    # In an image, the label image holds the steps beside the ome-zarr metadata.
    path = str(tmp_path / "image.ome.zarr")
    write_ome_zarr(path, raw, version=version)
    write_segmentation(path, labels, label_name="result", provenance=steps)
    attrs = _attrs(path, "labels/result")
    assert attrs["provenance:steps"] == steps
    assert attrs["zarr_conventions"] == [CONVENTION]
    assert _ome_attrs(path, "labels/result")["multiscales"][0]["name"] == "result"
    assert "provenance:steps" not in _attrs(path)

    # A result that replaces another result does not keep the steps of this result.
    write_segmentation(path, labels, label_name="result", overwrite=True)
    assert "provenance:steps" not in _attrs(path, "labels/result")

    path = str(tmp_path / "new.ome.zarr")
    write_segmentation(path, labels, provenance=steps)
    assert _attrs(path)["provenance:steps"] == steps
    write_segmentation(path, labels, overwrite=True)
    attrs = _attrs(path)
    assert "provenance:steps" not in attrs and "zarr_conventions" not in attrs
    assert _ome_attrs(path)["multiscales"][0]["name"] == "segmentation"

    # An image file does not store the steps.
    write_segmentation(str(tmp_path / "out.tif"), labels, provenance=steps)


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_write_into_ome_zarr_image(tmp_path, raw, labels, version):
    path = str(tmp_path / "image.ome.zarr")
    write_ome_zarr(path, raw, labels, version=version, n_scales=2)
    scale_keys = ["s0", "s1"] if version == "0.5" else ["0", "1"]
    segmentation = labels.astype("uint32") + 1
    write_segmentation(path, segmentation, label_name="result")

    assert zarr.open_group(path, mode="r").metadata.zarr_format == (3 if version == "0.5" else 2)
    assert _ome_attrs(path, "labels")["labels"] == ["cells", "result"]
    attrs = _ome_attrs(path, "labels/result")
    assert attrs["image-label"] == {"version": version, "source": {"image": "../../"}}

    # The label image has the scale levels of the image, with the same scales.
    datasets = attrs["multiscales"][0]["datasets"]
    assert [dataset["path"] for dataset in datasets] == scale_keys
    assert datasets[1]["coordinateTransformations"] == [{"type": "scale", "scale": [2.0, 2.0]}]
    group = zarr.open_group(path, mode="r")
    np.testing.assert_array_equal(group[f"labels/result/{scale_keys[0]}"][:], segmentation)
    np.testing.assert_array_equal(group[f"labels/result/{scale_keys[1]}"][:], segmentation[::2, ::2])

    np.testing.assert_array_equal(load_image_data(path, key="labels/cells"), labels)
    np.testing.assert_array_equal(load_image_data(path, key="labels/result"), segmentation)
    np.testing.assert_array_equal(load_image_data(path), raw)


def test_write_into_ome_zarr_image_from_lower_level(tmp_path, raw, labels):
    path = str(tmp_path / "image.ome.zarr")
    write_ome_zarr(path, raw, n_scales=2)
    segmentation = labels[::2, ::2].astype("uint32")
    write_segmentation(path, segmentation, label_name="result")

    group = zarr.open_group(path, mode="r")
    np.testing.assert_array_equal(group["labels/result/s1"][:], segmentation)
    # Each pixel is repeated, so the upsampled level is not shifted against the image.
    np.testing.assert_array_equal(group["labels/result/s0"][:], segmentation.repeat(2, axis=0).repeat(2, axis=1))

    with pytest.raises(ValueError, match="shapes"):
        write_segmentation(path, labels[:10], label_name="other")


def test_write_into_ome_zarr_image_keeps_the_shards(tmp_path):
    path = str(tmp_path / "image.ome.zarr")
    root = zarr.open_group(path, mode="w")
    datasets = []
    # The last level is smaller than a chunk of the image.
    for level, shape in enumerate([(64, 64), (32, 32), (8, 8)]):
        root.create_array(f"s{level}", shape=shape, chunks=(16, 16), shards=(32, 32), dtype="uint8")
        scale = [{"type": "scale", "scale": [64 / shape[0]] * 2}]
        datasets.append({"path": f"s{level}", "coordinateTransformations": scale})
    axes = [{"name": name, "type": "space"} for name in "yx"]
    root.attrs["ome"] = {"version": "0.5", "multiscales": [{"axes": axes, "datasets": datasets}]}

    segmentation = _data(2, (64, 64)).astype("uint32")
    write_segmentation(path, segmentation, label_name="cells", num_workers=4)
    group = zarr.open_group(path, mode="r")["labels/cells"]
    assert (group["s0"].chunks, group["s0"].shards) == ((16, 16), (32, 32))
    assert (group["s1"].chunks, group["s1"].shards) == ((16, 16), (32, 32))
    assert (group["s2"].chunks, group["s2"].shards) == ((8, 8), (16, 16))
    np.testing.assert_array_equal(group["s0"][:], segmentation)
    np.testing.assert_array_equal(group["s1"][:], segmentation[::2, ::2])
    np.testing.assert_array_equal(group["s2"][:], segmentation[::8, ::8])


def test_write_into_ome_zarr_image_drops_the_channel_axis(tmp_path, labels):
    path = str(tmp_path / "image.ome.zarr")
    root = zarr.open_group(path, mode="w")
    root.create_array("0", data=np.zeros((3,) + SHAPE, dtype="uint8"))
    axes = [{"name": "c", "type": "channel"}] + [{"name": name, "type": "space", "unit": "micrometer"} for name in "yx"]
    transforms = [{"type": "scale", "scale": [1.0, 0.5, 0.25]}, {"type": "translation", "translation": [0, 1, 2]}]
    datasets = [{"path": "0", "coordinateTransformations": transforms}]
    root.attrs["ome"] = {"version": "0.5", "multiscales": [{"axes": axes, "datasets": datasets}]}

    write_segmentation(path, labels, label_name="cells")
    multiscale = _ome_attrs(path, "labels/cells")["multiscales"][0]
    assert multiscale["axes"] == axes[1:]
    assert multiscale["datasets"][0]["coordinateTransformations"] == [
        {"type": "scale", "scale": [0.5, 0.25]}, {"type": "translation", "translation": [1.0, 2.0]}
    ]
    np.testing.assert_array_equal(load_image_data(path, is_segmentation=True), labels)


def test_invalid_ome_zarr_outputs(tmp_path, raw, labels):
    path = str(tmp_path / "plain.ome.zarr")
    zarr.open_group(path, mode="w").create_array("raw", data=raw)
    with pytest.raises(ValueError, match="does not contain an ome-zarr image"):
        skip_existing_output(path, "cells", overwrite=False)

    path = str(tmp_path / "collection.ome.zarr")
    root = zarr.open_group(path, mode="w")
    root.attrs["ome"] = {"version": "0.5", "bioformats2raw.layout": 3}
    write_ome_zarr(os.path.join(path, "0"), raw)
    with pytest.raises(ValueError, match="does not contain an ome-zarr image"):
        write_segmentation(path, labels, label_name="cells")

    # A root label image only takes a result with its own name.
    path = str(tmp_path / "out.ome.zarr")
    write_segmentation(path, labels, label_name="cells")
    with pytest.raises(ValueError, match="root image is the label image 'cells'"):
        skip_existing_output(path, "nuclei", overwrite=False)


@pytest.mark.parametrize("output", ["out.ome.zarr", "image.ome.zarr"])
def test_incomplete_ome_zarr_output_is_replaced(tmp_path, raw, labels, output, monkeypatch):
    path = str(tmp_path / output)
    if output == "image.ome.zarr":
        write_ome_zarr(path, raw, n_scales=2)
    write_segmentation(path, labels, label_name="cells")

    # A write that fails before its metadata is written leaves a label image without metadata.
    def fail(*args, **kwargs):
        raise RuntimeError("The write failed.")

    with monkeypatch.context() as context:
        context.setattr("clearscale.OmeZarrGroup.from_single", fail)
        with pytest.raises(RuntimeError, match="The write failed."):
            write_segmentation(path, labels + 1, label_name="cells", overwrite=True)

    # The incomplete result is not stored, and the next write replaces it without 'overwrite'.
    assert not skip_existing_output(path, "cells", overwrite=False)
    write_segmentation(path, labels + 2, label_name="cells")
    assert skip_existing_output(path, "cells", overwrite=False)
    np.testing.assert_array_equal(load_output(path, "cells"), labels + 2)


@pytest.mark.parametrize("output", ["out.ome.zarr", "image.ome.zarr"])
def test_ome_zarr_output_rejects_a_time_axis(tmp_path, raw, labels, output):
    path = str(tmp_path / output)
    if output == "image.ome.zarr":
        write_ome_zarr(path, raw)
    write_segmentation(path, labels, label_name="cells")

    # The check runs before the stored result is replaced.
    with pytest.raises(ValueError, match="only 2d and 3d segmentations"):
        write_segmentation(path, np.zeros((2, 2) + SHAPE, dtype="uint32"), label_name="cells", overwrite=True)
    np.testing.assert_array_equal(load_output(path, "cells"), labels)


def test_label_name(tmp_path, raw):
    path = str(tmp_path / "my image.ome.zarr")
    write_ome_zarr(path, raw)
    assert get_label_name(path) == "my_image_segmentation"

    root = zarr.open_group(path, mode="a")
    ome = root.attrs["ome"]
    root.attrs["ome"] = {**ome, "multiscales": [{**ome["multiscales"][0], "name": "nuclei"}]}
    assert get_label_name(path) == "nuclei_segmentation"
    # The suffix keeps the default names of different results apart.
    assert get_label_name(path, suffix="pixel_classification") == "nuclei_pixel_classification"

    assert get_label_name(str(tmp_path / "a (1).tif")) == "a_1_segmentation"
    assert get_label_name(raw) == "segmentation"
    assert get_label_name(raw, suffix="object_classification") == "object_classification"
    assert get_label_name(path, label_name="cells", suffix="pixel_classification") == "cells"
    with pytest.raises(ValueError):
        get_label_name(path, label_name="a/b")


def test_load_output_at_the_scale_level_of_the_input(tmp_path, raw, labels):
    path = str(tmp_path / "image.ome.zarr")
    write_ome_zarr(path, raw, n_scales=2)
    segmentation = labels[::2, ::2].astype("uint32")
    write_segmentation(path, segmentation, label_name="cells")

    # The result of a lower scale level of the same file has the shape of this level.
    np.testing.assert_array_equal(load_output(path, "cells", path, "s1"), segmentation)
    assert load_output(path, "cells", path).shape == SHAPE
    # Any other input gives the highest resolution.
    assert load_output(path, "cells").shape == SHAPE
    assert load_output(path, "cells", raw).shape == SHAPE
    assert load_output(path, "cells", str(tmp_path / "other.tif")).shape == SHAPE


@pytest.mark.parametrize("output", ["out.tif", "out.ome.zarr", "image.ome.zarr"])
def test_skip_and_overwrite_output(tmp_path, raw, labels, output, capsys):
    path = str(tmp_path / output)
    if output == "image.ome.zarr":
        write_ome_zarr(path, raw, n_scales=2)
    assert not skip_existing_output(path, "cells", overwrite=False)
    write_segmentation(path, labels, label_name="cells")

    assert skip_existing_output(path, "cells", overwrite=False)
    assert "already stored" in capsys.readouterr().out
    assert not skip_existing_output(path, "cells", overwrite=True)
    if output == "image.ome.zarr":
        assert not skip_existing_output(path, "nuclei", overwrite=False)

    with pytest.raises(FileExistsError):
        write_segmentation(path, labels, label_name="cells")
    write_segmentation(path, labels + 1, label_name="cells", overwrite=True)
    np.testing.assert_array_equal(load_output(path, "cells"), labels + 1)
    if output == "image.ome.zarr":
        assert _ome_attrs(path, "labels")["labels"] == ["cells"]
        np.testing.assert_array_equal(load_image_data(path, key="labels/cells/s1"), (labels + 1)[::2, ::2])
