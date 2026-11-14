import os

import h5py
import numpy as np
import pytest
import imageio.v3 as imageio
from skimage.measure import label
from click.testing import CliRunner
from skimage.data import binary_blobs

from micro_sam.v2.training.loaders import (
    default_sam2_dataset, default_sam2_loader, resolve_data_paths, split_train_val_files,
)

SHAPE_2D = (96, 96)
SHAPE_3D = (6, 96, 96)
PATCH_2D = (64, 64)


def _data(shape, seed):
    raw = binary_blobs(shape[-1], blob_size_fraction=0.15, n_dim=len(shape), rng=seed)
    raw = raw[tuple(slice(0, size) for size in shape)]
    labels = label(raw).astype("uint32")
    raw = (raw * 200 + np.random.default_rng(seed).integers(0, 50, size=shape)).astype("uint8")
    return raw, labels


def _write_container(path, raw, labels):
    import zarr
    import z5py

    if path.endswith(".h5"):
        with h5py.File(path, "w") as f:
            f.create_dataset("raw", data=raw)
            f.create_dataset("labels", data=labels)
    elif path.endswith(".n5"):
        with z5py.File(path, "w") as f:
            f.create_dataset("raw", data=raw, chunks=raw.shape)
            f.create_dataset("labels", data=labels, chunks=labels.shape)
    else:
        group = zarr.open_group(path, mode="w")
        group.create_array("raw", data=raw)
        group.create_array("labels", data=labels)


def _write_ome_zarr(path, raw, labels):
    import zarr
    from ome_zarr.writer import write_image, write_labels

    axes = "zyx" if raw.ndim == 3 else "yx"
    group = zarr.open_group(path, mode="w")
    write_image(raw, group, axes=axes, scale_factors=[])
    write_labels(labels, group, name="cells", axes=axes, scale_factors=[])


@pytest.fixture(scope="module")
def data_folder(tmp_path_factory):
    folder = tmp_path_factory.mktemp("training_data")
    for sub in ("tif_images", "tif_labels", "png_images", "png_labels", "rgb_images"):
        os.makedirs(folder / sub)
    for i in range(3):
        raw, labels = _data(SHAPE_2D, seed=i)
        imageio.imwrite(folder / "tif_images" / f"im{i}.tif", raw)
        imageio.imwrite(folder / "tif_labels" / f"im{i}.tif", labels)
        imageio.imwrite(folder / "png_images" / f"im{i}.png", raw)
        imageio.imwrite(folder / "png_labels" / f"im{i}.png", labels.astype("uint16"))
        imageio.imwrite(folder / "rgb_images" / f"im{i}.png", np.stack([raw] * 3, axis=-1))
    raw, labels = _data(SHAPE_3D, seed=7)
    imageio.imwrite(folder / "volume.tif", raw)
    imageio.imwrite(folder / "volume_labels.tif", labels)
    for name in ("data.h5", "data.zarr", "data.n5"):
        _write_container(str(folder / name), raw, labels)
    _write_ome_zarr(str(folder / "data.ome.zarr"), raw, labels)
    raw_2d, labels_2d = _data(SHAPE_2D, seed=11)
    _write_ome_zarr(str(folder / "image.ome.zarr"), raw_2d, labels_2d)
    return folder


def _check_sample(dataset, mode, depth):
    x, y = dataset[0]
    assert x.shape == (3, depth) + PATCH_2D
    assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0
    n_channels = {"interactive": 1, "automatic": 5, "joint": 6, "semantic": 1}[mode]
    assert y.shape == (n_channels, depth) + PATCH_2D
    if mode in ("interactive", "joint"):
        instances = y[0]
        assert len(np.unique(instances.numpy())) > 1
        assert np.allclose(instances, instances.round())


@pytest.mark.parametrize("mode", ["interactive", "automatic", "joint", "semantic"])
def test_files_2d(data_folder, mode):
    images = sorted(str(p) for p in (data_folder / "tif_images").iterdir())
    labels = sorted(str(p) for p in (data_folder / "tif_labels").iterdir())
    dataset = default_sam2_dataset(images, None, labels, None, PATCH_2D, mode, num_classes=4)
    assert len(dataset) == 100
    _check_sample(dataset, mode, depth=1)


@pytest.mark.parametrize("image_folder", ["png_images", "rgb_images", "tif_images"])
def test_folder_with_pattern(data_folder, image_folder):
    pattern = "*.tif" if image_folder == "tif_images" else "*.png"
    label_folder, label_pattern = ("tif_labels", "*.tif") if image_folder == "tif_images" else ("png_labels", "*.png")
    dataset = default_sam2_dataset(
        str(data_folder / image_folder), pattern, str(data_folder / label_folder), label_pattern,
        (1,) + PATCH_2D, "interactive", is_train=False,
    )
    assert len(dataset) == 5
    _check_sample(dataset, "interactive", depth=1)


@pytest.mark.parametrize("name", ["data.h5", "data.zarr", "data.n5"])
@pytest.mark.parametrize("patch_shape", [PATCH_2D, (4,) + PATCH_2D])
def test_containers(data_folder, name, patch_shape):
    path = str(data_folder / name)
    dataset = default_sam2_dataset(path, "raw", path, "labels", patch_shape, "joint")
    _check_sample(dataset, "joint", depth=1 if len(patch_shape) == 2 else 4)


@pytest.mark.parametrize("patch_shape", [PATCH_2D, (4,) + PATCH_2D])
def test_ome_zarr(data_folder, patch_shape):
    # The groups select their full resolution scale, whose array name depends on the ome-zarr version.
    path = str(data_folder / "data.ome.zarr")
    dataset = default_sam2_dataset(path, None, path, "labels/cells", patch_shape, "automatic")
    _check_sample(dataset, "automatic", depth=1 if len(patch_shape) == 2 else 4)

    path = str(data_folder / "image.ome.zarr")
    dataset = default_sam2_dataset(path, None, path, "labels/cells", PATCH_2D, "interactive")
    _check_sample(dataset, "interactive", depth=1)

    # An explicit scale is used as it is.
    scale = resolve_data_paths(path, None)[1]
    assert resolve_data_paths(path, scale)[1] == scale
    assert resolve_data_paths(path, "labels/cells")[1] == f"labels/cells/{scale}"


def test_tif_volume(data_folder):
    raw, labels = str(data_folder / "volume.tif"), str(data_folder / "volume_labels.tif")
    _check_sample(default_sam2_dataset(raw, None, labels, None, PATCH_2D, "semantic", num_classes=2), "semantic", 1)
    _check_sample(default_sam2_dataset(raw, None, labels, None, (4,) + PATCH_2D, "interactive"), "interactive", 4)


def test_semantic_labels_transform(data_folder):
    from micro_sam.v2.transforms.labels import semantic_labels

    path = str(data_folder / "data.h5")
    dataset = default_sam2_dataset(
        path, "raw", path, "labels", PATCH_2D, "semantic", num_classes=3, label_transform=semantic_labels,
    )
    _, y = dataset[0]
    assert set(np.unique(y.numpy())) <= {0, 1, 2}


def test_arrays():
    data = [_data(SHAPE_2D, seed=i) for i in range(2)]
    dataset = default_sam2_dataset([d[0] for d in data], None, [d[1] for d in data], None, PATCH_2D, "automatic")
    _check_sample(dataset, "automatic", depth=1)


def test_loader(data_folder):
    path = str(data_folder / "data.h5")
    loader = default_sam2_loader(path, "raw", path, "labels", PATCH_2D, "interactive", batch_size=2, num_workers=0)
    x, y = next(iter(loader))
    assert x.shape == (2, 3, 1) + PATCH_2D and y.shape == (2, 1, 1) + PATCH_2D
    assert loader.shuffle


def test_invalid_inputs(data_folder):
    path = str(data_folder / "data.h5")
    with pytest.raises(ValueError, match="training mode"):
        default_sam2_dataset(path, "raw", path, "labels", PATCH_2D, "instances")
    with pytest.raises(ValueError, match="num_classes"):
        default_sam2_dataset(path, "raw", path, "labels", PATCH_2D, "semantic")
    with pytest.raises(ValueError, match="needs volumes"):
        images = str(data_folder / "tif_images")
        default_sam2_dataset(images, "*.tif", images, "*.tif", (4,) + PATCH_2D, "interactive")
    with pytest.raises(ValueError, match="no files matching"):
        resolve_data_paths(str(data_folder), "*.jpg")


@pytest.mark.parametrize("kwargs, message", [
    ({}, "either 'input_path'"),
    ({"input_path": "data", "train_loader": [], "val_loader": []}, "either 'input_path'"),
    ({"train_loader": []}, "both 'train_loader'"),
])
def test_train_joint_sam2_data_arguments(kwargs, message):
    from micro_sam.v2.training import train_joint_sam2

    with pytest.raises(ValueError, match=message):
        train_joint_sam2("name", "hvit_t_cells", **kwargs)


def test_split_train_val_files():
    paths = [f"im{i}.tif" for i in range(20)]
    labels = [f"lab{i}.tif" for i in range(20)]
    train, train_labels, val, val_labels = split_train_val_files(paths, labels, val_fraction=0.1)
    assert len(val) == 2 and len(train) == 18
    assert sorted(train + val) == sorted(paths)
    assert [p.replace("im", "lab") for p in val] == val_labels
    with pytest.raises(ValueError, match="only one file"):
        split_train_val_files(paths[:1], labels[:1])


@pytest.fixture
def captured_training(monkeypatch):
    import micro_sam.v2.training as training

    calls = {}

    def capture(name):
        def fake(**kwargs):
            calls[name] = kwargs
        return fake

    for name in ("train_sam2", "train_automatic", "train_joint_sam2", "train_semantic"):
        monkeypatch.setattr(training, name, capture(name))
    return calls


@pytest.mark.parametrize("mode, trainer", [
    ("interactive", "train_sam2"), ("automatic", "train_automatic"),
    ("joint", "train_joint_sam2"), ("semantic", "train_semantic"),
])
def test_cli_train(data_folder, captured_training, mode, trainer):
    from micro_sam._cli import cli

    args = [
        "train", "-i", str(data_folder / "tif_images"), "--image_key", "*.tif",
        "-l", str(data_folder / "tif_labels"), "--label_key", "*.tif", "--mode", mode,
        "--patch_shape", "64,64", "--n_iterations", "5", "--num_workers", "0", "--num_classes", "2",
    ]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    kwargs = captured_training[trainer]
    assert kwargs["name"] == f"sam2_{mode}" and kwargs["model_type"] == "hvit_t_cells"
    assert kwargs["n_iterations"] == 5
    x, y = next(iter(kwargs["train_loader"]))
    assert x.shape == (1, 3, 1, 64, 64)
    # Without validation data, one of the three files is used for validation.
    assert "Training on 2 files and validating on 1 files." in result.output


def test_cli_train_containers(data_folder, captured_training):
    from micro_sam._cli import cli

    path = str(data_folder / "data.h5")
    args = [
        "train", "-i", path, "--image_key", "raw", "-l", path, "--label_key", "labels",
        "--val_images", str(data_folder / "data.n5"), "--val_labels", str(data_folder / "data.n5"),
        "--mode", "automatic", "--patch_shape", "4,64,64", "--num_workers", "0", "-m", "hvit_s_cells",
    ]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    kwargs = captured_training["train_automatic"]
    assert kwargs["model_type"] == "hvit_s_cells" and kwargs["with_boundaries"]
    x, y = next(iter(kwargs["val_loader"]))
    assert x.shape == (1, 3, 4, 64, 64) and y.shape == (1, 5, 4, 64, 64)


@pytest.mark.parametrize("extra_args, message", [
    (["--mode", "semantic"], "--num_classes"),
    (["--mode", "automatic", "-c", "model.pt"], "--checkpoint_path"),
    (["--val_images", "a.tif"], "--val_labels"),
    (["--patch_shape", "64"], "--patch_shape"),
])
def test_cli_train_errors(data_folder, captured_training, extra_args, message):
    from micro_sam._cli import cli

    path = str(data_folder / "data.h5")
    args = ["train", "-i", path, "--image_key", "raw", "-l", path, "--label_key", "labels"] + extra_args
    result = CliRunner().invoke(cli, args)
    assert result.exit_code != 0
    assert message in result.output
    assert not captured_training


def test_cli_train_single_file_needs_validation_data(data_folder, captured_training):
    from micro_sam._cli import cli

    path = str(data_folder / "data.h5")
    result = CliRunner().invoke(cli, ["train", "-i", path, "--image_key", "raw", "-l", path, "--label_key", "labels"])
    assert result.exit_code != 0
    assert "Pass separate validation data" in result.output
