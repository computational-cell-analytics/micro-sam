import os
import sys
import platform
import unittest
from shutil import rmtree
from subprocess import run

import z5py
import pytest
import numpy as np
import imageio.v3 as imageio
from skimage.data import binary_blobs

import micro_sam.util as util


def _cli(*args):
    return run([sys.executable, "-m", "micro_sam._cli", *args])


class TestCLI(unittest.TestCase):
    model_type = "vit_t_lm" if util.VIT_T_SUPPORT else "vit_b_lm"
    default_model_type = "vit_t" if util.VIT_T_SUPPORT else "vit_b"
    tmp_folder = "tmp-files"

    def setUp(self):
        os.makedirs(self.tmp_folder, exist_ok=True)

    def tearDown(self):
        rmtree(self.tmp_folder)

    def _test_help(self, *args):
        result = _cli(*args, "--help")
        self.assertEqual(result.returncode, 0)

    def test_help(self):
        self._test_help()
        self._test_help("annotator")
        self._test_help("inference")
        for cmd in [
            ["annotator", "segmentation"], ["annotator", "tracking"], ["annotator", "batch"],
            ["annotator", "pixel-classification"], ["annotator", "object-classification"],
            ["inference", "segmentation"], ["inference", "tracking"],
            ["inference", "pixel-classification"], ["inference", "object-classification"],
            ["precompute-embeddings"], ["train"], ["train-generalist"], ["info"],
        ]:
            self._test_help(*cmd)

    @pytest.mark.v1
    def test_v1_help(self):
        self._test_help("v1")
        for cmd in ["train", "automatic_segmentation", "evaluate", "benchmark_sam"]:
            self._test_help("v1", cmd)

    @pytest.mark.skipif(platform.system() == "Windows", reason="CLI test is not working on windows.")
    def test_precompute_embeddings(self):
        # Create 2 images as testdata.
        n_images = 2
        for i in range(n_images):
            im_path = os.path.join(self.tmp_folder, f"image-{i}.tif")
            image_data = binary_blobs(512).astype("uint8") * 255
            imageio.imwrite(im_path, image_data)

        # Test precomputation with a single (2d) image.
        emb_path1 = os.path.join(self.tmp_folder, "embedddings1.zarr")
        _cli("precompute-embeddings", "-i", im_path, "-e", emb_path1, "-m", "hvit_t")
        self.assertTrue(os.path.exists(emb_path1))
        with z5py.File(emb_path1, "r") as f:
            self.assertIn("features", f)
            self.assertIn("high_res_feats", f)

        # Test precomputation with an image stack (loaded as a 3d volume).
        emb_path2 = os.path.join(self.tmp_folder, "embedddings2.zarr")
        _cli("precompute-embeddings", "-i", self.tmp_folder, "-e", emb_path2, "-m", "hvit_t", "-k", "*.tif")
        self.assertTrue(os.path.exists(emb_path2))
        with z5py.File(emb_path2, "r") as f:
            self.assertIn("features", f)
            self.assertEqual(f["features"].shape[0], n_images)

        # Test precomputation with a pattern to process multiple images.
        emb_path3 = os.path.join(self.tmp_folder, "embedddings3")
        _cli(
            "precompute-embeddings", "-i", self.tmp_folder, "-e", emb_path3, "-m", "hvit_t", "--pattern", "*.tif",
        )
        for i in range(n_images):
            self.assertTrue(os.path.exists(os.path.join(emb_path3, f"image-{i}.zarr")))

    @pytest.mark.v1
    @pytest.mark.skipif(platform.system() == "Windows", reason="CLI test is not working on windows.")
    def test_v1_automatic_segmentation(self):
        # Create 1 image as testdata.
        im_path = os.path.join(self.tmp_folder, "image.tif")
        image_data = binary_blobs(512).astype("uint8") * 255
        imageio.imwrite(im_path, image_data)

        # Path to save automatic segmentation outputs.
        out_path = "output.tif"

        # Test AMG with default model in default mode.
        _cli(
            "v1", "automatic_segmentation", "-i", im_path, "-o", out_path,
            "-m", self.default_model_type, "--points_per_side", "4"
        )
        self.assertTrue(os.path.exists(out_path))
        os.remove(out_path)

        # Test AMG with default model exclusively in AMG mode.
        _cli(
            "v1", "automatic_segmentation", "-i", im_path, "-o", out_path,
            "-m", self.default_model_type, "--mode", "amg", "--points_per_side", "4"
        )
        self.assertTrue(os.path.exists(out_path))
        os.remove(out_path)

        # Test AIS with 'micro-sam' model in default mode.
        _cli("v1", "automatic_segmentation", "-i", im_path, "-o", out_path, "-m", self.model_type)
        self.assertTrue(os.path.exists(out_path))
        os.remove(out_path)

        # Test AIS with 'micro-sam' model exclusively in AMG mode.
        _cli(
            "v1", "automatic_segmentation", "-i", im_path, "-o", out_path,
            "-m", self.model_type, "--mode", "amg", "--points_per_side", "4"
        )
        self.assertTrue(os.path.exists(out_path))
        os.remove(out_path)

        # Test AIS with 'micro-sam' model exclusively in AIS mode.
        _cli("v1", "automatic_segmentation", "-i", im_path, "-o", out_path, "-m", self.model_type, "--mode", "ais")
        self.assertTrue(os.path.exists(out_path))
        os.remove(out_path)

    @pytest.mark.skipif(platform.system() == "Windows", reason="CLI test is not working on windows.")
    def test_parse_shape_turns_tiling_off(self):
        from micro_sam._cli import _parse_shape

        self.assertIsNone(_parse_shape(None))  # not given, i.e. decide automatically
        self.assertEqual(_parse_shape("384,384"), (384, 384))
        self.assertEqual(_parse_shape("4, 384, 384"), (4, 384, 384))
        # 'none' / 'off' is the readable spelling of the all-zero 'run untiled' request.
        for value in ("none", "None", " off "):
            self.assertEqual(_parse_shape(value), (0, 0))
        self.assertEqual(_parse_shape("none", ndim=3), (0, 0, 0))

    def test_automatic_segmentation(self):
        # Create 1 image as testdata.
        im_path = os.path.join(self.tmp_folder, "image.tif")
        image_data = binary_blobs(256).astype("uint8") * 255
        imageio.imwrite(im_path, image_data)

        out_path = os.path.join(self.tmp_folder, "seg.tif")

        # Test 'sparse' (flow) mode with a pass-through postproc param in '--key=value' form.
        _cli(
            "inference", "segmentation", "-i", im_path, "-o", out_path, "-m", "hvit_t_cells",
            "-n", "2", "--mode", "sparse", "--foreground_threshold=0.5"
        )
        self.assertTrue(os.path.exists(out_path))
        os.remove(out_path)

        # Test 'dense' (multicut) mode with a pass-through postproc param in '--key value' form.
        _cli(
            "inference", "segmentation", "-i", im_path, "-o", out_path, "-m", "hvit_t_cells",
            "-n", "2", "--mode", "dense", "--beta", "0.6"
        )
        self.assertTrue(os.path.exists(out_path))
        os.remove(out_path)

        # Test the embedding-reuse path (only the decoder is run on cached embeddings).
        emb_path = os.path.join(self.tmp_folder, "seg-embeddings.zarr")
        _cli(
            "inference", "segmentation", "-i", im_path, "-o", out_path, "-m", "hvit_t_cells",
            "-n", "2", "--mode", "sparse", "-e", emb_path
        )
        self.assertTrue(os.path.exists(out_path))
        self.assertTrue(os.path.exists(emb_path))
        os.remove(out_path)


if __name__ == "__main__":
    unittest.main()


def test_precompute_embeddings_forwards_the_tiling(monkeypatch):
    from click.testing import CliRunner

    from micro_sam._cli import cli
    import micro_sam.precompute_state as precompute_state

    calls = []
    monkeypatch.setattr(precompute_state, "precompute_state", lambda *args, **kwargs: calls.append(kwargs))

    runner = CliRunner()
    for options, expected in [
        ([], (None, None)),  # resolved per image, like the other tools
        (["--tile_shape", "256,256", "--overlap", "32,32"], ((256, 256), (32, 32))),
        (["--tile_shape", "none", "--overlap", "none"], ((0, 0), (0, 0))),
    ]:
        result = runner.invoke(cli, ["precompute-embeddings", "-i", "image.tif", "-e", "embeddings.zarr", *options])
        assert result.exit_code == 0, result.output
        assert (calls[-1]["tile_shape"], calls[-1]["halo"]) == expected


def test_v1_automatic_segmentation_forwards_the_tiling(monkeypatch):
    from click.testing import CliRunner

    from micro_sam._cli import cli
    import micro_sam.v1.automatic_segmentation as v1_automatic_segmentation

    calls = []
    monkeypatch.setattr(v1_automatic_segmentation, "main", lambda argv: calls.append(argv))

    runner = CliRunner()
    command = ["v1", "automatic_segmentation", "-i", "image.tif", "-o", "out.tif"]
    tiling = ["--tile_shape", "384", "384", "--halo", "64", "64"]
    for options, expected in [([], []), (tiling, tiling)]:  # without options, the v1 command decides the tiling
        result = runner.invoke(cli, [*command, *options])
        assert result.exit_code == 0, result.output
        assert calls[-1] == ["--input_path", "image.tif", "--output_path", "out.tif", *expected]

    # The tile shape and the halo take exactly two values.
    result = runner.invoke(cli, [*command, "--tile_shape", "384"])
    assert result.exit_code != 0


@pytest.fixture
def container_folder(tmp_path):
    from micro_sam._test_util import write_ome_zarr

    data = binary_blobs(32, n_dim=2, rng=0).astype("uint8")
    folder = tmp_path / "images"
    folder.mkdir()
    for name in ("a", "b"):
        write_ome_zarr(str(folder / f"{name}.ome.zarr"), data, data)
    imageio.imwrite(folder / "c.tif", data)
    return folder


def test_inference_segmentation_resolves_containers(monkeypatch, tmp_path, container_folder):
    from click.testing import CliRunner

    from micro_sam._cli import cli
    import micro_sam.v2.automatic_segmentation as automatic_segmentation

    calls = []
    monkeypatch.setattr(automatic_segmentation, "get_predictor_and_segmenter", lambda **kwargs: (None, None))
    monkeypatch.setattr(
        automatic_segmentation, "automatic_instance_segmentation", lambda **kwargs: calls.append(kwargs)
    )

    runner = CliRunner()
    out = str(tmp_path / "out")
    result = runner.invoke(cli, ["inference", "segmentation", "-i", str(container_folder), "-o", out])
    assert result.exit_code == 0, result.output
    assert [call["input_path"] for call in calls] == [
        str(container_folder / name) for name in ("a.ome.zarr", "b.ome.zarr", "c.tif")
    ]
    assert [call["output_path"] for call in calls] == [os.path.join(out, f"{name}.tif") for name in "abc"]

    calls.clear()
    path = str(container_folder / "a.ome.zarr")
    result = runner.invoke(cli, ["inference", "segmentation", "-i", path, "-o", out, "--pattern", "*.tif"])
    assert result.exit_code == 0, result.output
    assert [call["input_path"] for call in calls] == [path]


def test_inference_segmentation_ome_zarr_output(monkeypatch, tmp_path, container_folder):
    import click
    from click.testing import CliRunner

    from micro_sam._cli import cli, _resolve_output_paths
    import micro_sam.v2.automatic_segmentation as automatic_segmentation

    calls = []
    monkeypatch.setattr(automatic_segmentation, "get_predictor_and_segmenter", lambda **kwargs: (None, None))
    monkeypatch.setattr(
        automatic_segmentation, "automatic_instance_segmentation", lambda **kwargs: calls.append(kwargs)
    )

    path = str(container_folder / "a.ome.zarr")
    result = CliRunner().invoke(
        cli, ["inference", "segmentation", "-i", path, "-o", path, "--label_name", "cells", "--overwrite"]
    )
    assert result.exit_code == 0, result.output
    assert calls[0]["output_path"] == path
    assert calls[0]["label_name"] == "cells" and calls[0]["overwrite"]

    output_path = str(tmp_path / "out" / "result.ome.zarr")
    assert _resolve_output_paths([path], output_path) == [output_path]
    assert os.path.isdir(tmp_path / "out")
    with pytest.raises(click.UsageError, match="single input"):
        _resolve_output_paths([path, path], output_path)


def test_inference_pixel_classification_skips_existing_outputs(monkeypatch, tmp_path, container_folder):
    from click.testing import CliRunner

    import micro_sam._cli as micro_sam_cli
    import micro_sam.pixel_classification as pixel_classification

    calls = []

    def _predict(**kwargs):
        calls.append(kwargs["images"])
        return [util.load_image_data(path, key=kwargs["image_key"]) for path in kwargs["images"]]

    monkeypatch.setattr(micro_sam_cli, "_classifier_predictor", lambda *args: (None, None, None, "hvit_t", None, None))
    monkeypatch.setattr(pixel_classification, "run_prediction_with_pixel_classifier", _predict)

    out = tmp_path / "out"
    out.mkdir()
    imageio.imwrite(out / "a.tif", imageio.imread(container_folder / "c.tif"))
    args = [
        "inference", "pixel-classification", "-i", str(container_folder), "-o", str(out), "--classifier", "rf.joblib"
    ]
    runner = CliRunner()

    result = runner.invoke(micro_sam_cli.cli, args)
    assert result.exit_code == 0, result.output
    assert calls == [[str(container_folder / name) for name in ("b.ome.zarr", "c.tif")]]
    assert "already stored" in result.output

    # All results exist now, so nothing is computed.
    result = runner.invoke(micro_sam_cli.cli, args)
    assert result.exit_code == 0, result.output
    assert len(calls) == 1

    result = runner.invoke(micro_sam_cli.cli, args + ["--overwrite"])
    assert result.exit_code == 0, result.output
    assert len(calls) == 2 and len(calls[-1]) == 3

    path = str(container_folder / "a.ome.zarr")
    args = ["inference", "pixel-classification", "-i", path, "-o", path, "--classifier", "rf.joblib"]
    result = runner.invoke(micro_sam_cli.cli, args)
    assert result.exit_code == 0, result.output
    np.testing.assert_array_equal(util.load_image_data(path, key="labels/a"), util.load_image_data(path))


def test_inference_object_classification_resolves_containers(monkeypatch, tmp_path, container_folder):
    from click.testing import CliRunner

    import micro_sam._cli as micro_sam_cli
    import micro_sam.object_classification as object_classification

    calls = []

    def _predict(**kwargs):
        calls.append(kwargs)
        return [util.load_image_data(path, key=kwargs["segmentation_key"], is_segmentation=True)
                for path in kwargs["segmentations"]]

    monkeypatch.setattr(micro_sam_cli, "_classifier_predictor", lambda *args: (None, None, None, "hvit_t", None, None))
    monkeypatch.setattr(object_classification, "run_prediction_with_object_classifier", _predict)

    out = str(tmp_path / "out")
    result = CliRunner().invoke(micro_sam_cli.cli, [
        "inference", "object-classification", "-i", str(container_folder), "--pattern", "*.zarr",
        "-s", str(container_folder), "--segmentation_pattern", "*.zarr", "-o", out, "--classifier", "rf.joblib",
    ])
    assert result.exit_code == 0, result.output
    expected = [str(container_folder / name) for name in ("a.ome.zarr", "b.ome.zarr")]
    assert calls[0]["images"] == expected and calls[0]["segmentations"] == expected
    assert sorted(os.listdir(out)) == ["a.tif", "b.tif"]


def test_annotator_batch_forwards_the_keys(monkeypatch, tmp_path, container_folder):
    from click.testing import CliRunner

    import importlib
    from micro_sam._cli import cli

    # The package exports functions with the same names as these modules.
    batch_annotator = importlib.import_module("micro_sam.sam_annotator.batch_annotator")
    object_classifier = importlib.import_module("micro_sam.sam_annotator.object_classifier")

    calls = []
    monkeypatch.setattr(batch_annotator, "image_folder_annotator", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(object_classifier, "batch_object_classifier", lambda *args, **kwargs: calls.append(args))

    runner = CliRunner()
    command = ["annotator", "batch", "-i", str(container_folder), "-o", str(tmp_path / "out")]
    result = runner.invoke(cli, [*command, "-k", "s0", "--segmentation_key", "labels/cells"])
    assert result.exit_code == 0, result.output
    assert (calls[-1]["key"], calls[-1]["segmentation_key"]) == ("s0", "labels/cells")

    result = runner.invoke(cli, [
        *command, "--task", "object-classification", "-p", "*.zarr",
        "--initial_segmentation_folder", str(container_folder), "--initial_segmentation_pattern", "*.zarr",
    ])
    assert result.exit_code == 0, result.output
    expected = [str(container_folder / name) for name in ("a.ome.zarr", "b.ome.zarr")]
    assert list(calls[-1][:2]) == [expected, expected]
