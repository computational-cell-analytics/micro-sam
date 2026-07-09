import os
import sys
import platform
import unittest
from subprocess import run
from shutil import rmtree

import z5py
import pytest
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
            ["precompute-embeddings"], ["train"], ["info"],
        ]:
            self._test_help(*cmd)

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

    @pytest.mark.skipif(platform.system() == "Windows", reason="CLI test is not working on windows.")
    def test_v1_automatic_segmentation(self):
        # Create 1 image as testdata.
        im_path = os.path.join(self.tmp_folder, "image.tif")
        image_data = binary_blobs(512).astype("uint8") * 255
        imageio.imwrite(im_path, image_data)

        # Path to save automatic segmentation outputs.
        out_path = "output.tif"

        # Test AMG with default model in default mode.
        _cli("v1", "automatic_segmentation", "-i", im_path, "-o", out_path,
             "-m", self.default_model_type, "--points_per_side", "4")
        self.assertTrue(os.path.exists(out_path))
        os.remove(out_path)

        # Test AMG with default model exclusively in AMG mode.
        _cli("v1", "automatic_segmentation", "-i", im_path, "-o", out_path,
             "-m", self.default_model_type, "--mode", "amg", "--points_per_side", "4")
        self.assertTrue(os.path.exists(out_path))
        os.remove(out_path)

        # Test AIS with 'micro-sam' model in default mode.
        _cli("v1", "automatic_segmentation", "-i", im_path, "-o", out_path, "-m", self.model_type)
        self.assertTrue(os.path.exists(out_path))
        os.remove(out_path)

        # Test AIS with 'micro-sam' model exclusively in AMG mode.
        _cli("v1", "automatic_segmentation", "-i", im_path, "-o", out_path,
             "-m", self.model_type, "--mode", "amg", "--points_per_side", "4")
        self.assertTrue(os.path.exists(out_path))
        os.remove(out_path)

        # Test AIS with 'micro-sam' model exclusively in AIS mode.
        _cli("v1", "automatic_segmentation", "-i", im_path, "-o", out_path, "-m", self.model_type, "--mode", "ais")
        self.assertTrue(os.path.exists(out_path))
        os.remove(out_path)

    @pytest.mark.skipif(platform.system() == "Windows", reason="CLI test is not working on windows.")
    def test_automatic_segmentation(self):
        # Create 1 image as testdata.
        im_path = os.path.join(self.tmp_folder, "image.tif")
        image_data = binary_blobs(256).astype("uint8") * 255
        imageio.imwrite(im_path, image_data)

        out_path = os.path.join(self.tmp_folder, "seg.tif")

        # Test 'sparse' (flow) mode with a pass-through postproc param in '--key=value' form.
        _cli("inference", "segmentation", "-i", im_path, "-o", out_path, "-m", "hvit_t_cells",
             "-n", "2", "--mode", "sparse", "--foreground_threshold=0.5")
        self.assertTrue(os.path.exists(out_path))
        os.remove(out_path)

        # Test 'dense' (multicut) mode with a pass-through postproc param in '--key value' form.
        _cli("inference", "segmentation", "-i", im_path, "-o", out_path, "-m", "hvit_t_cells",
             "-n", "2", "--mode", "dense", "--beta", "0.6")
        self.assertTrue(os.path.exists(out_path))
        os.remove(out_path)

        # Test the embedding-reuse path (only the decoder is run on cached embeddings).
        emb_path = os.path.join(self.tmp_folder, "seg-embeddings.zarr")
        _cli("inference", "segmentation", "-i", im_path, "-o", out_path, "-m", "hvit_t_cells",
             "-n", "2", "--mode", "sparse", "-e", emb_path)
        self.assertTrue(os.path.exists(out_path))
        self.assertTrue(os.path.exists(emb_path))
        os.remove(out_path)


if __name__ == "__main__":
    unittest.main()
