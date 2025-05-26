import os
import platform
import unittest
from subprocess import run
from shutil import which, rmtree

import zarr
import pytest
import imageio.v3 as imageio
from skimage.data import binary_blobs

import micro_sam.util as util


class TestCLI(unittest.TestCase):
    model_type = "vit_t_lm" if util.VIT_T_SUPPORT else "vit_b_lm"
    default_model_type = "vit_t" if util.VIT_T_SUPPORT else "vit_b"
    tmp_folder = "tmp-files"

    def setUp(self):
        os.makedirs(self.tmp_folder, exist_ok=True)

    def tearDown(self):
        rmtree(self.tmp_folder)

    def _test_command(self, cmd):
        self.assertTrue(which(cmd) is not None)

    def test_annotator_2d(self):
        self._test_command("micro_sam.annotator_2d")

    def test_annotator_3d(self):
        self._test_command("micro_sam.annotator_3d")

    def test_annotator_tracking(self):
        self._test_command("micro_sam.annotator_tracking")

    def test_image_series_annotator(self):
        self._test_command("micro_sam.image_series_annotator")

    @pytest.mark.skipif(platform.system() == "Windows", reason="CLI test is not working on windows.")
    def test_precompute_embeddings(self):
        self._test_command("micro_sam.precompute_embeddings")

        # Create 2 images as testdata.
        n_images = 2
        for i in range(n_images):
            im_path = os.path.join(self.tmp_folder, f"image-{i}.tif")
            image_data = binary_blobs(512).astype("uint8") * 255
            imageio.imwrite(im_path, image_data)

        # Test precomputation with a single image.
        emb_path1 = os.path.join(self.tmp_folder, "embedddings1.zarr")
        run([
            "micro_sam.precompute_embeddings", "-i", im_path, "-e", emb_path1,
            "-m", self.model_type, "--precompute_amg_state"
        ])
        self.assertTrue(os.path.exists(emb_path1))
        f = zarr.open(emb_path1, mode="r")
        self.assertIn("features", f)

        ais_path = os.path.join(emb_path1, "is_state.h5")
        self.assertTrue(os.path.exists(ais_path))

        # Test precomputation with image stack.
        emb_path2 = os.path.join(self.tmp_folder, "embedddings2.zarr")
        run([
            "micro_sam.precompute_embeddings", "-i", self.tmp_folder, "-e", emb_path2,
            "-m", self.model_type, "-k", "*.tif", "--precompute_amg_state"
        ])
        self.assertTrue(os.path.exists(emb_path2))
        f = zarr.open(emb_path2, mode="r")
        self.assertIn("features", f)
        self.assertEqual(f["features"].shape[0], n_images)

        ais_path = os.path.join(emb_path2, "is_state.h5")
        self.assertTrue(os.path.exists(ais_path))

        # Test precomputation with pattern to process multiple image.
        emb_path3 = os.path.join(self.tmp_folder, "embedddings3")
        run([
            "micro_sam.precompute_embeddings", "-i", self.tmp_folder, "-e", emb_path3,
            "-m", self.model_type, "--pattern", "*.tif", "--precompute_amg_state"
        ])
        for i in range(n_images):
            self.assertTrue(os.path.exists(os.path.join(emb_path3, f"image-{i}.zarr")))
            ais_path = os.path.join(emb_path3, f"image-{i}.zarr", "is_state.h5")
            self.assertTrue(os.path.exists(ais_path))

    @pytest.mark.skipif(platform.system() == "Windows", reason="CLI test is not working on windows.")
    def test_automatic_segmentation(self):
        self._test_command("micro_sam.automatic_segmentation")

        # Create 1 image as testdata.
        im_path = os.path.join(self.tmp_folder, "image.tif")
        image_data = binary_blobs(512).astype("uint8") * 255
        imageio.imwrite(im_path, image_data)

        # Path to save automatic segmentation outputs.
        out_path = "output.tif"

        # Test AMG with default model in default mode.
        run(["micro_sam.automatic_segmentation", "-i", im_path, "-o", out_path,
             "-m", self.default_model_type, "--points_per_side", "4"])
        self.assertTrue(os.path.exists(out_path))
        os.remove(out_path)

        # Test AMG with default model exclusively in AMG mode.
        run(["micro_sam.automatic_segmentation", "-i", im_path, "-o", out_path,
             "-m", self.default_model_type, "--mode", "amg", "--points_per_side", "4"])
        self.assertTrue(os.path.exists(out_path))
        os.remove(out_path)

        # Test AIS with 'micro-sam' model in default mode.
        run(["micro_sam.automatic_segmentation", "-i", im_path, "-o", out_path, "-m", self.model_type])
        self.assertTrue(os.path.exists(out_path))
        os.remove(out_path)

        # Test AIS with 'micro-sam' model exclusively in AMG mode.
        run(["micro_sam.automatic_segmentation", "-i", im_path, "-o", out_path,
             "-m", self.model_type, "--mode", "amg", "--points_per_side", "4"])
        self.assertTrue(os.path.exists(out_path))
        os.remove(out_path)

        # Test AIS with 'micro-sam' model exclusively in AIS mode.
        run(["micro_sam.automatic_segmentation", "-i", im_path, "-o", out_path, "-m", self.model_type, "--mode", "ais"])
        self.assertTrue(os.path.exists(out_path))
        os.remove(out_path)


if __name__ == "__main__":
    unittest.main()
