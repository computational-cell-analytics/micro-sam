import os
import platform
import unittest
from shutil import which, rmtree
from subprocess import run

import imageio.v3 as imageio
import micro_sam.util as util
import zarr
from skimage.data import binary_blobs


class TestCLI(unittest.TestCase):
    model_type = "vit_t_lm" if util.VIT_T_SUPPORT else "vit_b_lm"
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

    def test_precompute_embeddings(self):
        self._test_command("micro_sam.precompute_embeddings")

        # The filepaths can't be found on windows, probably due different filepath conventions.
        # The actual functionality likely works despite this issue.
        if platform.system() == "Windows":
            return

        # Create 3 images as testdata.
        for i in range(3):
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
        with zarr.open(emb_path1, "r") as f:
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
        with zarr.open(emb_path2, "r") as f:
            self.assertIn("features", f)
            self.assertEqual(f["features"].shape[0], 3)
        ais_path = os.path.join(emb_path2, "is_state.h5")
        self.assertTrue(os.path.exists(ais_path))

        # Test precomputation with pattern to process multiple image.
        emb_path3 = os.path.join(self.tmp_folder, "embedddings3")
        run([
            "micro_sam.precompute_embeddings", "-i", self.tmp_folder, "-e", emb_path3,
            "-m", self.model_type, "--pattern", "*.tif", "--precompute_amg_state"
        ])
        for i in range(3):
            self.assertTrue(os.path.exists(os.path.join(emb_path3, f"image-{i}.zarr")))
            ais_path = os.path.join(emb_path3, f"image-{i}.zarr", "is_state.h5")
            self.assertTrue(os.path.exists(ais_path))


if __name__ == "__main__":
    unittest.main()
