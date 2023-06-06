import os
import unittest
from shutil import rmtree

import numpy as np
import zarr


class TestUtil(unittest.TestCase):
    tmp_folder = "tmp-files"

    def setUp(self):
        os.makedirs(self.tmp_folder, exist_ok=True)

    def tearDown(self):
        rmtree(self.tmp_folder)

    def test_compute_iou(self):
        from micro_sam.util import compute_iou

        x1, x2 = np.zeros((32, 32), dtype="uint32"), np.zeros((32, 32), dtype="uint32")
        x1[:16] = 1
        x2[16:] = 1

        self.assertTrue(np.isclose(compute_iou(x1, x1), 1.0))
        self.assertTrue(np.isclose(compute_iou(x1, x2), 0.0))

        n_samples = 10
        for _ in range(n_samples):
            x1, x2 = (np.random.rand(32, 32) > 0.5), (np.random.rand(32, 32) > 0.5)
            self.assertTrue(0.0 < compute_iou(x1, x2) < 1.0)

    def test_tiled_prediction(self):
        from micro_sam.util import precompute_image_embeddings, get_sam_model

        predictor = get_sam_model(model_type="vit_b")

        tile_shape, halo = (256, 256), (16, 16)
        input_ = np.random.rand(512, 512).astype("float32")
        save_path = os.path.join(self.tmp_folder, "emebd.zarr")
        precompute_image_embeddings(predictor, input_, save_path=save_path, tile_shape=tile_shape, halo=halo)

        self.assertTrue(os.path.exists(save_path))
        with zarr.open(save_path, "r") as f:
            self.assertIn("features", f)
            self.assertEqual(len(f["features"]), 4)


if __name__ == "__main__":
    unittest.main()
