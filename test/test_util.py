import os
import unittest
from shutil import rmtree

import numpy as np
import torch
import zarr

from skimage.data import binary_blobs
from skimage.measure import label


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
        from micro_sam.util import precompute_image_embeddings, get_sam_model, VIT_T_SUPPORT

        predictor = get_sam_model(model_type="vit_t" if VIT_T_SUPPORT else "vit_b")

        tile_shape, halo = (256, 256), (16, 16)
        input_ = np.random.rand(512, 512).astype("float32")
        save_path = os.path.join(self.tmp_folder, "emebd.zarr")
        precompute_image_embeddings(predictor, input_, save_path=save_path, tile_shape=tile_shape, halo=halo)

        self.assertTrue(os.path.exists(save_path))
        with zarr.open(save_path, "r") as f:
            self.assertIn("features", f)
            self.assertEqual(len(f["features"]), 4)

    def test_segmentation_to_one_hot(self):
        from micro_sam.util import segmentation_to_one_hot

        labels = label(binary_blobs(256, blob_size_fraction=0.05, volume_fraction=0.15))
        label_ids = np.unique(labels)[1:]

        mask = segmentation_to_one_hot(labels.astype("int64"), label_ids).numpy()

        expected_mask = np.zeros((len(label_ids), 1) + labels.shape, dtype="float32")
        for idx, label_id in enumerate(label_ids):
            expected_mask[idx, 0, labels == label_id] = 1
        self.assertEqual(expected_mask.shape, mask.shape)

        self.assertTrue(np.allclose(mask, expected_mask))

    def test_get_device(self):
        from micro_sam.util import get_device

        # check that device without argument works
        get_device()

        # check passing device as string
        device = get_device("cpu")
        self.assertEqual(device, "cpu")

        # check passing device as torch.device works
        device = get_device(torch.device("cpu"))
        self.assertTrue(isinstance(device, torch.device))
        self.assertEqual(device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
