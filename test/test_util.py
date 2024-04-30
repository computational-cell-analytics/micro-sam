import os
import unittest
from shutil import rmtree

import numpy as np
import requests
import torch
import zarr

from skimage.data import binary_blobs
from skimage.measure import label
from micro_sam.util import VIT_T_SUPPORT, SamPredictor, get_cache_directory, get_sam_model, set_precomputed


class TestUtil(unittest.TestCase):
    model_type = "vit_t" if VIT_T_SUPPORT else "vit_b"
    tmp_folder = "tmp-files"

    def setUp(self):
        os.makedirs(self.tmp_folder, exist_ok=True)

    def tearDown(self):
        rmtree(self.tmp_folder)

    # Check that the URLs for all models are valid.
    def test_model_registry(self):
        from micro_sam.util import models

        def check_url(url):
            try:
                # Make a HEAD request to the URL, which fetches HTTP headers but no content.
                response = requests.head(url, allow_redirects=True)
                # Check if the HTTP status code is one that indicates availability (200 <= code < 400).
                return response.status_code < 400
            except requests.RequestException:
                # Handle connection exceptions
                return False

        registry = models()
        for name in registry.registry.keys():
            url_exists = check_url(registry.get_url(name))
            self.assertTrue(url_exists)

    def test_get_sam_model(self):
        from micro_sam.util import get_sam_model

        def check_predictor(predictor):
            self.assertTrue(isinstance(predictor, SamPredictor))
            self.assertEqual(predictor.model_type, self.model_type)
            self.assertTrue(predictor._hash.startswith("xxh128"))

        # Check predictor with download.
        predictor = get_sam_model(model_type=self.model_type)
        check_predictor(predictor)

        # Check predictor with checkpoint path (using the cached model).
        checkpoint_path = os.path.join(get_cache_directory(), "models", self.model_type)
        predictor = get_sam_model(model_type=self.model_type, checkpoint_path=checkpoint_path)
        check_predictor(predictor)

        # Check predictor for one of our models.
        model_type = self.model_type + "_lm"
        predictor = get_sam_model(model_type=model_type)
        check_predictor(predictor)

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

    def _check_predictor_initialization(self, predictor, embeddings, i=None, tile_id=None):
        # We need to do a full reset of the predictor; the orginal_size and input_size
        # are not being reset.
        predictor.reset_image()
        predictor.input_size = None
        predictor.original_size = None

        set_precomputed(predictor, embeddings, i=i, tile_id=tile_id)
        self.assertTrue(predictor.is_image_set)
        self.assertEqual(predictor.features.shape, (1, 256, 64, 64))
        self.assertTrue(predictor.original_size is not None)
        self.assertTrue(predictor.input_size is not None)

        predictor.reset_image()
        predictor.input_size = None
        predictor.original_size = None

    def test_precompute_image_embeddings(self):
        from micro_sam.util import precompute_image_embeddings

        # Load model and create test data.
        predictor = get_sam_model(model_type=self.model_type)
        input_ = np.random.rand(512, 512).astype("float32")

        # Compute the image embeddings without save path.
        embeddings = precompute_image_embeddings(predictor, input_)
        self._check_predictor_initialization(predictor, embeddings)

        # Compute the image embeddings with save path.
        save_path = os.path.join(self.tmp_folder, "emebd.zarr")
        embeddings = precompute_image_embeddings(predictor, input_, save_path=save_path)
        self._check_predictor_initialization(predictor, embeddings)

        # Check the contents of the saved embeddings.
        self.assertTrue(os.path.exists(save_path))
        with zarr.open(save_path, "r") as f:
            self.assertIn("features", f)
            self.assertEqual(f["features"].shape, (1, 256, 64, 64))

        # Check that everything still works when we load the image embeddings from file.
        embeddings = precompute_image_embeddings(predictor, input_, save_path=save_path)
        self._check_predictor_initialization(predictor, embeddings)

    def test_precompute_image_embeddings_3d(self):
        from micro_sam.util import precompute_image_embeddings

        # Load model and create test data.
        predictor = get_sam_model(model_type=self.model_type)
        input_ = np.random.rand(3, 512, 512).astype("float32")

        # Compute the image embeddings without save path.
        embeddings = precompute_image_embeddings(predictor, input_, ndim=3)
        for i in range(input_.shape[0]):
            self._check_predictor_initialization(predictor, embeddings, i=i)

        # Compute the image embeddings with save path.
        save_path = os.path.join(self.tmp_folder, "emebd.zarr")
        embeddings = precompute_image_embeddings(predictor, input_, save_path=save_path, ndim=3)
        for i in range(input_.shape[0]):
            self._check_predictor_initialization(predictor, embeddings, i=i)

        # Check the contents of the saved embeddings.
        self.assertTrue(os.path.exists(save_path))
        with zarr.open(save_path, "r") as f:
            self.assertIn("features", f)
            self.assertEqual(f["features"].shape, (3, 1, 256, 64, 64))

        # Check that everything still works when we load the image embeddings from file.
        embeddings = precompute_image_embeddings(predictor, input_, save_path=save_path, ndim=3)
        for i in range(input_.shape[0]):
            self._check_predictor_initialization(predictor, embeddings, i=i)

    def test_precompute_image_embeddings_tiled(self):
        from micro_sam.util import precompute_image_embeddings

        # Load model and create test data.
        predictor = get_sam_model(model_type=self.model_type)
        tile_shape, halo = (256, 256), (16, 16)
        input_ = np.random.rand(512, 512).astype("float32")

        # Compute the image embeddings without save path.
        embeddings = precompute_image_embeddings(predictor, input_, tile_shape=tile_shape, halo=halo)
        for tile_id in range(4):
            self._check_predictor_initialization(predictor, embeddings, tile_id=tile_id)

        # Compute the image embeddings with save path.
        save_path = os.path.join(self.tmp_folder, "emebd.zarr")
        precompute_image_embeddings(predictor, input_, save_path=save_path, tile_shape=tile_shape, halo=halo)
        for tile_id in range(4):
            self._check_predictor_initialization(predictor, embeddings, tile_id=tile_id)

        # Check the contents of the saved embeddings.
        self.assertTrue(os.path.exists(save_path))
        with zarr.open(save_path, "r") as f:
            self.assertIn("features", f)
            self.assertEqual(len(f["features"]), 4)

        # Check that everything still works when we load the image embeddings from file.
        precompute_image_embeddings(predictor, input_, save_path=save_path, tile_shape=tile_shape, halo=halo)
        for tile_id in range(4):
            self._check_predictor_initialization(predictor, embeddings, tile_id=tile_id)

    def test_precompute_image_embeddings_tiled_3d(self):
        from micro_sam.util import precompute_image_embeddings

        # Load model and create test data.
        predictor = get_sam_model(model_type=self.model_type)
        tile_shape, halo = (256, 256), (16, 16)
        input_ = np.random.rand(2, 512, 512).astype("float32")

        # Compute the image embeddings without save path.
        embeddings = precompute_image_embeddings(predictor, input_, tile_shape=tile_shape, halo=halo)
        for i in range(2):
            for tile_id in range(4):
                self._check_predictor_initialization(predictor, embeddings, i=i, tile_id=tile_id)

        # Compute the image embeddings with save path.
        save_path = os.path.join(self.tmp_folder, "emebd.zarr")
        embeddings = precompute_image_embeddings(
            predictor, input_, save_path=save_path, tile_shape=tile_shape, halo=halo
        )
        for i in range(2):
            for tile_id in range(4):
                self._check_predictor_initialization(predictor, embeddings, i=i, tile_id=tile_id)

        # Check the contents of the saved embeddings.
        self.assertTrue(os.path.exists(save_path))
        with zarr.open(save_path, "r") as f:
            self.assertIn("features", f)
            self.assertEqual(len(f["features"]), 4)

        # Check that everything still works when we load the image embeddings from file.
        embeddings = precompute_image_embeddings(
            predictor, input_, save_path=save_path, tile_shape=tile_shape, halo=halo
        )
        for i in range(2):
            for tile_id in range(4):
                self._check_predictor_initialization(predictor, embeddings, i=i, tile_id=tile_id)

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
