import os
import unittest
from shutil import rmtree

import numpy as np
import requests
import torch
import zarr
import z5py

from skimage.data import binary_blobs
from skimage.measure import label
from micro_sam.util import VIT_T_SUPPORT, SamPredictor, get_cache_directory, _open_embeddings
from micro_sam.v1.util import get_sam_model, set_precomputed

ZARR_MAJOR = int(zarr.__version__.split(".")[0])


class TestUtil(unittest.TestCase):
    model_type = "vit_t" if VIT_T_SUPPORT else "vit_b"
    tmp_folder = "tmp-files"

    def setUp(self):
        os.makedirs(self.tmp_folder, exist_ok=True)

    def tearDown(self):
        rmtree(self.tmp_folder)

    # Check that the URLs for all models are valid.
    def test_model_registry(self):
        from micro_sam.v1.util import models

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
        from micro_sam.v1.util import get_sam_model

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

    def test_normalize_raw(self):
        from micro_sam.v2.normalization import normalize_raw

        raw = np.arange(10_000, dtype="float32").reshape(100, 100)
        raw[-1, -1] = 10_000  # An outlier must not determine the useful intensity range.

        normalized = normalize_raw(raw)
        self.assertEqual(normalized.dtype, np.float32)
        self.assertGreaterEqual(normalized.min(), 0.0)
        self.assertLessEqual(normalized.max(), 1.0)
        self.assertEqual(normalized[-1, -1], 1.0)
        self.assertGreater(normalized[50, 0], 0.45)

        # An integer output dtype rescales the same normalized data to the full dtype range.
        normalized_uint8 = normalize_raw(raw, output_dtype="uint8")
        self.assertEqual(normalized_uint8.dtype, np.uint8)
        self.assertTrue(np.array_equal(normalized_uint8, np.round(normalized * 255).astype("uint8")))

        empty = normalize_raw(np.empty((0, 4), dtype="uint16"))
        self.assertEqual(empty.shape, (0, 4))
        self.assertEqual(empty.dtype, np.float32)

    def test_normalize_raw_dtypes(self):
        from micro_sam.v2.normalization import normalize_raw

        raw = np.arange(10_000, dtype="float32").reshape(100, 100)

        # Floating and 8-/16-bit integer output dtypes are all supported.
        for dtype in ["float16", "float32", "float64", "uint8", "int8", "uint16", "int16"]:
            normalized = normalize_raw(raw, output_dtype=dtype)
            self.assertEqual(normalized.dtype, np.dtype(dtype))

        # 32-/64-bit integer, boolean and complex output dtypes are rejected.
        for dtype in ["int32", "uint32", "int64", "uint64", "bool", "complex64", "complex128"]:
            with self.assertRaises(ValueError):
                normalize_raw(raw, output_dtype=dtype)

    def test_normalize_raw_output_ranges(self):
        from micro_sam.v2.normalization import normalize_raw

        raw = np.arange(10_000, dtype="float32").reshape(100, 100)

        # Integer output dtypes are normalized to their full representable range.
        full_ranges = {"uint8": (0, 255), "int8": (-128, 127), "uint16": (0, 65535), "int16": (-32768, 32767)}
        for dtype, (low, high) in full_ranges.items():
            normalized = normalize_raw(raw, output_dtype=dtype)
            self.assertEqual(normalized.min(), low)
            self.assertEqual(normalized.max(), high)

        # Floating output dtypes are normalized to [0, 1].
        for dtype in ["float16", "float32", "float64"]:
            normalized = normalize_raw(raw, output_dtype=dtype)
            self.assertAlmostEqual(float(normalized.min()), 0.0, places=3)
            self.assertAlmostEqual(float(normalized.max()), 1.0, places=3)

    def test_normalize_raw_input_ranges(self):
        from micro_sam.v2.normalization import normalize_raw

        # Common microscopy input dtypes over sensible ranges normalize to [0, 1].
        ranges = {
            "uint8": (0, 255),
            "int8": (-128, 127),
            "uint16": (0, 65535),
            "int16": (-32768, 32767),
            "float16": (0.0, 1.0),
            "float32": (0.0, 1.0),
            "float64": (-5.0, 5.0),
        }
        for dtype, (low, high) in ranges.items():
            raw = np.linspace(low, high, 10_000, dtype=dtype).reshape(100, 100)
            normalized = normalize_raw(raw)
            self.assertEqual(normalized.dtype, np.float32)
            self.assertGreaterEqual(normalized.min(), 0.0)
            self.assertLessEqual(normalized.max(), 1.0)
            # A monotonic input ramp stays monotonic after normalization.
            self.assertTrue(np.all(np.diff(normalized.ravel()) >= 0))

    def test_normalize_raw_percentile_params(self):
        from micro_sam.v2.normalization import normalize_raw

        raw = np.arange(10_000, dtype="float32").reshape(100, 100)

        # Explicit defaults match the hard-coded 2nd/98th percentiles.
        default = normalize_raw(raw)
        explicit = normalize_raw(raw, lower_percentile=2.0, upper_percentile=98.0)
        self.assertTrue(np.array_equal(default, explicit))

        # Wider percentiles clip less, so intermediate values differ.
        wide = normalize_raw(raw, lower_percentile=0.0, upper_percentile=100.0)
        self.assertFalse(np.array_equal(default, wide))
        self.assertGreaterEqual(wide.min(), 0.0)
        self.assertLessEqual(wide.max(), 1.0)

    def test_normalize_raw_per_channel(self):
        from micro_sam.v2.normalization import normalize_raw, to_image
        from micro_sam.util import _to_image

        channel = np.arange(100, dtype="float32").reshape(10, 10)
        image = np.stack([channel, channel * 100 + 42], axis=-1)
        normalized = normalize_raw(image, axis=(0, 1))
        self.assertTrue(np.allclose(normalized[..., 0], normalized[..., 1]))

        rgb = to_image(image)
        self.assertEqual(rgb.shape, (10, 10, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertTrue(np.array_equal(rgb[..., 0], rgb[..., 1]))
        self.assertFalse(np.any(rgb[..., 2]))
        self.assertTrue(np.array_equal(rgb, _to_image(image)))

    def test_normalization_invalidates_incompatible_embeddings(self):
        from types import SimpleNamespace

        from micro_sam.util import _get_embedding_signature
        from micro_sam.v2.normalization import IMAGE_PREPROCESSING, VIDEO_PREPROCESSING
        from micro_sam.v2.util import _check_saved_embeddings

        self.assertEqual(IMAGE_PREPROCESSING, "minmax_per_channel")
        self.assertEqual(VIDEO_PREPROCESSING, "percentile_2_98_per_channel_torch_resize_v1")

        predictor = SimpleNamespace(model_type="hvit_t", model_name="hvit_t", _hash="test")
        raw = np.arange(100).reshape(10, 10)
        signature = _get_embedding_signature(raw, predictor, tile_shape=None, halo=None)

        def run(embeddings, preprocessing):
            return _check_saved_embeddings(raw, predictor, embeddings, "cache.zarr", None, None, preprocessing)

        def full_cache(normalization):
            attrs = {"input_size": [10, 10], **signature}
            if normalization is not None:
                attrs["normalization"] = normalization
            return SimpleNamespace(attrs=attrs)

        # A complete cache is reused only under the policy it was written with.
        self.assertFalse(run(full_cache(IMAGE_PREPROCESSING), IMAGE_PREPROCESSING))
        self.assertFalse(run(full_cache(VIDEO_PREPROCESSING), VIDEO_PREPROCESSING))
        # A 2d min-max cache is not reused for the 3d percentile policy and vice versa.
        self.assertTrue(run(full_cache(IMAGE_PREPROCESSING), VIDEO_PREPROCESSING))
        self.assertTrue(run(full_cache(VIDEO_PREPROCESSING), IMAGE_PREPROCESSING))
        # A missing tag is stale.
        self.assertTrue(run(full_cache(None), IMAGE_PREPROCESSING))

        class PartialEmbeddings(dict):
            def __init__(self, normalization=None):
                super().__init__(features=object())
                self.attrs = {} if normalization is None else {"normalization": normalization}

        # Partial caches (no 'input_size') resume only when the tag matches the requested policy.
        self.assertTrue(run(PartialEmbeddings(), IMAGE_PREPROCESSING))
        self.assertTrue(run(PartialEmbeddings(VIDEO_PREPROCESSING), IMAGE_PREPROCESSING))
        self.assertFalse(run(PartialEmbeddings(IMAGE_PREPROCESSING), IMAGE_PREPROCESSING))

        # An empty cache (no features) is never stale.
        empty_cache = PartialEmbeddings(IMAGE_PREPROCESSING)
        del empty_cache["features"]
        self.assertFalse(run(empty_cache, IMAGE_PREPROCESSING))

    def test_apply_nms_tiled_border_masks(self):
        from micro_sam.util import apply_nms

        predictions = [
            {
                "segmentation": torch.ones((4, 4), dtype=torch.bool),
                "bbox": [0, 0, 4, 4],
                "global_bbox": [0, 0, 4, 4],
                "predicted_iou": 1.0,
                "stability_score": 1.0,
            },
            {
                "segmentation": torch.ones((4, 2), dtype=torch.bool),
                "bbox": [0, 0, 2, 4],
                "global_bbox": [5, 0, 2, 4],
                "predicted_iou": 1.0,
                "stability_score": 1.0,
            },
        ]

        segmentation = apply_nms(predictions, min_size=0)

        self.assertEqual(segmentation.shape, (4, 7))
        self.assertEqual(segmentation.max(), 2)

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
        from micro_sam.v1.util import precompute_image_embeddings

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
        f = _open_embeddings(save_path, mode="r")
        self.assertIn("features", f)
        self.assertEqual(f["features"].shape, (1, 256, 64, 64))

        # Check that everything still works when we load the image embeddings from file.
        embeddings = precompute_image_embeddings(predictor, input_, save_path=save_path)
        self._check_predictor_initialization(predictor, embeddings)

    def test_precompute_image_embeddings_3d(self):
        from micro_sam.v1.util import precompute_image_embeddings

        # Load model and create test data.
        predictor = get_sam_model(model_type=self.model_type)
        input_ = np.random.rand(3, 512, 512).astype("float32")

        # Compute the image embeddings without save path.
        # We run this test with a batch size of 2.
        embeddings = precompute_image_embeddings(predictor, input_, ndim=3, batch_size=2)
        for i in range(input_.shape[0]):
            self._check_predictor_initialization(predictor, embeddings, i=i)

        # Compute the image embeddings with save path.
        save_path = os.path.join(self.tmp_folder, "emebd.zarr")
        embeddings = precompute_image_embeddings(predictor, input_, save_path=save_path, ndim=3)
        for i in range(input_.shape[0]):
            self._check_predictor_initialization(predictor, embeddings, i=i)

        # Check the contents of the saved embeddings.
        self.assertTrue(os.path.exists(save_path))
        f = _open_embeddings(save_path, mode="r")
        self.assertIn("features", f)
        self.assertEqual(f["features"].shape, (3, 1, 256, 64, 64))

        # Check that everything still works when we load the image embeddings from file.
        embeddings = precompute_image_embeddings(predictor, input_, save_path=save_path, ndim=3)
        for i in range(input_.shape[0]):
            self._check_predictor_initialization(predictor, embeddings, i=i)

    def test_precompute_image_embeddings_tiled(self):
        from micro_sam.v1.util import precompute_image_embeddings

        # Load model and create test data.
        predictor = get_sam_model(model_type=self.model_type)
        tile_shape, halo = (256, 256), (16, 16)
        input_ = np.random.rand(512, 512).astype("float32")

        # Compute the image embeddings without save path.
        # We run this test with a batch size of 2.
        embeddings = precompute_image_embeddings(predictor, input_, tile_shape=tile_shape, halo=halo, batch_size=2)
        for tile_id in range(4):
            self._check_predictor_initialization(predictor, embeddings, tile_id=tile_id)

        # Compute the image embeddings with save path.
        save_path = os.path.join(self.tmp_folder, "emebd.zarr")
        precompute_image_embeddings(predictor, input_, save_path=save_path, tile_shape=tile_shape, halo=halo)
        for tile_id in range(4):
            self._check_predictor_initialization(predictor, embeddings, tile_id=tile_id)

        # Check the contents of the saved embeddings.
        self.assertTrue(os.path.exists(save_path))
        f = _open_embeddings(save_path, mode="r")
        self.assertIn("features", f)
        self.assertEqual(len(f["features"]), 4)

        # Check that everything still works when we load the image embeddings from file.
        precompute_image_embeddings(predictor, input_, save_path=save_path, tile_shape=tile_shape, halo=halo)
        for tile_id in range(4):
            self._check_predictor_initialization(predictor, embeddings, tile_id=tile_id)

    def test_precompute_image_embeddings_tiled_3d(self):
        from micro_sam.v1.util import precompute_image_embeddings

        # Load model and create test data.
        predictor = get_sam_model(model_type=self.model_type)
        tile_shape, halo = (256, 256), (16, 16)
        input_ = np.random.rand(2, 512, 512).astype("float32")

        # Compute the image embeddings without save path.
        # We run this test with a batch size of 2.
        embeddings = precompute_image_embeddings(predictor, input_, tile_shape=tile_shape, halo=halo, batch_size=2)
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
        f = _open_embeddings(save_path, mode="r")
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
        from unittest import mock

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

        # Indexed accelerator strings are valid device selections too.
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            self.assertEqual(get_device("cuda:1"), "cuda:1")

    def test_device_type(self):
        from micro_sam.util import device_type

        self.assertEqual(device_type("cpu"), "cpu")
        self.assertEqual(device_type(torch.device("cpu")), "cpu")

        # Indexed accelerators must report the plain type: torch reports a model's parameters as
        # living on 'mps:0' / 'cuda:0', so 'str(device) == "mps"' silently fails.
        self.assertEqual(device_type("mps"), "mps")
        self.assertEqual(device_type(torch.device("mps")), "mps")
        self.assertEqual(device_type(torch.device("mps", 0)), "mps")
        self.assertEqual(device_type(torch.device("cuda", 3)), "cuda")

    def test_configure_mps_memory(self):
        from micro_sam.util import _configure_mps_memory

        key = "PYTORCH_MPS_HIGH_WATERMARK_RATIO"
        original = os.environ.get(key)
        try:
            # non-mps devices must not touch the watermark
            os.environ.pop(key, None)
            _configure_mps_memory("cpu")
            self.assertNotIn(key, os.environ)
            _configure_mps_memory(torch.device("cpu"))
            self.assertNotIn(key, os.environ)

            # mps disables the watermark when it is unset
            _configure_mps_memory("mps")
            self.assertEqual(os.environ.get(key), "0.0")

            # an explicit user-provided value is kept
            os.environ[key] = "1.9"
            _configure_mps_memory("mps")
            self.assertEqual(os.environ[key], "1.9")
        finally:
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original


try:
    import sam2  # noqa
    SAM2_SUPPORT = True
except ImportError:
    SAM2_SUPPORT = False


@unittest.skipUnless(SAM2_SUPPORT, "Requires the sam2 package.")
class TestSAM2Util(unittest.TestCase):
    model_type = "hvit_t"
    tmp_folder = "tmp-files-sam2"

    def setUp(self):
        os.makedirs(self.tmp_folder, exist_ok=True)

    def tearDown(self):
        rmtree(self.tmp_folder)

    def _get_predictor(self, ndim):
        # Build the SAM2 predictor exactly as the precompute CLI / annotator do.
        from micro_sam.sam_annotator._state import _get_sam_model
        predictor, _ = _get_sam_model(
            model_type=self.model_type, ndim=ndim, device="cpu",
            checkpoint_path=None, decoder_path=None, use_cli=True,
        )
        return predictor

    def _check_predictor_initialization_2d(self, predictor, embeddings):
        from micro_sam.v2.util import set_precomputed
        predictor.reset_predictor()
        set_precomputed(predictor, embeddings)
        self.assertTrue(predictor._is_image_set)
        self.assertIsNotNone(predictor._features)
        self.assertIsNotNone(predictor._orig_hw)
        predictor.reset_predictor()

    def test_precompute_image_embeddings_2d(self):
        from micro_sam.v2.normalization import IMAGE_PREPROCESSING
        from micro_sam.v2.util import precompute_image_embeddings

        predictor = self._get_predictor(ndim=2)
        input_ = np.random.rand(512, 512).astype("float32")

        # Compute the image embeddings without save path.
        embeddings = precompute_image_embeddings(predictor, input_, ndim=2)
        for key in ("features", "high_res_feats", "input_size", "original_size"):
            self.assertIn(key, embeddings)
        self.assertEqual(embeddings["features"].ndim, 4)
        self.assertEqual(embeddings["features"].shape, (1, 256, 64, 64))
        self._check_predictor_initialization_2d(predictor, embeddings)

        # Compute the image embeddings with save path.
        save_path = os.path.join(self.tmp_folder, "embed.zarr")
        embeddings = precompute_image_embeddings(predictor, input_, save_path=save_path, ndim=2)
        self._check_predictor_initialization_2d(predictor, embeddings)

        # Check the contents of the saved embeddings.
        self.assertTrue(os.path.exists(save_path))
        f = _open_embeddings(save_path, mode="r")
        self.assertIn("features", f)
        self.assertIn("high_res_feats", f)
        self.assertEqual(f["features"].shape, (1, 256, 64, 64))
        # The signature is written so the GUI / CLI can validate a reload.
        self.assertEqual(f.attrs["model_name"], self.model_type)
        self.assertIn("data_signature", f.attrs)
        self.assertEqual(f.attrs["normalization"], IMAGE_PREPROCESSING)

        # Check that everything still works when we load the image embeddings from file.
        embeddings = precompute_image_embeddings(predictor, input_, save_path=save_path, ndim=2)
        self.assertEqual(embeddings["features"].shape, (1, 256, 64, 64))
        self._check_predictor_initialization_2d(predictor, embeddings)

    def test_precompute_image_embeddings_3d(self):
        from micro_sam.v2.normalization import VIDEO_PREPROCESSING
        from micro_sam.v2.util import precompute_image_embeddings, set_precomputed

        predictor = self._get_predictor(ndim=3)
        input_ = np.random.rand(2, 256, 256).astype("float32")

        def check_slices(embeddings):
            for i in range(input_.shape[0]):
                _, inference_state = set_precomputed(predictor, embeddings, i=i, input_=input_)
                self.assertIn("cached_features", inference_state)

        # Compute the image embeddings without save path.
        # Note: the in-memory form stacks the per-slice features along z (4 dims),
        # while the saved form keeps an explicit singleton dim (5 dims).
        embeddings = precompute_image_embeddings(predictor, input_, ndim=3)
        for key in ("features", "pos_enc", "fpn", "input_size", "original_size"):
            self.assertIn(key, embeddings)
        self.assertEqual(embeddings["features"].shape[0], input_.shape[0])

        # Compute the image embeddings with save path.
        save_path = os.path.join(self.tmp_folder, "embed_3d.zarr")
        embeddings = precompute_image_embeddings(predictor, input_, save_path=save_path, ndim=3)
        check_slices(embeddings)

        # Check the contents of the saved embeddings.
        self.assertTrue(os.path.exists(save_path))
        f = _open_embeddings(save_path, mode="r")
        self.assertIn("features", f)
        self.assertIn("pos_enc", f)
        self.assertIn("fpn", f)
        self.assertEqual(f["features"].shape, (2, 1, 256, 64, 64))
        self.assertEqual(f.attrs["model_name"], self.model_type)
        self.assertEqual(f.attrs["normalization"], VIDEO_PREPROCESSING)

        # Check that everything still works when we load the image embeddings from file.
        embeddings = precompute_image_embeddings(predictor, input_, save_path=save_path, ndim=3)
        check_slices(embeddings)


class TestEmbeddingBackend(unittest.TestCase):
    """Direct tests of the z5py embedding backend, without needing a SAM model."""
    tmp_folder = "tmp-backend"

    def setUp(self):
        os.makedirs(self.tmp_folder, exist_ok=True)

    def tearDown(self):
        rmtree(self.tmp_folder)

    @staticmethod
    def _write_legacy_cache(save_path, data, attrs, zarr_format):
        # Mimic a cache written by the old zarr-python backend. zarr-python v2 has no zarr_format
        # argument and always writes v2, so only pass it on v3.
        if ZARR_MAJOR >= 3:
            group = zarr.open(save_path, mode="w", zarr_format=zarr_format)
            arr = group.create_array("features", shape=data.shape, dtype=data.dtype, chunks=data.shape)
            arr[:] = data
        else:
            group = zarr.open(save_path, mode="w")
            group.create_dataset("features", data=data, shape=data.shape, chunks=data.shape)
        for key, val in attrs.items():
            group.attrs[key] = val

    def test_image_embeddings_owns_temporary_store(self):
        from micro_sam.v2.util import ImageEmbeddings

        class Store:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        temporary_path = os.path.join(self.tmp_folder, "implicit.zarr")
        os.makedirs(temporary_path)
        store = Store()
        resource = ImageEmbeddings({"features": object()}, store=store, temporary_path=temporary_path)

        self.assertTrue(os.path.exists(temporary_path))
        with resource as embeddings:
            self.assertIs(embeddings, resource)
            self.assertFalse(embeddings.closed)

        self.assertTrue(store.closed)
        self.assertTrue(resource.closed)
        self.assertFalse(os.path.exists(temporary_path))
        resource.close()  # Idempotent.

        persistent_path = os.path.join(self.tmp_folder, "persistent.zarr")
        os.makedirs(persistent_path)
        persistent_store = Store()
        with ImageEmbeddings({}, store=persistent_store):
            pass
        self.assertTrue(persistent_store.closed)
        self.assertTrue(os.path.exists(persistent_path))

    def test_open_in_memory(self):
        from micro_sam.util import _open_embeddings
        # save_path=None returns an in-memory zarr group (z5py has no in-memory store).
        f = _open_embeddings(None)
        self.assertIsInstance(f, zarr.Group)

    def test_roundtrip(self):
        from micro_sam.util import _open_embeddings, _create_dataset_with_data, _create_dataset_without_data
        save_path = os.path.join(self.tmp_folder, "embed.zarr")
        data = np.random.rand(1, 256, 64, 64).astype("float32")

        # On-disk embeddings use the z5py backend.
        f = _open_embeddings(save_path, mode="a")
        self.assertIsInstance(f, z5py.Group)
        _create_dataset_with_data(f, "features", data=data)
        _create_dataset_without_data(f, "empty", shape=data.shape, dtype="float32", chunks=data.shape)
        f.attrs["input_size"] = [1024, 1024]
        f.attrs["tile_shape"] = None

        # Reload with the backend and check the contents (including tricky attrs: a list and None).
        g = _open_embeddings(save_path, mode="r")
        self.assertTrue(np.allclose(g["features"][:], data))
        self.assertEqual(list(g.attrs["input_size"]), [1024, 1024])
        self.assertIsNone(g.attrs["tile_shape"])

        # z5py-written caches are zarr v3 with blosc. Verify via zarr-python where it can read v3
        # (zarr-python v2 cannot read the v3 format, so this cross-check only applies on v3).
        if ZARR_MAJOR >= 3:
            z = zarr.open(save_path, mode="r")
            self.assertTrue(np.allclose(z["features"][:], data))
            self.assertEqual(z.metadata.zarr_format, 3)
            self.assertTrue(any("blosc" in repr(codec).lower() for codec in z["features"].metadata.codecs))

    def test_open_metadataless_dir(self):
        # z5py cannot open a directory without zarr metadata. _open_embeddings must handle it
        # gracefully, as the old zarr-python backend did (implicit group creation).
        from micro_sam.util import _open_embeddings
        save_path = os.path.join(self.tmp_folder, "empty.zarr")
        os.makedirs(save_path)
        f = _open_embeddings(save_path, mode="a")
        self.assertNotIn("features", f)

    def test_read_legacy_zarr_python_cache(self):
        # Caches written by the old zarr-python backend must still load via z5py. Only test the
        # formats the installed zarr-python can write (v2 alone on zarr-python v2, v2 and v3 on v3).
        from micro_sam.util import _open_embeddings
        data = np.random.rand(1, 256, 64, 64).astype("float32")
        attrs = {
            "input_size": [1024, 1024], "original_size": [512, 512], "tile_shape": None, "model_name": "vit_t",
        }
        formats = (2, 3) if ZARR_MAJOR >= 3 else (2,)
        for zarr_format in formats:
            save_path = os.path.join(self.tmp_folder, f"legacy_v{zarr_format}.zarr")
            self._write_legacy_cache(save_path, data, attrs, zarr_format)

            f = _open_embeddings(save_path, mode="a")
            self.assertTrue(np.allclose(f["features"][:], data))
            self.assertEqual(f.attrs["model_name"], "vit_t")
            self.assertEqual(list(f.attrs["original_size"]), [512, 512])
            self.assertIsNone(f.attrs["tile_shape"])


if __name__ == "__main__":
    unittest.main()
