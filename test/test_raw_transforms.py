import unittest
from functools import partial
from unittest.mock import patch

import numpy as np


class TestGaussianPercentileNormalization(unittest.TestCase):
    def test_sampled_percentiles_are_symmetric(self):
        from micro_sam.v2.transforms.raw import GaussianPercentileNormalization

        transform = GaussianPercentileNormalization(mean_lower_percentile=2.0, std_lower_percentile=1.0)
        with patch("micro_sam.v2.transforms.raw.np.random.normal", return_value=3.26) as sample:
            lower, upper = transform.sample_percentiles()

        sample.assert_called_once_with(2.0, 1.0)
        self.assertEqual(lower, 3.3)
        self.assertEqual(upper, 96.7)
        self.assertEqual(lower + upper, 100.0)

    def test_sampled_percentiles_are_clipped_to_valid_range(self):
        from micro_sam.v2.transforms.raw import GaussianPercentileNormalization

        transform = GaussianPercentileNormalization()
        with patch("micro_sam.v2.transforms.raw.np.random.normal", return_value=-10.0):
            self.assertEqual(transform.sample_percentiles(), (0.0, 100.0))

        with patch("micro_sam.v2.transforms.raw.np.random.normal", return_value=70.0):
            lower, upper = transform.sample_percentiles()
        self.assertEqual(lower, 5.0)
        self.assertEqual(upper, 95.0)
        self.assertEqual(lower + upper, 100.0)

    def test_normalizes_with_sampled_percentiles(self):
        from micro_sam.v2.normalization import normalize_raw
        from micro_sam.v2.transforms.raw import GaussianPercentileNormalization

        raw = np.arange(200, dtype="uint16").reshape(2, 10, 10) * 100
        transform = GaussianPercentileNormalization(mean_lower_percentile=4.0, std_lower_percentile=0.0, axis=(1, 2))

        transformed = transform(raw)
        expected = normalize_raw(raw, axis=(1, 2), lower_percentile=4.0, upper_percentile=96.0)
        self.assertTrue(np.allclose(transformed, expected))
        self.assertEqual(transformed.dtype, np.float32)
        self.assertGreaterEqual(transformed.min(), 0.0)
        self.assertLessEqual(transformed.max(), 1.0)

    def test_preprocessing_sees_original_dynamic_range(self):
        from micro_sam.v2.transforms.raw import GaussianPercentileNormalization

        raw = np.arange(16, dtype="uint16").reshape(4, 4) * 4000
        seen = {}

        def preprocessing(x):
            seen["dtype"] = x.dtype
            seen["maximum"] = x.max()
            return np.stack([x] * 3)

        transform = GaussianPercentileNormalization(
            mean_lower_percentile=0.0,
            std_lower_percentile=0.0,
            axis=(1, 2),
            preprocessing=preprocessing,
        )
        transformed = transform(raw)

        self.assertEqual(seen, {"dtype": np.dtype("uint16"), "maximum": np.uint16(60000)})
        self.assertEqual(transformed.shape, (3, 4, 4))
        self.assertTrue(np.isfinite(transformed).all())

    def test_factory_preserves_data_specific_preprocessing(self):
        from micro_sam.v2.transforms.raw import (
            GaussianPercentileNormalization,
            _normalize_percentile,
            _to_8bit,
            get_gaussian_percentile_normalization,
        )

        raw = np.arange(16, dtype="uint16").reshape(1, 4, 4)
        transform = get_gaussian_percentile_normalization(
            _to_8bit, mean_lower_percentile=0.0, std_lower_percentile=0.0
        )
        self.assertIsInstance(transform, GaussianPercentileNormalization)
        self.assertEqual(transform(raw).shape, (3, 4, 4))

        transform = get_gaussian_percentile_normalization(
            partial(_normalize_percentile, axis=(1, 2)),
            mean_lower_percentile=0.0,
            std_lower_percentile=0.0,
        )
        self.assertEqual(transform.axis, (1, 2))

    def test_invalid_distribution_parameters(self):
        from micro_sam.v2.transforms.raw import GaussianPercentileNormalization

        for mean, std in [(-1.0, 1.0), (5.1, 1.0), (np.inf, 1.0), (2.0, -1.0), (2.0, np.nan)]:
            with self.subTest(mean=mean, std=std), self.assertRaises(ValueError):
                GaussianPercentileNormalization(mean_lower_percentile=mean, std_lower_percentile=std)


class TestGeneralistNormalizationConfiguration(unittest.TestCase):
    def test_training_is_random_and_validation_matches_minmax(self):
        from micro_sam.v2.datasets.generalist_loader import (
            TRAIN_LOWER_PERCENTILE_MEAN,
            TRAIN_LOWER_PERCENTILE_STD,
            _configure_training_normalization,
        )
        from micro_sam.v2.datasets.wrapper import UniDataWrapper
        from micro_sam.v2.transforms.raw import GaussianPercentileNormalization, _identity

        class Dataset:
            def __init__(self):
                self.raw_transform = _identity

        train_leaf, val_leaf = Dataset(), Dataset()
        train_datasets = [UniDataWrapper(train_leaf)]
        val_datasets = [UniDataWrapper(val_leaf)]
        _configure_training_normalization(train_datasets, val_datasets)

        self.assertIsInstance(train_leaf.raw_transform, GaussianPercentileNormalization)
        self.assertEqual(train_leaf.raw_transform.mean_lower_percentile, TRAIN_LOWER_PERCENTILE_MEAN)
        self.assertEqual(train_leaf.raw_transform.std_lower_percentile, TRAIN_LOWER_PERCENTILE_STD)

        self.assertIsInstance(val_leaf.raw_transform, GaussianPercentileNormalization)
        self.assertEqual(val_leaf.raw_transform.mean_lower_percentile, 0.0)
        self.assertEqual(val_leaf.raw_transform.std_lower_percentile, 0.0)


if __name__ == "__main__":
    unittest.main()
