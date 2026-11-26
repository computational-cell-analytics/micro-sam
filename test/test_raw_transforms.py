import unittest
from functools import partial

import numpy as np


class TestRandomPercentileNormalization(unittest.TestCase):
    def test_factory_preserves_data_specific_preprocessing(self):
        from torch_em.transform.raw import RandomPercentileNormalization, RawTransform

        from micro_sam.v2.transforms.raw import _normalize_percentile, _to_8bit, get_random_percentile_normalization

        raw = np.arange(16, dtype="uint16").reshape(1, 4, 4)
        transform = get_random_percentile_normalization(_to_8bit)
        self.assertIsInstance(transform, RawTransform)
        self.assertIsInstance(transform.normalizer, RandomPercentileNormalization)
        self.assertEqual(transform.normalizer.lower_percentile_bounds, (0.0, 5.0))
        self.assertEqual(transform.normalizer.distribution, "uniform")
        self.assertIsNone(transform.normalizer.distribution_kwargs)
        self.assertEqual(transform.normalizer.rounding_decimals, 1)
        self.assertEqual(transform(raw).shape, (3, 4, 4))

        transform = get_random_percentile_normalization(
            partial(_normalize_percentile, axis=(1, 2)), lower_percentile_bounds=(0.0, 0.0),
        )
        self.assertEqual(transform.normalizer.axis, (1, 2))

        def postprocessing(x):
            return x

        transform.augmentation2 = postprocessing
        reconfigured = get_random_percentile_normalization(transform, lower_percentile_bounds=(0.0, 1.0))
        self.assertIs(reconfigured.augmentation1, transform.augmentation1)
        self.assertIs(reconfigured.augmentation2, postprocessing)


class TestGeneralistNormalizationConfiguration(unittest.TestCase):
    def test_training_is_random_and_validation_is_deterministic(self):
        from torch_em.transform.raw import RandomPercentileNormalization, RawTransform

        from micro_sam.v2.datasets.generalist_loader import (
            TRAIN_LOWER_PERCENTILE_BOUNDS, VALIDATION_LOWER_PERCENTILE_BOUNDS,
            _configure_training_normalization,
        )
        from micro_sam.v2.datasets.wrapper import UniDataWrapper
        from micro_sam.v2.transforms.raw import _identity

        class Dataset:
            def __init__(self):
                self.raw_transform = _identity

        train_leaf, val_leaf = Dataset(), Dataset()
        train_datasets = [UniDataWrapper(train_leaf)]
        val_datasets = [UniDataWrapper(val_leaf)]
        _configure_training_normalization(train_datasets, val_datasets)

        self.assertIsInstance(train_leaf.raw_transform, RawTransform)
        self.assertIsInstance(train_leaf.raw_transform.normalizer, RandomPercentileNormalization)
        self.assertEqual(train_leaf.raw_transform.normalizer.distribution, "uniform")
        self.assertEqual(train_leaf.raw_transform.normalizer.lower_percentile_bounds, TRAIN_LOWER_PERCENTILE_BOUNDS)

        self.assertIsInstance(val_leaf.raw_transform, RawTransform)
        self.assertIsInstance(val_leaf.raw_transform.normalizer, RandomPercentileNormalization)
        self.assertEqual(val_leaf.raw_transform.normalizer.distribution, "uniform")
        self.assertEqual(
            val_leaf.raw_transform.normalizer.lower_percentile_bounds, VALIDATION_LOWER_PERCENTILE_BOUNDS
        )
        self.assertEqual(val_leaf.raw_transform.normalizer.sample_percentiles(), (2.0, 98.0))


if __name__ == "__main__":
    unittest.main()
