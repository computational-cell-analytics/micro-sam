import unittest

import numpy as np
from skimage.data import coins
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops_table

from micro_sam.classical_features import (
    OBJECT_FEATURES, PIXEL_FEATURES, compute_classical_object_features, compute_classical_pixel_features,
)


class TestClassicalFeatures(unittest.TestCase):
    def setUp(self):
        self.image = coins()
        self.segmentation = label(self.image > threshold_otsu(self.image))

    def test_pixel_features_2d(self):
        grid_shape = (75, 96)
        features = compute_classical_pixel_features(self.image, grid_shape)
        # 6 filters per sigma, two of them with 2 eigenvalues, at 4 sigmas.
        self.assertEqual(features.shape, (75 * 96, 4 * 8))
        self.assertTrue(np.isfinite(features).all())

        # The features are independent of the order of the selection.
        subset = compute_classical_pixel_features(self.image, grid_shape, ["gaussian_smoothing"])
        reordered = compute_classical_pixel_features(
            self.image, grid_shape, ["hessian_of_gaussian_eigenvalues", "gaussian_smoothing"]
        )
        self.assertEqual(subset.shape, (75 * 96, 4))
        np.testing.assert_allclose(reordered[:, 0], subset[:, 0])

    def test_pixel_features_channels_and_3d(self):
        rgb = np.stack(3 * [self.image], axis=-1)
        features = compute_classical_pixel_features(rgb, (75, 96), ["gaussian_gradient_magnitude"])
        self.assertEqual(features.shape, (75 * 96, 3 * 4))

        volume = np.stack(2 * [self.image])
        features = compute_classical_pixel_features(volume, (2, 75, 96), ["difference_of_gaussians"])
        self.assertEqual(features.shape, (2 * 75 * 96, 4))

        with self.assertRaises(ValueError):
            compute_classical_pixel_features(volume, (3, 75, 96), ["difference_of_gaussians"])

    def test_invalid_features(self):
        with self.assertRaises(ValueError):
            compute_classical_pixel_features(self.image, (75, 96), ["size"])
        with self.assertRaises(ValueError):
            compute_classical_object_features(self.image, self.segmentation, features=[])

    def test_object_features_match_skimage(self):
        seg_ids = np.unique(self.segmentation)[1:][::-1]
        features = compute_classical_object_features(self.image, self.segmentation, seg_ids)
        self.assertEqual(features.shape, (len(seg_ids), len(OBJECT_FEATURES)))

        normalized = self.image.astype("float32")
        normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min())
        expected = regionprops_table(
            self.segmentation, normalized, properties=(
                "label", "area", "intensity_mean", "intensity_std", "intensity_min", "intensity_max", "extent",
                "equivalent_diameter_area", "axis_major_length", "axis_minor_length", "eccentricity", "solidity",
            )
        )
        order = [list(expected["label"]).index(seg_id) for seg_id in seg_ids]
        for i, name in enumerate(list(expected)[1:]):
            np.testing.assert_allclose(features[:, i], np.asarray(expected[name])[order], rtol=1e-4, atol=1e-5)

    def test_object_features_3d_and_missing_ids(self):
        volume = np.stack(3 * [self.image])
        segmentation = np.stack(3 * [self.segmentation])
        seg_ids = np.array([1, 2, 10_000])
        features = compute_classical_object_features(volume, segmentation, seg_ids, ["size", "solidity"])
        self.assertEqual(features.shape, (3, 2))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(features[0, 0], 3 * (self.segmentation == 1).sum())
        np.testing.assert_array_equal(features[2], 0)

    def test_supported_features(self):
        self.assertEqual(len(set(PIXEL_FEATURES)), len(PIXEL_FEATURES))
        self.assertEqual(len(set(OBJECT_FEATURES)), len(OBJECT_FEATURES))


if __name__ == "__main__":
    unittest.main()
