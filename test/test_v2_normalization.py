import unittest

import numpy as np

from micro_sam.v2.normalization import compute_percentile_bounds


class TestNormalizePercentile(unittest.TestCase):
    """Pin the inlined percentile normalization to the torch_em implementation it replaced.

    `compute_percentile_bounds` exists only to keep the torch_em training stack out of the inference
    import path. If the two ever diverge, cached embeddings computed by different versions would
    silently disagree, so the equivalence is asserted rather than assumed.
    """

    def _reference(self, raw, lower, upper, axis=None, eps=1e-7):
        from torch_em.transform.raw import normalize_percentile

        return normalize_percentile(raw.copy(), lower=lower, upper=upper, axis=axis, eps=eps)

    def _actual(self, raw, lower, upper, axis=None, eps=1e-7):
        v_lower, v_upper = compute_percentile_bounds(raw, lower, upper, axis=axis)
        return (raw - v_lower) / (v_upper - v_lower + eps)

    def test_matches_torch_em_for_various_shapes(self):
        rng = np.random.default_rng(0)
        cases = [
            (rng.random((32, 32)).astype("float32"), None),
            (rng.random((3, 32, 32)).astype("float32"), None),
            (rng.random((3, 32, 32)).astype("float32"), (1, 2)),
            (rng.random((5, 16, 16)).astype("float32"), 0),
            ((rng.random((32, 32)) * 1000).astype("float32"), None),
        ]
        for raw, axis in cases:
            with self.subTest(shape=raw.shape, axis=axis):
                expected = self._reference(raw, 2.0, 98.0, axis=axis)
                actual = self._actual(raw, 2.0, 98.0, axis=axis)
                # atol tolerates float32-level rounding noise between the two independent
                # implementations (observed up to ~1e-7), not a real algorithmic divergence.
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)

    def test_matches_torch_em_for_constant_input(self):
        # A constant image makes the percentile span zero, so this exercises the eps guard.
        raw = np.full((16, 16), 7.0, dtype="float32")
        expected = self._reference(raw, 2.0, 98.0)
        actual = self._actual(raw, 2.0, 98.0)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

    def test_does_not_import_torch_em(self):
        # The whole point of inlining: normalizing must not drag in the training stack.
        import subprocess
        import sys

        code = (
            "import sys; import numpy as np;"
            "from micro_sam.v2.normalization import normalize_raw;"
            "normalize_raw(np.random.rand(16, 16).astype('float32'));"
            "print('torch_em' in sys.modules)"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertEqual(result.stdout.strip().splitlines()[-1], "False")


class TestImageNormalization(unittest.TestCase):
    def test_image_preprocessing_matches_training(self):
        from micro_sam.v2.normalization import to_image
        from torch_em.transform.raw import normalize_percentile

        rng = np.random.default_rng(188)
        for n_channels in (1, 3):
            with self.subTest(n_channels=n_channels):
                raw = rng.normal(150, 20, (32, 48, n_channels)).astype("float32")
                # Rare dark/bright pixels should not set the scale of the whole tissue image.
                raw[0, 0] = 0
                raw[-1, -1] = 1000
                if n_channels == 1:
                    raw = raw[..., 0]
                # Training normalizes each channel at the 2nd and 98th percentiles, then clips.
                rgb = np.repeat(raw[..., None], 3, axis=-1) if n_channels == 1 else raw
                expected = np.clip(normalize_percentile(rgb.copy(), lower=2, upper=98, axis=(0, 1)), 0, 1)
                actual = to_image(raw)
                self.assertEqual(actual.dtype, np.uint8)
                np.testing.assert_allclose(actual.astype("float32") / 255, expected, atol=1 / 255, rtol=0)

    def test_image_preprocessing_preserves_constant_and_missing_channels(self):
        from micro_sam.v2.normalization import to_image

        image = np.stack([np.arange(64).reshape(8, 8), np.full((8, 8), 17)], axis=-1)
        before = image.copy()
        actual = to_image(image)
        self.assertEqual(actual.shape, (8, 8, 3))
        self.assertFalse(actual[..., 1:].any())
        self.assertEqual(actual[..., 0].min(), 0)
        self.assertEqual(actual[..., 0].max(), 255)
        np.testing.assert_array_equal(image, before)


if __name__ == "__main__":
    unittest.main()
