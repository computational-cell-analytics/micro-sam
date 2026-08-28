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
                np.testing.assert_allclose(actual, expected, rtol=0, atol=0)

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


if __name__ == "__main__":
    unittest.main()
