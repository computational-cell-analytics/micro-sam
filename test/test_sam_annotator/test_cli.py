import unittest
from shutil import which


class TestCLI(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
