import os
import unittest

import numpy as np
from elf.io import open_file
from elf.evaluation import symmetric_best_dice_score

import micro_sam.util as util
from micro_sam.sample_data import fetch_hela_2d_example_data

COMMIT_PATH_2D = "commit-for-test.zarr"


@unittest.skipUnless(util.VIT_T_SUPPORT, "Needs vit_t")
class TestReproducibility(unittest.TestCase):

    @unittest.skipUnless(os.path.exists(COMMIT_PATH_2D), "Needs commit test file")
    def test_rerun_segmentation_2d(self):
        from micro_sam.sam_annotator.reproducibility import rerun_segmentation_from_commit_file

        base_data_directory = os.path.join(util.get_cache_directory(), "sample_data")
        input_path = fetch_hela_2d_example_data(base_data_directory)
        segmentation = rerun_segmentation_from_commit_file(COMMIT_PATH_2D, input_path)

        with open_file(COMMIT_PATH_2D, "r") as f:
            expected_segmentation = f["committed_objects"][:]

        self.assertEqual(segmentation.shape, expected_segmentation.shape)
        score = symmetric_best_dice_score(segmentation, expected_segmentation)
        self.assertTrue(np.isclose(score, 1))


if __name__ == "__main__":
    unittest.main()
