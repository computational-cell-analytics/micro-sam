import os
import unittest

import micro_sam.util as util
from elf.io import open_file
from micro_sam.sample_data import fetch_hela_2d_example_data


@unittest.skipUnless(util.VIT_T_SUPPORT, "Needs vit_t")
class TestReproducibility(unittest.TestCase):
    commit_path = "commit-for-test.zarr"

    def test_automatic_mask_generator_2d(self):
        from micro_sam.sam_annotator.reproducibility import rerun_segmentation_from_commit_file

        base_data_directory = os.path.join(util.get_cache_directory(), "sample_data")
        input_path = fetch_hela_2d_example_data(base_data_directory)
        segmentation = rerun_segmentation_from_commit_file(self.commit_path, input_path)

        with open_file(self.commit_path, "r") as f:
            expected_segmentation = f["committed_objects"][:]

        breakpoint()

        # TODO check that segmentation and expected_segmentation are equivalent


if __name__ == "__main__":
    unittest.main()
