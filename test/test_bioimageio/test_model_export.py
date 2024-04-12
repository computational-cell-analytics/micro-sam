import os
import unittest

from shutil import rmtree

import bioimageio.spec
import micro_sam.util as util
from micro_sam.sample_data import synthetic_data

spec_minor = int(bioimageio.spec.__version__.split(".")[1])


@unittest.skipIf(spec_minor < 5, "Needs bioimagio.spec >= 0.5")
class TestModelExport(unittest.TestCase):
    tmp_folder = "tmp"
    model_type = "vit_t" if util.VIT_T_SUPPORT else "vit_b"

    def setUp(self):
        os.makedirs(self.tmp_folder, exist_ok=True)

    def tearDown(self):
        rmtree(self.tmp_folder)

    def test_model_export(self):
        from micro_sam.bioimageio import export_sam_model
        image, labels = synthetic_data(shape=(1024, 1022))

        export_path = os.path.join(self.tmp_folder, "test_export.zip")
        export_sam_model(
            image, labels,
            model_type=self.model_type, name="test-export",
            output_path=export_path,
        )

        self.assertTrue(os.path.exists(export_path))

        # TODO more tests: run prediction with models for different prompt settings


if __name__ == "__main__":
    unittest.main()
