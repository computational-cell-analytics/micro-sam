import os
import platform
import tempfile
import unittest
from shutil import rmtree
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

import bioimageio.spec

import torch

import micro_sam.util as util
from micro_sam.sample_data import synthetic_data

spec_minor = int(bioimageio.spec.__version__.split(".")[1])


@unittest.skipIf(
    spec_minor < 5 or platform.system() == "Windows",
    "Needs bioimagio.spec >= 0.5 and is not working on windows"
)
class TestModelExport(unittest.TestCase):
    tmp_folder = "tmp"
    model_type = "vit_t" if util.VIT_T_SUPPORT else "vit_b"

    def setUp(self):
        os.makedirs(self.tmp_folder, exist_ok=True)

    def tearDown(self):
        rmtree(self.tmp_folder, ignore_errors=True)

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

    def test_model_export_with_decoder(self):
        from micro_sam.bioimageio import export_sam_model
        image, labels = synthetic_data(shape=(1024, 1022))

        # Export a generalist model, which has an instance segmentation decoder,
        # so that the exported model supports automatic instance segmentation.
        model_type = f"{self.model_type}_lm"
        export_path = os.path.join(self.tmp_folder, "test_export_ais.zip")
        export_sam_model(image, labels, model_type=model_type, name="test-export-ais", output_path=export_path)

        self.assertTrue(os.path.exists(export_path))


class TestModelExportRegressions(unittest.TestCase):
    def test_prompt_free_prediction_without_decoder(self):
        from micro_sam.bioimageio.predictor_adaptor import PredictorAdaptor

        image = torch.zeros((1, 3, 8, 8), dtype=torch.uint8)
        embeddings = torch.zeros((1, 256, 64, 64), dtype=torch.float32)

        sam = MagicMock()
        sam.is_image_set = True
        sam.orig_h = sam.orig_w = 8
        sam.input_h = sam.input_w = 8
        sam.transform.apply_image_torch.return_value = image.float()
        sam.predict_torch.return_value = (
            torch.zeros((1, 1, 8, 8), dtype=torch.bool),
            torch.zeros((1, 1), dtype=torch.float32),
            torch.empty(0),
        )
        sam.get_image_embedding.return_value = embeddings

        adaptor = PredictorAdaptor.__new__(PredictorAdaptor)
        torch.nn.Module.__init__(adaptor)
        adaptor.sam = sam
        adaptor.decoder = None
        # Cache identity includes the transformed input and original size.
        adaptor._cached_input = image.float()
        adaptor._cached_original_size = (8, 8)
        adaptor._automatic_instance_segmentation = MagicMock(
            side_effect=AssertionError("AIS must not be called without a decoder")
        )

        masks, scores, output_embeddings = adaptor(image)

        adaptor._automatic_instance_segmentation.assert_not_called()
        sam.predict_torch.assert_called_once()
        self.assertEqual(masks.shape, (1, 1, 1, 8, 8))
        self.assertEqual(scores.shape, (1, 1, 1))
        self.assertIs(output_embeddings, embeddings)

    def test_embedding_cache_uses_bounded_transformed_input(self):
        from micro_sam.bioimageio.predictor_adaptor import PredictorAdaptor

        embeddings = torch.zeros((1, 256, 64, 64), dtype=torch.float32)
        sam = MagicMock()
        sam.is_image_set = False
        sam.transform.apply_image_torch.side_effect = lambda image: image[..., :4, :4]

        def set_torch_image(input_, original_image_size):
            sam.is_image_set = True
            sam.original_size = tuple(original_image_size)
            sam.input_size = tuple(input_.shape[2:])

        def predict_torch(**kwargs):
            height, width = sam.original_size
            return (
                torch.zeros((1, 1, height, width), dtype=torch.bool),
                torch.zeros((1, 1), dtype=torch.float32),
                torch.empty(0),
            )

        sam.set_torch_image.side_effect = set_torch_image
        sam.predict_torch.side_effect = predict_torch
        sam.get_image_embedding.return_value = embeddings

        adaptor = PredictorAdaptor.__new__(PredictorAdaptor)
        torch.nn.Module.__init__(adaptor)
        adaptor.sam = sam
        adaptor.decoder = None
        adaptor._cached_input = None
        adaptor._cached_original_size = None

        image = torch.zeros((1, 3, 8, 8), dtype=torch.uint8)
        adaptor(image)
        adaptor(image.clone())

        # The second call reuses the embeddings for the same image.
        self.assertEqual(sam.set_torch_image.call_count, 1)
        self.assertEqual(adaptor._cached_input.shape, (1, 3, 4, 4))
        self.assertEqual(adaptor._cached_original_size, (8, 8))

        # Different transformed content invalidates the embeddings.
        adaptor(torch.ones_like(image))
        self.assertEqual(sam.set_torch_image.call_count, 2)

        # A different original size invalidates identical transformed content.
        adaptor(torch.ones((1, 3, 16, 8), dtype=torch.uint8))
        self.assertEqual(sam.set_torch_image.call_count, 3)
        self.assertEqual(adaptor._cached_input.shape, (1, 3, 4, 4))
        self.assertEqual(adaptor._cached_original_size, (16, 8))

    def test_model_check_accepts_empty_automatic_segmentation(self):
        from micro_sam.bioimageio import model_export

        input_arrays = {
            "image": np.zeros((1, 3, 8, 8), dtype="uint8"),
            "embeddings": np.zeros((1, 256, 64, 64), dtype="float32"),
            "box_prompts": np.zeros((1, 1, 4), dtype="int64"),
            "point_prompts": np.zeros((1, 1, 1, 2), dtype="int64"),
            "point_labels": np.ones((1, 1, 1), dtype="int64"),
            "mask_prompts": np.zeros((1, 1, 1, 256, 256), dtype="float32"),
        }
        reference_mask = np.zeros((1, 1, 1, 8, 8), dtype="uint8")
        empty_mask = np.zeros((1, 0, 1, 8, 8), dtype="uint8")

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_paths = {}
            for name, array in input_arrays.items():
                path = os.path.join(tmp_dir, f"{name}.npy")
                np.save(path, array)
                input_paths[name] = path

            mask_path = os.path.join(tmp_dir, "mask.npy")
            np.save(mask_path, reference_mask)

            regular_prediction = SimpleNamespace(members={"masks": SimpleNamespace(data=reference_mask)})
            empty_prediction = SimpleNamespace(members={"masks": SimpleNamespace(data=empty_mask)})
            pipeline = MagicMock()
            pipeline.predict_sample_without_blocking.side_effect = [*[regular_prediction] * 7, empty_prediction]

            with (
                patch.object(model_export, "create_sample_for_model", return_value=object()),
                patch.object(model_export.bioimageio.core, "create_prediction_pipeline") as create_pipeline,
            ):
                create_pipeline.return_value.__enter__.return_value = pipeline
                model_export._check_model(
                    model_description=object(),
                    input_paths=input_paths,
                    result_paths={"embeddings": input_paths["embeddings"], "mask": mask_path},
                )

            self.assertEqual(pipeline.predict_sample_without_blocking.call_count, 8)
            self.assertEqual(create_pipeline.call_count, 2)


if __name__ == "__main__":
    unittest.main()
