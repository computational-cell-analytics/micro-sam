import os
import unittest
from functools import partial
from glob import glob
from shutil import rmtree

import imageio.v3 as imageio

from micro_sam.sample_data import synthetic_data
from micro_sam.util import VIT_T_SUPPORT, get_sam_model, SamPredictor


@unittest.skipUnless(VIT_T_SUPPORT, "Integration test is only run with vit_t support, otherwise it takes too long.")
class TestTraining(unittest.TestCase):
    """Integration test for training a SAM model.
    """
    tmp_folder = "./tmp-training"

    def setUp(self):
        image_root = os.path.join(self.tmp_folder, "synthetic-data", "images")
        label_root = os.path.join(self.tmp_folder, "synthetic-data", "labels")

        shape = (512, 512)
        self.n_images_train = 4
        self.n_images_val = 1
        self.n_images_test = 1

        n_images = self.n_images_train + self.n_images_val + self.n_images_test
        for i in range(n_images):
            if i < self.n_images_train:
                image_dir, label_dir = os.path.join(image_root, "train"), os.path.join(label_root, "train")
                idx = i
            elif i < self.n_images_train + self.n_images_val:
                image_dir, label_dir = os.path.join(image_root, "val"), os.path.join(label_root, "val")
                idx = i - self.n_images_train
            else:
                image_dir, label_dir = os.path.join(image_root, "test"), os.path.join(label_root, "test")
                idx = i - self.n_images_train - self.n_images_val

            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)

            image_path = os.path.join(image_dir, f"data-{idx}.tif")
            label_path = os.path.join(label_dir, f"data-{idx}.tif")

            image, labels = synthetic_data(shape)
            imageio.imwrite(image_path, image)
            imageio.imwrite(label_path, labels)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def _get_dataloader(self, split, patch_shape, batch_size):
        import micro_sam.training as sam_training

        # Create the synthetic training data and get the corresponding folders.
        image_root = os.path.join(self.tmp_folder, "synthetic-data", "images", split)
        label_root = os.path.join(self.tmp_folder, "synthetic-data", "labels", split)
        raw_key, label_key = "*.tif", "*.tif"

        loader = sam_training.default_sam_loader(
            raw_paths=image_root, raw_key=raw_key,
            label_paths=label_root, label_key=label_key,
            patch_shape=patch_shape, batch_size=batch_size,
            with_segmentation_decoder=False,
            shuffle=True, num_workers=1,
            n_samples=self.n_images_train if split == "train" else self.n_images_val
        )
        return loader

    def _train_model(self, model_type, device):
        import micro_sam.training as sam_training

        batch_size = 1
        n_sub_iteration = 3
        patch_shape = (512, 512)
        n_objects_per_batch = 2

        # Get the dataloaders.
        train_loader = self._get_dataloader("train", patch_shape, batch_size)
        val_loader = self._get_dataloader("val", patch_shape, batch_size)

        # Run the training.
        sam_training.train_sam(
            name="test",
            model_type=model_type,
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=1,
            n_objects_per_batch=n_objects_per_batch,
            n_sub_iteration=n_sub_iteration,
            with_segmentation_decoder=False,
            device=device,
            save_root=self.tmp_folder
        )

    def _export_model(self, checkpoint_path, export_path, model_type):
        from micro_sam.util import export_custom_sam_model

        export_custom_sam_model(
            checkpoint_path=checkpoint_path,
            model_type=model_type,
            save_path=export_path,
        )

    def _run_inference_and_check_results(
        self, model_path, model_type, inference_function, prediction_dir, expected_sa
    ):
        import micro_sam.evaluation as evaluation

        predictor = get_sam_model(model_type=model_type, checkpoint_path=model_path)

        image_paths = sorted(glob(os.path.join(self.tmp_folder, "synthetic-data", "images", "test", "*.tif")))
        label_paths = sorted(glob(os.path.join(self.tmp_folder, "synthetic-data", "labels", "test", "*.tif")))

        embedding_dir = os.path.join(self.tmp_folder, "embeddings")
        inference_function(predictor, image_paths, label_paths, embedding_dir, prediction_dir)

        pred_paths = sorted(glob(os.path.join(prediction_dir, "*.tif")))
        if len(pred_paths) == 0:  # we need to go to subfolder for iterative inference
            pred_paths = sorted(glob(os.path.join(prediction_dir, "iteration02", "*.tif")))

        self.assertEqual(len(pred_paths), len(label_paths))
        eval_res = evaluation.run_evaluation(label_paths, pred_paths, verbose=False)
        result = eval_res["sa50"].values.item()
        # We check against the expected segmentation accuracy.
        self.assertGreater(result, expected_sa)

    def test_training(self):
        import micro_sam.evaluation as evaluation

        model_type = "vit_t"
        device = "cpu"

        # Fine-tune the model.
        self._train_model(model_type=model_type, device=device)
        checkpoint_path = os.path.join(self.tmp_folder, "checkpoints", "test", "best.pt")
        self.assertTrue(os.path.exists(checkpoint_path))

        # Check that the model can be loaded from a custom checkpoint.
        predictor = get_sam_model(model_type=model_type, checkpoint_path=checkpoint_path, device=device)
        self.assertTrue(isinstance(predictor, SamPredictor))

        # Export the model.
        export_path = os.path.join(self.tmp_folder, "exported_model.pth")
        self._export_model(checkpoint_path, export_path, model_type)
        self.assertTrue(os.path.exists(export_path))

        # Check the model with inference with a single point prompt.
        prediction_dir = os.path.join(self.tmp_folder, "predictions-points")
        point_inference = partial(
            evaluation.run_inference_with_prompts,
            use_points=True, use_boxes=False,
            n_positives=1, n_negatives=0,
            batch_size=64,
        )
        self._run_inference_and_check_results(
            export_path, model_type, prediction_dir=prediction_dir,
            inference_function=point_inference, expected_sa=0.9
        )

        # Check the model with inference with a box point prompt.
        prediction_dir = os.path.join(self.tmp_folder, "predictions-boxes")
        box_inference = partial(
            evaluation.run_inference_with_prompts,
            use_points=False, use_boxes=True,
            n_positives=1, n_negatives=0,
            batch_size=64,
        )
        self._run_inference_and_check_results(
            export_path, model_type, prediction_dir=prediction_dir,
            inference_function=box_inference, expected_sa=0.95,
        )

        # Check the model with interactive inference.
        prediction_dir = os.path.join(self.tmp_folder, "predictions-iterative")
        iterative_inference = partial(
            evaluation.run_inference_with_iterative_prompting,
            start_with_box_prompt=False,
            n_iterations=3,
        )
        self._run_inference_and_check_results(
            export_path, model_type, prediction_dir=prediction_dir,
            inference_function=iterative_inference, expected_sa=0.95,
        )


if __name__ == "__main__":
    unittest.main()
