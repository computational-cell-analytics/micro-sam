import os
import unittest
from functools import partial
from glob import glob
from shutil import rmtree

import imageio.v3 as imageio

from micro_sam.sample_data import synthetic_data
from micro_sam.util import VIT_T_SUPPORT, get_sam_model, SamPredictor


class TestDataset(unittest.TestCase):
    tmp_folder = "./tmp-dataset"

    def setUp(self):
        self.image_dir = os.path.join(self.tmp_folder, "synthetic-data", "images")
        self.label_dir = os.path.join(self.tmp_folder, "synthetic-data", "labels")
        shape = (512, 512)

        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)

        n_images = 5
        for idx in range(n_images):
            image_path = os.path.join(self.image_dir, f"data-{idx}.tif")
            label_path = os.path.join(self.label_dir, f"data-{idx}.tif")

            image, labels = synthetic_data(shape)
            imageio.imwrite(image_path, image)
            imageio.imwrite(label_path, labels)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def _check_dataset(self, ds, patch_shape, exp_type):
        self.assertIsInstance(ds, exp_type)
        self.assertEqual(ds._ndim, 2)

        expected_im_shape = (1,) + patch_shape
        expected_label_shape = (4,) + patch_shape
        for i in range(5):
            x, y = ds[i]
            self.assertEqual(x.shape, expected_im_shape)
            self.assertEqual(y.shape, expected_label_shape)

    def test_default_sam_dataset(self):
        from micro_sam.training.training import default_sam_dataset
        from torch_em.data import SegmentationDataset

        patch_shape = (512, 512)
        ds = default_sam_dataset(
            self.image_dir, "*.tif", self.label_dir, "*.tif", patch_shape, with_segmentation_decoder=True
        )
        self._check_dataset(ds, patch_shape, SegmentationDataset)

    def test_default_sam_dataset_with_numpy_data(self):
        from micro_sam.training.training import default_sam_dataset
        from torch_em.data import TensorDataset

        patch_shape = (512, 512)
        images = sorted(glob(os.path.join(self.image_dir, "*.tif")))
        images = [imageio.imread(im) for im in images]
        labels = sorted(glob(os.path.join(self.label_dir, "*.tif")))
        labels = [imageio.imread(lab) for lab in labels]
        ds = default_sam_dataset(
            images, None, labels, None, patch_shape, with_segmentation_decoder=True
        )
        self._check_dataset(ds, patch_shape, TensorDataset)


@unittest.skip("Not working in CI")
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

    def _get_dataloader(self, split, patch_shape, batch_size, train_instance_segmentation_only=False):
        import micro_sam.training as sam_training

        # Create the synthetic training data and get the corresponding folders.
        image_root = os.path.join(self.tmp_folder, "synthetic-data", "images", split)
        label_root = os.path.join(self.tmp_folder, "synthetic-data", "labels", split)
        raw_key, label_key = "*.tif", "*.tif"

        loader = sam_training.default_sam_loader(
            raw_paths=image_root, raw_key=raw_key,
            label_paths=label_root, label_key=label_key,
            patch_shape=patch_shape, batch_size=batch_size,
            with_segmentation_decoder=train_instance_segmentation_only,
            shuffle=True, num_workers=1,
            n_samples=self.n_images_train if split == "train" else self.n_images_val,
            train_instance_segmentation_only=train_instance_segmentation_only,
        )
        return loader

    def _train_model(self, model_type, device):
        import micro_sam.training as sam_training

        batch_size = 1
        n_sub_iteration = 2
        patch_shape = (512, 512)
        n_objects_per_batch = 1

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
            freeze=["image_encoder"],
            device=device,
            save_root=self.tmp_folder,
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
        result = eval_res["SA50"].values.item()
        # We check against the expected segmentation accuracy.
        self.assertGreater(result, expected_sa)

    def test_training(self):
        import micro_sam.evaluation as evaluation

        model_type, device = "vit_t", "cpu"

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

        # Check the model with interactive inference.
        prediction_dir = os.path.join(self.tmp_folder, "predictions-iterative")
        iterative_inference = partial(
            evaluation.run_inference_with_iterative_prompting,
            start_with_box_prompt=False,
            n_iterations=3,
        )
        self._run_inference_and_check_results(
            export_path, model_type, prediction_dir=prediction_dir,
            inference_function=iterative_inference, expected_sa=0.8,
        )

    def test_train_instance_segmentation(self):
        from micro_sam.training.training import train_instance_segmentation, export_instance_segmentation_model
        from micro_sam.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter

        model_type, device = "vit_t", "cpu"
        batch_size, patch_shape = 1, (512, 512)

        # Get the dataloaders.
        train_loader = self._get_dataloader("train", patch_shape, batch_size, train_instance_segmentation_only=True)
        val_loader = self._get_dataloader("val", patch_shape, batch_size, train_instance_segmentation_only=True)

        # Run the training.
        # We freeze the image encoder to speed up the training process.
        name = "test-instance-seg-only"
        train_instance_segmentation(
            name=name,
            model_type=model_type,
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=1,
            device=device,
            save_root=self.tmp_folder,
            freeze=["image_encoder"],
        )

        checkpoint_path = os.path.join(self.tmp_folder, "checkpoints", name, "best.pt")
        self.assertTrue(os.path.exists(checkpoint_path))

        export_path = os.path.join(self.tmp_folder, "instance_segmentation_model.pt")
        export_instance_segmentation_model(checkpoint_path, export_path, model_type)
        self.assertTrue(os.path.exists(export_path))

        # Check that this model works for AIS.
        predictor, segmenter = get_predictor_and_segmenter(model_type, export_path, segmentation_mode="ais")
        image_path = os.path.join(self.tmp_folder, "synthetic-data", "images", "test", "data-0.tif")
        segmentation = automatic_instance_segmentation(predictor, segmenter, image_path)
        expected_shape = imageio.imread(image_path).shape
        self.assertEqual(segmentation.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
