import unittest
from shutil import rmtree

import micro_sam.util as util
import numpy as np

from elf.evaluation.matching import matching
from skimage.draw import disk
from skimage.measure import label


class TestInstanceSegmentation(unittest.TestCase):
    model_type = "vit_t" if util.VIT_T_SUPPORT else "vit_b"
    embedding_path = "./tmp_embeddings.zarr"
    tile_shape = (512, 512)
    halo = (96, 96)

    # create an input image with three objects
    @staticmethod
    def _get_input(shape=(256, 256)):
        mask = np.zeros(shape, dtype="uint8")

        def write_object(center, radius):
            circle = disk(center, radius, shape=shape)
            mask[circle] = 1

        center = tuple(sh // 4 for sh in shape)
        write_object(center, radius=29)

        center = tuple(sh // 2 for sh in shape)
        write_object(center, radius=33)

        center = tuple(3 * sh // 4 for sh in shape)
        write_object(center, radius=35)

        image = mask * 255
        mask = label(mask)
        return mask, image

    @staticmethod
    def _get_model(image, model_type):
        predictor = util.get_sam_model(model_type=model_type, device=util.get_device(None))
        image_embeddings = util.precompute_image_embeddings(predictor, image)
        return predictor, image_embeddings

    # we compute the default mask and predictor once for the class
    # so that we don't have to precompute it every time
    @classmethod
    def setUpClass(cls):
        cls.mask, cls.image = cls._get_input()
        cls.predictor, cls.image_embeddings = cls._get_model(cls.image, cls.model_type)
        cls.large_mask, cls.large_image = cls._get_input(shape=(1024, 1024))
        cls.tiled_embeddings = util.precompute_image_embeddings(
            cls.predictor, cls.large_image, save_path=cls.embedding_path, tile_shape=cls.tile_shape, halo=cls.halo
        )

    @classmethod
    def tearDownClass(cls):
        try:
            rmtree(cls.embedding_path)
        except OSError:
            pass

    def test_automatic_mask_generator(self):
        from micro_sam.instance_segmentation import AutomaticMaskGenerator, mask_data_to_segmentation

        mask, image = self.mask, self.image
        predictor, image_embeddings = self.predictor, self.image_embeddings

        amg = AutomaticMaskGenerator(predictor, points_per_side=10, points_per_batch=16)
        amg.initialize(image, image_embeddings=image_embeddings, verbose=False)

        predicted = amg.generate()
        predicted = mask_data_to_segmentation(predicted, with_background=True)
        self.assertGreater(matching(predicted, mask, threshold=0.75)["segmentation_accuracy"], 0.99)

        # check that regenerating the segmentation works
        predicted2 = amg.generate()
        predicted2 = mask_data_to_segmentation(predicted2, with_background=True)
        self.assertTrue(np.array_equal(predicted, predicted2))

        # check that serializing and reserializing the state works
        state = amg.get_state()
        amg = AutomaticMaskGenerator(predictor, points_per_side=10, points_per_batch=16)
        amg.set_state(state)
        predicted3 = amg.generate()
        predicted3 = mask_data_to_segmentation(predicted3, with_background=True)
        self.assertTrue(np.array_equal(predicted, predicted3))

    def test_tiled_automatic_mask_generator(self):
        from micro_sam.instance_segmentation import TiledAutomaticMaskGenerator, mask_data_to_segmentation

        # Release all unoccupied cached memory, tiling requires a lot of memory
        device = util.get_device(None)
        if device == "cuda":
            import torch.cuda
            torch.cuda.empty_cache()
        elif device == "mps":
            import torch.mps
            torch.mps.empty_cache()

        mask, image = self.large_mask, self.large_image
        predictor, image_embeddings = self.predictor, self.tiled_embeddings

        pred_iou_thresh = 0.75

        amg = TiledAutomaticMaskGenerator(predictor, points_per_side=8)
        amg.initialize(image, image_embeddings=image_embeddings, verbose=False)
        predicted = amg.generate(pred_iou_thresh=pred_iou_thresh)
        predicted = mask_data_to_segmentation(predicted, with_background=True)
        self.assertGreater(matching(predicted, mask, threshold=0.75)["segmentation_accuracy"], 0.99)

        predicted2 = amg.generate(pred_iou_thresh=pred_iou_thresh)
        predicted2 = mask_data_to_segmentation(predicted2, with_background=True)
        self.assertTrue(np.array_equal(predicted, predicted2))

        # check that serializing and reserializing the state works
        state = amg.get_state()
        amg = TiledAutomaticMaskGenerator(predictor)
        amg.set_state(state)
        predicted3 = amg.generate(pred_iou_thresh=pred_iou_thresh)
        predicted3 = mask_data_to_segmentation(predicted3, with_background=True)
        self.assertTrue(np.array_equal(predicted, predicted3))


if __name__ == "__main__":
    unittest.main()
