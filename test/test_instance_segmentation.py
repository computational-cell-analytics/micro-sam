import os
import unittest
from shutil import rmtree

import micro_sam.util as util
import numpy as np

from elf.evaluation.matching import matching
from skimage.draw import disk
from skimage.measure import label


class TestInstanceSegmentation(unittest.TestCase):
    embedding_path = "./tmp_embeddings.zarr"

    # create an input image with three objects
    @staticmethod
    def _get_input(shape=(256, 256)):
        mask = np.zeros(shape, dtype="uint8")

        def write_object(center, radius):
            circle = disk(center, radius, shape=shape)
            mask[circle] = 1

        center = tuple(sh // 4 for sh in shape)
        write_object(center, radius=19)

        center = tuple(sh // 2 for sh in shape)
        write_object(center, radius=27)

        center = tuple(3 * sh // 4 for sh in shape)
        write_object(center, radius=22)

        image = mask * 255
        mask = label(mask)
        return mask, image

    @staticmethod
    def _get_model(image, tile_shape=None, halo=None, save_path=None):
        predictor = util.get_sam_model(model_type="vit_b")

        image_embeddings = util.precompute_image_embeddings(
            predictor, image, tile_shape=tile_shape, halo=halo, save_path=save_path
        )
        return predictor, image_embeddings

    # we compute the default mask and predictor once for the class
    # so that we don't have to precompute it every time
    @classmethod
    def setUpClass(cls):
        cls.mask, cls.image = cls._get_input()
        cls.predictor, cls.image_embeddings = cls._get_model(cls.image)

    # remove temp embeddings if any
    def tearDown(self):
        if os.path.exists(self.embedding_path):
            rmtree(self.embedding_path)

    def test_automatic_mask_generator(self):
        from micro_sam.instance_segmentation import AutomaticMaskGenerator, mask_data_to_segmentation

        mask, image = self.mask, self.image
        predictor, image_embeddings = self.predictor, self.image_embeddings

        amg = AutomaticMaskGenerator(predictor, points_per_side=10, points_per_batch=16)
        amg.initialize(image, image_embeddings=image_embeddings, verbose=False)

        predicted = amg.generate()
        predicted = mask_data_to_segmentation(predicted, image.shape, with_background=True)
        self.assertGreater(matching(predicted, mask, threshold=0.75)["precision"], 0.99)

        predicted2 = amg.generate()
        predicted2 = mask_data_to_segmentation(predicted2, image.shape, with_background=True)
        self.assertTrue(np.array_equal(predicted, predicted2))

    def test_embedding_mask_generator(self):
        from micro_sam.instance_segmentation import EmbeddingMaskGenerator, mask_data_to_segmentation

        mask, image = self.mask, self.image
        predictor, image_embeddings = self.predictor, self.image_embeddings

        amg = EmbeddingMaskGenerator(predictor)
        amg.initialize(image, image_embeddings=image_embeddings, verbose=False)
        predicted = amg.generate(pred_iou_thresh=0.96)
        predicted = mask_data_to_segmentation(predicted, image.shape, with_background=True)

        self.assertGreater(matching(predicted, mask, threshold=0.75)["precision"], 0.99)

        initial_seg = amg.get_initial_segmentation()
        self.assertEqual(initial_seg.shape, image.shape)

        predicted2 = amg.generate(pred_iou_thresh=0.96)
        predicted2 = mask_data_to_segmentation(predicted2, image.shape, with_background=True)

        self.assertTrue(np.array_equal(predicted, predicted2))

    def test_tiled_embedding_mask_generator(self):
        from micro_sam.instance_segmentation import TiledEmbeddingMaskGenerator

        tile_shape, halo = (576, 576), (64, 64)
        mask, image = self._get_input(shape=(1024, 1024))
        predictor, image_embeddings = self._get_model(image, tile_shape, halo, self.embedding_path)

        amg = TiledEmbeddingMaskGenerator(predictor)
        amg.initialize(image, image_embeddings=image_embeddings)
        predicted = amg.generate(pred_iou_thresh=0.96)
        initial_seg = amg.get_initial_segmentation()

        self.assertGreater(matching(predicted, mask, threshold=0.75)["precision"], 0.99)
        self.assertEqual(initial_seg.shape, image.shape)

        predicted2 = amg.generate(pred_iou_thresh=0.96)
        self.assertTrue(np.array_equal(predicted, predicted2))


if __name__ == "__main__":
    unittest.main()
