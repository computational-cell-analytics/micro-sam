import unittest

import micro_sam.util as util
import numpy as np

from elf.evaluation.matching import matching
from skimage.draw import disk
from skimage.measure import label


class TestInstanceSegmentation(unittest.TestCase):

    # create an input image with three objects
    def _get_input(self, shape=(512, 512)):
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

    def _get_model(self):
        return util.get_sam_model(model_type="vit_b", return_sam=False)

    def test_automatic_mask_generator(self):
        from micro_sam.instance_segmentation import AutomaticMaskGenerator, mask_data_to_segmentation

        mask, image = self._get_input(shape=(256, 256))
        predictor = self._get_model()

        amg = AutomaticMaskGenerator(predictor, points_per_side=10, points_per_batch=16)
        amg.initialize(image, verbose=False)
        predicted = amg.generate()
        predicted = mask_data_to_segmentation(predicted, image.shape, with_background=True)

        self.assertGreater(matching(predicted, mask, threshold=0.75)["precision"], 0.99)

    def test_embedding_based_mask_generator(self):
        from micro_sam.instance_segmentation import EmbeddingBasedMaskGenerator, mask_data_to_segmentation

        mask, image = self._get_input()
        predictor = self._get_model()

        amg = EmbeddingBasedMaskGenerator(predictor)
        amg.initialize(image, verbose=False)
        predicted = amg.generate(pred_iou_thresh=0.96)
        predicted = mask_data_to_segmentation(predicted, image.shape, with_background=True)

        self.assertGreater(matching(predicted, mask, threshold=0.75)["precision"], 0.99)

        initial_seg = amg.get_initial_segmentation()
        self.assertEqual(initial_seg.shape, image.shape)


if __name__ == "__main__":
    unittest.main()
