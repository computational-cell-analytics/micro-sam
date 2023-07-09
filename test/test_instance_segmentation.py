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
        predictor, sam = util.get_sam_model(model_type="vit_b", return_sam=True)
        return predictor, sam

    @unittest.skip("This test takes very long.")
    def test_instance_segmentation_sam(self):
        from micro_sam.instance_segmentation import instance_segmentation_sam

        mask, image = self._get_input()
        _, sam = self._get_model()

        predicted = instance_segmentation_sam(sam, image)
        self.assertGreater(matching(predicted, mask, threshold=0.75)["precision"], 0.99)

    def test_instance_segmentation_from_embeddings(self):
        from micro_sam.instance_segmentation import instance_segmentation_from_embeddings

        mask, image = self._get_input()
        predictor, _ = self._get_model()

        image_embeddings = util.precompute_image_embeddings(predictor, image)
        util.set_precomputed(predictor, image_embeddings)

        predicted = instance_segmentation_from_embeddings(
            predictor, image_embeddings, min_initial_size=0, with_background=True, box_extension=5,
        )

        self.assertGreater(matching(predicted, mask, threshold=0.75)["precision"], 0.99)


if __name__ == "__main__":
    unittest.main()
