import unittest

import micro_sam.util as util
import numpy as np

from elf.evaluation.matching import matching
from skimage.draw import disk


class TestSegmentInstances(unittest.TestCase):

    # create an input image with three objects
    def _get_input(self, shape=(96, 96)):
        mask = np.zeros(shape, dtype="uint8")

        def write_object(center, radius):
            circle = disk(center, radius, shape=shape)
            mask[circle] = 1

        center = tuple(sh // 4 for sh in shape)
        write_object(center, radius=8)

        center = tuple(sh // 2 for sh in shape)
        write_object(center, radius=9)

        center = tuple(3 * sh // 4 for sh in shape)
        write_object(center, radius=7)

        image = mask * 255
        return mask, image

    def _get_model(self):
        predictor, sam = util.get_sam_model(model_type="vit_b", return_sam=True)
        return predictor, sam

    @unittest.skip("This test takes very long.")
    def test_segment_instances_sam(self):
        from micro_sam.segment_instances import segment_instances_sam

        mask, image = self._get_input()
        _, sam = self._get_model()

        predicted = segment_instances_sam(sam, image)
        self.assertGreater(matching(predicted, mask, threshold=0.75)["precision"], 0.99)

    # @unittest.skip("Needs some more debugging.")
    def test_segment_instances_from_embeddings(self):
        from micro_sam.segment_instances import segment_instances_from_embeddings

        mask, image = self._get_input()
        predictor, _ = self._get_model()

        image_embeddings = util.precompute_image_embeddings(predictor, image)
        util.set_precomputed(predictor, image_embeddings)

        predicted = segment_instances_from_embeddings(predictor, image_embeddings)
        # import napari
        # v = napari.Viewer()
        # v.add_image(image)
        # v.add_labels(mask)
        # v.add_labels(predicted)
        # napari.run()
        self.assertGreater(matching(predicted, mask, threshold=0.75)["precision"], 0.99)


if __name__ == "__main__":
    unittest.main()
