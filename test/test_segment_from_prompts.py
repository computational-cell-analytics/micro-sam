import unittest

import micro_sam.util as util
import numpy as np

from skimage.draw import disk


class TestSegmentFromPrompts(unittest.TestCase):
    def _get_input(self):
        shape = (256, 256)
        mask = np.zeros(shape, dtype="uint8")
        circle = disk((128, 128), radius=20, shape=shape)
        mask[circle] = 1
        image = mask * 255
        return mask, image

    def _get_model(self, image):
        predictor = util.get_sam_model(model_type="vit_b")
        image_embeddings = util.precompute_image_embeddings(predictor, image)
        util.set_precomputed(predictor, image_embeddings)
        return predictor

    def test_segment_from_points(self):
        from micro_sam.segment_from_prompts import segment_from_points

        mask, image = self._get_input()
        predictor = self._get_model(image)

        points = np.array([[128, 128], [64, 64], [192, 192], [64, 192], [192, 64]])
        labels = np.array([1, 0, 0, 0, 0])

        predicted = segment_from_points(predictor, points, labels)
        self.assertGreater(util.compute_iou(mask, predicted), 0.9)

    def test_segment_from_mask(self):
        from micro_sam.segment_from_prompts import segment_from_mask

        mask, image = self._get_input()
        predictor = self._get_model(image)

        # with mask and bounding box (default setting)
        predicted = segment_from_mask(predictor, mask)
        self.assertGreater(util.compute_iou(mask, predicted), 0.9)

        # with  bounding box (default setting)
        predicted = segment_from_mask(predictor, mask, use_mask=False, use_box=True)
        self.assertGreater(util.compute_iou(mask, predicted), 0.9)

        # with  bounding box (default setting)
        predicted = segment_from_mask(predictor, mask, use_mask=True, use_box=False)
        self.assertGreater(util.compute_iou(mask, predicted), 0.9)

    def test_segment_from_box(self):
        from micro_sam.segment_from_prompts import segment_from_box

        mask, image = self._get_input()
        predictor = self._get_model(image)

        box = np.array([106, 106, 150, 150])
        predicted = segment_from_box(predictor, box)
        self.assertGreater(util.compute_iou(mask, predicted), 0.9)


if __name__ == "__main__":
    unittest.main()
