import unittest

import micro_sam.util as util
import numpy as np

from skimage.draw import disk


class TestSegmentFromPrompts(unittest.TestCase):
    @staticmethod
    def _get_input(shape=(256, 256)):
        mask = np.zeros(shape, dtype="uint8")
        center = tuple(sh // 2 for sh in shape)
        circle = disk(center, radius=20, shape=shape)
        mask[circle] = 1
        image = mask * 255
        return mask, image

    @staticmethod
    def _get_model(image):
        predictor = util.get_sam_model(model_type="vit_b")
        image_embeddings = util.precompute_image_embeddings(predictor, image)
        util.set_precomputed(predictor, image_embeddings)
        return predictor

    # we compute the default mask and predictor once for the class
    # so that we don't have to precompute it every time
    @classmethod
    def setUpClass(cls):
        cls.mask, cls.image = cls._get_input()
        cls.predictor = cls._get_model(cls.image)

    def test_segment_from_points(self):
        from micro_sam.segment_from_prompts import segment_from_points

        points = np.array([[128, 128], [64, 64], [192, 192], [64, 192], [192, 64]])
        labels = np.array([1, 0, 0, 0, 0])

        predicted = segment_from_points(self.predictor, points, labels)
        self.assertGreater(util.compute_iou(self.mask, predicted), 0.9)

    def _test_segment_from_mask(self, shape=(256, 256), use_mask=True):
        from micro_sam.segment_from_prompts import segment_from_mask

        if shape == (256, 256):
            mask, image = self.mask, self.image
            predictor = self.predictor
        else:
            mask, image = self._get_input(shape)
            predictor = self._get_model(image)

        # with mask and bounding box (default setting)
        if use_mask:
            predicted = segment_from_mask(predictor, mask)
            self.assertGreater(util.compute_iou(mask, predicted), 0.9)

        # with bounding box
        predicted = segment_from_mask(predictor, mask, use_mask=False, use_box=True)
        self.assertGreater(util.compute_iou(mask, predicted), 0.9)

        # with mask
        if use_mask:
            predicted = segment_from_mask(predictor, mask, use_mask=True, use_box=False)
            self.assertGreater(util.compute_iou(mask, predicted), 0.9)

    def test_segment_from_mask(self):
        self._test_segment_from_mask()

    # segmenting from mask prompts is not working for non-square inputs yet
    # that's why it's deactivated here
    def test_segment_from_mask_non_square(self):
        self._test_segment_from_mask((256, 384), use_mask=False)

    def test_segment_from_box(self):
        from micro_sam.segment_from_prompts import segment_from_box

        box = np.array([106, 106, 150, 150])
        predicted = segment_from_box(self.predictor, box)
        self.assertGreater(util.compute_iou(self.mask, predicted), 0.9)

    def test_segment_from_box_and_points(self):
        from micro_sam.segment_from_prompts import segment_from_box_and_points

        box = np.array([106, 106, 150, 150])
        points = np.array([[128, 128], [64, 64], [192, 192], [64, 192], [192, 64]])
        labels = np.array([1, 0, 0, 0, 0])

        predicted = segment_from_box_and_points(self.predictor, box, points, labels)
        self.assertGreater(util.compute_iou(self.mask, predicted), 0.9)


if __name__ == "__main__":
    unittest.main()
