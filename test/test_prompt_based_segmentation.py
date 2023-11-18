import unittest

import micro_sam.util as util
import numpy as np

from skimage.draw import disk


class TestPromptBasedSegmentation(unittest.TestCase):
    model_type = "vit_t" if util.VIT_T_SUPPORT else "vit_b"

    @staticmethod
    def _get_input(shape=(256, 256)):
        mask = np.zeros(shape, dtype="uint8")
        center = tuple(sh // 2 for sh in shape)
        circle = disk(center, radius=20, shape=shape)
        mask[circle] = 1
        image = mask * 255
        return mask, image

    @staticmethod
    def _get_model(image, model_type):
        predictor = util.get_sam_model(model_type=model_type, device=util.get_device(None))
        image_embeddings = util.precompute_image_embeddings(predictor, image)
        util.set_precomputed(predictor, image_embeddings)
        return predictor

    # we compute the default mask and predictor once for the class
    # so that we don't have to precompute it every time
    @classmethod
    def setUpClass(cls):
        cls.mask, cls.image = cls._get_input()
        cls.predictor = cls._get_model(cls.image, cls.model_type)

    def test_segment_from_points(self):
        from micro_sam.prompt_based_segmentation import segment_from_points

        # segment with one positive and four negative points
        points = np.array([[128, 128], [64, 64], [192, 192], [64, 192], [192, 64]])
        labels = np.array([1, 0, 0, 0, 0])

        predicted = segment_from_points(self.predictor, points, labels)
        self.assertGreater(util.compute_iou(self.mask, predicted), 0.9)

        # segment with one positive point, using the best multimask
        points = np.array([[128, 128]])
        labels = np.array([1])

        predicted = segment_from_points(self.predictor, points, labels)
        self.assertGreater(util.compute_iou(self.mask, predicted), 0.9)

    def _test_segment_from_mask(self, shape=(256, 256)):
        from micro_sam.prompt_based_segmentation import segment_from_mask

        # we need to recompute the embedding if we have the non-square image
        # and we also need to set a lower expected iou when using only a mask prompt
        # (for some reason this does not work as well for non-square images)
        if shape == (256, 256):
            mask, image = self.mask, self.image
            predictor = self.predictor
            expected_iou_mask = 0.9
        else:
            mask, image = self._get_input(shape)
            predictor = self._get_model(image, self.model_type)
            expected_iou_mask = 0.8

        #
        # single prompts
        #

        # only with bounding box
        predicted = segment_from_mask(predictor, mask, use_box=True, use_mask=False, use_points=False)
        self.assertGreater(util.compute_iou(mask, predicted), 0.9)

        # only with mask
        predicted = segment_from_mask(predictor, mask, use_box=False, use_mask=True, use_points=False)
        self.assertGreater(util.compute_iou(mask, predicted), expected_iou_mask)

        # only with points
        predicted = segment_from_mask(predictor, mask, use_box=False, use_mask=False, use_points=True, box_extension=4)
        self.assertGreater(util.compute_iou(mask, predicted), 0.7)  # need to be more lenient for only points

        #
        # combinations of two and prompts
        #

        # with box and mask (default setting)
        predicted = segment_from_mask(predictor, mask, use_box=True, use_mask=True, use_points=False)
        self.assertGreater(util.compute_iou(mask, predicted), 0.9)

        # with box and points
        predicted = segment_from_mask(predictor, mask, use_box=True, use_mask=False, use_points=True)
        self.assertGreater(util.compute_iou(mask, predicted), 0.9)

        # with mask and points
        predicted = segment_from_mask(predictor, mask, use_box=False, use_mask=True, use_points=True)
        self.assertGreater(util.compute_iou(mask, predicted), 0.9)

        # with box, mask and points
        predicted = segment_from_mask(predictor, mask, use_mask=True, use_box=True, use_points=True)
        self.assertGreater(util.compute_iou(mask, predicted), 0.9)

    def test_segment_from_mask(self):
        self._test_segment_from_mask()

    def test_segment_from_mask_non_square(self):
        self._test_segment_from_mask((256, 384))

    def test_segment_from_box(self):
        from micro_sam.prompt_based_segmentation import segment_from_box

        box = np.array([106, 106, 150, 150])
        predicted = segment_from_box(self.predictor, box)
        self.assertGreater(util.compute_iou(self.mask, predicted), 0.9)

    def test_segment_from_box_and_points(self):
        from micro_sam.prompt_based_segmentation import segment_from_box_and_points

        box = np.array([106, 106, 150, 150])
        points = np.array([[128, 128], [64, 64], [192, 192], [64, 192], [192, 64]])
        labels = np.array([1, 0, 0, 0, 0])

        predicted = segment_from_box_and_points(self.predictor, box, points, labels)
        self.assertGreater(util.compute_iou(self.mask, predicted), 0.9)


if __name__ == "__main__":
    unittest.main()
