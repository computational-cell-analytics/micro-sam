import unittest

import micro_sam.util as util
import numpy as np

from skimage.draw import disk


class TestPromptBasedSegmentation(unittest.TestCase):
    model_type = "vit_t" if util.VIT_T_SUPPORT else "vit_b"

    @staticmethod
    def _get_input(shape=(256, 256), radius=20, mask_offset=None):
        mask = np.zeros(shape, dtype="uint8")
        center = tuple(sh // 2 for sh in shape)
        if mask_offset is not None:
            center = tuple(ce + off for ce, off in zip(center, mask_offset))
        circle = disk(center, radius=radius, shape=shape)
        mask[circle] = 1
        image = mask * 255
        return mask, image

    @staticmethod
    def _get_model(image, model_type, tile_shape=None, halo=None):
        predictor = util.get_sam_model(model_type=model_type, device=util.get_device(None))
        image_embeddings = util.precompute_image_embeddings(predictor, image, tile_shape=tile_shape, halo=halo)
        if tile_shape is None:
            util.set_precomputed(predictor, image_embeddings)
        return predictor, image_embeddings

    # we compute the default mask and predictor once for the class
    # so that we don't have to precompute it every time
    @classmethod
    def setUpClass(cls):
        cls.mask, cls.image = cls._get_input()
        cls.predictor, _ = cls._get_model(cls.image, cls.model_type)

        cls.mask_non_square, cls.image_non_square = cls._get_input(
            shape=(1040, 1408), mask_offset=(256, 256)
        )
        cls.predictor_non_square, _ = cls._get_model(cls.image_non_square, cls.model_type)

        cls.mask_tiled, cls.image_tiled = cls._get_input(shape=(1024, 1024), radius=64)
        cls.predictor_tiled, cls.tiled_embeddings = cls._get_model(
            cls.image_tiled, cls.model_type, tile_shape=(512, 512), halo=(96, 96)
        )

    #
    # Tests for 'segment_from_points':
    # normal test, non-square input, tiled
    #

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

    def test_segment_from_points_non_square(self):
        from micro_sam.prompt_based_segmentation import segment_from_points

        predictor, mask = self.predictor_non_square, self.mask_non_square

        mask_coords = np.where(mask)
        # Get the center from the mask coordinates for positive point.
        center = [mask_coords[0].mean(), mask_coords[1].mean()]

        # Put negative points at the bounding box edges.
        border_points = [
            [mask_coords[0].min(), mask_coords[1].min()],
            [mask_coords[0].max(), mask_coords[1].min()],
            [mask_coords[0].min(), mask_coords[1].max()],
            [mask_coords[0].max(), mask_coords[1].max()],
        ]

        points = np.array([center] + border_points)
        labels = np.array([1] + [0] * len(border_points))

        predicted = segment_from_points(predictor, points, labels)
        self.assertGreater(util.compute_iou(mask, predicted), 0.9)

    def test_segment_from_points_tiled(self):
        from micro_sam.prompt_based_segmentation import segment_from_points

        # segment with one positive and two negative points
        points = np.array([[510, 510], [400, 200], [200, 400]])
        labels = np.array([1, 0, 0])

        predicted = segment_from_points(
            self.predictor_tiled, points, labels, image_embeddings=self.tiled_embeddings
        )

        self.assertGreater(util.compute_iou(self.mask_tiled, predicted), 0.9)

        # segment with one positive point, using the best multimask
        points = np.array([[512, 512]])
        labels = np.array([1])

        predicted = segment_from_points(
            self.predictor_tiled, points, labels, image_embeddings=self.tiled_embeddings
        )
        self.assertGreater(util.compute_iou(self.mask_tiled, predicted), 0.9)

        # segment with multimasking
        predicted = segment_from_points(
            self.predictor_tiled, points, labels, image_embeddings=self.tiled_embeddings,
            multimask_output=True, use_best_multimask=False,
        )
        self.assertEqual(predicted.shape, (3,) + self.mask_tiled.shape)

    #
    # Tests for 'segment_from_mask':
    # normal test, non square inputs, tiled
    #

    def _test_segment_from_mask(self, predictor, mask, expected_iou_mask=0.9, embeddings=None):
        from micro_sam.prompt_based_segmentation import segment_from_mask

        #
        # single prompts
        #

        # only with bounding box
        predicted = segment_from_mask(
            predictor, mask, use_box=True, use_mask=False, use_points=False,
            image_embeddings=embeddings,
        )
        self.assertGreater(util.compute_iou(mask, predicted), 0.9)

        # only with mask
        predicted = segment_from_mask(
            predictor, mask, use_box=False, use_mask=True, use_points=False,
            image_embeddings=embeddings,
        )
        self.assertGreater(util.compute_iou(mask, predicted), expected_iou_mask)

        # only with points
        predicted = segment_from_mask(
            predictor, mask, use_box=False, use_mask=False, use_points=True, box_extension=4,
            image_embeddings=embeddings,
        )
        self.assertGreater(util.compute_iou(mask, predicted), 0.7)  # need to be more lenient for only points

        #
        # combinations of two and prompts
        #

        # with box and mask (default setting)
        predicted = segment_from_mask(
            predictor, mask, use_box=True, use_mask=True, use_points=False,
            image_embeddings=embeddings,
        )
        self.assertGreater(util.compute_iou(mask, predicted), 0.9)

        # with box and points
        predicted = segment_from_mask(
            predictor, mask, use_box=True, use_mask=False, use_points=True,
            image_embeddings=embeddings,
        )
        self.assertGreater(util.compute_iou(mask, predicted), 0.9)

        # with mask and points
        predicted = segment_from_mask(
            predictor, mask, use_box=False, use_mask=True, use_points=True,
            image_embeddings=embeddings,
        )
        self.assertGreater(util.compute_iou(mask, predicted), 0.9)

        # with box, mask and points
        predicted = segment_from_mask(
            predictor, mask, use_mask=True, use_box=True, use_points=True,
            image_embeddings=embeddings,
        )
        self.assertGreater(util.compute_iou(mask, predicted), 0.9)

    def test_segment_from_mask(self):
        self._test_segment_from_mask(self.predictor, self.mask)

    def test_segment_from_mask_non_square(self):
        self._test_segment_from_mask(self.predictor_non_square, self.mask_non_square, expected_iou_mask=0.8)

    def test_segment_from_mask_tiled(self):
        self._test_segment_from_mask(
            self.predictor_tiled, self.mask_tiled, embeddings=self.tiled_embeddings, expected_iou_mask=0.8
        )

    #
    # Tests for 'segment_from_box':
    # normal test, non square inputs, tiled
    #

    def test_segment_from_box(self):
        from micro_sam.prompt_based_segmentation import segment_from_box

        box = np.array([106, 106, 150, 150])
        predicted = segment_from_box(self.predictor, box)
        self.assertGreater(util.compute_iou(self.mask, predicted), 0.9)

    def test_segment_from_box_non_square(self):
        from micro_sam.prompt_based_segmentation import segment_from_box

        predictor, mask = self.predictor_non_square, self.mask_non_square
        box = np.where(mask)
        box = np.array([b.min() for b in box] + [b.max() for b in box])
        predicted = segment_from_box(predictor, box)

        self.assertGreater(util.compute_iou(mask, predicted), 0.9)

    def test_segment_from_box_tiled(self):
        from micro_sam.prompt_based_segmentation import segment_from_box

        box = np.array([450, 450, 570, 570])
        predicted = segment_from_box(self.predictor_tiled, box, image_embeddings=self.tiled_embeddings)
        self.assertGreater(util.compute_iou(self.mask_tiled, predicted), 0.9)

    #
    # Tests for 'segment_from_box_and_points':
    # normal test, non square inputs, tiled
    #

    def test_segment_from_box_and_points(self):
        from micro_sam.prompt_based_segmentation import segment_from_box_and_points

        box = np.array([106, 106, 150, 150])
        points = np.array([[128, 128], [64, 64], [192, 192], [64, 192], [192, 64]])
        labels = np.array([1, 0, 0, 0, 0])

        predicted = segment_from_box_and_points(self.predictor, box, points, labels)
        self.assertGreater(util.compute_iou(self.mask, predicted), 0.9)

    def test_segment_from_box_and_points_non_square(self):
        from micro_sam.prompt_based_segmentation import segment_from_box_and_points

        predictor, mask = self.predictor_non_square, self.mask_non_square

        mask_coords = np.where(mask)
        # Get the center from the mask coordinates for positive point.
        center = [mask_coords[0].mean(), mask_coords[1].mean()]

        # Put negative points at the bounding box edges.
        border_points = [
            [mask_coords[0].min(), mask_coords[1].min()],
            [mask_coords[0].max(), mask_coords[1].min()],
            [mask_coords[0].min(), mask_coords[1].max()],
            [mask_coords[0].max(), mask_coords[1].max()],
        ]

        box = np.array([m.min() for m in mask_coords] + [m.max() for m in mask_coords])
        points = np.array([center] + border_points)
        labels = np.array([1] + [0] * len(border_points))

        predicted = segment_from_box_and_points(predictor, box, points, labels)
        self.assertGreater(util.compute_iou(mask, predicted), 0.9)

    def test_segment_from_box_and_points_tiled(self):
        from micro_sam.prompt_based_segmentation import segment_from_box_and_points

        # Segment with one positive and two negative points + box
        box = np.array([450, 450, 570, 570])
        points = np.array([[510, 510], [400, 200], [200, 400]])
        labels = np.array([1, 0, 0])

        predicted = segment_from_box_and_points(
            self.predictor_tiled, box, points, labels, image_embeddings=self.tiled_embeddings
        )
        self.assertGreater(util.compute_iou(self.mask_tiled, predicted), 0.9)


if __name__ == "__main__":
    unittest.main()
