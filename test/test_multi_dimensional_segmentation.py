import unittest

import numpy as np
from skimage.data import binary_blobs
from skimage.draw import disk
from skimage.measure import label

from micro_sam.util import VIT_T_SUPPORT, precompute_image_embeddings, get_sam_model
from elf.evaluation import matching


class TestMultiDimensionalSegmentation(unittest.TestCase):

    def test_merge_instance_segmentation_3d(self):
        from micro_sam.multi_dimensional_segmentation import merge_instance_segmentation_3d

        n_slices = 5
        data = np.stack(n_slices * [binary_blobs(512)])
        seg = label(data)

        stacked_seg = []
        offset = 0
        for _ in range(n_slices):
            stack_seg = seg.copy()
            stack_seg[stack_seg != 0] += offset
            offset = stack_seg.max()
            stacked_seg.append(stack_seg)
        stacked_seg = np.stack(stacked_seg)

        merged_seg = merge_instance_segmentation_3d(stacked_seg)

        # Make sure that we don't have any new objects in z + 1.
        # Every object should be merged, since we have full overlap due to stacking.
        ids0 = np.unique(merged_seg[0])
        for z in range(1, n_slices):
            self.assertTrue(np.array_equal(ids0, np.unique(merged_seg)))

    def test_postprocess_volumetric_segmentation(self):
        from micro_sam.multi_dimensional_segmentation import postprocess_volumetric_segmentation

        # Create data with four different circle
        data = np.zeros((512, 512), dtype="uint8")
        data[disk((128, 128), 48)] = 255
        data[disk((384, 128), 48)] = 255
        data[disk((128, 384), 48)] = 255
        data[disk((384, 384), 48)] = 255
        n_slices = 7
        data = np.stack(n_slices * [data])

        expected_segmentation = label(data)

        model_type = "vit_t" if VIT_T_SUPPORT else "vit_b"
        predictor = get_sam_model(model_type=model_type)
        image_embeddings = precompute_image_embeddings(predictor, data, ndim=3)

        segmentation = expected_segmentation.copy()
        segmentation[4] = 0
        segmentation = label(segmentation)

        segmentation_pp = postprocess_volumetric_segmentation(
            segmentation, predictor, image_embeddings,
            projection="bounding_box", z_extension=2,
        )

        score = matching(expected_segmentation, segmentation_pp, threshold=0.90)["f1"]
        self.assertAlmostEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
