import unittest

import numpy as np
from skimage.data import binary_blobs
from skimage.measure import label


class TestMultiDimensionalSegmentation(unittest.TestCase):

    def test_merge_instance_segmentation_3d(self):
        from micro_sam.multi_dimensional_segmentation import merge_instance_segmentation_3d

        n_slices = 5
        data = np.stack(n_slices * binary_blobs(512))
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
            self.assertTrue(np.array_equal(ids0, np.unique(merged_seg[z])))

    def test_merge_instance_segmentation_3d_with_closing(self):
        from micro_sam.multi_dimensional_segmentation import merge_instance_segmentation_3d

        n_slices = 5
        data = np.stack(n_slices * binary_blobs(512))
        seg = label(data)

        stacked_seg = []
        offset = 0
        for z in range(n_slices):
            # Leave the middle slice blank, so that we can check that it
            # gets merged via closing.
            if z == 2:
                stack_seg = np.zeros_like(seg)
            else:
                stack_seg = seg.copy()
                stack_seg[stack_seg != 0] += offset
                offset = stack_seg.max()
            stacked_seg.append(stack_seg)
        stacked_seg = np.stack(stacked_seg)

        merged_seg = merge_instance_segmentation_3d(stacked_seg, gap_closing=1)

        # Make sure that we don't have any new objects in z + 1.
        # Every object should be merged, since we have full overlap due to stacking.
        ids0 = np.unique(merged_seg[0])
        for z in range(1, n_slices):
            self.assertTrue(np.array_equal(ids0, np.unique(merged_seg[z])))


if __name__ == "__main__":
    unittest.main()
