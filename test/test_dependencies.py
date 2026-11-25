import unittest

import torch

from segment_anything.utils.transforms import ResizeLongestSide


class TestDependencies(unittest.TestCase):
    def test_resize_longest_side_torch(self):
        transform = ResizeLongestSide(16)
        image = torch.zeros((1, 3, 5, 7))

        resized = transform.apply_image_torch(image)

        self.assertEqual(tuple(resized.shape), (1, 3, 11, 16))
