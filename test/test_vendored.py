import os
import unittest

import numpy as np
import torch

from segment_anything.utils.amg import mask_to_rle_pytorch as mask_to_rle_pytorch_sam
from skimage.draw import random_shapes


class TestVendored(unittest.TestCase):
    def _get_mask_to_box_data(self):
        mask_numpy = np.zeros((10, 10)).astype(bool)
        mask_numpy[7:9, 3:5] = True
        expected_result = [3, 7, 4, 8]
        return mask_numpy, expected_result

    def _test_batched_mask_to_box(self, device):
        from micro_sam._vendored import batched_mask_to_box

        mask, expected_result = self._get_mask_to_box_data()
        mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
        expected_result = torch.as_tensor(expected_result, dtype=torch.int, device=device)
        result = batched_mask_to_box(mask)
        assert all(result == expected_result)

    def test_cpu_batched_mask_to_box(self):
        self._test_batched_mask_to_box(device="cpu")

    @unittest.skipIf(not torch.cuda.is_available(),
                     "CUDA Pytorch backend is not available")
    def test_cuda_batched_mask_to_box(self):
        self._test_batched_mask_to_box(device="cuda")


    @unittest.skipIf((os.getenv("GITHUB_ACTIONS") == "true"),
                     "Test fails on Github Actions macos-14 runner " + \
                     "https://github.com/computational-cell-analytics/micro-sam/issues/380")
    @unittest.skipIf(not (torch.backends.mps.is_available() and torch.backends.mps.is_built()),
                     "MPS Pytorch backend is not available")
    def test_mps_batched_mask_to_box(self):
        self._test_batched_mask_to_box(device="mps")

    def _get_mask_to_rle_pytorch_data(self):
        shape = (128, 256)

        # randm shapes for 6 masks
        n_masks = 6
        masks, _ = random_shapes(shape, min_shapes=n_masks, max_shapes=n_masks)
        masks = masks.astype("uint32").sum(axis=-1)

        bg_id = 765  # bg val is 3 * 255 = 765
        mask_ids = np.setdiff1d(np.unique(masks), bg_id)

        one_hot = np.zeros((len(mask_ids),) + shape, dtype=bool)
        for i, idx in enumerate(mask_ids):
            one_hot[i, masks == idx] = 1
        one_hot = torch.from_numpy(one_hot)

        expected_result = mask_to_rle_pytorch_sam(one_hot)
        return one_hot, expected_result

    def test_mask_to_rle_pytorch(self):
        from micro_sam._vendored import mask_to_rle_pytorch

        masks, expected_result = self._get_mask_to_rle_pytorch_data()
        expected_size = masks.shape[1] * masks.shape[2]

        # make sure that the RLE's are consistent (their sum needs to be equal to the number of pixels)
        for rle in expected_result:
            assert sum(rle["counts"]) == expected_size, f"{sum(rle['counts'])}, {expected_size}"

        result = mask_to_rle_pytorch(masks)
        for rle in result:
            assert sum(rle["counts"]) == expected_size, f"{sum(rle['counts'])}, {expected_size}"

        # make sure that the RLE's agree
        assert result == expected_result


if __name__ == "__main__":
    unittest.main()
