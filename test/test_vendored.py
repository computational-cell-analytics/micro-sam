import unittest

import numpy as np
import torch


class TestVendored(unittest.TestCase):
    def setUp(self):
        mask_numpy = np.zeros((10,10)).astype(bool)
        mask_numpy[7:9, 3:5] = True
        self.mask = mask_numpy
        self.expected_result = [3, 7, 4, 8]

    def test_cpu_batched_mask_to_box(self):
        from micro_sam._vendored import batched_mask_to_box

        device = "cpu"
        mask = torch.as_tensor(self.mask, dtype=torch.bool, device=device)
        expected_result = torch.as_tensor(self.expected_result, dtype=torch.int, device=device)
        result = batched_mask_to_box(mask)
        assert all(result == expected_result)

    @unittest.skipIf(not torch.cuda.is_available(),
                     "CUDA Pytorch backend is not available")
    def test_cuda_batched_mask_to_box(self):
        from micro_sam._vendored import batched_mask_to_box

        device = "cuda"
        mask = torch.as_tensor(self.mask, dtype=torch.bool, device=device)
        expected_result = torch.as_tensor(self.expected_result, dtype=torch.int, device=device)
        result = batched_mask_to_box(mask)
        assert all(result == expected_result)

    @unittest.skipIf(not (torch.backends.mps.is_available() and torch.backends.mps.is_built()),
                     "MPS Pytorch backend is not available")
    def test_mps_batched_mask_to_box(self):
        from micro_sam._vendored import batched_mask_to_box

        device = "mps"
        mask = torch.as_tensor(self.mask, dtype=torch.bool, device=device)
        expected_result = torch.as_tensor(self.expected_result, dtype=torch.int, device=device)
        result = batched_mask_to_box(mask)
        assert all(result == expected_result)
