import unittest

import torch


class TestSAM3DWrapper(unittest.TestCase):
    model_type = "vit_b"

    def test_sam_3d_wrapper(self):
        from micro_sam.models.sam_3d_wrapper import get_sam_3d_model

        image_size = 256
        n_classes = 2
        sam_3d = get_sam_3d_model(device="cpu", model_type=self.model_type, image_size=image_size, n_classes=n_classes)

        # Shape: C X D X H X W
        shape = (3, 4, image_size, image_size)
        expected_shape = (1, n_classes, 4, image_size, image_size)
        with torch.no_grad():
            batched_input = [{"image": torch.rand(*shape), "original_size": shape[-2:]}]
            output = sam_3d(batched_input, multimask_output=True)
            masks = output[0]["masks"]
            self.assertEqual(masks.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
