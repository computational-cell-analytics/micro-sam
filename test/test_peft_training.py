import unittest

import torch

from micro_sam.util import get_sam_model
from micro_sam.training.peft_sam import PEFT_Sam


class TestPEFTModule(unittest.TestCase):
    """Integraton test for instantiating a PEFT SAM model.
    """
    def _fetch_sam_model(self, model_type, device):
        _, sam_model = get_sam_model(model_type=model_type, device=device, return_sam=True)
        return sam_model

    def _create_dummy_inputs(self, shape):
        input_image = torch.ones(shape)
        return input_image

    def test_peft_sam(self):
        model_type = "vit_b"
        device = "cpu"

        # Load the dummy inputs.
        input_shape = (1, 512, 512)
        inputs = self._create_dummy_inputs(shape=input_shape)

        # Convert to the inputs expected by Segment Anything
        batched_inputs = [
            {"image": inputs, "original_size": input_shape[1:]}
        ]

        # Load the Segment Anything model.
        sam_model = self._fetch_sam_model(model_type=model_type, device=device)

        # Wrap the Segment Anything model with PEFT methods.
        peft_sam_model = PEFT_Sam(model=sam_model, rank=4)

        # Get the model outputs
        outputs = peft_sam_model(batched_input=batched_inputs, multimask_output=False)

        # Check the expected shape of the outputs
        mask_shapes = [output["masks"].shape[-2:] for output in outputs]
        for shape in mask_shapes:
            self.assertEqual(shape, input_shape[1:])


if __name__ == "__main__":
    unittest.main()
