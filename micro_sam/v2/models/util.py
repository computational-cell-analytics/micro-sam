from typing import Optional, Union

import torch
import torch.nn as nn

from torch_em.model.unetr import UNETR3D

from micro_sam.util import get_device
from micro_sam.v2.util import get_sam2_model


class CustomActivation(nn.Module):
    """Apply sigmoid to foreground and optional auxiliary channels, and tanh to distances."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # In bfloat16 the sigmoid saturates to exactly one from a logit of 6.5, which zeroes the loss gradient.
        x = x.float()
        return torch.cat([torch.sigmoid(x[:, :1]), torch.tanh(x[:, 1:4]), torch.sigmoid(x[:, 4:])], dim=1)


def joint_unetr_state(state):
    """The UniSAM2 state of a joint checkpoint.

    The joint trainer saves the SAM2 weights as 'model_state' and the decoder as 'decoder_state'; the encoder of the
    UniSAM2 is the SAM2 image encoder, stored under the adapter's 'encoder.inner.' prefix. Joint checkpoints from
    before v6 hold the whole UniSAM2 state as 'unetr_state'.
    """
    if "unetr_state" in state:
        return state["unetr_state"]
    prefix = "image_encoder."
    encoder = {
        "encoder.inner." + key[len(prefix):]: value
        for key, value in state["model_state"].items() if key.startswith(prefix)
    }
    return {**encoder, **state["decoder_state"]}


class SAM2EncoderAdapter(nn.Module):
    """Wraps SAM2's ImageEncoder so UNETR3D can call encoder(x)[0].

    SAM2's ImageEncoder returns a dict; UNETR3D expects integer-indexed access
    where index 0 is the primary feature tensor.
    """
    def __init__(self, sam2_image_encoder: nn.Module, img_size: int = 1024):
        super().__init__()
        self.inner = sam2_image_encoder
        self.img_size = img_size

    def forward(self, x: torch.Tensor):
        out = self.inner(x)
        return [out["vision_features"]]


class UniSAM2(UNETR3D):
    """UNETR-based model for universal (2d + 3d) segmentation.
    """
    def __init__(
        self,
        encoder: Union[str, nn.Module] = "hvit_t",
        output_channels: int = 4,
        img_size: int = 1024,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        device = torch.device("cpu") if device is None else torch.device(get_device(device))

        # One encoder type for both callers, so the weights land under the same keys either way.
        if isinstance(encoder, str):
            encoder = get_sam2_model(model_type=encoder, input_type="images", device=device).image_encoder

        super().__init__(
            img_size=img_size,
            backbone="sam2",
            encoder=SAM2EncoderAdapter(encoder, img_size=img_size),
            final_activation=CustomActivation(),
            out_channels=output_channels,
            use_sam_stats=True,
            embed_dim=256,
            use_strip_pooling=True,
            **kwargs
        )
        self.to(device)


class SemanticSAM2(UNETR3D):
    """UNETR-based model for semantic (2d + 3d) segmentation.

    The model has no final activation, so it returns the raw class logits that the semantic losses expect.
    """
    def __init__(
        self,
        encoder: Union[str, nn.Module] = "hvit_t",
        num_classes: int = 3,
        img_size: int = 1024,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        device = torch.device("cpu") if device is None else torch.device(get_device(device))

        # One encoder type for both callers, so the weights land under the same keys either way.
        if isinstance(encoder, str):
            encoder = get_sam2_model(model_type=encoder, input_type="images", device=device).image_encoder

        super().__init__(
            img_size=img_size,
            backbone="sam2",
            encoder=SAM2EncoderAdapter(encoder, img_size=img_size),
            final_activation=None,
            out_channels=num_classes,
            use_sam_stats=True,
            embed_dim=256,
            use_strip_pooling=True,
            **kwargs
        )
        self.to(device)
