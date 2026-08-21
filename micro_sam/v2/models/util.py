from typing import Union

import torch
import torch.nn as nn

from torch_em.model.unetr import UNETR3D

from micro_sam.v2.util import get_sam2_model


class CustomActivation(nn.Module):
    """Applies 'Sigmoid' activation for channel 0 (i.e. foreground), and
    'Tanh' for the remaining channels (i.e. distances).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.sigmoid(x[:, :1]), torch.tanh(x[:, 1:])], dim=1)


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
        self, encoder: Union[str, nn.Module] = "hvit_t", output_channels: int = 4, img_size: int = 1024, **kwargs
    ):
        # One encoder type for both callers, so the weights land under the same keys either way.
        if isinstance(encoder, str):
            encoder = get_sam2_model(model_type=encoder, input_type="images").image_encoder

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
