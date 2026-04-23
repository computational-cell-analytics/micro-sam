import torch
import torch.nn as nn

from torch_em.model.unetr import UNETR3D

from micro_sam.v2.util import _get_checkpoint


class CustomActivation(nn.Module):
    """Applies 'Sigmoid' activation for channel 0 (i.e. foreground), and
    'Tanh' for the remaining channels (i.e. distances).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.sigmoid(x[:, :1]), torch.tanh(x[:, 1:])], dim=1)


class UniSAM2(UNETR3D):
    """UNETR-based model for universal (2d + 3d) segmentation.
    """
    def __init__(self, model_type: str = "hvit_t", output_channels: int = 4, **kwargs):
        super().__init__(
            img_size=1024,
            backbone="sam2",
            encoder=model_type,
            encoder_checkpoint=_get_checkpoint(model_type=model_type, backbone="sam2.1"),
            final_activation=CustomActivation(),
            out_channels=output_channels,
            use_sam_stats=True,
            embed_dim=256,
            use_strip_pooling=True,
            **kwargs
        )
        self.init_kwargs = {"model_type": model_type, "output_channels": output_channels}
