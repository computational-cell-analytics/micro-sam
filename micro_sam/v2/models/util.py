from typing import Optional, Union

import torch
import torch.nn as nn

from torch_em.model.unetr import UNETR3D

from micro_sam.util import get_device
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

    Args:
        encoder: The SAM2 backbone name, e.g. 'hvit_t', or a prebuilt SAM2 image encoder.
        output_channels: The number of output channels (foreground + directed distances).
        img_size: The input size the encoder expects.
        device: The device to build the model on.
        initial_features: Width of the convolutional decoder: the features per level are
            'initial_features * 2 ** i'. None keeps torch_em's default width (64). The joint
            checkpoints from 2026-08 on were trained at 32; a torch_em that does not take the
            width as an argument gets its decoder rebuilt here, so the same checkpoints load
            regardless of the installed version.
        kwargs: Forwarded to `torch_em.model.unetr.UNETR3D`.
    """
    def __init__(
        self,
        encoder: Union[str, nn.Module] = "hvit_t",
        output_channels: int = 4,
        img_size: int = 1024,
        device: Optional[Union[str, torch.device]] = None,
        initial_features: Optional[int] = None,
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
            **({} if initial_features is None else {"initial_features": initial_features}),
            **kwargs
        )
        if initial_features is not None and self.out_conv.in_channels != initial_features:
            self._rebuild_decoder(initial_features, output_channels)
        self.to(device)

    def _rebuild_decoder(self, initial_features: int, output_channels: int) -> None:
        """Rebuild the convolutional decoder at another width, mirroring `UNETR3D.__init__`.

        torch_em 0.10 fixes the decoder width at 64 and ignores the argument; the blocks are the
        library's own, so a rebuilt decoder loads a checkpoint trained at that width unchanged.
        """
        from functools import partial
        from torch_em.model.unet import Decoder, Upsampler3d
        from torch_em.model.unetr import ConvBlock3dWithStrip, Deconv3DBlock

        embed_dim, depth, gain, scale_factors, use_strip_pooling = 256, 3, 2, [1, 2, 2], True
        features = [initial_features * gain ** i for i in range(depth + 1)][::-1]
        deconv = partial(Deconv3DBlock, scale_factor=scale_factors, use_strip_pooling=use_strip_pooling)
        self.deconv1 = deconv(in_channels=embed_dim, out_channels=features[0])
        self.deconv2 = deconv(in_channels=features[0], out_channels=features[1])
        self.deconv3 = deconv(in_channels=features[1], out_channels=features[2])
        self.deconv4 = deconv(in_channels=features[2], out_channels=features[3])
        self.decoder = Decoder(
            features=features,
            scale_factors=[scale_factors] * depth,
            conv_block_impl=partial(ConvBlock3dWithStrip, use_strip_pooling=use_strip_pooling),
            sampler_impl=Upsampler3d,
        )
        self.deconv_out = deconv(in_channels=features[-1], out_channels=features[-1])
        self.base = ConvBlock3dWithStrip(
            in_channels=embed_dim, out_channels=features[0], use_strip_pooling=use_strip_pooling,
        )
        self.decoder_head = ConvBlock3dWithStrip(
            in_channels=2 * features[-1], out_channels=features[-1], use_strip_pooling=use_strip_pooling,
        )
        self.out_conv = nn.Conv3d(features[-1], output_channels, 1)
