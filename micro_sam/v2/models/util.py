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

    Args:
        encoder: The SAM2 model type, e.g. 'hvit_t', or a SAM2 image encoder module.
        output_channels: The number of output channels.
        img_size: The input size of the image encoder.
        device: The device of the model.
        decoder_type: The decoder, one of `micro_sam.v2.models.decoders.DECODER_TYPES`. 'unetr' (the default) is the
            UNETR3D decoder on the stride-16 features; 'unetr_skip', 'hiera_u', 'cagd_3d' and 'c3f2' also read
            the stride-8 and stride-4 features. Their projections `conv_s0` / `conv_s1` are taken from the SAM2
            model when the encoder is given by name, and are initialised freshly for an encoder module.
        decoder_kwargs: Keyword arguments for the decoders in `decoders.SEG_DECODERS` ('hiera_u', 'cagd_3d',
            'c3f2'), e.g. their widths.
        kwargs: Further keyword arguments for `UNETR3D`, e.g. 'initial_features'.
    """
    def __init__(
        self,
        encoder: Union[str, nn.Module] = "hvit_t",
        output_channels: int = 4,
        img_size: int = 1024,
        device: Optional[Union[str, torch.device]] = None,
        decoder_type: str = "unetr",
        decoder_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        from . import decoders

        if decoder_type not in decoders.DECODER_TYPES:
            raise ValueError(f"Invalid decoder_type '{decoder_type}', choose from {decoders.DECODER_TYPES}.")
        if decoder_kwargs and decoder_type not in decoders.SEG_DECODERS:
            raise ValueError(f"The decoder '{decoder_type}' takes no decoder_kwargs, got {decoder_kwargs}.")
        device = torch.device("cpu") if device is None else torch.device(get_device(device))

        # One encoder type for both callers, so the weights land under the same keys either way.
        conv_s0 = conv_s1 = None
        if isinstance(encoder, str):
            sam2_model = get_sam2_model(model_type=encoder, input_type="images", device=device)
            encoder = sam2_model.image_encoder
            conv_s0, conv_s1 = sam2_model.sam_mask_decoder.conv_s0, sam2_model.sam_mask_decoder.conv_s1

        if decoder_type in decoders.MULTI_SCALE_DECODER_TYPES:
            adapter = decoders.SAM2MultiScaleEncoderAdapter(encoder, conv_s0, conv_s1, img_size=img_size)
        else:
            adapter = SAM2EncoderAdapter(encoder, img_size=img_size)

        super().__init__(
            img_size=img_size,
            backbone="sam2",
            encoder=adapter,
            final_activation=CustomActivation(),
            out_channels=output_channels,
            use_sam_stats=True,
            embed_dim=256,
            use_strip_pooling=True,
            **kwargs
        )
        self.decoder_type = decoder_type
        self.decoder_kwargs = dict(decoder_kwargs or {})

        if decoder_type == "unetr_skip":
            # The synthesised pyramid is replaced by the real features; the rest of the decoder stays.
            for name in ("deconv1", "deconv2", "deconv3", "deconv4"):
                delattr(self, name)
            self.skip_inputs = decoders.UNETRSkipInputs(self.initial_features)
        elif decoder_type in decoders.SEG_DECODERS:
            for name in ("deconv1", "deconv2", "deconv3", "deconv4", "base", "decoder", "deconv_out",
                         "decoder_head", "out_conv"):
                delattr(self, name)
            self.seg_decoder = decoders.SEG_DECODERS[decoder_type](out_channels=output_channels, **self.decoder_kwargs)
        # The deep supervision outputs of the last forward pass in training mode (activated, at the output shape),
        # for decoders that have them ('cagd_3d'); None otherwise.
        self.auxiliary_outputs = None

        self.init_kwargs = {
            "encoder": encoder, "output_channels": output_channels, "img_size": img_size,
            "decoder_type": decoder_type, "decoder_kwargs": self.decoder_kwargs,
            **{key: value for key, value in kwargs.items() if key != "encoder_checkpoint"},
        }
        self.to(device)

    @property
    def decoder_config(self) -> dict:
        """The arguments that rebuild this decoder, recorded in checkpoints."""
        return {
            "decoder_type": self.decoder_type, "decoder_kwargs": dict(self.decoder_kwargs),
            "output_channels": self.out_channels, "initial_features": self.initial_features,
        }

    def encode(self, x: torch.Tensor):
        """Preprocess the input and run the image encoder on every z-slice.

        Args:
            x: The input tensor with shape (B, C, Z, Y, X).

        Returns:
            The encoder features and the spatial shape after preprocessing. For the default decoder the features
            are the stride-16 features (B, 256, Z, Y', X'); for the multi-scale decoders they are a dict of the
            stride-16 / 8 / 4 features and the preprocessed image, with z folded into the batch.
        """
        if self.decoder_type == "unetr":
            return super().encode(x)

        from .decoders import fold_z

        n_z = x.shape[2]
        x, input_shape = self.preprocess(x)
        image = fold_z(x)
        features = self.encoder(image)
        features["image"] = image
        features["n_z"] = n_z
        return features, input_shape

    def decode(self, features, input_shape, original_shape) -> torch.Tensor:
        """Run the decoder on the encoder features.

        Args:
            features: The encoder features returned by `encode`.
            input_shape: The spatial shape (Z, Y, X) after preprocessing.
            original_shape: The spatial shape (Z, Y, X) of the original input.

        Returns:
            The output, resized to `original_shape`.
        """
        if self.decoder_type == "unetr":
            return super().decode(features, input_shape, original_shape)

        from .decoders import unfold_z

        n_z = features["n_z"]
        self.auxiliary_outputs = None
        if self.decoder_type != "unetr_skip":
            x = unfold_z(self.seg_decoder(features, features["image"], n_z), n_z)
            aux = getattr(self.seg_decoder, "aux_outputs", None)
            if self.training and aux:
                self.auxiliary_outputs = [
                    self.postprocess_masks(self._activate(unfold_z(a, n_z)), input_shape, original_shape)
                    for a in aux
                ]
        else:
            skips, z0 = self.skip_inputs(
                unfold_z(features["s8"], n_z), unfold_z(features["s4"], n_z), unfold_z(features["image"], n_z),
            )
            x = self.base(unfold_z(features["s16"], n_z))
            x = self.decoder(x, encoder_inputs=skips)
            x = self.deconv_out(x)
            x = self.decoder_head(torch.cat([x, z0], dim=1))
            x = self.out_conv(x)

        return self.postprocess_masks(self._activate(x), input_shape, original_shape)

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.final_activation is None else self.final_activation(x)


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
        self.init_kwargs = {
            "encoder": encoder, "num_classes": num_classes, "img_size": img_size,
            **{key: value for key, value in kwargs.items() if key != "encoder_checkpoint"},
        }
        self.to(device)
