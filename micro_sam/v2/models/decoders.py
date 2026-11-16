"""Segmentation decoders for UniSAM2 that read the multi-scale SAM2 features.

The default UniSAM2 decoder (`decoder_type="unetr"`) decodes only the stride-16 `vision_features` and synthesises
the finer levels by upsampling. The decoders here also read the stride-8 and stride-4 features of the SAM2 image
encoder (after SAM2's `conv_s0` / `conv_s1` projections) and the preprocessed image:

- `"unetr_skip"`: the default decoder with one change, the synthesised pyramid (`deconv1` to `deconv4`) is replaced
  by projections of the real stride-8 and stride-4 features and an image stem for strides 2 and 1.
- `"hiera_u"`: a plain skip U-decoder built with the hybrid 2D/3D recipe: all spatial convolutions are 2D and run
  on the slices folded into the batch, normalisation is a GroupNorm with per-slice statistics, and the z context
  comes from zero-initialised residual branches that are skipped for a single slice. At initialisation a volume
  therefore decodes exactly to the stack of its decoded slices.
- `"cagd_3d"`: MedCAGD (Wazir et al., ECCV 2026, arXiv 2607.00409, github.com/saadwazir/MedCAGD) adapted to the
  hybrid recipe: gated skip fusion, context aggregators, refinement blocks with channel recalibration and deep
  supervision. All global pooling is per slice, the bottleneck gets attention along z and the blocks get
  zero-initialised z branches.
- `"c3f2"`: a coarse-3D / fine-2D fusion head. ConvNeXt blocks with attention along z at stride 16, with z
  convolutions at strides 8 and 4, and a purely 2D image-guided upsampling head.

`"hiera_u"`, `"cagd_3d"` and `"c3f2"` follow the same interface (`forward(features, image, n_z)` on z folded into
the batch) and are registered in `SEG_DECODERS`.
"""

import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_em.model.unetr import DepthStripPooling

#: The decoder types UniSAM2 can build.
DECODER_TYPES = ("unetr", "unetr_skip", "hiera_u", "cagd_3d", "c3f2")

#: The decoder types that need the stride-8 and stride-4 features of the encoder.
MULTI_SCALE_DECODER_TYPES = ("unetr_skip", "hiera_u", "cagd_3d", "c3f2")

# Channels of the SAM2 stride-4 and stride-8 features after `conv_s0` / `conv_s1`, and of the FPN levels before.
SAM2_FPN_CHANNELS = 256
SAM2_S4_CHANNELS = 32
SAM2_S8_CHANNELS = 64


class SAM2MultiScaleEncoderAdapter(nn.Module):
    """Wrap SAM2's image encoder so it returns the stride-16, stride-8 and stride-4 features.

    The stride-8 and stride-4 FPN levels are projected by SAM2's `conv_s1` and `conv_s0`, as SAM2's
    `forward_image` does for its mask decoder. These convolutions live on the SAM2 model, not on the image
    encoder, so the caller passes them in; without them they are initialised freshly (their weights then
    come from a checkpoint).

    Args:
        sam2_image_encoder: The SAM2 image encoder.
        conv_s0: The projection of the stride-4 FPN level (256 to 32 channels).
        conv_s1: The projection of the stride-8 FPN level (256 to 64 channels).
        img_size: The input size of the image encoder.
    """
    def __init__(
        self,
        sam2_image_encoder: nn.Module,
        conv_s0: Optional[nn.Module] = None,
        conv_s1: Optional[nn.Module] = None,
        img_size: int = 1024,
    ):
        super().__init__()
        self.inner = sam2_image_encoder
        self.conv_s0 = nn.Conv2d(SAM2_FPN_CHANNELS, SAM2_S4_CHANNELS, 1) if conv_s0 is None else conv_s0
        self.conv_s1 = nn.Conv2d(SAM2_FPN_CHANNELS, SAM2_S8_CHANNELS, 1) if conv_s1 is None else conv_s1
        self.img_size = img_size

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode a batch of images.

        Args:
            x: The preprocessed images, shape (N, 3, H, W).

        Returns:
            The features at stride 16 (256 channels), 8 (64 channels) and 4 (32 channels).
        """
        out = self.inner(x)
        s4, s8 = out["backbone_fpn"][0], out["backbone_fpn"][1]
        return {"s16": out["vision_features"], "s8": self.conv_s1(s8), "s4": self.conv_s0(s4)}


def fold_z(x: torch.Tensor) -> torch.Tensor:
    """Fold z into the batch: (B, C, Z, H, W) to (B * Z, C, H, W)."""
    b, c, z, h, w = x.shape
    return x.transpose(1, 2).reshape(b * z, c, h, w)


def unfold_z(x: torch.Tensor, n_z: int) -> torch.Tensor:
    """Unfold z from the batch: (B * Z, C, H, W) to (B, C, Z, H, W)."""
    bz, c, h, w = x.shape
    return x.reshape(bz // n_z, n_z, c, h, w).transpose(1, 2)


#
# Arm B: the default decoder with real skips.
#


class SkipProjection3d(nn.Module):
    """Map a feature or image level to the channels of a decoder skip.

    Built from the components of the default decoder's `Deconv3DBlock` without its upsampler: an in-plane
    (1, 3, 3) convolution, batch norm, ReLU and depth strip pooling.

    Args:
        in_channels: The input channels.
        out_channels: The output channels.
        stride: The in-plane stride.
        use_strip_pooling: Whether to add depth strip pooling.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_strip_pooling: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(True),
            DepthStripPooling(out_channels) if use_strip_pooling else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNETRSkipInputs(nn.Module):
    """The skip inputs of the `"unetr_skip"` decoder, replacing `deconv1` to `deconv4` of the default decoder.

    The default decoder feeds its levels at stride 8, 4, 2 with the middle 128, 64, 32 channels of the synthesised
    pyramid (its `_crop` drops the rest) and its output head with 32 channels at stride 1. These modules provide
    the same widths from the real features: stride 8 and 4 from the encoder, stride 2 and 1 from the image.

    Args:
        initial_features: The width of the default decoder.
        use_strip_pooling: Whether to add depth strip pooling.
    """
    def __init__(self, initial_features: int = 32, use_strip_pooling: bool = True):
        super().__init__()
        f = initial_features
        self.skip8 = SkipProjection3d(SAM2_S8_CHANNELS, 4 * f, use_strip_pooling=use_strip_pooling)
        self.skip4 = SkipProjection3d(SAM2_S4_CHANNELS, 2 * f, use_strip_pooling=use_strip_pooling)
        self.stem1 = SkipProjection3d(3, f, use_strip_pooling=use_strip_pooling)
        self.stem2 = SkipProjection3d(f, f, stride=2, use_strip_pooling=use_strip_pooling)

    def forward(self, s8: torch.Tensor, s4: torch.Tensor, image: torch.Tensor):
        """Compute the skips.

        Args:
            s8: The stride-8 features, (B, 64, Z, H/8, W/8).
            s4: The stride-4 features, (B, 32, Z, H/4, W/4).
            image: The preprocessed image, (B, 3, Z, H, W).

        Returns:
            The skips at stride 8, 4, 2 (decoder inputs) and the stride-1 skip of the output head.
        """
        z0 = self.stem1(image)
        z3 = self.stem2(z0)
        return [self.skip8(s8), self.skip4(s4), z3], z0


#
# Arm C: Hiera-U with the hybrid 2D/3D recipe.
#


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, channels), channels)


class ZResidual(nn.Module):
    """A residual branch along z: x + pointwise(GELU(depthwise_z(x))).

    The pointwise convolution is zero-initialised, so the branch starts as the identity. For a single slice it
    is skipped, so 2D inputs never see it.

    Args:
        channels: The channels.
        kernel_size: The kernel size along z.
    """
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.depthwise = nn.Conv3d(
            channels, channels, kernel_size=(kernel_size, 1, 1), padding=(kernel_size // 2, 0, 0), groups=channels,
        )
        self.act = nn.GELU()
        self.pointwise = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, x: torch.Tensor, n_z: int) -> torch.Tensor:
        """Apply the branch.

        Args:
            x: The features with z folded into the batch, (B * Z, C, H, W).
            n_z: The number of slices Z.

        Returns:
            The features, same shape.
        """
        if n_z == 1:
            return x
        volume = unfold_z(x, n_z)
        return x + fold_z(self.pointwise(self.act(self.depthwise(volume))))


class HybridBlock(nn.Module):
    """A residual 2D block with per-slice GroupNorm and an optional zero-initialised z branch.

    Args:
        in_channels: The input channels.
        out_channels: The output channels.
        with_z: Whether to add the z branch.
    """
    def __init__(self, in_channels: int, out_channels: int, with_z: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = _group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = _group_norm(out_channels)
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
        self.act = nn.GELU()
        self.z_branch = ZResidual(out_channels) if with_z else None

    def forward(self, x: torch.Tensor, n_z: int) -> torch.Tensor:
        y = self.act(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        y = self.act(y + self.shortcut(x))
        if self.z_branch is not None:
            y = self.z_branch(y, n_z)
        return y


class HieraUDecoder(nn.Module):
    """A plain skip U-decoder on the SAM2 stride-16 / 8 / 4 features and an image stem.

    All spatial operations are 2D on the slices folded into the batch. The z branches of the blocks at
    strides 16 to 2 carry the 3D context; the full-resolution block is 2D only.

    Args:
        out_channels: The output channels.
        widths: The decoder widths at strides 16, 8, 4, 2 and 1.
        stem_widths: The widths of the image stem at strides 1 and 2.
    """
    def __init__(
        self,
        out_channels: int = 5,
        widths: Sequence[int] = (192, 128, 64, 32, 16),
        stem_widths: Sequence[int] = (16, 32),
    ):
        super().__init__()
        w16, w8, w4, w2, w1 = widths
        st1, st2 = stem_widths
        self.stem1 = nn.Sequential(nn.Conv2d(3, st1, 3, padding=1, bias=False), _group_norm(st1), nn.GELU())
        self.stem2 = nn.Sequential(
            nn.Conv2d(st1, st2, 3, stride=2, padding=1, bias=False), _group_norm(st2), nn.GELU(),
        )
        self.block16 = HybridBlock(SAM2_FPN_CHANNELS, w16)
        self.block8 = HybridBlock(w16 + SAM2_S8_CHANNELS, w8)
        self.block4 = HybridBlock(w8 + SAM2_S4_CHANNELS, w4)
        self.block2 = HybridBlock(w4 + st2, w2)
        self.block1 = HybridBlock(w2 + st1, w1, with_z=False)
        self.out_conv = nn.Conv2d(w1, out_channels, 1)

    @staticmethod
    def _up(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, features: Dict[str, torch.Tensor], image: torch.Tensor, n_z: int) -> torch.Tensor:
        """Decode.

        Args:
            features: The encoder features 's16', 's8', 's4' with z folded into the batch.
            image: The preprocessed image with z folded into the batch, (B * Z, 3, H, W).
            n_z: The number of slices Z.

        Returns:
            The decoder output before the activation, (B * Z, out_channels, H, W).
        """
        stem1 = self.stem1(image)
        stem2 = self.stem2(stem1)
        x = self.block16(features["s16"], n_z)
        x = self.block8(self._up(x, features["s8"]), n_z)
        x = self.block4(self._up(x, features["s4"]), n_z)
        x = self.block2(self._up(x, stem2), n_z)
        x = self.block1(self._up(x, stem1), n_z)
        return self.out_conv(x)


#
# Shared hybrid blocks of arms D and E.
#


class ZAttention(nn.Module):
    """Residual self-attention along z at every xy position, with a learned relative position bias.

    The output projection is zero-initialised, so the block starts as the identity; for a single slice it is
    skipped.

    Args:
        channels: The channels.
        n_heads: The number of attention heads; by default one head per 64 channels.
        max_distance: The largest z distance with its own position bias; larger distances share the last one.
    """
    def __init__(self, channels: int, n_heads: Optional[int] = None, max_distance: int = 16):
        super().__init__()
        self.n_heads = n_heads or max(1, channels // 64)
        if channels % self.n_heads != 0:
            raise ValueError(f"{channels} channels cannot be split into {self.n_heads} heads.")
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.max_distance = max_distance
        self.position_bias = nn.Parameter(torch.zeros(self.n_heads, 2 * max_distance + 1))

    def forward(self, x: torch.Tensor, n_z: int) -> torch.Tensor:
        """Apply the attention.

        Args:
            x: The features with z folded into the batch, (B * Z, C, H, W).
            n_z: The number of slices Z.

        Returns:
            The features, same shape.
        """
        if n_z == 1:
            return x
        bz, c, h, w = x.shape
        b = bz // n_z
        tokens = x.reshape(b, n_z, c, h, w).permute(0, 3, 4, 1, 2).reshape(b * h * w, n_z, c)
        qkv = self.qkv(self.norm(tokens)).reshape(-1, n_z, 3, self.n_heads, c // self.n_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        index = torch.arange(n_z, device=x.device)
        distance = (index[None] - index[:, None]).clamp(-self.max_distance, self.max_distance) + self.max_distance
        bias = self.position_bias[:, distance].to(q.dtype)[None]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        out = self.proj(out.transpose(1, 2).reshape(-1, n_z, c))
        out = out.reshape(b, h, w, n_z, c).permute(0, 3, 4, 1, 2).reshape(bz, c, h, w)
        return x + out


def _cagd_group_norm(channels: int) -> nn.GroupNorm:
    """GroupNorm with the group count of MedCAGD (4 to 32 channels per group). Per slice on folded inputs."""
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0 and 4 <= channels // groups <= 32:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


def _resize_to(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == reference.shape[-2:]:
        return x
    return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)


#
# Arm D: CAGD-3D, MedCAGD adapted to the hybrid 2D/3D recipe.
#


class ECA(nn.Module):
    """Efficient channel attention of MedCAGD with a learned temperature, pooled per slice.

    MedCAGD averages three convolutions of the global mean (its pooling scales 1, 2, 4 all reduce to the global
    mean); being linear, this equals one convolution, which is used here.

    Args:
        channels: The channels.
        is_encoder: Selects MedCAGD's kernel size rule for encoder (skip) or decoder features.
    """
    def __init__(self, channels: int, is_encoder: bool = True):
        super().__init__()
        gamma, b = (2.0, 1.0) if is_encoder else (2.4, 0.8)
        k = int(abs(math.log2(channels) / gamma + b))
        k = k + 1 if k % 2 == 0 else k
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.log_tau = nn.Parameter(torch.zeros(()))
        self.tau_min, self.tau_max = 0.5, 2.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x.mean(dim=(2, 3))[:, None])[:, 0]  # (N, C), N = slices
        tau = self.log_tau.exp().clamp(self.tau_min, self.tau_max)
        attention = torch.sigmoid(logits / tau)
        attention = attention / (attention.mean(dim=1, keepdim=True) + 1e-6)
        return x * attention[..., None, None]


class GlobalContext(nn.Module):
    """MedCAGD's residual global-context attention (a GCNet block), pooled per slice.

    Args:
        channels: The channels.
        reduction: The channel reduction of the transform.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        inter = max(4, channels // reduction)  # MedCAGD: max(1, ...); same for the default widths (>= 64)
        self.mask = nn.Conv2d(channels, 1, kernel_size=1)
        self.transform = nn.Sequential(
            nn.Conv2d(channels, inter, kernel_size=1, bias=False),
            nn.GroupNorm(1, inter),  # = LayerNorm([inter, 1, 1]) of MedCAGD
            nn.ReLU(inplace=True),
            nn.Conv2d(inter, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        weights = torch.softmax(self.mask(x).reshape(n, 1, h * w), dim=-1)
        context = torch.bmm(x.reshape(n, c, h * w), weights.transpose(1, 2)).reshape(n, c, 1, 1)
        return x + self.transform(context)


class SCAGate(nn.Module):
    """MedCAGD's attention gate (SCA-Gate): gates the skip features with guidance from the decoder.

    Changed for 1024 px inputs: the spatial competition is a softmax over the H * W positions of a slice, whose
    values are about 1 / (H * W), i.e. numerically zero at this resolution. It is rescaled by H * W (mean one) and
    clamped at 4.

    Args:
        guide_channels: The channels of the upsampled decoder features.
        skip_channels: The channels of the skip features.
        inter_channels: The latent width.
    """
    def __init__(self, guide_channels: int, skip_channels: int, inter_channels: int, max_spatial: float = 4.0):
        super().__init__()
        self.eca_skip = ECA(skip_channels, is_encoder=True)
        self.eca_guide = ECA(guide_channels, is_encoder=False)

        def branch(channels):
            return nn.Sequential(
                nn.Conv2d(channels, inter_channels, 3, padding=1, bias=False), _cagd_group_norm(inter_channels),
                nn.SiLU(),
                nn.Conv2d(inter_channels, inter_channels, 3, padding=2, dilation=2, groups=inter_channels,
                          bias=False),
                _cagd_group_norm(inter_channels), nn.SiLU(),
            )
        self.theta, self.phi = branch(guide_channels), branch(skip_channels)
        self.global_fc = nn.Sequential(
            nn.Conv2d(guide_channels + skip_channels, inter_channels, 1), nn.SiLU(), nn.Sigmoid(),
        )
        self.dw3 = nn.Conv2d(inter_channels, inter_channels, 3, padding=1, groups=inter_channels, bias=False)
        self.dw5 = nn.Conv2d(inter_channels, inter_channels, 5, padding=2, groups=inter_channels, bias=False)
        self.score = nn.Conv2d(inter_channels, 1, 1, bias=False)
        self.log_tau = nn.Parameter(torch.tensor(0.7).log())
        self.tau_min, self.tau_max = 0.3, 2.0
        self.max_spatial = max_spatial
        self.att = nn.Sequential(nn.GroupNorm(1, inter_channels), nn.Conv2d(inter_channels, 1, 1), nn.Hardsigmoid())

    def forward(self, guide: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        skip = self.eca_skip(skip)
        guide = self.eca_guide(guide)
        fused = F.normalize(self.theta(guide) * self.phi(skip), p=2, dim=1)
        fused = fused + self.global_fc(torch.cat([guide, skip], dim=1).mean(dim=(2, 3), keepdim=True)) * fused
        score = self.score(self.dw3(fused) + self.dw5(fused))
        n, _, h, w = score.shape
        tau = self.log_tau.exp().clamp(self.tau_min, self.tau_max)
        spatial = torch.softmax(score.reshape(n, h * w) / tau, dim=-1).reshape(n, 1, h, w) * (h * w)
        fused = fused * (1.0 + spatial.clamp(max=self.max_spatial))
        return skip * self.att(fused + 1e-5)


class DWBlock(nn.Module):
    """MedCAGD's conv block (depth-wise 3x3 / point-wise, twice) followed by a zero-initialised z branch.

    Args:
        in_channels: The input channels.
        out_channels: The output channels.
        with_z: Whether to add the z branch.
    """
    def __init__(self, in_channels: int, out_channels: int, with_z: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            _cagd_group_norm(in_channels), nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 1, bias=False), _cagd_group_norm(out_channels), nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
            _cagd_group_norm(out_channels), nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 1, bias=False), _cagd_group_norm(out_channels), nn.SiLU(),
        )
        self.z_branch = ZResidual(out_channels) if with_z else None

    def forward(self, x: torch.Tensor, n_z: int) -> torch.Tensor:
        x = self.block(x)
        return x if self.z_branch is None else self.z_branch(x, n_z)


class RefinementBlock(nn.Module):
    """MedCAGD's refinement block: conv block, optional depth-wise spatial conv, channel recalibration."""
    def __init__(self, channels: int, spatial_focus: bool, with_z: bool = True):
        super().__init__()
        self.conv = DWBlock(channels, channels, with_z=with_z)
        self.spatial = nn.Conv2d(channels, channels, 3, padding=1, groups=channels) if spatial_focus else None
        self.eca = ECA(channels)

    def forward(self, x: torch.Tensor, n_z: int) -> torch.Tensor:
        x = self.conv(x, n_z)
        if self.spatial is not None:
            x = self.spatial(x)
        return self.eca(x)


class CAGDStage(nn.Module):
    """MedCAGD's decoder block: transposed-conv upsampling, gated skip fusion (for skips wider than 64 channels as
    in MedCAGD) and the conv block."""
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, with_z: bool = True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.gate = SCAGate(out_channels, skip_channels, max(1, skip_channels // 2)) if skip_channels > 64 else None
        self.conv = DWBlock(out_channels + skip_channels, out_channels, with_z=with_z)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, n_z: int) -> torch.Tensor:
        x = _resize_to(self.up(x), skip)
        if self.gate is not None:
            skip = self.gate(x, skip)
        return self.conv(torch.cat([x, skip], dim=1), n_z)


class ContextAggregator(nn.Module):
    """MedCAGD's context aggregator: all pyramid levels projected, resized, averaged, plus global context."""
    def __init__(self, in_channels: Sequence[int], out_channels: int):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels])
        self.global_context = GlobalContext(out_channels)

    def forward(self, features: Sequence[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
        fused = sum(_resize_to(proj(f), reference) for f, proj in zip(features, self.proj))
        return self.global_context(fused / len(features))


class CAGD3DDecoder(nn.Module):
    """MedCAGD's decoder on the SAM2 features, adapted to the hybrid 2D/3D recipe.

    Pyramid: 1x1 projections of the stride-4 / 8 / 16 features to 64 / 128 / 320 channels; SAM2 has no stride-32
    level, so it is a strided 3x3 convolution of the stride-16 features (512 channels). The bottleneck (conv,
    attention along z, global context) and the stages at strides 16, 8 and 4 (gated fusion, context aggregator,
    refinement) follow MedCAGD. Changed against MedCAGD: the last stage runs at stride 2 with an image-stem skip
    instead of at full resolution (MedCAGD's full-resolution stage costs 1.7x the FLOPs), followed by a transposed
    convolution to full resolution fused with the stride-1 stem.

    Deep supervision: 1x1 heads on the stages at strides 16, 8 and 4 predict all output channels (the boundary
    channel takes the role of MedCAGD's separate edge heads). In training mode their outputs, upsampled to the
    image size as in MedCAGD, are kept in `aux_outputs` (deepest first, MedCAGD weights 0.4 / 0.3 / 0.2).

    Args:
        out_channels: The output channels.
        widths: The pyramid widths at strides 4, 8, 16 and 32.
        stem_widths: The widths of the image stem at strides 1 and 2.
        deep_supervision: Whether to build the deep supervision heads.
    """
    def __init__(
        self,
        out_channels: int = 5,
        widths: Sequence[int] = (64, 128, 320, 512),
        stem_widths: Sequence[int] = (16, 32),
        deep_supervision: bool = True,
    ):
        super().__init__()
        w4, w8, w16, w32 = widths
        st1, st2 = stem_widths
        self.stem1 = nn.Sequential(nn.Conv2d(3, st1, 3, padding=1, bias=False), _cagd_group_norm(st1), nn.SiLU())
        self.stem2 = nn.Sequential(
            nn.Conv2d(st1, st2, 3, stride=2, padding=1, bias=False), _cagd_group_norm(st2), nn.SiLU(),
        )
        self.p4 = nn.Conv2d(SAM2_S4_CHANNELS, w4, 1)
        self.p8 = nn.Conv2d(SAM2_S8_CHANNELS, w8, 1)
        self.p16 = nn.Conv2d(SAM2_FPN_CHANNELS, w16, 1)
        self.p32 = nn.Conv2d(SAM2_FPN_CHANNELS, w32, 3, stride=2, padding=1)

        self.bottleneck_proj = nn.Conv2d(w32, w32, 1)
        self.bottleneck_conv = nn.Sequential(nn.Conv2d(w32, w32, 3, padding=1), _cagd_group_norm(w32), nn.SiLU())
        self.bottleneck_z = ZAttention(w32)
        self.bottleneck_context = GlobalContext(w32)

        levels = (w4, w8, w16, w32)
        self.stage16, self.context16, self.ref16 = (
            CAGDStage(w32, w16, w16), ContextAggregator(levels, w16), RefinementBlock(w16, spatial_focus=False),
        )
        self.stage8, self.context8, self.ref8 = (
            CAGDStage(w16, w8, w8), ContextAggregator(levels, w8), RefinementBlock(w8, spatial_focus=True),
        )
        self.stage4, self.context4, self.ref4 = (
            CAGDStage(w8, w4, w4), ContextAggregator(levels, w4), RefinementBlock(w4, spatial_focus=True),
        )
        self.stage2 = CAGDStage(w4, st2, st2)
        self.ref2 = RefinementBlock(st2, spatial_focus=True)

        self.head_up = nn.Sequential(
            nn.ConvTranspose2d(st2, st1, kernel_size=2, stride=2), _cagd_group_norm(st1), nn.SiLU(),
        )
        self.head_fuse = nn.Sequential(
            nn.Conv2d(2 * st1, st1, 3, padding=1, bias=False), _cagd_group_norm(st1), nn.SiLU(),
        )
        self.out_conv = nn.Conv2d(st1, out_channels, 1)
        self.aux_heads = nn.ModuleList(
            [nn.Conv2d(c, out_channels, 1) for c in (w16, w8, w4)]
        ) if deep_supervision else None
        self.aux_outputs: Optional[List[torch.Tensor]] = None

    def forward(self, features: Dict[str, torch.Tensor], image: torch.Tensor, n_z: int) -> torch.Tensor:
        """Decode.

        Args:
            features: The encoder features 's16', 's8', 's4' with z folded into the batch.
            image: The preprocessed image with z folded into the batch, (B * Z, 3, H, W).
            n_z: The number of slices Z.

        Returns:
            The decoder output before the activation, (B * Z, out_channels, H, W).
        """
        stem1 = self.stem1(image)
        stem2 = self.stem2(stem1)
        p4, p8, p16 = self.p4(features["s4"]), self.p8(features["s8"]), self.p16(features["s16"])
        b = self.bottleneck_conv(self.bottleneck_proj(self.p32(features["s16"])))
        b = self.bottleneck_context(self.bottleneck_z(b, n_z))
        levels = [p4, p8, p16, b]

        d16 = self.stage16(b, p16, n_z)
        d16 = self.ref16(d16 + self.context16(levels, d16), n_z)
        d8 = self.stage8(d16, p8, n_z)
        d8 = self.ref8(d8 + self.context8(levels, d8), n_z)
        d4 = self.stage4(d8, p4, n_z)
        d4 = self.ref4(d4 + self.context4(levels, d4), n_z)
        d2 = self.ref2(self.stage2(d4, stem2, n_z), n_z)

        x = _resize_to(self.head_up(d2), stem1)
        out = self.out_conv(self.head_fuse(torch.cat([x, stem1], dim=1)))

        self.aux_outputs = None
        if self.training and self.aux_heads is not None:
            self.aux_outputs = [
                _resize_to(head(d), image) for head, d in zip(self.aux_heads, (d16, d8, d4))
            ]
        return out


#
# Arm E: C3F2, coarse-3D / fine-2D fusion head.
#


class NeXtBlock(nn.Module):
    """A ConvNeXt block in xy (7x7 depth-wise, per-slice norm, inverted bottleneck), then an optional z module.

    Args:
        channels: The channels.
        z_mode: 'attention' (`ZAttention`), 'conv' (`ZResidual`) or None.
    """
    def __init__(self, channels: int, z_mode: Optional[str] = None):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.GroupNorm(1, channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, 4 * channels, 1), nn.GELU(), nn.Conv2d(4 * channels, channels, 1),
        )
        if z_mode == "attention":
            self.z_module = ZAttention(channels)
        elif z_mode == "conv":
            self.z_module = ZResidual(channels)
        elif z_mode is None:
            self.z_module = None
        else:
            raise ValueError(f"Invalid z_mode {z_mode}")

    def forward(self, x: torch.Tensor, n_z: int) -> torch.Tensor:
        x = x + self.mlp(self.norm(self.depthwise(x)))
        return x if self.z_module is None else self.z_module(x, n_z)


class C3F2Decoder(nn.Module):
    """Coarse-3D / fine-2D fusion head on the SAM2 features.

    Stride 16: two ConvNeXt blocks, each followed by attention along z; all long-range z reasoning happens here.
    Stride 8 and 4: additive top-down fusion with the lateral features and ConvNeXt blocks with z convolutions
    (the last block at stride 4 is 2D). Finer: two transposed convolutions to full resolution, each fused with the
    image stem, purely 2D.

    Args:
        out_channels: The output channels.
        widths: The widths at strides 16, 8 and 4.
        head_width: The width of the stride-2 head; the stride-1 head uses the stride-1 stem width.
        stem_widths: The widths of the image stem at strides 1 and 2.
    """
    def __init__(
        self,
        out_channels: int = 5,
        widths: Sequence[int] = (256, 128, 64),
        head_width: int = 32,
        stem_widths: Sequence[int] = (16, 32),
    ):
        super().__init__()
        w16, w8, w4 = widths
        st1, st2 = stem_widths
        self.stem1 = nn.Sequential(nn.Conv2d(3, st1, 3, padding=1, bias=False), _group_norm(st1), nn.GELU())
        self.stem2 = nn.Sequential(
            nn.Conv2d(st1, st2, 3, stride=2, padding=1, bias=False), _group_norm(st2), nn.GELU(),
        )
        self.lateral16 = nn.Conv2d(SAM2_FPN_CHANNELS, w16, 1)
        self.blocks16 = nn.ModuleList([NeXtBlock(w16, "attention"), NeXtBlock(w16, "attention")])
        self.lateral8 = nn.Conv2d(SAM2_S8_CHANNELS, w8, 1)
        self.reduce8 = nn.Conv2d(w16, w8, 1)
        self.blocks8 = nn.ModuleList([NeXtBlock(w8, "conv"), NeXtBlock(w8, "conv")])
        self.lateral4 = nn.Conv2d(SAM2_S4_CHANNELS, w4, 1)
        self.reduce4 = nn.Conv2d(w8, w4, 1)
        self.blocks4 = nn.ModuleList([NeXtBlock(w4, "conv"), NeXtBlock(w4, None)])
        self.up2 = nn.Sequential(nn.ConvTranspose2d(w4, head_width, 2, stride=2), _group_norm(head_width), nn.GELU())
        self.fuse2 = nn.Sequential(
            nn.Conv2d(head_width + st2, head_width, 3, padding=1, bias=False), _group_norm(head_width), nn.GELU(),
        )
        self.up1 = nn.Sequential(nn.ConvTranspose2d(head_width, st1, 2, stride=2), _group_norm(st1), nn.GELU())
        self.fuse1 = nn.Sequential(
            nn.Conv2d(2 * st1, st1, 3, padding=1, bias=False), _group_norm(st1), nn.GELU(),
        )
        self.out_conv = nn.Conv2d(st1, out_channels, 1)

    @staticmethod
    def _run(blocks, x, n_z):
        for block in blocks:
            x = block(x, n_z)
        return x

    def forward(self, features: Dict[str, torch.Tensor], image: torch.Tensor, n_z: int) -> torch.Tensor:
        """Decode.

        Args:
            features: The encoder features 's16', 's8', 's4' with z folded into the batch.
            image: The preprocessed image with z folded into the batch, (B * Z, 3, H, W).
            n_z: The number of slices Z.

        Returns:
            The decoder output before the activation, (B * Z, out_channels, H, W).
        """
        stem1 = self.stem1(image)
        stem2 = self.stem2(stem1)
        x = self._run(self.blocks16, self.lateral16(features["s16"]), n_z)
        lateral = self.lateral8(features["s8"])
        x = self._run(self.blocks8, _resize_to(self.reduce8(x), lateral) + lateral, n_z)
        lateral = self.lateral4(features["s4"])
        x = self._run(self.blocks4, _resize_to(self.reduce4(x), lateral) + lateral, n_z)
        x = self.fuse2(torch.cat([_resize_to(self.up2(x), stem2), stem2], dim=1))
        x = self.fuse1(torch.cat([_resize_to(self.up1(x), stem1), stem1], dim=1))
        return self.out_conv(x)


#: The decoders with the `forward(features, image, n_z)` interface, by decoder type.
SEG_DECODERS = {"hiera_u": HieraUDecoder, "cagd_3d": CAGD3DDecoder, "c3f2": C3F2Decoder}
