import os
from contextlib import nullcontext
from typing import Any, List, Dict, Union, Optional

import torch
import torch.nn as nn

from ..util import get_sam_model
from .peft_sam import LoRASurgery


def get_simple_sam_3d_model(
    device: Union[str, torch.device],
    n_classes: int,
    image_size: int,
    lora_rank: Optional[int] = None,
    freeze_encoder: bool = False,
    model_type: str = "vit_b",
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
):
    if lora_rank is None:
        peft_kwargs = {}
    else:
        peft_kwargs = {"rank": lora_rank, "peft_module": LoRASurgery}

    _, sam = get_sam_model(
        model_type=model_type,
        device=device,
        checkpoint_path=checkpoint_path,
        return_sam=True,
        image_size=image_size,
        flexible_load_checkpoint=True,
        peft_kwargs=peft_kwargs,
    )

    # Make sure not to freeze the encoder when using LoRA.
    freeze_encoder_ = freeze_encoder if lora_rank is None else False
    sam_3d = SimpleSam3DWrapper(sam, num_classes=n_classes, freeze_encoder=freeze_encoder_)
    sam_3d.to(device)

    return sam_3d


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3, 3),
        stride=(1, 1, 1),
        padding=(1, 1, 1),
        bias=True,
        mode="nearest"
    ):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm3d(out_channels)
        )

        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=bias),
            nn.InstanceNorm3d(out_channels)
        )

        self.leakyrelu = nn.LeakyReLU()

        self.up = nn.Upsample(scale_factor=(1, 2, 2), mode=mode)

    def forward(self, x):
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.conv2(out)
        out += residual

        out = self.leakyrelu(out)
        out = self.up(out)
        return out


class SegmentationHead(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3, 3),
        stride=(1, 1, 1),
        padding=(1, 1, 1),
        bias=True
    ):
        super().__init__()

        self.conv_pred = nn.Sequential(
            nn.Conv3d(
                in_channels, in_channels // 2, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
            ),
            nn.InstanceNorm3d(in_channels // 2),
            nn.LeakyReLU()
        )
        self.segmentation_head = nn.Conv3d(in_channels // 2, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv_pred(x)
        return self.segmentation_head(x)


class SimpleSam3DWrapper(nn.Module):
    def __init__(self, sam, num_classes, freeze_encoder):
        super().__init__()

        self.sam = sam
        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False
            self.no_grad = torch.no_grad

        else:
            self.no_grad = nullcontext

        self.decoders = nn.ModuleList([
            BasicBlock(in_channels=256, out_channels=128),
            BasicBlock(in_channels=128, out_channels=64),
            BasicBlock(in_channels=64, out_channels=32),
            BasicBlock(in_channels=32, out_channels=16),
        ])
        self.out_conv = SegmentationHead(in_channels=16, out_channels=num_classes)

    def _apply_image_encoder(self, x, D):
        encoder_features = []
        for d in range(D):
            image = x[:, :, d]
            feature = self.sam.image_encoder(image)
            encoder_features.append(feature)
        encoder_features = torch.stack(encoder_features, 2)
        return encoder_features

    def forward(
        self, batched_input: List[Dict[str, Any]], multimask_output: bool
    ) -> List[Dict[str, torch.Tensor]]:
        """Predict 3D masks for the current inputs.

        Unlike original SAM this model only supports automatic segmentation and does not support prompts.

        Args:
            batched_input: A list over input images, each a dictionary with the following keys.L
                'image': The image as a torch tensor in 3xDxHxW format. Already transformed for the input to the model.
            multimask_output: Wheterh to predict with the multi- or single-mask head of the maks decoder.

        Returns:
            A list over input images, where each element is as dictionary with the following keys:
                'masks': Mask prediction for this object.
        """
        x = torch.stack([inp["image"] for inp in batched_input], dim=0)

        B, C, D, H, W = x.shape
        assert C == 3

        with self.no_grad():
            features = self._apply_image_encoder(x, D)

        out = features
        for decoder in self.decoders:
            out = decoder(out)
        logits = self.out_conv(out)

        outputs = [{"masks": mask.unsqueeze(0)} for mask in logits]
        return outputs
