import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type
#from segment_anything import build_sam, SamPredictor
#from segment_anything import sam_model_registry
from segment_anything.modeling.image_encoder import window_partition, window_unpartition
from segment_anything.modeling.image_encoder import Block as SamBlock
from segment_anything.modeling import Sam
from segment_anything import SamPredictor


class Predictor3D(SamPredictor):
    def __init__(
        self,
        #predictor: SamPredictor,
        sam_model: Sam,
        d_size,
    ):
        super().__init__(sam_model)
        
        self.d_size = d_size
        # predictor.model = Sam3DWrapper(predictor.model, self.d_size)
        # self.predictor = predictor


class Sam3DWrapper(nn.Module):
    def __init__(
        self,
        sam_model: Sam,
        d_size,
    ):
        super().__init__()
        self.sam_model = sam_model
        self.d_size = d_size
        sam_model.image_encoder = ImageEncoderViT3DWrapper(image_encoder=self.sam_model.image_encoder, d_size=self.d_size)
        self.sam_model = sam_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sam_model(x, d_size=self.d_size)


class ImageEncoderViT3DWrapper(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        num_heads: int = 12,
        embed_dim: int = 768,
        **kwargs
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.img_size = self.image_encoder.img_size

        # freeze sam blocks
        for k, v in self.image_encoder.named_parameters():
            if '.adapter_' not in k:
                v.requires_grad = False

        # replace default blocks with 3d adapter blocks
        for i, blk in enumerate(self.image_encoder.blocks):
            self.image_encoder.blocks[i] = NDBlock(Block=blk, num_heads=num_heads, dim=embed_dim)

    def forward(self, x: torch.Tensor, d_size) -> torch.Tensor:
        x = self.image_encoder.patch_embed(x)
        if self.image_encoder.pos_embed is not None:
            x = x + self.image_encoder.pos_embed

        for blk in self.image_encoder.blocks:
            x = blk(x, d_size)

        x = self.image_encoder.neck(x.permute(0, 3, 1, 2))

        return x


class NDBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            Block: nn.Module,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            **kwargs
    ):
        super().__init__()
        self.Block = Block
        
        self.adapter_channels = 384
        self.adapter_linear_down = nn.Linear(dim, self.adapter_channels, bias=False)
        self.adapter_linear_up = nn.Linear(self.adapter_channels, dim, bias=False)
        self.adapter_conv = nn.Conv3d(self.adapter_channels, self.adapter_channels, kernel_size=(3, 1, 1), padding='same')
        self.adapter_act = nn.GELU()
        self.adapter_norm = norm_layer(dim)

        self.adapter_linear_down_2 = nn.Linear(dim, self.adapter_channels, bias=False)
        self.adapter_linear_up_2 = nn.Linear(self.adapter_channels, dim, bias=False)
        self.adapter_conv_2 = nn.Conv3d(self.adapter_channels, self.adapter_channels, kernel_size=(3, 1, 1), padding='same')
        self.adapter_act_2 = nn.GELU()
        self.adapter_norm_2 = norm_layer(dim)

    def forward(self, x: torch.Tensor, d_size) -> torch.Tensor:
        b_size, hw_size = x.shape[0], x.shape[1]
        
        # 3D adapter
        shortcut = x
        x = self.adapter_norm(x)
        x = self.adapter_linear_down(x)
        x = x.contiguous().view(int(b_size/d_size), d_size, hw_size, hw_size, self.adapter_channels)
        x = torch.permute(x, (0, -1, 1, 2, 3))
        x = self.adapter_conv(x)
        x = torch.permute(x, (0, 2, 3, 4, 1))
        x = x.contiguous().view(b_size, hw_size, hw_size, self.adapter_channels)
        x = self.adapter_act(x)
        x = self.adapter_linear_up(x)
        x = shortcut + x
        # end 3D adapter
        
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        
        # 3D adapter
        shortcut = x
        x = self.adapter_norm_2(x)
        x = self.adapter_linear_down_2(x)
        x = x.contiguous().view(int(b_size/d_size), d_size, hw_size, hw_size, self.Block.adapter_channels)
        x = torch.permute(x, (0, -1, 1, 2, 3))
        x = self.adapter_conv_2(x)
        x = torch.permute(x, (0, 2, 3, 4, 1))
        x = x.contiguous().view(b_size, hw_size, hw_size, self.Block.adapter_channels)
        x = self.adapter_act_2(x)
        x = self.adapter_linear_up_2(x)
        x = shortcut + x
        # end 3D adapter
        
        x = x + self.mlp(self.norm2(x))

        return x
