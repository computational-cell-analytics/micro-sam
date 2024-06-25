import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Tuple
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
        model_type: str = "vit_b",
    ):
        super().__init__(sam_model)
        self.model_type = model_type
        self.d_size = d_size
        self.model = Sam3DWrapper(sam_model, d_size) 

        # predictor.model = Sam3DWrapper(predictor.model, self.d_size)
        # self.predictor = predictor


class Sam3DWrapper(nn.Module):
    def __init__(
        self,
        sam_model: Sam,
        d_size,
    ):
        """
        Initializes the Sam3DWrapper object.
        
        Args:
            sam_model (Sam): The Sam model to be wrapped.
            d_size (int):  Factor controlling the 3D adapter reshaping.
                d_size determines how the input features are grouped along a new dimension
                for processing within the 3D adapter modules. A larger d_size creates fewer groups
                with more features per group.

        """
        super().__init__()
        self.sam_model = sam_model
        self.d_size = d_size
        sam_model.image_encoder = ImageEncoderViT3DWrapper(image_encoder=self.sam_model.image_encoder, d_size=self.d_size)
        self.sam_model = sam_model

    def forward(self, batched_input, multimask_output, image_size) -> torch.Tensor:
        return self._forward_train(batched_input, multimask_output, image_size)

    def _forward_train(self, batched_input, multimask_output, image_size):
        b_size, hw_size, d_size = batched_input.shape[0], batched_input.shape[-2], batched_input.shape[1] # [b, d, 3, h, w]
        batched_input = batched_input.contiguous().view(-1, 3, hw_size, hw_size)

        input_images = self.preprocess(batched_input)
        image_embeddings = self.image_encoder(input_images, d_size)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None, boxes=None, masks=None
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output
        )
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )
        outputs = {
            'masks': masks,
            'iou_predictions': iou_predictions,
            'low_res_logits': low_res_masks
        }
        # print(low_res_masks.shape)
        return outputs


    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


class ImageEncoderViT3DWrapper(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        d_size: int,
        num_heads: int = 12,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.d_size = d_size
        self.img_size = self.image_encoder.img_size

        # replace default blocks with 3d adapter blocks
        for i, blk in enumerate(self.image_encoder.blocks):
            self.image_encoder.blocks[i] = NDBlockWrapper(block=blk, num_heads=num_heads, dim=embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.image_encoder.patch_embed(x)
        if self.image_encoder.pos_embed is not None:
            x = x + self.image_encoder.pos_embed

        for blk in self.image_encoder.blocks:
            x = blk(x, self.d_size)

        x = self.image_encoder.neck(x.permute(0, 3, 1, 2))

        return x


class NDBlockWrapper(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        dim: int,
        num_heads: int,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        adapter_channels: int = 384,
    ):
        super().__init__()
        self.block = block
        
        self.adapter_channels = adapter_channels
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
        x = self.block.norm1(x)
        # Window partition
        if self.block.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.block.window_size)

        x = self.block.attn(x)
        # Reverse window partition
        if self.block.window_size > 0:
            x = window_unpartition(x, self.block.window_size, pad_hw, (H, W))

        x = shortcut + x
        
        # 3D adapter
        shortcut = x
        x = self.adapter_norm_2(x)
        x = self.adapter_linear_down_2(x)
        x = x.contiguous().view(int(b_size/d_size), d_size, hw_size, hw_size, self.adapter_channels)
        x = torch.permute(x, (0, -1, 1, 2, 3))
        x = self.adapter_conv_2(x)
        x = torch.permute(x, (0, 2, 3, 4, 1))
        x = x.contiguous().view(b_size, hw_size, hw_size, self.adapter_channels)
        x = self.adapter_act_2(x)
        x = self.adapter_linear_up_2(x)
        x = shortcut + x
        # end 3D adapter
        
        x = x + self.block.mlp(self.block.norm2(x))

        return x
