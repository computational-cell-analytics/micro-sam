from typing import Type

import torch
import torch.nn as nn

from segment_anything.modeling.image_encoder import window_partition, window_unpartition
from segment_anything.modeling import Sam

from .util import get_sam_model


def get_3d_sam_model(
    device,
    n_classes,
    image_size,
    lora_rank=None,
    freeze_encoder=False,
    model_type="vit_b",
    checkpoint_path=None,
):
    if lora_rank is None:
        use_lora = False
        rank = None
        freeze_encoder_ = freeze_encoder
    else:
        use_lora = True
        rank = lora_rank
        freeze_encoder_ = False

    _, sam = get_sam_model(
        model_type=model_type,
        device=device,
        checkpoint_path=checkpoint_path,
        return_sam=True,
        flexible_load_checkpoint=True,
        num_multimask_outputs=n_classes,
        image_size=image_size,
        use_lora=use_lora,
        rank=rank,
    )

    sam_3d = Sam3DWrapper(sam, freeze_encoder=freeze_encoder_)
    sam_3d.to(device)
    return sam_3d


class Sam3DWrapper(nn.Module):
    def __init__(self, sam_model: Sam, freeze_encoder: bool):
        """
        Initializes the Sam3DWrapper object.

        Args:
            sam_model (Sam): The Sam model to be wrapped.
        """
        super().__init__()
        sam_model.image_encoder = ImageEncoderViT3DWrapper(
            image_encoder=sam_model.image_encoder
        )
        self.sam_model = sam_model

        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            for param in self.sam_model.image_encoder.parameters():
                param.requires_grad = False

    # FIXME
    # - handling of the image size here is wrong, this only works for square images
    # - this does not take care of resizing
    # unclear how batches are handled
    def forward(self, batched_input, multimask_output, image_size) -> torch.Tensor:
        return self._forward_train(batched_input, multimask_output, image_size)

    def _forward_train(self, batched_input, multimask_output, image_size):
        # dimensions: [b, d, 3, h, w]
        shape = batched_input.shape
        batch_size, d_size, hw_size = shape[0], shape[1], shape[-2]
        batched_input = batched_input.contiguous().view(-1, 3, hw_size, hw_size)

        input_images = self.sam_model.preprocess(batched_input)
        image_embeddings = self.sam_model.image_encoder(input_images, d_size)
        sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
            points=None, boxes=None, masks=None
        )
        low_res_masks, iou_predictions = self.sam_model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output
        )
        masks = self.sam_model.postprocess_masks(
            low_res_masks,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )

        # Bring the masks and low-res masks into the correct shape:
        # - disentangle batches and z-slices
        # - rearrange output channels and z-slices

        n_channels = masks.shape[1]
        masks = masks.view(*(batch_size, d_size, n_channels, masks.shape[-2], masks.shape[-1]))
        low_res_masks = low_res_masks.view(
            *(batch_size, d_size, n_channels, low_res_masks.shape[-2], low_res_masks.shape[-1])
        )

        masks = masks.transpose(1, 2)
        low_res_masks = low_res_masks.transpose(1, 2)

        outputs = {
            "masks": masks,
            "iou_predictions": iou_predictions,
            "low_res_logits": low_res_masks
        }
        return outputs


class ImageEncoderViT3DWrapper(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        num_heads: int = 12,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.img_size = self.image_encoder.img_size

        # replace default blocks with 3d adapter blocks
        for i, blk in enumerate(self.image_encoder.blocks):
            self.image_encoder.blocks[i] = NDBlockWrapper(block=blk, num_heads=num_heads, dim=embed_dim)

    def forward(self, x: torch.Tensor, d_size: int) -> torch.Tensor:
        x = self.image_encoder.patch_embed(x)
        if self.image_encoder.pos_embed is not None:
            x = x + self.image_encoder.pos_embed

        for blk in self.image_encoder.blocks:
            x = blk(x, d_size)

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
        self.adapter_conv = nn.Conv3d(
            self.adapter_channels, self.adapter_channels, kernel_size=(3, 1, 1), padding="same"
        )
        self.adapter_act = nn.GELU()
        self.adapter_norm = norm_layer(dim)

        self.adapter_linear_down_2 = nn.Linear(dim, self.adapter_channels, bias=False)
        self.adapter_linear_up_2 = nn.Linear(self.adapter_channels, dim, bias=False)
        self.adapter_conv_2 = nn.Conv3d(
            self.adapter_channels, self.adapter_channels, kernel_size=(3, 1, 1), padding="same"
        )
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
