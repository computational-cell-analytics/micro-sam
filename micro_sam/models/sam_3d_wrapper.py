import os
from collections import OrderedDict
from typing import Any, List, Dict, Type, Union, Optional, Literal

import torch
import torch.nn as nn

from segment_anything.modeling import Sam
from segment_anything.modeling.image_encoder import window_partition, window_unpartition

from .peft_sam import LoRASurgery
from ..instance_segmentation import get_decoder
from ..util import get_sam_model, _DEFAULT_MODEL


def get_sam_3d_model(
    image_size: int,
    n_classes: int,
    model_type: str = _DEFAULT_MODEL,
    lora_rank: Optional[int] = None,
    decoder_choice: Literal["default", "unetr"] = "default",
    device: Optional[Union[str, torch.device]] = None,
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
) -> nn.Module:
    """Get the SAM 3D model for semantic segmentation.

    Args:
        image_size: The size of height / width of the input image.
        n_classes: The number of output classes.
        model_type: The choice of SAM model.
        decoder_choice: Whether to use the SAM mask decoder, i.e. chosen by 'default' value,
            or the UNETR decoder, i.e. chosen by 'unetr' value.
        device: The torch device.
        checkpoint_path: Optional, whether to load a finetuned model.

    Returns:
        The SAM 3D model.
    """
    if decoder_choice not in ["default", "unetr"]:
        raise ValueError(
            f"'{decoder_choice}' as the decoder choice is not supported. Please choose either 'default' or 'unetr'."
        )

    kwargs = {}
    if decoder_choice == "default":
        kwargs["num_multimask_outputs"] = n_classes

    peft_kwargs = {}
    if lora_rank is not None:
        peft_kwargs["rank"] = lora_rank
        peft_kwargs["peft_module"] = LoRASurgery

    _, sam, state = get_sam_model(
        model_type=model_type,
        device=device,
        checkpoint_path=checkpoint_path,
        return_sam=True,
        return_state=True,
        flexible_load_checkpoint=True,
        image_size=image_size,
        peft_kwargs=peft_kwargs,
        **kwargs
    )

    if decoder_choice == "default":
        model = Sam3DClassicWrapper(sam_model=sam, model_type=model_type)
    else:
        model = Sam3DUNETRWrapper(
            sam_model=sam,
            model_type=model_type,
            decoder_state=state.get("decoder_state", None),  # Loads the decoder state automatically, if weights found.
            output_channels=n_classes,
        )

    return model.to(device)


class Sam3DWrapperBase(nn.Module):
    """Sam3DWrapperBase is a base class to implement specific SAM-based 3d semantic segmentation models.
    """
    def __init__(self, model_type: str = "vit_b"):
        super().__init__()
        self.embed_dim, self.num_heads = self._get_model_config(model_type)

    def _get_model_config(self, model_type: str):
        # Returns the model configuration.
        if model_type == "vit_b":
            return 768, 12
        elif model_type == "vit_l":
            return 1024, 16
        elif model_type == "vit_h":
            return 1280, 16
        else:
            raise ValueError(f"'{model_type}' is not a supported choice of model.")

    def _prepare_inputs(self, batched_input: List[Dict[str, Any]]):
        batched_images = torch.stack([inp["image"] for inp in batched_input], dim=0)
        original_size = batched_input[0]["original_size"]
        assert all(inp["original_size"] == original_size for inp in batched_input)

        shape = batched_images.shape
        assert shape[1] == 3
        batch_size, d_size, hw_size = shape[0], shape[2], shape[-2]

        batched_images = batched_images.transpose(1, 2).contiguous().view(-1, 3, hw_size, hw_size)
        return batched_images, original_size, batch_size, d_size

    def forward(
        self, batched_input: List[Dict[str, Any]], multimask_output: bool = False,
    ) -> List[Dict[str, torch.Tensor]]:
        """Predicts 3D masks for the provided inputs.

        Unlike original SAM, this model only supports automatic segmentation and does not support prompts.

        Args:
            batched_input: A list over input images, each a dictionary with the following keys.
                'image': The image as a torch tensor in 3xDxHxW format. Already transformed for the input to the model.
                'original_size': The original size of the image (HxW) before transformation.
            multimask_output: Whether to predict with the multi- or single-mask head of the maks decoder.

        Returns:
            A list over input images, where each element is as dictionary with the following keys:
                'masks': Mask prediction for this object (IMPORTANT, in accordance to SAM output style).
                'iou_predictions': IOU score prediction for this object for the default mask decoder (OPTIONAL).
                'low_res_masks': Low resolution mask prediction for this object for the default mask decoder (OPTIONAL).
        """
        raise NotImplementedError(
            "Sam3DWrapperBase is just a class template. Use a child class that implements the forward pass."
        )


class Sam3DClassicWrapper(Sam3DWrapperBase):
    def __init__(self, sam_model: Sam, model_type: str = "vit_b"):
        """Initializes the Sam3DClassicWrapper object.

        Args:
            sam_model: The SAM model to be wrapped.
            model_type: The choice of segment anything model to wrap adapters for respective model configuration.
        """
        super().__init__(model_type)

        sam_model.image_encoder = ImageEncoderViT3DWrapper(
            image_encoder=sam_model.image_encoder,
            num_heads=self.num_heads,
            embed_dim=self.embed_dim,
        )
        self.sam_model = sam_model

    def forward(
        self, batched_input: List[Dict[str, Any]], multimask_output: bool = False
    ) -> List[Dict[str, torch.Tensor]]:
        """Predict 3D masks for the current inputs.

        Unlike original SAM this model only supports automatic segmentation and does not support prompts.

        Args:
            batched_input: A list over input images, each a dictionary with the following keys.
                'image': The image as a torch tensor in 3xDxHxW format. Already transformed for the input to the model.
                'original_size': The original size of the image (HxW) before transformation.
            multimask_output: Whether to predict with the multi- or single-mask head of the maks decoder.

        Returns:
            A list over input images, where each element is as dictionary with the following keys:
                'masks': Mask prediction for this object.
                'iou_predictions': IOU score prediction for this object for the default mask decoder.
                'low_res_masks': Low resolution mask prediction for this object for the default mask decoder.
        """
        batched_images, original_size, batch_size, d_size = self._prepare_inputs(batched_input)

        input_images = self.sam_model.preprocess(batched_images)
        image_embeddings = self.sam_model.image_encoder(input_images, d_size)
        sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(points=None, boxes=None, masks=None)
        low_res_masks, iou_predictions = self.sam_model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output
        )
        masks = self.sam_model.postprocess_masks(
            masks=low_res_masks, input_size=batched_images.shape[-2:], original_size=original_size,
        )

        # Bring the masks and low-res masks into the correct shape:
        # - disentangle batches and z-slices
        # - rearrange output channels and z-slices

        n_channels = masks.shape[1]
        masks = masks.view(*(batch_size, d_size, n_channels, masks.shape[-2], masks.shape[-1]))
        masks = masks.transpose(1, 2)

        low_res_masks = low_res_masks.view(
            *(batch_size, d_size, n_channels, low_res_masks.shape[-2], low_res_masks.shape[-1])
        )
        low_res_masks = low_res_masks.transpose(1, 2)

        # Make the output compatible with the SAM output.
        outputs = [{
            "masks": mask.unsqueeze(0), "iou_predictions": iou_pred, "low_res_logits": low_res_mask.unsqueeze(0)
        } for mask, iou_pred, low_res_mask in zip(masks, iou_predictions, low_res_masks)]

        return outputs


class Sam3DUNETRWrapper(Sam3DWrapperBase):
    def __init__(
        self,
        sam_model: Sam,
        model_type: str = "vit_b",
        decoder_state: Optional[OrderedDict[str, torch.Tensor]] = None,
        output_channels: int = 3,
    ):
        """Initializes the Sam3DUNETRWrapper object.

        Args:
            sam_model: The SAM model to be wrapped.
            model_type: The choice of segment anything model to wrap adapters for respective model configuration.
            decoder_state: Optional, whether to load the UNETR decoder with provided pretrained weights.
            output_channels: The choice of output classes.
        """
        super().__init__(model_type)

        self.image_encoder = ImageEncoderViT3DWrapper(
            image_encoder=sam_model.image_encoder,
            num_heads=self.num_heads,
            embed_dim=self.embed_dim,
        )
        self._preprocess = sam_model.preprocess

        # NOTE: Remove the output layer weights as we have new target class for the new task.
        if decoder_state is not None:
            decoder_state = OrderedDict(
                [(k, v) for k, v in decoder_state.items() if not k.startswith("out_conv.")]
            )

        # Get a custom decoder, which overtakes the SAM mask decoder.
        self.decoder = get_decoder(
            image_encoder=sam_model.image_encoder,
            decoder_state=decoder_state,
            out_channels=output_channels,
            flexible_load_checkpoint=True,
        )

    def forward(
        self, batched_input: List[Dict[str, Any]], multimask_output: bool = False
    ) -> List[Dict[str, torch.Tensor]]:
        """Predict 3D masks for the current inputs.

        Unlike original SAM this model only supports automatic segmentation and does not support prompts.

        Args:
            batched_input: A list over input images, each a dictionary with the following keys.
                'image': The image as a torch tensor in 3xDxHxW format. Already transformed for the input to the model.
                'original_size': The original size of the image (HxW) before transformation.
            multimask_output: Whether to predict with the multi- or single-mask head of the maks decoder.

        Returns:
            A list over input images, where each element is as dictionary with the following key:
                'masks': Mask prediction for this object.
        """
        batched_images, original_size, batch_size, d_size = self._prepare_inputs(batched_input)

        input_images = self._preprocess(batched_images)
        image_embeddings = self.image_encoder(input_images, d_size)
        masks = self.decoder(image_embeddings, batched_images.shape[-2:], original_size)

        # Bring the masks and low-res masks into the correct shape:
        # - disentangle batches and z-slices
        # - rearrange output channels and z-slices

        n_channels = masks.shape[1]
        masks = masks.view(*(batch_size, d_size, n_channels, masks.shape[-2], masks.shape[-1]))
        masks = masks.transpose(1, 2)

        # Make the output compatable with the SAM output.
        outputs = [{"masks": mask.unsqueeze(0)} for mask in masks]

        return outputs


class ImageEncoderViT3DWrapper(nn.Module):
    def __init__(self, image_encoder: nn.Module, num_heads: int = 12, embed_dim: int = 768):

        super().__init__()
        self.image_encoder = image_encoder
        self.img_size = self.image_encoder.img_size

        # Replace default blocks with 3d adapter blocks
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
