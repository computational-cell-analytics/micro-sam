from typing import List, Optional

import torch.nn as nn

from micro_sam.models.peft import (
    AttentionLoRA, MLPLoRA, FacTSurgery, AdaptFormer, ScaleShiftLayer, SelectiveSurgery,
    AttentionSurgery, BiasSurgery, LayerNormSurgery, ClassicalSurgery, quantize_linear_layers,
)


class LoRASurgery(nn.Module):
    """Operates on the linear layers (attention and/or feed forward) of a SAM2 (Hiera) block.

    (Inspired from: https://github.com/JamesQFreeman/Sam_LoRA/)

    Args:
        rank: The rank of the decomposition matrices for updating weights in each block.
        block: The chosen Hiera block for implementing LoRA.
        update_matrices: Which specific matrices to update in the block. Choice of "q", "k", "v", "mlp".
    """
    def __init__(self, rank: int, block: nn.Module, update_matrices: List[str] = ["q", "v"]):
        super().__init__()
        # Check whether all values for "update_matrices" are as expected.
        if set(update_matrices) - set(["q", "k", "v", "mlp"]):
            raise ValueError(f"Some of the expected keys for updating matrics in '{update_matrices}' are not expected.")

        self.block = block
        block.attn.qkv = AttentionLoRA(rank=rank, block=block.attn.qkv, update_matrices=update_matrices)

        if "mlp" in update_matrices:
            # SAM2's MLP stores its two linear layers in a ModuleList ('layers') with activation 'act'.
            block.mlp = MLPLoRA(
                rank=rank, mlp_layer=block.mlp, get_layers=lambda m: (m.layers[0], m.layers[1], m.act),
            )

    def forward(self, x):
        return x


class SSFSurgery(nn.Module):
    """Adds learnable scale and shift parameters to every sub-layer of a SAM2 (Hiera) block.

    Args:
        rank: This parameter is not used in `SSFSurgery`. This is kept here for consistency.
        block: A Hiera block, or the trunk's PatchEmbed for the patch-embedding scale and shift.
    """
    def __init__(self, rank: int, block: nn.Module):
        super().__init__()
        self.block = block

        # A transformer block: wrap the qkv/proj/mlp/norm sub-layers. The qkv scale matches the qkv
        # OUTPUT dimension ('out_features' = dim_out * 3), which differs from the input at stage
        # transitions. SAM2's MLP stores its two linear layers in a ModuleList ('layers').
        if hasattr(block, "attn"):
            block.attn.qkv = ScaleShiftLayer(block.attn.qkv, block.attn.qkv.out_features)
            block.attn.proj = ScaleShiftLayer(block.attn.proj, block.attn.proj.out_features)
            block.mlp.layers[0] = ScaleShiftLayer(block.mlp.layers[0], block.mlp.layers[0].out_features)
            block.mlp.layers[1] = ScaleShiftLayer(block.mlp.layers[1], block.mlp.layers[1].out_features)
            block.norm1 = ScaleShiftLayer(block.norm1, block.norm1.normalized_shape[0])
            block.norm2 = ScaleShiftLayer(block.norm2, block.norm2.normalized_shape[0])

        # The PatchEmbed: wrap its convolution (channels-first output).
        elif hasattr(block, "proj"):
            block.proj = ScaleShiftLayer(block.proj, block.proj.out_channels)

    def forward(self, x):
        return x


class PEFT_Sam2(nn.Module):
    """Wraps SAM2's Hiera image encoder for different parameter efficient finetuning methods.

    All pretrained image encoder parameters are frozen first, then the chosen PEFT method is applied
    to the selected Hiera blocks (found at `model.image_encoder.trunk.blocks`).

    Args:
        model: The Segment Anything 2 model.
        rank: The rank for low-rank adaptation.
        peft_module: Wrapper to operate on the image encoder blocks for the PEFT method.
        attention_layers_to_update: Which specific blocks we apply PEFT methods to.
            For reference, the total number of blocks is 12 for 'hvit_t'/'hvit_s'/'hvit_b' and 48 for 'hvit_l'.
            By default, applies the PEFT method to all blocks.
        quantize: Whether to quantize the image encoder to 4 bit precision for QLoRA-style training.
            Requires 'bitsandbytes' and is supported on CUDA devices only. By default, does not quantize.
        module_kwargs: The additional arguments for the respective PEFT modules.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: Optional[int] = None,
        peft_module: nn.Module = LoRASurgery,
        attention_layers_to_update: Optional[List[int]] = None,
        quantize: bool = False,
        **module_kwargs
    ):
        super().__init__()

        if issubclass(peft_module, (LoRASurgery, FacTSurgery)) and (not rank or rank <= 0):
            raise RuntimeError("The chosen PEFT method cannot run without a valid rank choice.")

        assert issubclass(
            peft_module, (LoRASurgery, FacTSurgery, SelectiveSurgery, SSFSurgery, AdaptFormer)
        ), "Invalid PEFT module"

        blocks = model.image_encoder.trunk.blocks

        if attention_layers_to_update:
            self.peft_layers = attention_layers_to_update
        else:  # Applies PEFT to all Hiera blocks by default.
            self.peft_layers = list(range(len(blocks)))

        self.peft_module = peft_module
        self.peft_blocks = []

        # Whether to quantize the linear layers to 4 bit precision (QLoRA).
        # NOTE: This is currently supported for CUDA-supported devices only.
        if quantize:
            quantize_linear_layers(model.image_encoder)

        # Let's freeze all the pretrained image encoder layers first.
        for param in model.image_encoder.parameters():
            param.requires_grad = False

        # Add scale and shift parameters to the patch embedding layer (SSF only).
        if issubclass(self.peft_module, SSFSurgery):
            self.peft_blocks.append(self.peft_module(rank=rank, block=model.image_encoder.trunk.patch_embed))

        # If specified, the blocks to update should match the available blocks.
        if attention_layers_to_update and (set(attention_layers_to_update) - set(list(range(len(blocks))))):
            raise ValueError("The chosen layer(s) to apply PEFT method is not a valid Hiera block id.")

        for t_layer_i, blk in enumerate(blocks):

            # If we only want specific layers with PEFT instead of all.
            if t_layer_i not in self.peft_layers:
                continue

            if issubclass(self.peft_module, SelectiveSurgery):
                self.peft_blocks.append(self.peft_module(block=blk))
            else:
                self.peft_blocks.append(self.peft_module(rank=rank, block=blk, **module_kwargs))

        self.peft_blocks = nn.ModuleList(self.peft_blocks)
        self.sam = model

    def forward(self, *args, **kwargs):
        return self.sam(*args, **kwargs)


# Registry mapping PEFT module class names to classes, for (de)serializing a peft config stored in a
# checkpoint (see `micro_sam.models.peft.deserialize_peft_kwargs`).
PEFT_MODULES = {
    cls.__name__: cls
    for cls in (
        LoRASurgery, FacTSurgery, SSFSurgery, AdaptFormer,
        ClassicalSurgery, AttentionSurgery, BiasSurgery, LayerNormSurgery,
    )
}
