from typing import List, Optional

import torch.nn as nn

from micro_sam.models.peft import quantize_linear_layers
from micro_sam.models.peft import (  # noqa
    AttentionLoRA, MLPLoRA, SelectiveSurgery, AttentionSurgery, BiasSurgery, LayerNormSurgery, ClassicalSurgery,
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

        if issubclass(peft_module, LoRASurgery) and (not rank or rank <= 0):
            raise RuntimeError("The chosen PEFT method cannot run without a valid rank choice.")

        assert issubclass(peft_module, (LoRASurgery, SelectiveSurgery)), "Invalid PEFT module"

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
    for cls in (LoRASurgery, ClassicalSurgery, AttentionSurgery, BiasSurgery, LayerNormSurgery)
}
