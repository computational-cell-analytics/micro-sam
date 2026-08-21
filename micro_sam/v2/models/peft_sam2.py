from typing import List, Optional

import torch.nn as nn

from micro_sam.models.peft import (
    MLPLoRA, AdaptFormer, FacTSurgery, AttentionLoRA, BiasSurgery, ScaleShiftLayer, ClassicalSurgery,
    AttentionSurgery, SelectiveSurgery, LayerNormSurgery, quantize_linear_layers,
)


class LoRASurgery(nn.Module):
    """Apply low-rank adaptation to the linear layers of a SAM2 Hiera block.

    Based on https://github.com/JamesQFreeman/Sam_LoRA/.

    Args:
        rank: The rank of the decomposition matrices for updating weights in each block.
        block: The chosen Hiera block for implementing LoRA.
        update_matrices: The matrices to update in the block. Choose "q", "k", "v", or "mlp".
    """
    def __init__(self, rank: int, block: nn.Module, update_matrices: List[str] = ["q", "v"]):
        super().__init__()
        invalid_matrices = set(update_matrices) - {"q", "k", "v", "mlp"}
        if invalid_matrices:
            raise ValueError(f"The update matrix names {sorted(invalid_matrices)} are not valid.")

        self.block = block
        block.attn.qkv = AttentionLoRA(rank=rank, block=block.attn.qkv, update_matrices=update_matrices)

        if "mlp" in update_matrices:
            block.mlp = MLPLoRA(
                rank=rank, mlp_layer=block.mlp, get_layers=lambda m: (m.layers[0], m.layers[1], m.act)
            )

    def forward(self, x):
        return x


class SSFSurgery(nn.Module):
    """Add scale and shift parameters to each sublayer of a SAM2 Hiera block.

    Args:
        rank: The unused rank. This argument keeps the PEFT module signatures consistent.
        block: The Hiera block or PatchEmbed to change.
    """
    def __init__(self, rank: int, block: nn.Module):
        super().__init__()
        self.block = block

        if hasattr(block, "attn"):
            block.attn.qkv = ScaleShiftLayer(block.attn.qkv, block.attn.qkv.out_features)
            block.attn.proj = ScaleShiftLayer(block.attn.proj, block.attn.proj.out_features)
            block.mlp.layers[0] = ScaleShiftLayer(block.mlp.layers[0], block.mlp.layers[0].out_features)
            block.mlp.layers[1] = ScaleShiftLayer(block.mlp.layers[1], block.mlp.layers[1].out_features)
            block.norm1 = ScaleShiftLayer(block.norm1, block.norm1.normalized_shape[0])
            block.norm2 = ScaleShiftLayer(block.norm2, block.norm2.normalized_shape[0])

        elif hasattr(block, "proj"):
            block.proj = ScaleShiftLayer(block.proj, block.proj.out_channels)

    def forward(self, x):
        return x


class PEFT_Sam2(nn.Module):
    """Wrap the SAM2 Hiera image encoder for parameter efficient finetuning.

    This class freezes the pretrained encoder parameters. It then applies the PEFT method to the selected Hiera blocks.

    Args:
        model: The Segment Anything 2 model.
        rank: The rank for low-rank adaptation.
        peft_module: The wrapper for the PEFT method.
        attention_layers_to_update: The blocks that use the PEFT method. The default is all blocks.
        quantize: The flag that enables 4-bit encoder precision. This option needs CUDA and bitsandbytes.
        module_kwargs: The extra arguments for the PEFT module.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: Optional[int] = None,
        peft_module: nn.Module = LoRASurgery,
        attention_layers_to_update: Optional[List[int]] = None,
        quantize: bool = False,
        **module_kwargs,
    ):
        super().__init__()

        if issubclass(peft_module, (LoRASurgery, FacTSurgery)) and (not rank or rank <= 0):
            raise RuntimeError(f"The rank {rank} is not valid for {peft_module.__name__}. Pass a positive integer.")

        valid_modules = (LoRASurgery, FacTSurgery, SelectiveSurgery, SSFSurgery, AdaptFormer)
        if not issubclass(peft_module, valid_modules):
            raise ValueError(f"The PEFT module {peft_module.__name__} is not valid.")

        blocks = model.image_encoder.trunk.blocks

        if attention_layers_to_update:
            self.peft_layers = attention_layers_to_update
        else:
            self.peft_layers = list(range(len(blocks)))

        self.peft_module = peft_module
        self.peft_blocks = []

        if quantize:
            quantize_linear_layers(model.image_encoder)

        for param in model.image_encoder.parameters():
            param.requires_grad = False

        if issubclass(self.peft_module, SSFSurgery):
            self.peft_blocks.append(self.peft_module(rank=rank, block=model.image_encoder.trunk.patch_embed))

        if attention_layers_to_update and (set(attention_layers_to_update) - set(list(range(len(blocks))))):
            raise ValueError(
                f"The Hiera block ids {attention_layers_to_update} are not valid for a model with {len(blocks)} blocks."
            )

        for t_layer_i, blk in enumerate(blocks):

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


PEFT_MODULES = {
    cls.__name__: cls
    for cls in (
        LoRASurgery, FacTSurgery, SSFSurgery, AdaptFormer,
        ClassicalSurgery, AttentionSurgery, BiasSurgery, LayerNormSurgery,
    )
}
