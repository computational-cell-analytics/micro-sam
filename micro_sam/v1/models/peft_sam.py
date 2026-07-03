from typing import List, Union, Optional

import torch.nn as nn

from segment_anything.modeling import Sam

from micro_sam.models.peft import (
    AttentionLoRA, MLPLoRA, FacTSurgery, AdaptFormer, ScaleShiftLayer, SelectiveSurgery, quantize_linear_layers,
)
from micro_sam.models.peft import (  # noqa
    AttentionSurgery, BiasSurgery, LayerNormSurgery, ClassicalSurgery,
)


class LoRASurgery(nn.Module):
    """Operates on the linear layers (attention and/or other feed forward) for performing low-rank adaptation.

    (Inspired from: https://github.com/JamesQFreeman/Sam_LoRA/)

    In SAM, it is implemented as:
    ```python
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    ```

    Args:
        rank: The rank of the decomposition matrices for updating weights in each attention layer.
        block: The chosen attention blocks for implementing LoRA.
        update_matrices: Which specific matrices to update in the attention layer. Choice of "q", "k", "v", "mlp".
    """
    def __init__(self, rank: int, block: nn.Module, update_matrices: List[str] = ["q", "v"]):
        super().__init__()
        # Check whether all values for "update_matrices" are as expected.
        if set(update_matrices) - set(["q", "k", "v", "mlp"]):
            raise ValueError(f"Some of the expected keys for updating matrics in '{update_matrices}' are not expected.")

        self.block = block
        block.attn.qkv = AttentionLoRA(rank=rank, block=block.attn.qkv, update_matrices=update_matrices)

        if "mlp" in update_matrices:
            # SAM's MLPBlock exposes its two linear layers as 'lin1'/'lin2' with activation 'act'.
            block.mlp = MLPLoRA(rank=rank, mlp_layer=block.mlp, get_layers=lambda m: (m.lin1, m.lin2, m.act))

    def forward(self, x):
        return x


class SSFSurgery(nn.Module):
    """Operates on all layers in the transformer block for adding learnable scale and shift parameters.

    Args:
        rank: This parameter is not used in `SSFSurgery`. This is kept here for consistency.
        block: The chosen attention blocks for implementing ssf.
    """
    def __init__(self, rank: int, block: nn.Module):
        super().__init__()
        self.block = block

        # If we get a transformer block (w. multiple sub-layers), we perform surgery on each layer.
        if hasattr(block, "attn"):  # the minimum assumption is to verify the attention layers.
            block.attn.qkv = ScaleShiftLayer(block.attn.qkv, block.attn.qkv.in_features*3)
            block.attn.proj = ScaleShiftLayer(block.attn.proj, block.attn.proj.in_features)
            block.mlp.lin1 = ScaleShiftLayer(block.mlp.lin1, block.mlp.lin1.out_features)
            block.mlp.lin2 = ScaleShiftLayer(block.mlp.lin2, block.mlp.lin2.out_features)
            block.norm1 = ScaleShiftLayer(block.norm1, block.norm1.normalized_shape[0])
            block.norm2 = ScaleShiftLayer(block.norm2, block.norm2.normalized_shape[0])

        # If we get the embedding block, add one ScaleShiftLayer
        elif hasattr(block, "patch_embed"):
            block.proj = ScaleShiftLayer(block.proj, block.proj.out_channels)

    def forward(self, x):
        return x


class PEFT_Sam(nn.Module):
    """Wraps the Segment Anything model's image encoder to different parameter efficient finetuning methods.

    Args:
        model: The Segment Anything model.
        rank: The rank for low-rank adaptation.
        peft_module: Wrapper to operate on the image encoder blocks for the PEFT method.
        attention_layers_to_update: Which specific layers we apply PEFT methods to.
            For reference, the total number of blocks for 'vit_b' is 12, for 'vit_l' is 24 and for 'vit_h' is 32.
            By default, applies the PEFT method to all attention layers.
        quantize: Whether to quantize the model for lower precision training. By default, does not quantize the model.
        module_kwargs: The additional arguments for the respective PEFT modules.
    """

    def __init__(
        self,
        model: Sam,
        rank: Optional[int] = None,
        peft_module: nn.Module = LoRASurgery,
        attention_layers_to_update: Optional[List[int]] = None,
        quantize: bool = False,
        **module_kwargs
    ):
        super().__init__()

        if issubclass(peft_module, Union[LoRASurgery, FacTSurgery]) and (not rank or rank <= 0):
            raise RuntimeError("The chosen PEFT method cannot run without a valid rank choice.")

        assert issubclass(peft_module, Union[LoRASurgery, FacTSurgery, SelectiveSurgery, SSFSurgery, AdaptFormer]), (
            "Invalid PEFT module"
        )
        if attention_layers_to_update:
            self.peft_layers = attention_layers_to_update
        else:  # Applies PEFT to the image encoder by default
            self.peft_layers = list(range(len(model.image_encoder.blocks)))

        self.peft_module = peft_module
        self.peft_blocks = []

        # Whether to quantize the linear layers to 4 bit precision.
        # NOTE: This is currently supported for CUDA-supported devices only.
        if quantize:
            quantize_linear_layers(model.image_encoder)

        # Let's freeze all the pretrained image encoder layers first
        for param in model.image_encoder.parameters():
            param.requires_grad = False

        # Add scale and shift parameters to the patch embedding layers.
        if issubclass(self.peft_module, SSFSurgery):
            self.peft_blocks.append(self.peft_module(rank=rank, block=model.image_encoder.patch_embed))

        # If specified, the attention layers to update should match the available blocks.
        if attention_layers_to_update and (
            set(attention_layers_to_update) - set(list(range(len(model.image_encoder.blocks))))
        ):
            raise ValueError("The chosen layer(s) to apply PEFT method is not a valid transformer block id.")

        for t_layer_i, blk in enumerate(model.image_encoder.blocks):

            # If we only want specific layers with PEFT instead of all
            if t_layer_i not in self.peft_layers:
                continue

            if issubclass(self.peft_module, SelectiveSurgery):
                self.peft_blocks.append(self.peft_module(block=blk))
            else:
                self.peft_blocks.append(self.peft_module(rank=rank, block=blk, **module_kwargs))

        self.peft_blocks = nn.ModuleList(self.peft_blocks)
        self.sam = model

    def forward(self, batched_input, multimask_output):
        return self.sam(batched_input, multimask_output)
