import math
from typing import List, Union, Optional

import torch.nn as nn

from segment_anything.modeling import Sam


class LoRASurgery(nn.Module):
    """Operates on the attention layers for performing low-rank adaptation.

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
        block: The chosen attention blocks for implementing lora.
    """
    def __init__(self, rank: int, block: nn.Module):
        super().__init__()
        self.qkv_proj = block.attn.qkv
        self.dim = self.qkv_proj.in_features

        self.w_a_linear_q = nn.Linear(self.dim, rank, bias=False)
        self.w_b_linear_q = nn.Linear(rank, self.dim, bias=False)
        self.w_a_linear_v = nn.Linear(self.dim, rank, bias=False)
        self.w_b_linear_v = nn.Linear(rank, self.dim, bias=False)

        self.reset_parameters()

        block.attn.qkv = self

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_a_linear_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_a_linear_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.w_b_linear_q.weight)
        nn.init.zeros_(self.w_b_linear_v.weight)

    def forward(self, x):
        qkv = self.qkv_proj(x)  # B, N, N, 3 * org_C
        new_q = self.w_b_linear_q(self.w_a_linear_q(x))
        new_v = self.w_b_linear_v(self.w_a_linear_v(x))
        qkv[:, :, :, :self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv


class FacTSurgery(nn.Module):
    """Operates on the attention layers for performing factorized attention.

    (Inspired from: https://github.com/cchen-cc/MA-SAM/blob/main/MA-SAM/sam_fact_tt_image_encoder.py)

    Args:
        rank: The rank of the decomposition matrices for updating weights in each attention layer.
        block: The chosen attention blocks for implementing fact.
    """

    def __init__(self, rank: int, block: nn.Module, dropout: Optional[float] = None):
        super().__init__()
        self.qkv_proj = block.attn.qkv
        self.dim = self.qkv_proj.in_features

        self.q_FacTs = nn.Linear(rank, rank, bias=False)
        self.v_FacTs = nn.Linear(rank, rank, bias=False)

        self.dropout = dropout
        if self.dropout is not None:
            # NOTE : Dropout is not included in the original implementation
            self.dp_q = nn.Dropout(self.dropout)
            self.dp_v = nn.Dropout(self.dropout)

        self.FacTu = nn.Linear(self.dim, rank, bias=False)
        self.FacTv = nn.Linear(rank, self.dim, bias=False)

        block.attn.qkv = self

    def forward(self, x):
        qkv = self.qkv_proj(x)  # B, N, N, 3 * org_C

        new_q = self.q_FacTs(self.FacTu(x))
        new_v = self.v_FacTs(self.FacTu(x))

        if self.dropout is not None:
            new_q = self.dp_q(new_q)
            new_v = self.dp_v(new_v)

        new_q = self.FacTv(new_q)
        new_v = self.FacTv(new_v)

        # NOTE : Scaling Factor was set to 1 as it can be tuned via the learning rate
        # Does it make sense to include it, in order to have similar learning rate as the original model?
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v

        return qkv


class SelectiveSurgery(nn.Module):
    """Base class for selectively allowing gradient updates for certain parameters.
    """
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def allow_gradient_update_for_parameters(
        self,
        prefix: Optional[List[str]] = None,
        suffix: Optional[List[str]] = None,
        infix: Optional[List[str]] = None,
    ):
        """
        """
        for k, v in self.block.named_parameters():
            if prefix is not None and k.startswith(tuple(prefix)):
                v.requires_grad = True

            if suffix is not None and k.endswith(tuple(suffix)):
                v.requires_grad = True

            if infix is not None:
                for per_infix in infix:
                    if k.find(per_infix) != -1:
                        v.requires_grad = True

    def forward(self, x):
        return x


class AttentionSurgery(SelectiveSurgery):
    """Child class for allowing gradient updates for parameters in attention layers.
    """
    def __init__(self, block: nn.Module):
        super().__init__(block=block)
        # Allow gradient updates for the attention layers in the image encoder.
        self.allow_gradient_update_for_parameters(prefix=["attn"])


class BiasSurgery(SelectiveSurgery):
    """Child class for allowing gradient updates for bias parameters.
    """
    def __init__(self, block: nn.Module):
        super().__init__(block=block)
        # Allow gradient updates for the bias parameters in the image encoder.
        self.allow_gradient_update_for_parameters(suffix=["bias"])


class LayerNormSurgery(SelectiveSurgery):
    """Child class for allowing gradient updates in normalization layers.
    """
    def __init__(self, block: nn.Module):
        super().__init__(block=block)
        # Allow gradient updates for the LayerNorm parameters in the image encoder.
        self.allow_gradient_update_for_parameters(infix=["norm1", "norm2"])


class PEFT_Sam(nn.Module):
    """Wraps the Segment Anything model's image encoder to different parameter efficient finetuning methods.

    Args:
        model: The Segment Anything model.
        rank: The rank for low-rank adaptation.
        peft_module: Wrapper to operate on the image encoder blocks for the PEFT method.
        attention_layers_to_update: Which specific layers we apply PEFT methods to.
    """

    def __init__(
        self,
        model: Sam,
        rank: int,
        peft_module: nn.Module = LoRASurgery,
        attention_layers_to_update: Union[List[int]] = None
    ):
        super().__init__()

        assert rank > 0
        assert issubclass(peft_module, Union[LoRASurgery, FacTSurgery, SelectiveSurgery]), "Invalid PEFT module."

        if attention_layers_to_update:
            self.peft_layers = attention_layers_to_update
        else:   # Applies PEFT to the image encoder by default
            self.peft_layers = list(range(len(model.image_encoder.blocks)))

        self.peft_module = peft_module
        self.peft_blocks = []

        # let's freeze all the pretrained image encoder layers first
        for param in model.image_encoder.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(model.image_encoder.blocks):
            # If we only want specific layers with PEFT instead of all
            if t_layer_i not in self.peft_layers:
                continue

            if issubclass(self.peft_module, SelectiveSurgery):
                peft_block = self.peft_module(block=blk)
            else:
                peft_block = self.peft_module(rank=rank, block=blk)

            self.peft_blocks.append(peft_block)

        self.peft_blocks = nn.ModuleList(self.peft_blocks)

        self.sam = model

    def forward(self, batched_input, multimask_output):
        return self.sam(batched_input, multimask_output)
