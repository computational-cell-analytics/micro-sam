import math
from typing import List, Union

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
    """
    def __init__(
        self,
        rank: int,
        block: nn.Module,
    ):
        super().__init__()
        self.qkv = block.attn.qkv
        self.dim = self.qkv.in_features

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
        qkv = self.qkv(x)  # B, N, N, 3 * org_C
        new_q = self.w_b_linear_q(self.w_a_linear_q(x))
        new_v = self.w_b_linear_v(self.w_a_linear_v(x))
        qkv[:, :, :, :self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv


class PEFT_Sam(nn.Module):
    """Inspired from: https://github.com/JamesQFreeman/Sam_LoRA/

    Wraps the Segment Anything model's image encoder to different parameter efficient finetuning methods.

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
        super(PEFT_Sam, self).__init__()

        assert rank > 0

        if attention_layers_to_update:
            self.peft_layers = attention_layers_to_update
        else:   # Applies PEFT to the image encoder by default
            self.peft_layers = list(
                range(len(model.image_encoder.blocks))
            )

        self.peft_module = peft_module
        self.peft_blocks = []

        # let's freeze all the pretrained image encoder layers first
        for param in model.image_encoder.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(model.image_encoder.blocks):
            # If we only want specific layers with PEFT instead of all
            if t_layer_i not in self.peft_layers:
                continue

            peft_block = self.peft_module(rank=rank, block=blk)
            self.peft_blocks.append(peft_block)

        self.peft_blocks = nn.ModuleList(self.peft_blocks)

        self.sam = model

    def forward(self, batched_input, multimask_output):
        return self.sam(batched_input, multimask_output)
