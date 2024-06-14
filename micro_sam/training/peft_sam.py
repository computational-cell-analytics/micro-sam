import math
from typing import Tuple, List, Union

import torch
import torch.nn as nn

from segment_anything.modeling import Sam


class PEFTBase(nn.Module):
    """PEFTBase is an interface to implement specific PEFT-based methods.
    """
    def __call__(
        self,
        rank: int,
        block: nn.Module
    ) -> Tuple[nn.Module, List[nn.Module]]:
        """Returns the attention block by updating the qkv pair per block and the respective linear layers.

        Args:
            rank: The rank of the per-layer covariance matrix for low-rank adaptation.
            block: The individual attention block.

        Returns:
            The attention block.
            The pair of values for the linear layer.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class LoRASurgery(PEFTBase):
    """Operates on the attention layers for performing low-rank adaptation.
    """
    def __call__(
        self,
        rank: int,
        block
    ):
        w_qkv_linear = block.attn.qkv
        dim = w_qkv_linear.in_features

        w_a_linear_q = nn.Linear(dim, rank, bias=False)
        w_b_linear_q = nn.Linear(rank, dim, bias=False)
        w_a_linear_v = nn.Linear(dim, rank, bias=False)
        w_b_linear_v = nn.Linear(rank, dim, bias=False)

        block.attn.qkv = _LoRA_Sam(
            w_qkv_linear, w_a_linear_q, w_b_linear_q, w_a_linear_v, w_b_linear_v,
        )

        return block, [w_a_linear_q, w_b_linear_q, w_a_linear_v, w_b_linear_v]


class _LoRA_Sam(nn.Module):
    """Inspired from: https://github.com/JamesQFreeman/Sam_LoRA/

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
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B, N, N, 3 * org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
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
        peft_module: PEFTBase = LoRASurgery(),
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

        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # let's freeze all the pretrained image encoder layers first
        for param in model.image_encoder.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(model.image_encoder.blocks):
            # If we only want specific layers for PEFT instead of all
            if t_layer_i not in self.peft_layers:
                continue

            blk, linear_layers = self.peft_module(rank=rank, block=blk)
            w_a_linear_q, w_b_linear_q, w_a_linear_v, w_b_linear_v = linear_layers

            self.w_As.extend([w_a_linear_q, w_a_linear_v])
            self.w_Bs.extend([w_b_linear_q, w_b_linear_v])

        self.reset_parameters()
        self.sam = model

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, batched_input, multimask_output):
        return self.sam(batched_input, multimask_output)
