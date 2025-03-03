import math
from typing import List, Union, Optional

import torch
import torch.nn as nn

from segment_anything.modeling import Sam

try:
    import bitsandbytes as bnb
    _have_bnb = True
except ImportError:
    _have_bnb = False


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
        block: The chosen attention blocks for implementing LoRA.
        start_layer: The layer from which to start applying LoRA if late lora is used.
    """
    def __init__(self, rank: int, block: nn.Module, start_layer: int = -1):
        super().__init__()

        self.block = block
        block.attn.qkv = AttnLoRA(rank, block.attn.qkv, late_lora=start_layer)
        if start_layer >= 0:
            block.attn.mlp = MLPLoRA(rank, block.mlp)

    def forward(self, x):
        return x


class AttnLoRA(nn.Module):

    def __init__(self, rank: int, layer: nn.Module, late_lora: int = -1):
        super().__init__()
        self.qkv_proj = layer
        self.dim = self.qkv_proj.in_features
        self.alpha = 1  # From our experiments, 'alpha' as 1 gives the best performance.
        self.rank = rank
        self.late_lora = late_lora

        self.w_a_linear_q = nn.Linear(self.dim, self.rank, bias=False)
        self.w_b_linear_q = nn.Linear(self.rank, self.dim, bias=False)
        self.w_a_linear_v = nn.Linear(self.dim, self.rank, bias=False)
        self.w_b_linear_v = nn.Linear(self.rank, self.dim, bias=False)
        if self.late_lora >= 0:
            self.w_a_linear_k = nn.Linear(self.dim, self.rank, bias=False)
            self.w_b_linear_k = nn.Linear(self.rank, self.dim, bias=False)

        self.reset_parameters()

        layer = self

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_a_linear_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_a_linear_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.w_b_linear_q.weight)
        nn.init.zeros_(self.w_b_linear_v.weight)
        if self.late_lora >= 0:
            nn.init.kaiming_uniform_(self.w_a_linear_k.weight, a=math.sqrt(5))
            nn.init.zeros_(self.w_b_linear_k.weight)

    def forward(self, x):
        qkv = self.qkv_proj(x)  # B, N, N, 3 * org_C
        new_q = self.alpha * self.w_b_linear_q(self.w_a_linear_q(x))
        new_v = self.alpha * self.w_b_linear_v(self.w_a_linear_v(x))
        new_k = self.alpha * self.w_b_linear_k(self.w_a_linear_k(x)) if self.late_lora >= 0 else 0
        qkv = torch.cat(
            [
                qkv[:, :, :, :self.dim] + new_q,  # replacing new q values
                qkv[:, :, :, self.dim:-self.dim] + new_k,  # leaving the middle part as identical
                qkv[:, :, :, -self.dim:] + new_v  # replacing new v values
            ], dim=-1
        )

        return qkv


class MLPLoRA(nn.Module):
    def __init__(self, rank, layer):
        super().__init__()
        self.layer = layer
        self.rank = rank
        self.w_a_linear_1 = nn.Linear(layer.lin1.in_features, rank, bias=False)
        self.w_b_linear_1 = nn.Linear(rank, layer.lin1.out_features, bias=False)
        self.w_a_linear_2 = nn.Linear(layer.lin2.in_features, rank, bias=False)
        self.w_b_linear_2 = nn.Linear(rank, layer.lin2.out_features, bias=False)
        self.act = layer.act

        self.reset_parameters()

        layer = self

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_a_linear_1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_a_linear_2.weight, a=math.sqrt(5))
        nn.init.zeros_(self.w_b_linear_1.weight)
        nn.init.zeros_(self.w_b_linear_2.weight)

    def forward(self, x):
        x = self.layer(x)
        x = self.w_b_linear_2(self.w_a_linear_2(x))
        x = self.w_b_linear_1(self.w_a_linear_1(x))
        return x


class FacTSurgery(nn.Module):
    """Operates on the attention layers for performing factorized attention.

    (Inspired from: https://github.com/cchen-cc/MA-SAM/blob/main/MA-SAM/sam_fact_tt_image_encoder.py)

    Args:
        rank: The rank of the decomposition matrices for updating weights in each attention layer.
        block: The chosen attention blocks for implementing fact.
        dropout: The dropout rate for the factorized attention.
    """
    def __init__(
        self,
        rank: int,
        block: nn.Module,
        dropout: Optional[float] = 0.1,
    ):
        super().__init__()
        self.qkv_proj = block.attn.qkv
        self.dim = self.qkv_proj.in_features

        self.q_FacTs = nn.Linear(rank, rank, bias=False)
        self.v_FacTs = nn.Linear(rank, rank, bias=False)

        self.dropout = dropout
        if self.dropout is not None:
            self.dp_q = nn.Dropout(self.dropout)
            self.dp_v = nn.Dropout(self.dropout)

        self.FacTu = nn.Linear(self.dim, rank, bias=False)
        self.FacTv = nn.Linear(rank, self.dim, bias=False)

        block.attn.qkv = self

    def forward(self, x):
        qkv = self.qkv_proj(x)

        new_q = self.q_FacTs(self.FacTu(x))
        new_v = self.v_FacTs(self.FacTu(x))

        if self.dropout is not None:
            new_q = self.dp_q(new_q)
            new_v = self.dp_v(new_v)

        new_q = self.FacTv(new_q)
        new_v = self.FacTv(new_v)

        # NOTE : Scaling Factor is set to 1 as it can be tuned via the learning rate.
        qkv = torch.cat(
            [
                qkv[:, :, :, :self.dim] + new_q,  # replacing new q values
                qkv[:, :, :, self.dim:-self.dim],  # leaving the middle part as identical
                qkv[:, :, :, -self.dim:] + new_v  # replacing new v values
            ], dim=-1
        )

        return qkv


class ScaleShiftLayer(nn.Module):
    def __init__(self, layer, dim):
        super().__init__()
        self.layer = layer
        self.scale = nn.Parameter(torch.normal(mean=1.0, std=0.2, size=(dim,)))
        self.shift = nn.Parameter(torch.normal(mean=0.0, std=0.2, size=(dim,)))
        layer = self

    def forward(self, x):
        x = self.layer(x)
        assert self.scale.shape == self.shift.shape
        if x.shape[-1] == self.scale.shape[0]:
            return x * self.scale + self.shift
        elif x.shape[1] == self.scale.shape[0]:
            return x * self.scale.view(1, -1, 1, 1) + self.shift.view(1, -1, 1, 1)
        else:
            raise ValueError('Input tensors do not match the shape of the scale factors.')


class SSFSurgery(nn.Module):
    """Operates on all layers in the transformer block for adding learnable scale and shift parameters.

    Args:
        rank: This parameter is not used in `SSFSurgery`. This is kept here for consistency.
        block: The chosen attention blocks for implementing ssf.
        dim: The input dimensions determining the shape of scale and shift parameters.
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
        """This function decides the parameter attributes to match for allowing gradient updates.

        Args:
            prefix: Matches the part of parameter name in front.
            suffix: Matches the part of parameter name at the end.
            infix: Matches parts of parameter name occuring in between.
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


class AdaptFormer(nn.Module):
    """Adds AdaptFormer Module in place of the MLP Layers

    Args:
        rank: The rank is not used in this class but kept here for consistency.
        block: The chosen encoder block for implementing AdaptFormer.
        alpha: A parameters that scales the Adapter path. Can be either learnable or some fixed value.
        dropout: The dropout rate for the dropout layer between down and up projection layer.
        projection_size: The size of the projection layer.
    """
    def __init__(
        self,
        rank: int,
        block: nn.Module,
        alpha: Optional[Union[str, float]] = "learnable_scalar",  # Stable choice from our preliminary exp.
        dropout: Optional[float] = None,  # Does not have an obvious advantage.
        projection_size: int = 64,  # Stable choice from our preliminary exp.
    ):
        super().__init__()

        self.mlp_proj = block.mlp
        self.n_embd = block.mlp.lin1.in_features

        if alpha == 'learnable_scalar':
            self.alpha = nn.Parameter(torch.ones(1))
        else:
            self.alpha = alpha

        self.projection_size = projection_size
        self.dropout = dropout

        self.down_proj = nn.Linear(self.n_embd, self.projection_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.projection_size, self.n_embd)

        block.mlp = self

        if self.dropout is not None:
            self.dropout_layer = nn.Dropout(self.dropout)

        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        residual = x
        mlp_output = self.mlp_proj(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)

        if self.dropout is not None:
            down = self.dropout_layer(down)

        up = self.up_proj(down)
        up = up * self.alpha
        output = up + residual + mlp_output

        return output


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
        quantize: Whether to quantize the model for lower precision training.
    """

    def __init__(
        self,
        model: Sam,
        rank: Optional[int] = None,
        peft_module: nn.Module = LoRASurgery,
        attention_layers_to_update: Union[List[int]] = None,
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
        else:   # Applies PEFT to the image encoder by default
            self.peft_layers = list(range(len(model.image_encoder.blocks)))

        self.peft_module = peft_module
        self.peft_blocks = []

        # Whether to quantize the linear layers to 4 bit precision.
        # NOTE: This is currently supported for CUDA-supported devices only.
        if quantize:
            if not _have_bnb:
                raise ModuleNotFoundError("Please install 'bitsandbytes'.")

            for name, module in model.image_encoder.named_modules():
                if isinstance(module, torch.nn.Linear):
                    *parent_path, layer_name = name.split(".")
                    parent_module = model.image_encoder

                    for sub_module in parent_path:
                        parent_module = getattr(parent_module, sub_module)

                    # Create the new Linear4bit layer
                    linear_q = bnb.nn.Linear4bit(
                        module.in_features,
                        module.out_features,
                        bias=False if module.bias is None else True,
                    )
                    # Assign weights and bias to the new layer
                    new_weight = bnb.nn.Params4bit(
                        data=module.weight,
                        requires_grad=False,
                    )
                    linear_q.weight = new_weight
                    if module.bias is not None:
                        linear_q.bias = torch.nn.Parameter(module.bias)

                    # Replace the original linear layer with the quantized one
                    setattr(parent_module, layer_name, linear_q)

        # Let's freeze all the pretrained image encoder layers first
        for param in model.image_encoder.parameters():
            param.requires_grad = False

        # Add scale and shift parameters to the patch embedding layers.
        if issubclass(self.peft_module, SSFSurgery):
            self.peft_blocks.append(self.peft_module(rank=rank, block=model.image_encoder.patch_embed))

        start_layer = module_kwargs.get("start_layer", None)

        for t_layer_i, blk in enumerate(model.image_encoder.blocks):
            # If we only want specific layers with PEFT instead of all

            # For late lora, we only apply lora on the layers after a certain "start layer".
            if start_layer is not None:
                if t_layer_i <= start_layer:
                    continue

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
