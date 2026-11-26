"""Shared modules for parameter efficient finetuning (PEFT) of SAM and SAM2."""
import math
from typing import Callable, List, Optional, Union

try:
    import bitsandbytes as bnb
    HAVE_BITSANDBYTES = True
except ImportError:
    HAVE_BITSANDBYTES = False

import torch
import torch.nn as nn


def quantize_linear_layers(image_encoder: nn.Module):
    """Replace each linear layer with a 4-bit bitsandbytes layer.

    Args:
        image_encoder: The image encoder module whose linear layers are quantized in place.
    """
    if not HAVE_BITSANDBYTES:
        raise ModuleNotFoundError("'bitsandbytes' is not available. Install it with 'pip install bitsandbytes'.")

    for name, module in image_encoder.named_modules():
        if isinstance(module, torch.nn.Linear):
            *parent_path, layer_name = name.split(".")
            parent_module = image_encoder

            for sub_module in parent_path:
                parent_module = getattr(parent_module, sub_module)

            linear_q = bnb.nn.Linear4bit(
                module.in_features,
                module.out_features,
                bias=False if module.bias is None else True,
            )
            linear_q.weight = bnb.nn.Params4bit(data=module.weight, requires_grad=False)
            if module.bias is not None:
                linear_q.bias = torch.nn.Parameter(module.bias)

            setattr(parent_module, layer_name, linear_q)


class AttentionLoRA(nn.Module):
    """Low-rank adaptation on a fused qkv projection.

    This module supports the symmetric SAM projection and the asymmetric SAM2 Hiera projection.

    Args:
        rank: The rank of the decomposition matrices for updating weights in each attention layer.
        block: The chosen qkv projection layer for implementing LoRA.
        update_matrices: The matrices to update in the attention layer. Choose "q", "k", or "v".
    """
    def __init__(self, rank: int, block: nn.Module, update_matrices: List[str] = ["q", "v"]):
        super().__init__()
        self.qkv_proj = block
        self.in_dim = block.in_features
        self.out_dim = block.out_features // 3
        self.alpha = 1  # The experiments gave the best results with alpha set to 1.
        self.rank = rank

        # By default, we follow LoRA's recommended setup, i.e. update the "q" and "v" matrices.
        if "q" in update_matrices:
            self.w_a_linear_q = nn.Linear(self.in_dim, self.rank, bias=False)
            self.w_b_linear_q = nn.Linear(self.rank, self.out_dim, bias=False)

        if "v" in update_matrices:
            self.w_a_linear_v = nn.Linear(self.in_dim, self.rank, bias=False)
            self.w_b_linear_v = nn.Linear(self.rank, self.out_dim, bias=False)

        if "k" in update_matrices:
            self.w_a_linear_k = nn.Linear(self.in_dim, self.rank, bias=False)
            self.w_b_linear_k = nn.Linear(self.rank, self.out_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, "w_a_linear_q"):
            nn.init.kaiming_uniform_(self.w_a_linear_q.weight, a=math.sqrt(5))
            nn.init.zeros_(self.w_b_linear_q.weight)

        if hasattr(self, "w_a_linear_v"):
            nn.init.kaiming_uniform_(self.w_a_linear_v.weight, a=math.sqrt(5))
            nn.init.zeros_(self.w_b_linear_v.weight)

        if hasattr(self, "w_a_linear_k"):
            nn.init.kaiming_uniform_(self.w_a_linear_k.weight, a=math.sqrt(5))
            nn.init.zeros_(self.w_b_linear_k.weight)

    def forward(self, x):
        qkv = self.qkv_proj(x)  # (..., 3 * out_dim)
        d = self.out_dim

        new_q = self.alpha * self.w_b_linear_q(self.w_a_linear_q(x)) if hasattr(self, "w_a_linear_q") else 0
        new_v = self.alpha * self.w_b_linear_v(self.w_a_linear_v(x)) if hasattr(self, "w_a_linear_v") else 0
        new_k = self.alpha * self.w_b_linear_k(self.w_a_linear_k(x)) if hasattr(self, "w_a_linear_k") else 0
        qkv = torch.cat(
            [
                qkv[..., :d] + new_q,
                qkv[..., d:2 * d] + new_k,
                qkv[..., 2 * d:] + new_v,
            ],
            dim=-1,
        )

        return qkv


class MLPLoRA(nn.Module):
    """Low-rank adaptation on a two-layer feed forward block.

    The `get_layers` function supports the different MLP layouts in SAM and SAM2.

    Args:
        rank: The rank of the decomposition matrices for updating weights in each feed forward layer.
        mlp_layer: The chosen MLP layer for implementing LoRA.
        get_layers: Callable mapping `mlp_layer` to the tuple `(lin1, lin2, activation)`.
    """
    def __init__(self, rank: int, mlp_layer: nn.Module, get_layers: Callable):
        super().__init__()
        self.mlp_layer = mlp_layer
        self.get_layers = get_layers
        self.rank = rank
        lin1, lin2, _ = get_layers(mlp_layer)
        self.w_a_linear_1 = nn.Linear(lin1.in_features, rank, bias=False)
        self.w_b_linear_1 = nn.Linear(rank, lin1.out_features, bias=False)
        self.w_a_linear_2 = nn.Linear(lin2.in_features, rank, bias=False)
        self.w_b_linear_2 = nn.Linear(rank, lin2.out_features, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_a_linear_1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_a_linear_2.weight, a=math.sqrt(5))
        nn.init.zeros_(self.w_b_linear_1.weight)
        nn.init.zeros_(self.w_b_linear_2.weight)

    def forward(self, x):
        lin1, lin2, activation = self.get_layers(self.mlp_layer)
        x = lin1(x) + self.w_b_linear_1(self.w_a_linear_1(x))
        x = activation(x)
        x = lin2(x) + self.w_b_linear_2(self.w_a_linear_2(x))
        return x


class ScaleShiftLayer(nn.Module):
    """Wraps a layer to apply learnable per-channel scale and shift to its output (used by SSF)."""
    def __init__(self, layer, dim):
        super().__init__()
        self.layer = layer
        self.scale = nn.Parameter(torch.normal(mean=1.0, std=0.2, size=(dim,)))
        self.shift = nn.Parameter(torch.normal(mean=0.0, std=0.2, size=(dim,)))

    def forward(self, x):
        x = self.layer(x)
        assert self.scale.shape == self.shift.shape
        if x.shape[-1] == self.scale.shape[0]:
            return x * self.scale + self.shift
        elif x.shape[1] == self.scale.shape[0]:
            return x * self.scale.view(1, -1, 1, 1) + self.shift.view(1, -1, 1, 1)
        else:
            raise ValueError(
                f"The input shape {tuple(x.shape)} does not match the scale shape {tuple(self.scale.shape)}."
            )


class SelectiveSurgery(nn.Module):
    """Allow gradient updates for selected parameters."""
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def allow_gradient_update_for_parameters(
        self,
        prefix: Optional[List[str]] = None,
        suffix: Optional[List[str]] = None,
        infix: Optional[List[str]] = None,
    ):
        """Allow gradient updates for parameter names that match the filters.

        Args:
            prefix: Matches the part of parameter name in front.
            suffix: Matches the part of parameter name at the end.
            infix: Matches parts of parameter name occurring in between.
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
    """Allow gradient updates for parameters in attention layers."""

    def __init__(self, block: nn.Module):
        super().__init__(block=block)
        self.allow_gradient_update_for_parameters(prefix=["attn"])


class BiasSurgery(SelectiveSurgery):
    """Allow gradient updates for bias parameters."""

    def __init__(self, block: nn.Module):
        super().__init__(block=block)
        self.allow_gradient_update_for_parameters(suffix=["bias"])


class LayerNormSurgery(SelectiveSurgery):
    """Allow gradient updates for parameters in normalization layers."""

    def __init__(self, block: nn.Module):
        super().__init__(block=block)
        self.allow_gradient_update_for_parameters(infix=["norm1", "norm2"])


class ClassicalSurgery(SelectiveSurgery):
    """Allow gradient updates for all parameters in selected blocks."""

    def __init__(self, block: nn.Module):
        super().__init__(block=block)
        self.block = block

        for k, v in self.block.named_parameters():
            v.requires_grad = True

    def forward(self, x):
        return x


class FacTSurgery(nn.Module):
    """Apply factorized attention to attention layers.

    Based on https://github.com/cchen-cc/MA-SAM/blob/main/MA-SAM/sam_fact_tt_image_encoder.py.
    This module supports the symmetric SAM projection and the asymmetric SAM2 Hiera projection.

    Args:
        rank: The rank of the decomposition matrices for updating weights in each attention layer.
        block: The chosen attention block for implementing fact.
        dropout: The dropout rate for the factorized attention.
    """
    def __init__(self, rank: int, block: nn.Module, dropout: Optional[float] = 0.1):
        super().__init__()
        self.qkv_proj = block.attn.qkv
        self.in_dim = self.qkv_proj.in_features
        self.out_dim = self.qkv_proj.out_features // 3

        self.q_FacTs = nn.Linear(rank, rank, bias=False)
        self.v_FacTs = nn.Linear(rank, rank, bias=False)

        self.dropout = dropout
        if self.dropout is not None:
            self.dp_q = nn.Dropout(self.dropout)
            self.dp_v = nn.Dropout(self.dropout)

        self.FacTu = nn.Linear(self.in_dim, rank, bias=False)
        self.FacTv = nn.Linear(rank, self.out_dim, bias=False)

        block.attn.qkv = self

    def forward(self, x):
        qkv = self.qkv_proj(x)
        d = self.out_dim

        new_q = self.q_FacTs(self.FacTu(x))
        new_v = self.v_FacTs(self.FacTu(x))

        if self.dropout is not None:
            new_q = self.dp_q(new_q)
            new_v = self.dp_v(new_v)

        new_q = self.FacTv(new_q)
        new_v = self.FacTv(new_v)

        # The learning rate controls the scale, so the scaling factor is 1.
        qkv = torch.cat(
            [
                qkv[..., :d] + new_q,
                qkv[..., d:2 * d],
                qkv[..., 2 * d:] + new_v,
            ],
            dim=-1,
        )

        return qkv


class AdaptFormer(nn.Module):
    """Adds an AdaptFormer module in place of the MLP layers.

    Args:
        rank: The rank is not used in this class but kept here for consistency.
        block: The chosen encoder block for implementing AdaptFormer.
        alpha: The value that scales the adapter path. Pass a number or `"learnable_scalar"`.
        dropout: The dropout rate for the dropout layer between the down and up projection layer.
        projection_size: The size of the projection layer.
    """
    def __init__(
        self,
        rank: int,
        block: nn.Module,
        alpha: Optional[Union[str, float]] = "learnable_scalar",  # Preliminary experiments support this value.
        dropout: Optional[float] = None,  # Preliminary experiments show no clear benefit.
        projection_size: int = 64,  # Preliminary experiments support this value.
    ):
        super().__init__()

        self.mlp_proj = block.mlp
        # SAM and SAM2 use different attributes for the first MLP layer.
        self.n_embd = block.mlp.lin1.in_features if hasattr(block.mlp, "lin1") else block.mlp.layers[0].in_features

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
        mlp_output = self.mlp_proj(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)

        if self.dropout is not None:
            down = self.dropout_layer(down)

        up = self.up_proj(down)
        up = up * self.alpha
        output = up + mlp_output

        return output


def serialize_peft_kwargs(peft_kwargs: Optional[dict]) -> Optional[dict]:
    """Convert `peft_kwargs` to a dictionary that a checkpoint can store.

    Args:
        peft_kwargs: The PEFT keyword arguments, or None.

    Returns:
        The serialized config, or None if `peft_kwargs` is empty.
    """
    if not peft_kwargs:
        return None
    serialized = dict(peft_kwargs)
    module = serialized.get("peft_module")
    if module is not None and not isinstance(module, str):
        serialized["peft_module"] = module.__name__
    return serialized


def deserialize_peft_kwargs(config: Optional[dict], module_registry: dict) -> Optional[dict]:
    """Build `peft_kwargs` from a saved configuration.

    Args:
        config: The serialized config (as produced by `serialize_peft_kwargs`), or None.
        module_registry: The mapping from each PEFT module name to its class.

    Returns:
        The deserialized `peft_kwargs`, or None if `config` is empty.
    """
    if not config:
        return None
    deserialized = dict(config)
    module = deserialized.get("peft_module")
    if isinstance(module, str):
        if module not in module_registry:
            raise ValueError(f"Unknown PEFT module '{module}'. Known modules: {sorted(module_registry)}")
        deserialized["peft_module"] = module_registry[module]
    return deserialized
