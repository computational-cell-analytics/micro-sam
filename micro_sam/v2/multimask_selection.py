"""Torch feature extraction and compact MLP models for APG mask scoring.

The SAM2 decoder returns three alternatives for an ambiguous point prompt. This module turns the
already-computed masks, quality scores and APG decoder output into a small, versioned feature vector.
It deliberately has no dependency on the APG generator so that evaluation tooling can train and
serialize selectors without duplicating the runtime feature definition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch


MULTIMASK_FEATURE_VERSION = 1
MASK_TOKEN_DIMENSION = 256
MULTIMASK_FEATURE_NAMES = (
    "predicted_iou",
    "stability",
    "predicted_iou_x_stability",
    "score_delta_from_best",
    "stability_delta_from_best",
    "log_area",
    "log_bounding_box_area",
    "bounding_box_occupancy",
    "alternative_index",
    "score_rank",
    "area_rank",
    "foreign_seed_count",
    "positive_seed_containment",
    "foreground_mean",
    "foreground_precision",
    "log_nearest_seed_distance",
    "log_area_per_seed_distance_squared",
    "pairwise_iou_mean",
    "area_over_triplet_median",
)
REFINEMENT_GATE_FEATURE_NAMES = MULTIMASK_FEATURE_NAMES + (
    "selection_score",
    "selection_margin",
    "selection_score_spread",
    "raw_and_selected_disagree",
)
POSTMERGE_REFINEMENT_GATE_FEATURE_NAMES = (
    "predicted_iou",
    "stability",
    "predicted_iou_x_stability",
    "selection_score",
    "selection_minus_predicted_iou",
    "merge_score",
    "score_filter_margin",
    "alternative_index",
    "log_source_area",
    "log_visible_area",
    "visible_fraction",
    "log_visible_box_area",
    "visible_box_occupancy",
    "log_box_aspect_ratio",
    "border_contact_fraction",
    "foreground_mean",
    "foreground_precision",
    "claimed_fraction",
    "log_instance_count",
    "log_nearest_instance_seed_distance",
    "grouped_prompt_count",
    "positive_prompt_count",
    "negative_prompt_count",
    "log_nearest_negative_distance",
    "log_mean_negative_distance",
)

MASK_TOKEN_FEATURE_NAMES = (
    "predicted_iou", "alternative_index",
) + tuple(f"mask_token_{index}" for index in range(MASK_TOKEN_DIMENSION))
MASK_TOKEN_LOWRES_FEATURE_NAMES = MULTIMASK_FEATURE_NAMES + tuple(
    f"mask_token_{index}" for index in range(MASK_TOKEN_DIMENSION)
)
SELECTOR_FEATURE_SCHEMAS = {
    "dense_v1": MULTIMASK_FEATURE_NAMES,
    "lowres_v1": MULTIMASK_FEATURE_NAMES,
    "token_v1": MASK_TOKEN_FEATURE_NAMES,
    "token_lowres_v1": MASK_TOKEN_LOWRES_FEATURE_NAMES,
}


def selector_input_schema(scorer) -> str:
    """Return the versioned runtime input schema requested by a selector artifact."""
    schema = getattr(scorer, "input_schema", "dense_v1")
    if schema not in SELECTOR_FEATURE_SCHEMAS:
        raise ValueError(f"Unsupported multimask selector input schema {schema!r}.")
    return schema


def combine_selector_features_torch(
    schema: str, lowres_features: Optional[torch.Tensor], scores: torch.Tensor,
    mask_tokens: Optional[torch.Tensor],
) -> torch.Tensor:
    """Assemble one of the compact three-alternative selector schemas on the decoder device."""
    if schema not in SELECTOR_FEATURE_SCHEMAS:
        raise ValueError(f"Unsupported multimask selector input schema {schema!r}.")
    scores = torch.as_tensor(scores, dtype=torch.float32)
    if scores.ndim != 2:
        raise ValueError(f"Expected grouped predicted-IoU scores, got {tuple(scores.shape)}.")
    n_prompts, n_alternatives = scores.shape
    if n_alternatives != 3:
        raise ValueError(f"Compact mask-token scoring expects three multimask alternatives, got {n_alternatives}.")
    if schema in ("dense_v1", "lowres_v1"):
        if lowres_features is None:
            raise ValueError(f"Selector schema {schema!r} requires mask features.")
        return lowres_features.to(torch.float32)
    if mask_tokens is None:
        raise ValueError(f"Selector schema {schema!r} requires SAM2 mask tokens.")
    mask_tokens = torch.as_tensor(mask_tokens, dtype=torch.float32, device=scores.device)
    expected = (n_prompts, n_alternatives, MASK_TOKEN_DIMENSION)
    if tuple(mask_tokens.shape) != expected:
        raise ValueError(f"Expected mask tokens with shape {expected}, got {tuple(mask_tokens.shape)}.")
    if schema == "token_v1":
        alternative = torch.arange(
            n_alternatives, dtype=torch.float32, device=scores.device,
        )[None, :, None].expand(n_prompts, -1, -1)
        return torch.cat((scores[:, :, None], alternative, mask_tokens), dim=2)
    if lowres_features is None:
        raise ValueError("The token_lowres_v1 schema requires low-resolution mask features.")
    return torch.cat((lowres_features.to(torch.float32), mask_tokens), dim=2)


def extract_multimask_features_torch(
    masks: torch.Tensor,
    scores: torch.Tensor,
    stability: torch.Tensor,
    points: Union[np.ndarray, torch.Tensor],
    foreground: Union[np.ndarray, torch.Tensor],
    foreground_threshold: float,
    context_points: Optional[Union[np.ndarray, torch.Tensor]] = None,
    prompt_indices: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> torch.Tensor:
    """Extract one feature vector per alternative without leaving the decoder device."""
    if not isinstance(masks, torch.Tensor):
        raise TypeError("The Torch multimask extractor expects masks as a torch.Tensor.")
    device = masks.device
    masks = masks.to(dtype=torch.bool)
    scores = torch.as_tensor(scores, dtype=torch.float32, device=device)
    stability = torch.as_tensor(stability, dtype=torch.float32, device=device)
    points = torch.as_tensor(points, dtype=torch.float32, device=device)
    foreground = torch.as_tensor(foreground, dtype=torch.float32, device=device)
    if masks.ndim != 4:
        raise ValueError(f"Expected multimasks with shape (N, K, Y, X), got {tuple(masks.shape)}.")
    n_prompts, n_alternatives, height, width = masks.shape
    expected = (n_prompts, n_alternatives)
    if tuple(scores.shape) != expected or tuple(stability.shape) != expected:
        raise ValueError(
            f"Expected score and stability shapes {expected}, got {tuple(scores.shape)} and "
            f"{tuple(stability.shape)}."
        )
    if tuple(points.shape) != (n_prompts, 2):
        raise ValueError(f"Expected point shape {(n_prompts, 2)}, got {tuple(points.shape)}.")
    if tuple(foreground.shape) != (height, width):
        raise ValueError(f"Expected foreground shape {(height, width)}, got {tuple(foreground.shape)}.")

    xy = points.round().to(dtype=torch.int64)
    xy[:, 0].clamp_(0, width - 1)
    xy[:, 1].clamp_(0, height - 1)
    context_xy = xy if context_points is None else torch.as_tensor(
        context_points, dtype=torch.float32, device=device,
    ).round().to(dtype=torch.int64)
    context_xy[:, 0].clamp_(0, width - 1)
    context_xy[:, 1].clamp_(0, height - 1)
    if prompt_indices is None:
        prompt_indices = torch.arange(n_prompts, dtype=torch.int64, device=device)
    else:
        prompt_indices = torch.as_tensor(prompt_indices, dtype=torch.int64, device=device)
    if tuple(prompt_indices.shape) != (n_prompts,):
        raise ValueError("prompt_indices must locate every decoded prompt in context_points.")
    # On CUDA the indexing operations below validate the values without forcing two device-wide
    # synchronizations per decoder batch. Retain the friendlier eager check on CPU.
    if device.type == "cpu" and (
        bool((prompt_indices < 0).any()) or bool((prompt_indices >= len(context_xy)).any())
    ):
        raise ValueError("prompt_indices must locate every decoded prompt in context_points.")

    areas = masks.sum(dim=(2, 3), dtype=torch.float32)
    median_area = areas.median(dim=1).values.clamp_min(1.0)
    image_scale = float(max(height, width))
    if len(context_xy) > 1:
        distances = torch.cdist(xy.to(torch.float32), context_xy.to(torch.float32))
        distances[torch.arange(n_prompts, device=device), prompt_indices] = torch.inf
        nearest = distances.min(dim=1).values.clamp_max(image_scale)
    else:
        nearest = torch.full((n_prompts,), image_scale, dtype=torch.float32, device=device)

    rows = masks.any(dim=3)
    columns = masks.any(dim=2)
    nonempty = rows.any(dim=2)
    y0 = rows.to(torch.int64).argmax(dim=2)
    y1 = height - rows.flip(2).to(torch.int64).argmax(dim=2)
    x0 = columns.to(torch.int64).argmax(dim=2)
    x1 = width - columns.flip(2).to(torch.int64).argmax(dim=2)
    box_area = ((y1 - y0) * (x1 - x0)).to(torch.float32) * nonempty

    score_ranks = torch.argsort(
        torch.argsort(scores, dim=1, descending=True, stable=True), dim=1, stable=True,
    ).to(torch.float32)
    area_ranks = torch.argsort(
        torch.argsort(areas, dim=1, stable=True), dim=1, stable=True,
    ).to(torch.float32)
    prompt_rows = torch.arange(n_prompts, device=device)[:, None]
    alternative_columns = torch.arange(n_alternatives, device=device)[None, :]
    contains_seed = masks[prompt_rows, alternative_columns, xy[:, 1, None], xy[:, 0, None]].to(torch.float32)
    foreign = masks[:, :, context_xy[:, 1], context_xy[:, 0]].sum(dim=2, dtype=torch.float32) - contains_seed

    foreground_means, foreground_precisions = [], []
    foreground_binary = foreground > foreground_threshold
    for alternative in range(n_alternatives):
        alternative_mask = masks[:, alternative]
        denominator = areas[:, alternative].clamp_min(1.0)
        foreground_means.append(
            (alternative_mask * foreground[None]).sum(dim=(1, 2)) / denominator
        )
        foreground_precisions.append(
            (alternative_mask & foreground_binary[None]).sum(dim=(1, 2), dtype=torch.float32) / denominator
        )
    foreground_mean = torch.stack(foreground_means, dim=1)
    foreground_precision = torch.stack(foreground_precisions, dim=1)

    if n_alternatives == 1:
        pairwise_iou = torch.ones_like(areas)
    else:
        pairwise_sums = torch.zeros_like(areas)
        for first in range(n_alternatives):
            for second in range(first + 1, n_alternatives):
                intersection = (masks[:, first] & masks[:, second]).sum(dim=(1, 2), dtype=torch.float32)
                union = areas[:, first] + areas[:, second] - intersection
                iou = torch.where(union > 0, intersection / union, torch.zeros_like(union))
                pairwise_sums[:, first] += iou
                pairwise_sums[:, second] += iou
        pairwise_iou = pairwise_sums / float(n_alternatives - 1)

    nearest_squared = (nearest * nearest).clamp_min(1.0)
    features = torch.stack((
        scores,
        stability,
        scores * stability,
        scores - scores.max(dim=1, keepdim=True).values,
        stability - stability.max(dim=1, keepdim=True).values,
        torch.log1p(areas),
        torch.log1p(box_area),
        torch.where(box_area > 0, areas / box_area, torch.zeros_like(areas)),
        torch.arange(n_alternatives, dtype=torch.float32, device=device)[None].expand(n_prompts, -1),
        score_ranks,
        area_ranks,
        foreign,
        contains_seed,
        foreground_mean,
        foreground_precision,
        torch.log1p(nearest)[:, None].expand(-1, n_alternatives),
        torch.log1p(areas / nearest_squared[:, None]),
        pairwise_iou,
        areas / median_area[:, None],
    ), dim=2).to(torch.float32)
    return features


def refinement_gate_features_torch(
    features: torch.Tensor, selection_scores: torch.Tensor, selected: torch.Tensor,
) -> torch.Tensor:
    """Build one pre-refinement feature row for each selected prompt mask."""
    if features.ndim != 3 or tuple(selection_scores.shape) != tuple(features.shape[:2]):
        raise ValueError("Expected grouped multimask features and selection scores.")
    selected = torch.as_tensor(selected, dtype=torch.int64, device=features.device)
    if tuple(selected.shape) != (len(features),):
        raise ValueError(f"Expected one selected index per prompt, got {tuple(selected.shape)}.")
    rows = torch.arange(len(features), device=features.device)
    selected_scores = selection_scores[rows, selected]
    competitors = selection_scores.clone()
    competitors[rows, selected] = -torch.inf
    margin = (
        selected_scores - competitors.max(dim=1).values
        if selection_scores.shape[1] > 1 else torch.ones_like(selected_scores)
    )
    raw_best = features[:, :, 0].argmax(dim=1)
    extras = torch.stack((
        selected_scores,
        margin,
        selection_scores.max(dim=1).values - selection_scores.min(dim=1).values,
        (raw_best != selected).to(torch.float32),
    ), dim=1)
    return torch.cat((features[rows, selected], extras), dim=1).to(torch.float32)


class TorchFeatureScorer:
    """A small standardized MLP feature regressor."""

    kind = "mlp"

    def __init__(
        self, module: torch.nn.Module, mean: Sequence[float], scale: Sequence[float], metadata=None,
        feature_names: Sequence[str] = MULTIMASK_FEATURE_NAMES,
    ):
        self.module = module.eval()
        self.feature_names = tuple(feature_names)
        parameter = next(self.module.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
        self.mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
        self.scale = torch.as_tensor(scale, dtype=torch.float32, device=device)
        self.scale = torch.where(self.scale == 0, torch.ones_like(self.scale), self.scale)
        self.metadata = dict(metadata or {})
        self.input_schema = str(self.metadata.get("input_schema", "dense_v1"))
        self.output_activation = str(self.metadata.get("output_activation", "clamp"))
        self.gate_stage = str(self.metadata.get("gate_stage", "premerge"))
        if self.output_activation not in ("clamp", "identity"):
            raise ValueError(f"Unsupported scorer output activation {self.output_activation!r}.")

    def predict_tensor(self, features: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        parameter = next(self.module.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
        values = torch.as_tensor(features, dtype=torch.float32, device=device)
        with torch.no_grad():
            prediction = self.module((values - self.mean) / self.scale).reshape(-1)
        if self.output_activation == "clamp":
            prediction = prediction.clamp(0.0, 1.0)
        return prediction.to(torch.float32)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_tensor(features).cpu().numpy().astype("float32")


class GroupwiseMLP(torch.nn.Module):
    """Permutation-equivariant scorer for a fixed group of multimask alternatives."""

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        encoder = [
            torch.nn.Linear(input_size, hidden_size), torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU(),
        ]
        if dropout:
            encoder.append(torch.nn.Dropout(dropout))
        head_size = max(hidden_size // 2, 16)
        head = [torch.nn.Linear(3 * hidden_size, head_size), torch.nn.ReLU()]
        if dropout:
            head.append(torch.nn.Dropout(dropout))
        head.append(torch.nn.Linear(head_size, 1))
        self.encoder = torch.nn.Sequential(*encoder)
        self.head = torch.nn.Sequential(*head)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"Expected grouped features with shape (N, K, F), got {tuple(features.shape)}.")
        encoded = self.encoder(features)
        mean = encoded.mean(dim=1, keepdim=True).expand_as(encoded)
        maximum = encoded.max(dim=1, keepdim=True).values.expand_as(encoded)
        return torch.sigmoid(self.head(torch.cat((encoded, mean, maximum), dim=2)).squeeze(2))


class GroupwiseTorchFeatureScorer:
    """Standardized Torch wrapper for a fixed-size alternative group MLP."""

    kind = "groupwise_mlp"

    def __init__(
        self, module: GroupwiseMLP, mean: Sequence[float], scale: Sequence[float], metadata=None,
        feature_names: Sequence[str] = MULTIMASK_FEATURE_NAMES, n_alternatives: int = 3,
    ) -> None:
        self.module = module.eval()
        self.feature_names = tuple(feature_names)
        parameter = next(self.module.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
        self.mean = torch.as_tensor(mean, dtype=torch.float32, device=device)
        self.scale = torch.as_tensor(scale, dtype=torch.float32, device=device)
        self.scale = torch.where(self.scale == 0, torch.ones_like(self.scale), self.scale)
        self.metadata = dict(metadata or {})
        self.input_schema = str(self.metadata.get("input_schema", "dense_v1"))
        self.n_alternatives = int(n_alternatives)
        if self.n_alternatives < 1:
            raise ValueError("A groupwise scorer requires at least one alternative.")

    def predict_grouped_tensor(self, features: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        parameter = next(self.module.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
        values = torch.as_tensor(features, dtype=torch.float32, device=device)
        if values.ndim != 3 or values.shape[1] != self.n_alternatives:
            raise ValueError(
                f"Expected grouped features with shape (N, {self.n_alternatives}, F), "
                f"got {tuple(values.shape)}."
            )
        with torch.no_grad():
            return self.module((values - self.mean) / self.scale).to(torch.float32)

    def predict_grouped(self, features: np.ndarray) -> np.ndarray:
        return self.predict_grouped_tensor(features).cpu().numpy().astype("float32")


def load_feature_scorer(path: Union[str, Path], device: Union[str, torch.device] = "cpu"):
    """Load a pointwise or groupwise MLP artifact."""
    path = Path(path)
    if path.suffix in (".pt", ".pth"):
        state = torch.load(path, map_location=device, weights_only=False)
        _validate_feature_state(state)
        metadata = dict(state.get("metadata") or {})
        metadata.setdefault("input_schema", state.get("input_schema", "dense_v1"))
        if state.get("kind") not in ("mlp", "groupwise_mlp"):
            raise ValueError(f"Unsupported torch feature scorer kind {state.get('kind')!r}.")
        if state["kind"] == "groupwise_mlp":
            module = GroupwiseMLP(
                len(state["feature_names"]), int(state["hidden_size"]), float(state.get("dropout", 0.0)),
            ).to(device)
            module.load_state_dict(state["state_dict"])
            return GroupwiseTorchFeatureScorer(
                module, state["mean"], state["scale"], metadata, state["feature_names"],
                int(state.get("n_alternatives", 3)),
            )
        layers = []
        width = len(state["feature_names"])
        for hidden in state["hidden_sizes"]:
            layers.extend((torch.nn.Linear(width, int(hidden)), torch.nn.ReLU()))
            if state.get("dropout", 0.0):
                layers.append(torch.nn.Dropout(float(state["dropout"])))
            width = int(hidden)
        layers.append(torch.nn.Linear(width, 1))
        module = torch.nn.Sequential(*layers).to(device)
        module.load_state_dict(state["state_dict"])
        return TorchFeatureScorer(
            module, state["mean"], state["scale"], metadata, state["feature_names"]
        )
    raise ValueError(f"Unsupported feature scorer extension {path.suffix!r}; expected '.pt' or '.pth'.")


def _validate_feature_state(state: Dict[str, Any]) -> None:
    if state.get("feature_version") != MULTIMASK_FEATURE_VERSION:
        raise ValueError(
            f"Feature scorer version {state.get('feature_version')} does not match "
            f"runtime version {MULTIMASK_FEATURE_VERSION}."
        )
    names = tuple(state.get("feature_names", ()))
    metadata = dict(state.get("metadata") or {})
    schema = str(state.get("input_schema", metadata.get("input_schema", "dense_v1")))
    if names in (REFINEMENT_GATE_FEATURE_NAMES, POSTMERGE_REFINEMENT_GATE_FEATURE_NAMES):
        return
    expected = SELECTOR_FEATURE_SCHEMAS.get(schema)
    if expected is None or names != expected:
        raise ValueError(
            f"Feature scorer names do not match runtime schema {schema!r}."
        )
