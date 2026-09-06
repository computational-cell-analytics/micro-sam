"""Extract Torch APG mask features and train the selected groupwise H64 scorer.

The three-mask and dedicated single-mask variants share this entry point. Five deterministic,
image-level folds produce leakage-safe out-of-fold predictions for threshold screening, followed by
one refit on the complete primary subset. The holdout is only consumed by the screening and
canonical benchmark programs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from micro_sam.v2.multimask_selection import (
    GroupwiseMLP, MULTIMASK_FEATURE_NAMES, MULTIMASK_FEATURE_VERSION,
    SELECTOR_FEATURE_SCHEMAS,
)

EVALUATION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVALUATION_ROOT))

import common  # noqa
from optimization.benchmark_apg_optimization import (  # noqa
    DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, _default_manifest_path, _load_2d_sample,
    _validate_roots, prepare_manifest, MANIFEST_SUBSETS,
)


ARCHITECTURE = {"hidden_size": 64, "dropout": 0.1}

_ABSOLUTE_SIZE_FEATURES = (
    "log_area", "log_bounding_box_area", "log_nearest_seed_distance", "log_area_per_seed_distance_squared",
)
_DECODER_FEATURES = ("foreground_mean", "foreground_precision")
# Named subsets of the 19 generic mask statistics, for the generalization ablation: which inputs let a
# selector transfer to a dataset it has never seen (leave-one-dataset-out) while still helping in-domain.
GENERIC_FEATURE_SETS = {
    "lowres_all": tuple(MULTIMASK_FEATURE_NAMES),
    "iou_stab": ("predicted_iou", "stability", "predicted_iou_x_stability"),
    "sam_scores": (
        "predicted_iou", "stability", "predicted_iou_x_stability", "score_delta_from_best",
        "stability_delta_from_best", "alternative_index", "score_rank",
    ),
    "scale_free": tuple(name for name in MULTIMASK_FEATURE_NAMES if name not in _ABSOLUTE_SIZE_FEATURES),
    "no_decoder": tuple(name for name in MULTIMASK_FEATURE_NAMES if name not in _DECODER_FEATURES),
    "scale_free_no_decoder": tuple(
        name for name in MULTIMASK_FEATURE_NAMES if name not in _ABSOLUTE_SIZE_FEATURES + _DECODER_FEATURES
    ),
}
PER_IMAGE_MODES = ("none", "replace", "append")
MODEL_KINDS = ("mlp", "linear")
TARGET_KINDS = ("iou", "matched")
MATCHED_IOU = 0.5


def _per_image_standardize(features: np.ndarray, sample_ids: np.ndarray) -> np.ndarray:
    """Z-score every column within its image, so dataset-level offsets and scales drop out."""
    standardized = np.empty_like(features)
    order = np.argsort(sample_ids, kind="stable")
    ordered = sample_ids[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    stops = np.r_[starts[1:], len(order)]
    for start, stop in zip(starts, stops):
        rows = order[start:stop]
        block = features[rows]
        mean = block.mean(axis=0, keepdims=True)
        scale = block.std(axis=0, keepdims=True)
        scale[scale < 1e-6] = 1.0
        standardized[rows] = (block - mean) / scale
    return standardized


class GroupwiseLinear(torch.nn.Module):
    """One linear score per alternative; the smallest model the screen compares the MLP against."""

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)


def _build_model(input_size: int, architecture: dict) -> torch.nn.Module:
    if architecture.get("model", "mlp") == "linear":
        return GroupwiseLinear(input_size)
    return GroupwiseMLP(input_size, hidden_size=architecture["hidden_size"], dropout=architecture["dropout"])


def _target_values(targets: np.ndarray, target_kind: str) -> np.ndarray:
    if target_kind == "iou":
        return targets
    if target_kind == "matched":
        return (targets >= MATCHED_IOU).astype("float32")
    raise ValueError(f"Unknown target kind {target_kind!r}.")


def _output_values(output: torch.Tensor, target_kind: str) -> torch.Tensor:
    return torch.sigmoid(output) if target_kind == "matched" else output


def _auc(scores: np.ndarray, positives: np.ndarray) -> float:
    """Rank AUC of 'scores' for the binary 'positives'; NaN when one class is missing."""
    positives = positives.astype(bool)
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    from scipy.stats import rankdata
    ranks = rankdata(scores)
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _stable_folds(samples: Iterable[Dict[str, Any]], n_folds: int = 5) -> Dict[str, int]:
    by_dataset: Dict[str, list] = {}
    for sample in samples:
        if sample["ndim"] == 2:
            by_dataset.setdefault(sample["dataset"], []).append(sample["sample_id"])
    folds = {}
    for sample_ids in by_dataset.values():
        ordered = sorted(sample_ids, key=lambda value: hashlib.sha256(value.encode()).hexdigest())
        folds.update({sample_id: index % n_folds for index, sample_id in enumerate(ordered)})
    return folds


def _record_target(record: dict, labels: np.ndarray) -> float:
    x, y = np.round(record["point"]).astype("int64")
    x, y = int(np.clip(x, 0, labels.shape[1] - 1)), int(np.clip(y, 0, labels.shape[0] - 1))
    object_id = int(labels[y, x])
    if object_id == 0:
        return 0.0
    mask = np.asarray(record["segmentation"], dtype=bool)
    target = labels[record["bounding_box"]] == object_id
    intersection = int(np.count_nonzero(mask & target))
    union = int(mask.sum()) + int(np.count_nonzero(labels == object_id)) - intersection
    return intersection / union if union else 0.0


PROPOSAL_SETTING_KEYS = ("candidate_threshold", "foreground_threshold", "n_iter", "dt", "sigma", "min_candidate_size")


def _seeded_and_proposed(proposals: list, labels: np.ndarray, targets: list) -> Tuple[int, int, int]:
    """Objects containing a prompt point, and objects some alternative matches at IoU >= 0.5."""
    seeded, proposed = set(), set()
    for record, target in zip(proposals, targets):
        x, y = np.round(record["point"]).astype("int64")
        x, y = int(np.clip(x, 0, labels.shape[1] - 1)), int(np.clip(y, 0, labels.shape[0] - 1))
        object_id = int(labels[y, x])
        if object_id:
            seeded.add(object_id)
            if target >= 0.5:
                proposed.add(object_id)
    return int(len(np.unique(labels)) - 1), len(seeded), len(proposed)


def extract_dataset(
    manifest: dict, data_root: Path, output: Path, device: str, multimasking: bool = True,
    input_schema: str = "dense_v1", proposal_settings: Optional[List[dict]] = None,
    outputs: Optional[List[Path]] = None,
) -> Path:
    """Extract the selector features of every proposal alternative on every manifest image.

    With 'proposal_settings', several candidate-generation settings (see PROPOSAL_SETTING_KEYS) are
    proposed from one encoding and decoder prediction per image, and each setting is written to its
    own feature dataset in 'outputs'. A recall diagnostic per (image, setting) - objects, seeded
    objects, proposed objects - lands beside the first output as 'recall_diagnostic.csv'.
    """
    if not multimasking and input_schema != "dense_v1":
        raise ValueError("Compact selector schemas require the three-mask output.")
    settings = proposal_settings or [{}]
    outputs = outputs or [output]
    if len(outputs) != len(settings):
        raise ValueError("One output path per proposal setting is required.")
    for setting in settings:
        unknown = set(setting) - set(PROPOSAL_SETTING_KEYS)
        if unknown:
            raise ValueError(f"Unknown proposal setting keys: {sorted(unknown)}.")
    samples = [sample for sample in manifest["samples"] if sample["ndim"] == 2]
    folds = _stable_folds(samples)
    checkpoint = common.get_joint_checkpoint("hvit_t", "best")
    segmenter = common.build_apg_segmenter(
        "hvit_t", 2, device, joint_checkpoint="best",
        joint_checksum=common.checkpoint_checksum(checkpoint),
        export_root=str(output.parent / "model_exports"),
    )
    rows_per_setting: List[list] = [[] for _ in settings]
    diagnostic = []
    started = time.perf_counter()
    try:
        for number, sample in enumerate(samples, 1):
            raw, labels = _load_2d_sample(sample, data_root)
            segmenter.clear_state()
            segmenter.initialize(raw, ndim=2)
            for setting_index, setting in enumerate(settings):
                proposals = segmenter.propose(
                    multimasking=multimasking, multimask_scorer="predicted_iou",
                    multimask_selection="deferred" if multimasking else "eager",
                    return_multimask_features=True, multimask_feature_schema=input_schema, **setting,
                )
                targets = []
                for record in proposals:
                    if "multimask_features" not in record:
                        raise RuntimeError("Proposal did not retain selector features.")
                    target = _record_target(record, labels)
                    targets.append(target)
                    rows_per_setting[setting_index].append({
                        "features": record["multimask_features"],
                        "target": target,
                        "sample_id": sample["sample_id"],
                        "dataset": sample["dataset"],
                        "fold": folds[sample["sample_id"]],
                        "prompt_group": f"{sample['sample_id']}:{record['prompt_index']}",
                        "alternative": record["multimask_index"],
                    })
                n_objects, seeded, proposed = _seeded_and_proposed(proposals, labels, targets)
                diagnostic.append({
                    "sample_id": sample["sample_id"], "dataset": sample["dataset"], "setting": setting_index,
                    **{key: setting.get(key) for key in PROPOSAL_SETTING_KEYS},
                    "n_prompts": len({record["prompt_index"] for record in proposals}),
                    "gt_objects": n_objects, "seeded": seeded, "proposed": proposed,
                })
            print(f"[{number}/{len(samples)}] {sample['sample_id']} "
                  f"alternatives={[len(rows) for rows in rows_per_setting]}", flush=True)
    finally:
        segmenter.clear_state()

    if proposal_settings is not None:
        import pandas as pd
        outputs[0].parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(diagnostic).to_csv(outputs[0].parent / "recall_diagnostic.csv", index=False)
    for setting, rows, path in zip(settings, rows_per_setting, outputs):
        _write_feature_dataset(rows, path, manifest, input_schema, multimasking, setting)
    print(f"Wrote {len(settings)} feature dataset(s) in {time.perf_counter() - started:.1f}s")
    return outputs[0]


def _write_feature_dataset(rows: list, output: Path, manifest: dict, input_schema: str, multimasking: bool,
                           setting: dict) -> None:
    features = np.stack([row["features"] for row in rows]).astype("float32")
    targets = np.asarray([row["target"] for row in rows], dtype="float32")
    sample_ids = np.asarray([row["sample_id"] for row in rows])
    datasets = np.asarray([row["dataset"] for row in rows])
    groups = np.asarray([row["prompt_group"] for row in rows])
    folds_array = np.asarray([row["fold"] for row in rows], dtype="int8")
    alternatives = np.asarray([row["alternative"] for row in rows], dtype="int8")

    weights = np.zeros(len(rows), dtype="float64")
    for dataset in np.unique(datasets):
        dataset_rows = np.flatnonzero(datasets == dataset)
        dataset_samples = np.unique(sample_ids[dataset_rows])
        for sample_id in dataset_samples:
            image_rows = dataset_rows[sample_ids[dataset_rows] == sample_id]
            weights[image_rows] = 1.0 / (len(np.unique(datasets)) * len(dataset_samples) * len(image_rows))
    weights /= weights.mean()
    n_alternatives = 3 if multimasking else 1
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, features=features, targets=targets, sample_ids=sample_ids, datasets=datasets,
        groups=groups, folds=folds_array, alternatives=alternatives, weights=weights.astype("float32"),
        feature_version=np.asarray(MULTIMASK_FEATURE_VERSION),
        feature_names=np.asarray(SELECTOR_FEATURE_SCHEMAS[input_schema]),
        input_schema=np.asarray(input_schema),
        manifest_checksum=np.asarray(manifest["manifest_checksum"]),
        n_alternatives=np.asarray(n_alternatives),
        proposal_setting=np.asarray(json.dumps(setting, sort_keys=True)),
    )
    print(f"Wrote {len(rows)} alternatives to {output}")


def _load_grouped_dataset(
    path: Path, requested_schema: str | None = None, feature_set: str | None = None, per_image: str = "none",
) -> dict:
    data = np.load(path, allow_pickle=False)
    if int(data["feature_version"]) != MULTIMASK_FEATURE_VERSION:
        raise ValueError("The feature dataset has a different runtime schema version.")
    input_schema = str(data["input_schema"]) if "input_schema" in data.files else "dense_v1"
    if input_schema not in SELECTOR_FEATURE_SCHEMAS:
        raise ValueError(f"Unknown selector input schema {input_schema!r}.")
    if tuple(data["feature_names"].tolist()) != SELECTOR_FEATURE_SCHEMAS[input_schema]:
        raise ValueError("The feature dataset does not match the runtime schema.")
    features = data["features"].astype("float32", copy=False)
    if requested_schema is not None and requested_schema != input_schema:
        if input_schema != "token_lowres_v1" or requested_schema not in ("lowres_v1", "token_v1"):
            raise ValueError(f"Cannot derive schema {requested_schema!r} from {input_schema!r}.")
        if requested_schema == "lowres_v1":
            features = features[:, :len(MULTIMASK_FEATURE_NAMES)]
        else:
            token_start = len(MULTIMASK_FEATURE_NAMES)
            features = np.concatenate(
                (features[:, 0:1], features[:, 8:9], features[:, token_start:]), axis=1,
            )
        input_schema = requested_schema
    feature_names = list(SELECTOR_FEATURE_SCHEMAS[input_schema])
    if feature_set is not None:
        wanted = GENERIC_FEATURE_SETS[feature_set]
        missing = [name for name in wanted if name not in feature_names]
        if missing:
            raise ValueError(f"Feature set {feature_set!r} needs {missing} which {input_schema!r} lacks.")
        columns = [feature_names.index(name) for name in wanted]
        features = features[:, columns]
        feature_names = list(wanted)
    if per_image not in PER_IMAGE_MODES:
        raise ValueError(f"Unknown per-image mode {per_image!r}.")
    if per_image != "none":
        standardized = _per_image_standardize(features, data["sample_ids"])
        if per_image == "replace":
            features, feature_names = standardized, [f"{name}_z" for name in feature_names]
        else:
            features = np.concatenate((features, standardized), axis=1)
            feature_names = feature_names + [f"{name}_z" for name in feature_names]
    n_alternatives = int(data["n_alternatives"]) if "n_alternatives" in data else 3
    if n_alternatives not in (1, 3):
        raise ValueError(f"Expected one or three alternatives per prompt, got {n_alternatives}.")

    groups, alternatives = data["groups"], data["alternatives"]
    order = np.lexsort((alternatives, groups))
    ordered_groups = groups[order]
    starts = np.r_[0, np.flatnonzero(ordered_groups[1:] != ordered_groups[:-1]) + 1]
    stops = np.r_[starts[1:], len(order)]
    # An alternative whose mask came back empty leaves no record, so its prompt has fewer rows. The
    # group keeps a slot for it (index -1): its features are the group's mean, its target 0 and it
    # carries no weight, so the model sees a complete triplet and the flat arrays stay aligned.
    rows = np.full((len(starts), n_alternatives), -1, dtype="int64")
    for group_index, (start, stop) in enumerate(zip(starts, stops)):
        present = order[start:stop]
        slots = alternatives[present].astype("int64")
        if len(present) > n_alternatives or len(np.unique(slots)) != len(slots) or slots.max() >= n_alternatives:
            raise ValueError(f"Every prompt must have at most {n_alternatives} distinct alternatives.")
        rows[group_index, slots] = present
    present_mask = rows >= 0
    if not present_mask.any(axis=1).all():
        raise ValueError("Every prompt must have at least one alternative.")
    first_present = rows[np.arange(len(rows)), present_mask.argmax(axis=1)]
    safe_rows = np.where(present_mask, rows, first_present[:, None])
    folds = data["folds"][safe_rows]
    sample_ids = data["sample_ids"][safe_rows]
    if not np.all(folds == folds[:, :1]) or not np.all(sample_ids == sample_ids[:, :1]):
        raise ValueError("All alternatives of a prompt must belong to the same image and fold.")
    grouped_features = features[safe_rows].astype("float32", copy=True)
    if not present_mask.all():
        counts = present_mask.sum(axis=1, keepdims=True)
        group_mean = (grouped_features * present_mask[..., None]).sum(axis=1, keepdims=True) / counts[..., None]
        grouped_features = np.where(present_mask[..., None], grouped_features, group_mean)
    grouped_targets = np.where(present_mask, data["targets"][safe_rows], 0.0).astype("float32")
    grouped_weights = (data["weights"][safe_rows] * present_mask).sum(axis=1) / present_mask.sum(axis=1)
    return {
        "features": grouped_features,
        "targets": grouped_targets,
        "weights": grouped_weights.astype("float32"),
        "folds": folds[:, 0].astype("int8"),
        "datasets": data["datasets"][safe_rows][:, 0],
        "rows": rows,
        "present": present_mask,
        "n_incomplete_groups": int((~present_mask.all(axis=1)).sum()),
        "groups": groups,
        "flat_targets": data["targets"].astype("float32", copy=False),
        "flat_weights": data["weights"].astype("float32", copy=False),
        "manifest_checksum": str(data["manifest_checksum"]),
        "n_alternatives": n_alternatives,
        "input_schema": input_schema,
        "feature_names": tuple(feature_names),
        "feature_set": feature_set, "per_image": per_image,
        "sample_ids": sample_ids[:, 0],
    }


def _selection_metrics(targets, predictions, groups, weights) -> Dict[str, float]:
    chosen_target, oracle_target, correct = [], [], []
    order = np.argsort(groups, kind="stable")
    ordered_groups = groups[order]
    starts = np.r_[0, np.flatnonzero(ordered_groups[1:] != ordered_groups[:-1]) + 1]
    stops = np.r_[starts[1:], len(order)]
    for start, stop in zip(starts, stops):
        indices = order[start:stop]
        chosen = indices[int(np.argmax(predictions[indices]))]
        oracle = indices[int(np.argmax(targets[indices]))]
        chosen_target.append(float(targets[chosen]))
        oracle_target.append(float(targets[oracle]))
        correct.append(chosen == oracle or targets[chosen] == targets[oracle])
    error = targets - predictions
    return {
        "weighted_mse": float(np.average(error * error, weights=weights)),
        "weighted_mae": float(np.average(np.abs(error), weights=weights)),
        "selection_accuracy": float(np.mean(correct)),
        "selected_target_iou": float(np.mean(chosen_target)),
        "oracle_target_iou": float(np.mean(oracle_target)),
        "selection_regret": float(np.mean(np.asarray(oracle_target) - chosen_target)),
        "correlation": float(np.corrcoef(predictions, targets)[0, 1]),
    }


def _normalization(features: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = features.reshape(-1, features.shape[-1])
    flat_weights = np.repeat(weights, features.shape[1])
    mean = np.average(flat, axis=0, weights=flat_weights).astype("float32")
    variance = np.average((flat - mean) ** 2, axis=0, weights=flat_weights)
    scale = np.sqrt(variance).astype("float32")
    scale[scale == 0] = 1.0
    return mean, scale


def _loss(prediction, target, weight, target_kind="iou"):
    if target_kind == "matched":
        per_group = F.binary_cross_entropy_with_logits(prediction, target, reduction="none").mean(dim=1)
    else:
        per_group = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=1)
    return (per_group * weight).sum() / weight.sum()


def _fit(features, targets, weights, train, validation, device, architecture, max_epochs=120):
    target_kind = architecture.get("target", "iou")
    mean, scale = _normalization(features[train], weights[train])
    x = torch.as_tensor((features - mean) / scale, dtype=torch.float32, device=device)
    y = torch.as_tensor(_target_values(targets, target_kind), dtype=torch.float32, device=device)
    w = torch.as_tensor(weights, dtype=torch.float32, device=device)
    torch.manual_seed(17)
    model = _build_model(features.shape[-1], architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(17)
    train_indices = torch.as_tensor(np.flatnonzero(train), dtype=torch.int64)
    validation_indices = torch.as_tensor(np.flatnonzero(validation), dtype=torch.int64, device=device)
    best_state, best_loss, best_epoch, stale = None, float("inf"), 0, 0
    for epoch in range(max_epochs):
        model.train()
        order = train_indices[torch.randperm(len(train_indices), generator=generator)]
        for start in range(0, len(order), 4096):
            index = order[start:start + 4096].to(device)
            loss = _loss(model(x[index]), y[index], w[index], target_kind)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(_loss(
                model(x[validation_indices]), y[validation_indices], w[validation_indices], target_kind,
            ).cpu())
        if validation_loss < best_loss - 1e-7:
            best_loss, best_epoch, stale = validation_loss, epoch + 1, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 10:
                break
    model.load_state_dict(best_state)
    return model.eval(), mean, scale, best_epoch


def _fit_full(features, targets, weights, device, epochs, architecture):
    target_kind = architecture.get("target", "iou")
    mean, scale = _normalization(features, weights)
    x = torch.as_tensor((features - mean) / scale, dtype=torch.float32, device=device)
    y = torch.as_tensor(_target_values(targets, target_kind), dtype=torch.float32, device=device)
    w = torch.as_tensor(weights, dtype=torch.float32, device=device)
    torch.manual_seed(17)
    model = _build_model(features.shape[-1], architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(17)
    indices = torch.arange(len(features), dtype=torch.int64)
    for _ in range(epochs):
        order = indices[torch.randperm(len(indices), generator=generator)]
        for start in range(0, len(order), 4096):
            index = order[start:start + 4096].to(device)
            loss = _loss(model(x[index]), y[index], w[index], target_kind)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval(), mean, scale


def _load_pooled_datasets(
    datasets: Sequence[Path], requested_schema: str | None, feature_set: str | None = None, per_image: str = "none",
) -> dict:
    """Concatenate several feature datasets of one schema; every dataset gets equal total weight."""
    parts = [
        _load_grouped_dataset(path, requested_schema=requested_schema, feature_set=feature_set, per_image=per_image)
        for path in datasets
    ]
    first = parts[0]
    for part in parts[1:]:
        if part["input_schema"] != first["input_schema"] or part["n_alternatives"] != first["n_alternatives"]:
            raise ValueError("Pooled feature datasets must share their schema and alternative count.")
    n_groups = [len(part["targets"]) for part in parts]
    mean_groups = float(np.mean(n_groups))
    pooled = {
        "features": np.concatenate([part["features"] for part in parts]),
        "targets": np.concatenate([part["targets"] for part in parts]),
        "weights": np.concatenate([part["weights"] * (mean_groups / n) for part, n in zip(parts, n_groups)]),
        "folds": np.concatenate([part["folds"] for part in parts]),
        "datasets": np.concatenate([part["datasets"] for part in parts]),
        "n_alternatives": first["n_alternatives"], "input_schema": first["input_schema"],
        "feature_names": first["feature_names"], "feature_set": first["feature_set"], "per_image": first["per_image"],
        "sample_ids": np.concatenate([part["sample_ids"] for part in parts]),
        "manifest_checksum": ",".join(sorted({part["manifest_checksum"] for part in parts})),
        "parts": parts, "group_offsets": np.r_[0, np.cumsum(n_groups)],
    }
    return pooled


def _scatter(part: dict, grouped: np.ndarray, fill: float = np.nan) -> np.ndarray:
    """Write grouped predictions back to a dataset's flat rows; padded slots have no flat row."""
    flat = np.full_like(part["flat_targets"], fill)
    present = part["rows"] >= 0
    flat[part["rows"][present]] = grouped[present]
    return flat


def train_selector(
    dataset: Path | Sequence[Path], output_dir: Path, device: str, hidden_size: int = 64,
    input_schema: str | None = None, lodo: bool = False, feature_set: str | None = None,
    per_image: str = "none", model_kind: str = "mlp", target_kind: str = "iou",
) -> Path:
    """Fit the groupwise selector with image-level out-of-fold predictions, then refit on everything.

    With 'lodo' a leave-one-dataset-out pass is added: for every dataset, a model fitted on the other
    datasets (fold 0 of those as validation) predicts its rows, written to '{name}_lodo.npy' aligned
    with the OOF file. It measures how much of the selector's signal is dataset identity.
    """
    datasets = [dataset] if isinstance(dataset, (str, Path)) else list(dataset)
    pooled = len(datasets) > 1
    if model_kind not in MODEL_KINDS or target_kind not in TARGET_KINDS:
        raise ValueError(f"Unknown model {model_kind!r} or target {target_kind!r}.")
    data = _load_pooled_datasets(datasets, input_schema, feature_set, per_image) if pooled else _load_grouped_dataset(
        datasets[0], requested_schema=input_schema, feature_set=feature_set, per_image=per_image,
    )
    architecture = {"hidden_size": int(hidden_size), "dropout": 0.1, "model": model_kind, "target": target_kind}
    features, targets = data["features"], data["targets"]
    weights, folds = data["weights"], data["folds"]
    grouped_oof = np.zeros_like(targets)
    fold_epochs = []
    for outer in range(5):
        validation_fold = (outer + 1) % 5
        train = (folds != outer) & (folds != validation_fold)
        validation, test = folds == validation_fold, folds == outer
        model, mean, scale, best_epoch = _fit(
            features, targets, weights, train, validation, device, architecture,
        )
        values = torch.as_tensor((features[test] - mean) / scale, dtype=torch.float32, device=device)
        with torch.no_grad():
            grouped_oof[test] = _output_values(model(values), target_kind).cpu().numpy()
        fold_epochs.append(best_epoch)
        print(f"selector fold {outer + 1}/5 epoch={best_epoch}", flush=True)

    if pooled:
        # One flat OOF array per input dataset, aligned with that dataset's rows.
        flat_oofs, metrics_parts = [], {}
        for part, start, stop, path in zip(
            data["parts"], data["group_offsets"][:-1], data["group_offsets"][1:], datasets,
        ):
            flat_oof = _scatter(part, grouped_oof[start:stop], fill=0.0)
            flat_oofs.append(flat_oof)
            metrics_parts[Path(path).stem] = _selection_metrics(
                part["flat_targets"], flat_oof, part["groups"], part["flat_weights"],
            )
        metrics = {"per_dataset": metrics_parts}
        flat_oof = np.concatenate(flat_oofs)
    else:
        flat_oof = _scatter(data, grouped_oof, fill=0.0)
        metrics = _selection_metrics(
            data["flat_targets"], flat_oof, data["groups"], data["flat_weights"],
        )
    metrics["fold_epochs"] = fold_epochs

    grouped_lodo = None
    feature_names_start_with_iou = tuple(data["feature_names"])[:1] == ("predicted_iou",)
    if lodo:
        grouped_lodo = np.full_like(targets, np.nan)
        metrics["lodo"] = {}
        for held_out in np.unique(data["datasets"]):
            test = data["datasets"] == held_out
            others = ~test
            train, validation = others & (folds != 0), others & (folds == 0)
            model, mean, scale, best_epoch = _fit(
                features, targets, weights, train, validation, device, architecture,
            )
            values = torch.as_tensor((features[test] - mean) / scale, dtype=torch.float32, device=device)
            with torch.no_grad():
                grouped_lodo[test] = _output_values(model(values), target_kind).cpu().numpy()
            oof_rows = grouped_oof[test].reshape(-1)
            lodo_rows = grouped_lodo[test].reshape(-1)
            target_rows = targets[test].reshape(-1)
            baseline_rows = features[test][..., 0].reshape(-1) if feature_names_start_with_iou else None
            metrics["lodo"][str(held_out)] = {
                "epoch": best_epoch, "n_groups": int(test.sum()),
                "oof_correlation": float(np.corrcoef(oof_rows, target_rows)[0, 1]),
                "lodo_correlation": float(np.corrcoef(lodo_rows, target_rows)[0, 1]),
                "oof_matched_auc": _auc(oof_rows, target_rows >= MATCHED_IOU),
                "lodo_matched_auc": _auc(lodo_rows, target_rows >= MATCHED_IOU),
                "predicted_iou_matched_auc": (
                    _auc(baseline_rows, target_rows >= MATCHED_IOU) if baseline_rows is not None else None
                ),
                "predicted_iou_selected_iou": (
                    float(targets[test][np.arange(int(test.sum())), features[test][..., 0].argmax(1)].mean())
                    if baseline_rows is not None else None
                ),
                "oof_selected_iou": float(
                    targets[test][np.arange(int(test.sum())), grouped_oof[test].argmax(1)].mean()
                ),
                "lodo_selected_iou": float(
                    targets[test][np.arange(int(test.sum())), grouped_lodo[test].argmax(1)].mean()
                ),
            }
            print(f"lodo {held_out}: {metrics['lodo'][str(held_out)]}", flush=True)
    refit_epochs = max(1, int(round(float(np.mean(fold_epochs)))))
    model, mean, scale = _fit_full(features, targets, weights, device, refit_epochs, architecture)

    prefix = "singlemask-" if data["n_alternatives"] == 1 else ""
    schema_prefix = "" if data["input_schema"] == "dense_v1" else f"{data['input_schema']}-"
    width = "linear" if model_kind == "linear" else f"h{hidden_size}-d0p1"
    objective = "regression" if target_kind == "iou" else "matched"
    name = f"{prefix}{schema_prefix}groupwise-{width}-{objective}"
    if feature_set is not None:
        name += f"-fs_{feature_set}"
    if per_image != "none":
        name += f"-z_{per_image}"
    if pooled:
        name += f"-pooled{len(datasets)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / f"{name}.pt"
    torch.save({
        "kind": "groupwise_linear" if model_kind == "linear" else "groupwise_mlp",
        "feature_version": MULTIMASK_FEATURE_VERSION,
        "input_schema": data["input_schema"], "feature_names": list(data["feature_names"]),
        "feature_set": feature_set, "per_image": per_image, "target": target_kind,
        "n_alternatives": data["n_alternatives"],
        "hidden_size": architecture["hidden_size"], "dropout": architecture["dropout"],
        "mean": mean, "scale": scale,
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "metadata": {
            "architecture": architecture, "epochs": refit_epochs,
            "loss": "direct-regression" if target_kind == "iou" else "matched-bce",
            "input_schema": data["input_schema"],
            "manifest_checksum": data["manifest_checksum"], "oof_metrics": metrics,
            "training_datasets": [str(path) for path in datasets],
        },
    }, artifact)
    np.save(output_dir / f"{name}_oof.npy", flat_oof.astype("float32"))
    if grouped_lodo is not None:
        flat_lodo = np.full_like(flat_oof, np.nan)
        if pooled:
            flat_lodo = np.concatenate([
                _scatter(part, grouped_lodo[start:stop])
                for part, start, stop in zip(data["parts"], data["group_offsets"][:-1], data["group_offsets"][1:])
            ])
        else:
            flat_lodo = _scatter(data, grouped_lodo)
        np.save(output_dir / f"{name}_lodo.npy", flat_lodo.astype("float32"))
        if pooled:
            for path, part, start, stop in zip(
                datasets, data["parts"], data["group_offsets"][:-1], data["group_offsets"][1:],
            ):
                np.save(
                    output_dir / f"{name}_lodo_{Path(path).stem}.npy",
                    _scatter(part, grouped_lodo[start:stop]).astype("float32"),
                )
    if pooled:
        for path, part_oof in zip(datasets, flat_oofs):
            np.save(output_dir / f"{name}_oof_{Path(path).stem}.npy", part_oof.astype("float32"))
    with open(output_dir / f"{name}_training_results.json", "w") as f:
        json.dump({
            "artifact": str(artifact), "metrics": metrics, "refit_epochs": refit_epochs,
            "oof_quantiles": {
                str(quantile): float(np.quantile(flat_oof, quantile))
                for quantile in np.linspace(0.0, 1.0, 11)
            },
        }, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("extract", "train", "all"), default="all")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, action="append", default=None,
                        help="Feature dataset(s) to train on; repeat to pool several.")
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument(
        "--single-mask", action="store_true",
        help="Extract and train for the dedicated single-mask decoder token.",
    )
    parser.add_argument(
        "--input-schema", choices=tuple(SELECTOR_FEATURE_SCHEMAS), default="dense_v1",
        help="Selector inputs to extract and train. Compact schemas support three masks only.",
    )
    parser.add_argument(
        "--hidden-size", action="append", type=int, default=[],
        help="Groupwise MLP width. Repeat to train several widths.",
    )
    parser.add_argument(
        "--train-schema", action="append", choices=tuple(SELECTOR_FEATURE_SCHEMAS), default=[],
        help="Schema to train from the extracted dataset. Hybrid extraction can derive token or lowres inputs.",
    )
    parser.add_argument("--lodo", action="store_true", help="Also write leave-one-dataset-out predictions.")
    parser.add_argument(
        "--feature-set", action="append", choices=tuple(GENERIC_FEATURE_SETS), default=[],
        help="Named subset of the generic mask statistics to train on; repeat for several.",
    )
    parser.add_argument(
        "--per-image", choices=PER_IMAGE_MODES, default="none",
        help="Standardize features within each image ('replace') or append the standardized copy.",
    )
    parser.add_argument("--model", choices=MODEL_KINDS, default="mlp")
    parser.add_argument("--target", choices=TARGET_KINDS, default="iou",
                        help="'iou' regresses the mask IoU; 'matched' classifies IoU >= 0.5.")
    parser.add_argument(
        "--subset", choices=MANIFEST_SUBSETS, default="primary",
        help="Manifest subset to extract; 'training_extra' adds datasets outside the benchmark for training only.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    manifest_path = args.manifest or _default_manifest_path(args.output_root, "standard", args.subset)
    data_root, output_root, manifest_path = _validate_roots(args.data_root, args.output_root, manifest_path)
    manifest = prepare_manifest(data_root, manifest_path, "standard", subset=args.subset)
    selection_root = output_root / "multimask_selection"
    if args.single_mask and args.input_schema != "dense_v1":
        raise ValueError("--single-mask only supports --input-schema dense_v1.")
    schema_root = args.input_schema if args.input_schema != "dense_v1" else None
    dataset_root = selection_root / "singlemask_v1" if args.single_mask else selection_root
    model_root = selection_root / ("singlemask_v1" if args.single_mask else "groupwise_v1")
    if schema_root is not None:
        dataset_root = dataset_root / schema_root
        model_root = model_root / schema_root
    datasets = args.dataset or [dataset_root / f"{args.subset}_features.npz"]
    dataset = datasets[0]
    artifact_dir = args.artifact_dir or model_root / "models"
    if args.stage in ("extract", "all"):
        extract_dataset(
            manifest, data_root, dataset, args.device, multimasking=not args.single_mask,
            input_schema=args.input_schema,
        )
    if args.stage in ("train", "all"):
        train_schemas = args.train_schema or [args.input_schema]
        for train_schema in train_schemas:
            hidden_sizes = args.hidden_size or ([64] if train_schema == "lowres_v1" else [32, 64, 128])
            feature_sets = args.feature_set or [None]
            for hidden_size in hidden_sizes:
                for feature_set in feature_sets:
                    artifact = train_selector(
                        [path.resolve(strict=True) for path in datasets], artifact_dir, args.device,
                        hidden_size=hidden_size, input_schema=train_schema, lodo=args.lodo,
                        feature_set=feature_set, per_image=args.per_image, model_kind=args.model,
                        target_kind=args.target,
                    )
                    print(f"Artifact: {artifact}")


if __name__ == "__main__":
    main()
