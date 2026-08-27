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
from typing import Any, Dict, Iterable, Tuple

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
    _validate_roots, prepare_manifest,
)


ARCHITECTURE = {"hidden_size": 64, "dropout": 0.1}


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


def extract_dataset(
    manifest: dict, data_root: Path, output: Path, device: str, multimasking: bool = True,
    input_schema: str = "dense_v1",
) -> Path:
    if not multimasking and input_schema != "dense_v1":
        raise ValueError("Compact selector schemas require the three-mask output.")
    samples = [sample for sample in manifest["samples"] if sample["ndim"] == 2]
    folds = _stable_folds(samples)
    checkpoint = common.get_joint_checkpoint("hvit_t", "best")
    segmenter = common.build_apg_segmenter(
        "hvit_t", 2, device, joint_checkpoint="best",
        joint_checksum=common.checkpoint_checksum(checkpoint),
        export_root=str(output.parent / "model_exports"),
    )
    rows = []
    started = time.perf_counter()
    try:
        for number, sample in enumerate(samples, 1):
            raw, labels = _load_2d_sample(sample, data_root)
            segmenter.clear_state()
            segmenter.initialize(raw, ndim=2)
            proposals = segmenter.propose(
                multimasking=multimasking, multimask_scorer="predicted_iou",
                multimask_selection="deferred" if multimasking else "eager",
                return_multimask_features=True, multimask_feature_schema=input_schema,
            )
            for record in proposals:
                if "multimask_features" not in record:
                    raise RuntimeError("Proposal did not retain selector features.")
                rows.append({
                    "features": record["multimask_features"],
                    "target": _record_target(record, labels),
                    "sample_id": sample["sample_id"],
                    "dataset": sample["dataset"],
                    "fold": folds[sample["sample_id"]],
                    "prompt_group": f"{sample['sample_id']}:{record['prompt_index']}",
                    "alternative": record["multimask_index"],
                })
            print(f"[{number}/{len(samples)}] {sample['sample_id']} alternatives={len(proposals)}", flush=True)
    finally:
        segmenter.clear_state()

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
    )
    print(f"Wrote {len(rows)} alternatives to {output} in {time.perf_counter() - started:.1f}s")
    return output


def _load_grouped_dataset(path: Path, requested_schema: str | None = None) -> dict:
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
    n_alternatives = int(data["n_alternatives"]) if "n_alternatives" in data else 3
    if n_alternatives not in (1, 3):
        raise ValueError(f"Expected one or three alternatives per prompt, got {n_alternatives}.")

    groups, alternatives = data["groups"], data["alternatives"]
    order = np.lexsort((alternatives, groups))
    ordered_groups = groups[order]
    starts = np.r_[0, np.flatnonzero(ordered_groups[1:] != ordered_groups[:-1]) + 1]
    stops = np.r_[starts[1:], len(order)]
    if any(stop - start != n_alternatives for start, stop in zip(starts, stops)):
        raise ValueError(f"Every prompt must have exactly {n_alternatives} alternatives.")
    rows = np.stack([order[start:stop] for start, stop in zip(starts, stops)])
    expected = np.broadcast_to(np.arange(n_alternatives), rows.shape)
    if not np.array_equal(alternatives[rows], expected):
        raise ValueError("Prompt alternatives are missing, repeated, or out of order.")
    folds = data["folds"][rows]
    sample_ids = data["sample_ids"][rows]
    if not np.all(folds == folds[:, :1]) or not np.all(sample_ids == sample_ids[:, :1]):
        raise ValueError("All alternatives of a prompt must belong to the same image and fold.")
    return {
        "features": features[rows].astype("float32", copy=False),
        "targets": data["targets"][rows].astype("float32", copy=False),
        "weights": data["weights"][rows].mean(axis=1).astype("float32"),
        "folds": folds[:, 0].astype("int8"),
        "rows": rows,
        "groups": groups,
        "flat_targets": data["targets"].astype("float32", copy=False),
        "flat_weights": data["weights"].astype("float32", copy=False),
        "manifest_checksum": str(data["manifest_checksum"]),
        "n_alternatives": n_alternatives,
        "input_schema": input_schema,
        "feature_names": SELECTOR_FEATURE_SCHEMAS[input_schema],
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


def _loss(prediction, target, weight):
    per_group = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=1)
    return (per_group * weight).sum() / weight.sum()


def _fit(features, targets, weights, train, validation, device, architecture, max_epochs=120):
    mean, scale = _normalization(features[train], weights[train])
    x = torch.as_tensor((features - mean) / scale, dtype=torch.float32, device=device)
    y = torch.as_tensor(targets, dtype=torch.float32, device=device)
    w = torch.as_tensor(weights, dtype=torch.float32, device=device)
    torch.manual_seed(17)
    model = GroupwiseMLP(features.shape[-1], **architecture).to(device)
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
            loss = _loss(model(x[index]), y[index], w[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(_loss(
                model(x[validation_indices]), y[validation_indices], w[validation_indices],
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
    mean, scale = _normalization(features, weights)
    x = torch.as_tensor((features - mean) / scale, dtype=torch.float32, device=device)
    y = torch.as_tensor(targets, dtype=torch.float32, device=device)
    w = torch.as_tensor(weights, dtype=torch.float32, device=device)
    torch.manual_seed(17)
    model = GroupwiseMLP(features.shape[-1], **architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(17)
    indices = torch.arange(len(features), dtype=torch.int64)
    for _ in range(epochs):
        order = indices[torch.randperm(len(indices), generator=generator)]
        for start in range(0, len(order), 4096):
            index = order[start:start + 4096].to(device)
            loss = _loss(model(x[index]), y[index], w[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval(), mean, scale


def train_selector(
    dataset: Path, output_dir: Path, device: str, hidden_size: int = 64,
    input_schema: str | None = None,
) -> Path:
    data = _load_grouped_dataset(dataset, requested_schema=input_schema)
    architecture = {"hidden_size": int(hidden_size), "dropout": 0.1}
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
            grouped_oof[test] = model(values).cpu().numpy()
        fold_epochs.append(best_epoch)
        print(f"selector fold {outer + 1}/5 epoch={best_epoch}", flush=True)

    flat_oof = np.zeros_like(data["flat_targets"])
    flat_oof[data["rows"].reshape(-1)] = grouped_oof.reshape(-1)
    metrics = _selection_metrics(
        data["flat_targets"], flat_oof, data["groups"], data["flat_weights"],
    )
    metrics["fold_epochs"] = fold_epochs
    refit_epochs = max(1, int(round(float(np.mean(fold_epochs)))))
    model, mean, scale = _fit_full(features, targets, weights, device, refit_epochs, architecture)

    prefix = "singlemask-" if data["n_alternatives"] == 1 else ""
    schema_prefix = "" if data["input_schema"] == "dense_v1" else f"{data['input_schema']}-"
    name = f"{prefix}{schema_prefix}groupwise-h{hidden_size}-d0p1-regression"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / f"{name}.pt"
    torch.save({
        "kind": "groupwise_mlp", "feature_version": MULTIMASK_FEATURE_VERSION,
        "input_schema": data["input_schema"], "feature_names": list(data["feature_names"]),
        "n_alternatives": data["n_alternatives"],
        "hidden_size": architecture["hidden_size"], "dropout": architecture["dropout"],
        "mean": mean, "scale": scale,
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "metadata": {
            "architecture": architecture, "loss": "direct-regression", "epochs": refit_epochs,
            "input_schema": data["input_schema"],
            "manifest_checksum": data["manifest_checksum"], "oof_metrics": metrics,
        },
    }, artifact)
    np.save(output_dir / f"{name}_oof.npy", flat_oof.astype("float32"))
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
    parser.add_argument("--dataset", type=Path, default=None)
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    manifest_path = args.manifest or _default_manifest_path(args.output_root, "standard", "primary")
    data_root, output_root, manifest_path = _validate_roots(args.data_root, args.output_root, manifest_path)
    manifest = prepare_manifest(data_root, manifest_path, "standard", subset="primary")
    selection_root = output_root / "multimask_selection"
    if args.single_mask and args.input_schema != "dense_v1":
        raise ValueError("--single-mask only supports --input-schema dense_v1.")
    schema_root = args.input_schema if args.input_schema != "dense_v1" else None
    dataset_root = selection_root / "singlemask_v1" if args.single_mask else selection_root
    model_root = selection_root / ("singlemask_v1" if args.single_mask else "groupwise_v1")
    if schema_root is not None:
        dataset_root = dataset_root / schema_root
        model_root = model_root / schema_root
    dataset = args.dataset or dataset_root / "primary_features.npz"
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
            for hidden_size in hidden_sizes:
                artifact = train_selector(
                    dataset.resolve(strict=True), artifact_dir, args.device, hidden_size=hidden_size,
                    input_schema=train_schema,
                )
                print(f"Artifact: {artifact}")


if __name__ == "__main__":
    main()
