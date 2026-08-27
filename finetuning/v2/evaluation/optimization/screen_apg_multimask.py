"""Screen the groupwise H64 APG scorer with eager and deferred merge strategies.

One rich predicted-IoU/deferred proposal pass is reused for every configuration. These timings are
screening diagnostics only; shortlisted configurations must be run through the serialized canonical
benchmark for an acceptance decision.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from micro_sam.v2.multimask_selection import load_feature_scorer

EVALUATION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVALUATION_ROOT))

import common  # noqa
from parameter_search import compute_metrics  # noqa
from optimization.benchmark_apg_optimization import (  # noqa
    DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, GT_MIN_SIZE_2D, _atomic_write_csv, _atomic_write_json,
    _content_checksum, _default_manifest_path, _git_revision, _implementation_checksum,
    _hardware_identity, _load_2d_sample, _validate_roots, prepare_manifest,
)


def _default_configs(models: dict) -> list:
    configs = [
        {"name": "predicted-iou-eager", "scorer": None, "selection": "eager", "merge": "raw"},
        {"name": "predicted-iou-deferred", "scorer": None, "selection": "deferred", "merge": "raw"},
    ]
    for name in models:
        configs.extend([
            {"name": f"{name}-eager-select", "scorer": name, "selection": "eager", "merge": "raw"},
            {"name": f"{name}-eager-rescore", "scorer": name, "selection": "eager", "merge": "learned"},
            {"name": f"{name}-deferred", "scorer": name, "selection": "deferred", "merge": "learned"},
        ])
    return configs


def _configured_records(proposals, config, predictions):
    records = [dict(record) for record in proposals]
    if predictions is None:
        selection_scores = np.asarray([record["predicted_iou"] for record in records], dtype="float32")
    else:
        selection_scores = predictions
    for record, score in zip(records, selection_scores):
        record["selection_score"] = float(score)
        record["merge_score"] = (
            float(score) if config["merge"] == "learned"
            else record["predicted_iou"] * record["stability_score"]
        )

    if config["selection"] == "deferred":
        return records
    by_group = {}
    for index, record in enumerate(records):
        by_group.setdefault(record["multimask_group"], []).append(index)
    chosen = []
    for indices in by_group.values():
        index = max(indices, key=lambda candidate: (records[candidate]["selection_score"], -candidate))
        record = records[index]
        record.pop("multimask_group", None)
        chosen.append(record)
    return chosen


def _summarize(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, frame in samples.groupby("config_name", sort=False):
        table = frame.groupby("dataset", sort=True).agg(
            n_samples=("sample_id", "count"), msa_mean=("msa", "mean"),
            selection_seconds=("selection_seconds", "sum"),
        ).reset_index()
        table.insert(0, "config_name", name)
        rows.append(table)
        rows.append(pd.DataFrame([{
            "config_name": name, "dataset": "__dataset_balanced__", "n_samples": len(frame),
            "msa_mean": float(table["msa_mean"].mean()),
            "selection_seconds": float(table["selection_seconds"].sum()),
        }]))
    summary = pd.concat(rows, ignore_index=True)
    ranks = summary[summary["dataset"] == "__dataset_balanced__"].sort_values(
        "msa_mean", ascending=False
    )["config_name"].tolist()
    order = {name: index for index, name in enumerate(ranks)}
    summary["_order"] = summary["config_name"].map(order)
    return summary.sort_values(["_order", "dataset"]).drop(columns="_order").reset_index(drop=True)


def _load_oof_lookup(feature_dataset, oof_artifacts, manifest_checksum):
    data = np.load(feature_dataset, allow_pickle=False)
    if str(data["manifest_checksum"]) != manifest_checksum:
        raise ValueError(
            "The selector feature dataset was extracted from a different manifest: "
            f"{data['manifest_checksum']} != {manifest_checksum}."
        )
    n_rows = len(data["sample_ids"])
    predictions = {}
    for name, path in oof_artifacts.items():
        values = np.load(path, allow_pickle=False).astype("float32", copy=False)
        if values.shape != (n_rows,):
            raise ValueError(f"OOF predictions for {name!r} have shape {values.shape}, expected {(n_rows,)}.")
        predictions[name] = values

    lookup = {}
    for index, (sample_id, group, alternative) in enumerate(
        zip(data["sample_ids"], data["groups"], data["alternatives"])
    ):
        prompt_index = int(str(group).rsplit(":", 1)[1])
        key = (str(sample_id), prompt_index, int(alternative))
        if key in lookup:
            raise ValueError(f"Duplicate feature-dataset key: {key}.")
        lookup[key] = index
    return data["features"], predictions, lookup


def _oof_predictions_for_sample(sample_id, proposals, feature_rows, predictions, lookup):
    indices = []
    for record in proposals:
        key = (sample_id, int(record["prompt_index"]), int(record["multimask_index"]))
        try:
            indices.append(lookup[key])
        except KeyError as error:
            raise ValueError(f"Proposal {key} is missing from the OOF feature dataset.") from error
    if indices:
        current = np.stack([record["multimask_features"] for record in proposals])
        expected = feature_rows[np.asarray(indices)]
        if not np.allclose(current, expected, rtol=1e-5, atol=1e-5):
            raise ValueError(
                f"Regenerated proposal features differ from the OOF dataset for sample {sample_id!r}."
            )
    return {name: values[np.asarray(indices)] for name, values in predictions.items()}


def _predict_records(model, proposals):
    """Predict record-aligned scores, preserving complete three-alternative groups."""
    if not hasattr(model, "predict_grouped") or model.n_alternatives != 3:
        raise ValueError("Multimask screening requires a three-alternative groupwise MLP.")
    grouped = {}
    for index, record in enumerate(proposals):
        grouped.setdefault(record["multimask_group"], []).append(index)
    rows, indices = [], []
    for group_indices in grouped.values():
        group_indices.sort(key=lambda index: proposals[index]["multimask_index"])
        alternatives = [proposals[index]["multimask_index"] for index in group_indices]
        if alternatives != [0, 1, 2]:
            raise ValueError(f"Groupwise scoring requires alternatives [0, 1, 2], got {alternatives}.")
        rows.append(np.stack([proposals[index]["multimask_features"] for index in group_indices]))
        indices.append(group_indices)
    prediction = model.predict_grouped(np.stack(rows))
    aligned = np.empty(len(proposals), dtype="float32")
    for group_indices, group_prediction in zip(indices, prediction):
        aligned[group_indices] = group_prediction
    return aligned


def run_screening(
    manifest, data_root, output_root, artifacts, device, subset, *, feature_dataset=None,
    oof_artifacts=None, only_configs=None,
):
    models = {name: load_feature_scorer(path, device=device) for name, path in artifacts.items()}
    configs = _default_configs(models)
    if only_configs:
        known = {config["name"] for config in configs}
        unknown = sorted(set(only_configs).difference(known))
        if unknown:
            raise ValueError(f"Unknown configuration names: {unknown}. Known names: {sorted(known)}")
        configs = [config for config in configs if config["name"] in only_configs]
    if not configs:
        raise ValueError("At least one screening configuration is required.")

    use_oof = subset == "primary"
    if use_oof:
        if feature_dataset is None or oof_artifacts is None:
            raise ValueError("Primary screening requires the feature dataset and OOF predictions.")
        feature_rows, oof_predictions, oof_lookup = _load_oof_lookup(
            feature_dataset, oof_artifacts, manifest["manifest_checksum"],
        )
    else:
        feature_rows = oof_predictions = oof_lookup = None
    checkpoint = common.get_joint_checkpoint("hvit_t", "best")
    checkpoint_id = common.checkpoint_checksum(checkpoint)
    identity = {
        "manifest_checksum": manifest["manifest_checksum"],
        "implementation_checksum": _implementation_checksum(),
        "screen_implementation_checksum": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "artifacts": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in artifacts.items()},
        "configs": configs,
        "device": device,
        "hardware": _hardware_identity(device),
        "prediction_source": "out-of-fold" if use_oof else "refit-model",
    }
    if use_oof:
        identity["feature_dataset"] = hashlib.sha256(Path(feature_dataset).read_bytes()).hexdigest()
        identity["oof_artifacts"] = {
            name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in oof_artifacts.items()
        }
    run_dir = output_root / "multimask_screening" / "hvit_t" / checkpoint_id / _content_checksum(identity)
    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(run_dir / "metadata.json", {
        **identity, "screening": True, "subset": subset,
        "git_revision": _git_revision(), "artifact_paths": {key: str(value) for key, value in artifacts.items()},
        "feature_dataset_path": str(feature_dataset) if feature_dataset is not None else None,
        "oof_artifact_paths": (
            {key: str(value) for key, value in oof_artifacts.items()} if oof_artifacts is not None else None
        ),
    })
    samples_path, summary_path = run_dir / "samples.csv", run_dir / "summary.csv"
    completed = pd.read_csv(samples_path) if samples_path.exists() else pd.DataFrame()
    completed_ids = set(completed["sample_id"]) if not completed.empty else set()
    samples = [sample for sample in manifest["samples"] if sample["ndim"] == 2]
    pending = [sample for sample in samples if sample["sample_id"] not in completed_ids]
    segmenter = common.build_apg_segmenter(
        "hvit_t", 2, device, joint_checkpoint="best", joint_checksum=checkpoint_id,
        export_root=str(output_root / "model_exports"),
    )
    try:
        for number, sample in enumerate(pending, 1):
            raw, labels = _load_2d_sample(sample, data_root)
            segmenter.clear_state()
            segmenter.initialize(raw, ndim=2)
            proposals = segmenter.propose(
                multimasking=True, multimask_scorer="predicted_iou", multimask_selection="deferred",
            )
            if use_oof:
                model_predictions = _oof_predictions_for_sample(
                    sample["sample_id"], proposals, feature_rows, oof_predictions, oof_lookup,
                )
            else:
                model_predictions = {
                    name: _predict_records(model, proposals)
                    for name, model in models.items()
                } if proposals else {}
            rows = []
            for config in configs:
                started = time.perf_counter()
                records = _configured_records(
                    proposals, config, model_predictions.get(config["scorer"]),
                )
                segmentation = segmenter.select(records).astype("uint32")
                elapsed = time.perf_counter() - started
                metrics = compute_metrics(
                    segmentation, labels, "sparse", border_min_size=GT_MIN_SIZE_2D.get(sample["dataset"], 0),
                )
                rows.append({
                    "sample_id": sample["sample_id"], "dataset": sample["dataset"],
                    "config_name": config["name"], "msa": metrics["msa"],
                    "selection_seconds": elapsed, "predicted_objects": int(segmentation.max()),
                })
            completed = pd.concat([completed, pd.DataFrame(rows)], ignore_index=True)
            _atomic_write_csv(samples_path, completed)
            print(f"[{number}/{len(pending)}] {sample['sample_id']}", flush=True)
    finally:
        segmenter.clear_state()
    summary = _summarize(completed)
    _atomic_write_csv(summary_path, summary)
    return run_dir, summary


def _parse_artifacts(values, artifact_dir):
    if values:
        artifacts = {}
        for value in values:
            name, separator, path = value.partition("=")
            if not separator or not name or not path:
                raise ValueError(f"Expected NAME=PATH for --model, got {value!r}.")
            artifacts[name] = Path(path).resolve(strict=True)
        return artifacts
    defaults = {"groupwise-h64": artifact_dir / "groupwise-h64-d0p1-regression.pt"}
    missing = [str(path) for path in defaults.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing selector artifacts: {missing}")
    return defaults


def _parse_oof_artifacts(values, artifact_dir, model_names):
    if values:
        artifacts = {}
        for value in values:
            name, separator, path = value.partition("=")
            if not separator or not name or not path:
                raise ValueError(f"Expected NAME=PATH for --oof, got {value!r}.")
            artifacts[name] = Path(path).resolve(strict=True)
    else:
        artifacts = {
            name: artifact_dir / "groupwise-h64-d0p1-regression_oof.npy"
            for name in model_names
        }
    missing_names = sorted(set(model_names).difference(artifacts))
    if missing_names:
        raise ValueError(f"Missing OOF predictions for selector models: {missing_names}.")
    missing_paths = [str(path) for path in artifacts.values() if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing OOF prediction artifacts: {missing_paths}")
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--subset", choices=("primary", "holdout"), default="primary")
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--feature-dataset", type=Path, default=None)
    parser.add_argument("--oof", action="append", default=[])
    parser.add_argument(
        "--only", action="append", default=[], help="Screen only the named configuration (repeatable).",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    manifest_path = args.manifest or _default_manifest_path(args.output_root, "standard", args.subset)
    data_root, output_root, manifest_path = _validate_roots(args.data_root, args.output_root, manifest_path)
    manifest = prepare_manifest(data_root, manifest_path, "standard", subset=args.subset)
    artifact_dir = args.artifact_dir or output_root / "multimask_selection" / "groupwise_v1" / "models"
    artifacts = _parse_artifacts(args.model, artifact_dir)
    if args.subset == "primary":
        feature_dataset = args.feature_dataset or output_root / "multimask_selection" / "primary_features.npz"
        feature_dataset = feature_dataset.resolve(strict=True)
        oof_artifacts = _parse_oof_artifacts(args.oof, artifact_dir, artifacts)
    else:
        feature_dataset, oof_artifacts = None, None
    run_dir, summary = run_screening(
        manifest, data_root, output_root, artifacts, args.device, args.subset,
        feature_dataset=feature_dataset, oof_artifacts=oof_artifacts, only_configs=args.only,
    )
    print(summary[summary["dataset"] == "__dataset_balanced__"].to_string(index=False))
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
