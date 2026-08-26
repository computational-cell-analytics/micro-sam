"""Screen single/three-mask APG heads and learned pre-merge filtering on established 2D data.

Primary runs replay image-level OOF scores. Holdout runs load refit artifacts. Every threshold reuses
the same two decoder passes per image; only record filtering and merging are repeated.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
import time

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
    _load_2d_sample, _validate_roots, prepare_manifest,
)
from optimization.screen_apg_multimask import _load_oof_lookup, _oof_predictions_for_sample  # noqa


FILTER_THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.20, 0.801, 0.05))


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _threshold_name(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _configs() -> list:
    configs = [
        {"name": "three-iou-default", "head": "three", "selection": "eager", "learned": False,
         "filter": "predicted_iou", "threshold": 0.6},
        {"name": "single-iou-default", "head": "single", "selection": "eager", "learned": False,
         "filter": "predicted_iou", "threshold": 0.6},
    ]
    for head, selection, prefix in (
        ("single", "eager", "single-mlp"),
        ("three", "eager", "three-mlp-eager"),
        ("three", "deferred", "three-mlp-deferred"),
    ):
        configs.extend([
            {"name": f"{prefix}-iou-filter", "head": head, "selection": selection,
             "learned": True, "filter": "predicted_iou", "threshold": 0.6},
            {"name": f"{prefix}-no-filter", "head": head, "selection": selection,
             "learned": True, "filter": "none", "threshold": 0.0},
        ])
        configs.extend({
            "name": f"{prefix}-mlp-filter-{_threshold_name(threshold)}", "head": head,
            "selection": selection, "learned": True, "filter": "selection_score",
            "threshold": threshold,
        } for threshold in FILTER_THRESHOLDS)
    return configs


def _configure(proposals: list, scores: np.ndarray, selection: str, learned: bool) -> list:
    records = [dict(record) for record in proposals]
    if len(records) != len(scores):
        raise ValueError(f"Expected {len(records)} scorer values, got {len(scores)}.")
    for record, score in zip(records, scores):
        record["selection_score"] = float(score)
        record["merge_score"] = (
            float(score) if learned else record["predicted_iou"] * record["stability_score"]
        )
    if selection == "deferred" or not records or "multimask_group" not in records[0]:
        return records
    groups = {}
    for index, record in enumerate(records):
        groups.setdefault(record["multimask_group"], []).append(index)
    chosen = []
    for indices in groups.values():
        index = max(indices, key=lambda candidate: (records[candidate]["selection_score"], -candidate))
        record = records[index]
        record.pop("multimask_group", None)
        chosen.append(record)
    return chosen


def _summary(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, frame in samples.groupby("config_name", sort=False):
        table = frame.groupby("dataset", sort=True).agg(
            n_samples=("sample_id", "count"), msa_mean=("msa", "mean"),
            selection_seconds=("selection_seconds", "sum"),
            eligible_candidates=("eligible_candidates", "sum"),
        ).reset_index()
        table.insert(0, "config_name", name)
        rows.append(table)
        rows.append(pd.DataFrame([{
            "config_name": name, "dataset": "__dataset_balanced__", "n_samples": len(frame),
            "msa_mean": float(table["msa_mean"].mean()),
            "selection_seconds": float(table["selection_seconds"].sum()),
            "eligible_candidates": int(table["eligible_candidates"].sum()),
        }]))
    summary = pd.concat(rows, ignore_index=True)
    ranks = summary[summary["dataset"] == "__dataset_balanced__"].sort_values(
        "msa_mean", ascending=False,
    )["config_name"].tolist()
    order = {name: index for index, name in enumerate(ranks)}
    summary["_order"] = summary["config_name"].map(order)
    return summary.sort_values(["_order", "dataset"]).drop(columns="_order").reset_index(drop=True)


def run_screening(
    manifest: dict, data_root: Path, output_root: Path, subset: str, device: str,
    single_model_path: Path, three_model_path: Path,
    single_feature_dataset: Path | None = None, single_oof_path: Path | None = None,
    three_feature_dataset: Path | None = None, three_oof_path: Path | None = None,
    only: list[str] | None = None,
) -> tuple[Path, pd.DataFrame]:
    configs = _configs()
    if only:
        unknown = sorted(set(only).difference(config["name"] for config in configs))
        if unknown:
            raise ValueError(f"Unknown configurations: {unknown}.")
        configs = [config for config in configs if config["name"] in only]
    use_oof = subset == "primary"
    if use_oof:
        required = (single_feature_dataset, single_oof_path, three_feature_dataset, three_oof_path)
        if any(path is None for path in required):
            raise ValueError("Primary screening requires both feature datasets and both OOF score files.")
        single_rows, single_values, single_lookup = _load_oof_lookup(
            single_feature_dataset, {"single": single_oof_path}, manifest["manifest_checksum"],
        )
        three_rows, three_values, three_lookup = _load_oof_lookup(
            three_feature_dataset, {"three": three_oof_path}, manifest["manifest_checksum"],
        )
    else:
        single_rows = single_values = single_lookup = None
        three_rows = three_values = three_lookup = None

    single_model = load_feature_scorer(single_model_path, device=device)
    three_model = load_feature_scorer(three_model_path, device=device)
    identity = {
        "manifest_checksum": manifest["manifest_checksum"], "implementation_checksum": _implementation_checksum(),
        "screen_implementation_checksum": _checksum(Path(__file__)), "subset": subset,
        "models": {"single": _checksum(single_model_path), "three": _checksum(three_model_path)},
        "configs": configs, "prediction_source": "out-of-fold" if use_oof else "refit-model",
    }
    if use_oof:
        identity["feature_datasets"] = {
            "single": _checksum(single_feature_dataset), "three": _checksum(three_feature_dataset),
        }
        identity["oof_scores"] = {"single": _checksum(single_oof_path), "three": _checksum(three_oof_path)}
    checkpoint = common.get_joint_checkpoint("hvit_t", "best")
    checkpoint_id = common.checkpoint_checksum(checkpoint)
    run_dir = output_root / "mask_head_filter_screening" / "hvit_t" / checkpoint_id / _content_checksum(identity)
    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(run_dir / "metadata.json", {
        **identity, "device": device, "git_revision": _git_revision(),
        "model_paths": {"single": str(single_model_path), "three": str(three_model_path)},
    })
    samples_path = run_dir / "samples.csv"
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
            segmenter.set_multimask_models(scorer=single_model)
            single = segmenter.propose(
                multimasking=False, multimask_scorer="microscopy", multimask_selection="eager",
                return_multimask_features=use_oof,
            )
            segmenter.set_multimask_models(scorer=three_model)
            three = segmenter.propose(
                multimasking=True, multimask_scorer="microscopy", multimask_selection="deferred",
            )
            if use_oof:
                single_scores = _oof_predictions_for_sample(
                    sample["sample_id"], single, single_rows, single_values, single_lookup,
                )["single"]
                three_scores = _oof_predictions_for_sample(
                    sample["sample_id"], three, three_rows, three_values, three_lookup,
                )["three"]
            else:
                single_scores = np.asarray([record["selection_score"] for record in single], dtype="float32")
                three_scores = np.asarray([record["selection_score"] for record in three], dtype="float32")

            result_rows = []
            for config in configs:
                proposals, learned_scores = (
                    (single, single_scores) if config["head"] == "single" else (three, three_scores)
                )
                ranking_scores = (
                    learned_scores if config["learned"]
                    else np.asarray([record["predicted_iou"] for record in proposals], dtype="float32")
                )
                started = time.perf_counter()
                records = _configure(proposals, ranking_scores, config["selection"], config["learned"])
                segmentation = segmenter.select(
                    records, score_filter=config["filter"], score_threshold=config["threshold"],
                ).astype("uint32")
                elapsed = time.perf_counter() - started
                if config["filter"] == "none":
                    eligible = len(records)
                else:
                    eligible = sum(record[config["filter"]] >= config["threshold"] for record in records)
                metrics = compute_metrics(
                    segmentation, labels, "sparse",
                    border_min_size=GT_MIN_SIZE_2D.get(sample["dataset"], 0),
                )
                result_rows.append({
                    "sample_id": sample["sample_id"], "dataset": sample["dataset"],
                    "config_name": config["name"], "msa": metrics["msa"],
                    "selection_seconds": elapsed, "eligible_candidates": eligible,
                    "predicted_objects": int(segmentation.max()),
                })
            completed = pd.concat([completed, pd.DataFrame(result_rows)], ignore_index=True)
            _atomic_write_csv(samples_path, completed)
            print(f"[{number}/{len(pending)}] {sample['sample_id']}", flush=True)
    finally:
        segmenter.clear_state()
    summary = _summary(completed)
    _atomic_write_csv(run_dir / "summary.csv", summary)
    return run_dir, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--subset", choices=("primary", "holdout"), default="primary")
    parser.add_argument("--single-model", type=Path, required=True)
    parser.add_argument("--three-model", type=Path, required=True)
    parser.add_argument("--single-feature-dataset", type=Path)
    parser.add_argument("--single-oof", type=Path)
    parser.add_argument("--three-feature-dataset", type=Path)
    parser.add_argument("--three-oof", type=Path)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    manifest_path = args.manifest or _default_manifest_path(args.output_root, "standard", args.subset)
    data_root, output_root, manifest_path = _validate_roots(args.data_root, args.output_root, manifest_path)
    manifest = prepare_manifest(data_root, manifest_path, "standard", subset=args.subset)
    run_dir, summary = run_screening(
        manifest, data_root, output_root, args.subset, args.device,
        args.single_model.resolve(strict=True), args.three_model.resolve(strict=True),
        args.single_feature_dataset.resolve(strict=True) if args.single_feature_dataset else None,
        args.single_oof.resolve(strict=True) if args.single_oof else None,
        args.three_feature_dataset.resolve(strict=True) if args.three_feature_dataset else None,
        args.three_oof.resolve(strict=True) if args.three_oof else None,
        args.only,
    )
    print(summary[summary["dataset"] == "__dataset_balanced__"].to_string(index=False))
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
