"""Screen the candidate supply of the 2d APG under the learned selector and filter.

The learned filter rejects poor masks far better than the predicted-IoU threshold did, which makes
lower candidate thresholds affordable: more density components are prompted, and the filter decides.
This screen re-extracts selector features for every proposal setting (prompts re-index when the
threshold changes, so the existing out-of-fold predictions do not apply), trains one pooled selector
with image-level out-of-fold predictions across all settings, and then screens the settings against
a grid of learned-score thresholds, overlap limits and size floors - each image encoded once, each
setting proposed once, each selection replayed from the proposals. It reports where the recall goes:
objects seeded, proposed, scored and merged.

Usage examples:
    python screen_apg_candidate_supply.py --stage extract
    python screen_apg_candidate_supply.py --stage train
    python screen_apg_candidate_supply.py --stage screen
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

EVALUATION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVALUATION_ROOT))

import common  # noqa
from common import GT_MIN_SIZE_2D, unmatched_objects  # noqa
from parameter_search import compute_metrics  # noqa
from optimization.benchmark_apg_optimization import (  # noqa
    DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, _atomic_write_csv, _atomic_write_json, _content_checksum,
    _default_manifest_path, _git_revision, _hardware_identity, _implementation_checksum, _load_2d_sample,
    _validate_roots, prepare_manifest,
)
from optimization.screen_apg_multimask import _configured_records, _load_oof_lookup, _oof_predictions_for_sample  # noqa
from optimization.train_apg_multimask_selector import _record_target, extract_dataset, train_selector  # noqa

SCHEMA = "token_lowres_v1"
# The accepted first pass, minus what the screen varies.
BASE_PARAMS = {"dt": 0.25, "sigma": 0.5, "min_candidate_size": 4, "n_iter": 50}
DEFAULT_CANDIDATE_THRESHOLDS = (3.0, 2.0, 1.5, 1.0, 0.5)
DEFAULT_FOREGROUND_THRESHOLDS = (0.7, 0.5)
DEFAULT_SCORE_THRESHOLDS = tuple(sorted({round(v, 3) for v in np.arange(0.25, 0.6001, 0.05)} | {0.375}))
DEFAULT_MAX_OVERLAPS = (0.15, 0.3, 0.5)
DEFAULT_MIN_SIZES = (25, 50)


def setting_name(candidate_threshold: float, foreground_threshold: float) -> str:
    return f"ct{candidate_threshold:g}_fg{foreground_threshold:g}".replace(".", "p")


def settings_grid(candidate_thresholds: Sequence[float], foreground_thresholds: Sequence[float]) -> List[dict]:
    return [
        {**BASE_PARAMS, "candidate_threshold": float(ct), "foreground_threshold": float(fg)}
        for ct in candidate_thresholds for fg in foreground_thresholds
    ]


def feature_path(root: Path, setting: dict) -> Path:
    name = setting_name(setting["candidate_threshold"], setting["foreground_threshold"])
    return root / f"primary_features_{name}.npz"


def stage_extract(manifest, data_root, feature_root, settings, device):
    outputs = [feature_path(feature_root, setting) for setting in settings]
    pending = [(setting, path) for setting, path in zip(settings, outputs) if not path.exists()]
    if not pending:
        print("All feature datasets exist.")
        return outputs
    extract_dataset(
        manifest, data_root, outputs[0], device, multimasking=True, input_schema=SCHEMA,
        proposal_settings=[setting for setting, _ in pending], outputs=[path for _, path in pending],
    )
    return outputs


def stage_train(feature_paths: Sequence[Path], model_root: Path, device: str, hidden_size: int) -> Path:
    return train_selector([path.resolve(strict=True) for path in feature_paths], model_root, device,
                          hidden_size=hidden_size, input_schema=SCHEMA)


def _oof_name_for(model_root: Path, artifact: Path, feature: Path) -> Path:
    return model_root / f"{artifact.stem}_oof_{feature.stem}.npy"


def _scored_objects(records: Sequence[dict], targets: Dict[int, float], labels: np.ndarray) -> int:
    scored = set()
    for index, record in enumerate(records):
        if targets.get(id(record), 0.0) >= 0.5:
            x, y = np.round(record["point"]).astype("int64")
            x, y = int(np.clip(x, 0, labels.shape[1] - 1)), int(np.clip(y, 0, labels.shape[0] - 1))
            if labels[y, x]:
                scored.add(int(labels[y, x]))
    return len(scored)


def stage_screen(
    manifest, data_root, output_root, feature_root, model_root, artifact: Path, settings, score_thresholds,
    max_overlaps, min_sizes, device,
) -> Path:
    samples = [sample for sample in manifest["samples"] if sample["ndim"] == 2]
    lookups = {}
    for setting in settings:
        feature = feature_path(feature_root, setting)
        oof = _oof_name_for(model_root, artifact, feature)
        lookups[setting_name(setting["candidate_threshold"], setting["foreground_threshold"])] = _load_oof_lookup(
            feature, {"selector": oof}, manifest["manifest_checksum"],
        )
    identity = _content_checksum({
        "settings": settings, "score_thresholds": list(score_thresholds), "max_overlaps": list(max_overlaps),
        "min_sizes": list(min_sizes), "artifact": artifact.name, "manifest": manifest["manifest_checksum"],
        "implementation": _implementation_checksum(),
    })
    run_dir = output_root / "candidate_supply_screening" / "hvit_t" / identity
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_path = run_dir / "samples.csv"
    done = pd.read_csv(samples_path) if samples_path.exists() else pd.DataFrame()
    done_ids = set(done["sample_id"]) if not done.empty else set()
    checkpoint = common.get_joint_checkpoint("hvit_t", "best")
    segmenter = common.build_apg_segmenter(
        "hvit_t", 2, device, joint_checkpoint="best", joint_checksum=common.checkpoint_checksum(checkpoint),
        export_root=str(DEFAULT_OUTPUT_ROOT / "model_exports"),
    )
    _atomic_write_json(run_dir / "metadata.json", {
        "settings": settings, "score_thresholds": list(score_thresholds), "max_overlaps": list(max_overlaps),
        "min_sizes": list(min_sizes), "artifact": str(artifact), "prediction_source": "out-of-fold",
        "manifest_checksum": manifest["manifest_checksum"], "implementation_checksum": _implementation_checksum(),
        "git_revision": _git_revision(), "hardware": _hardware_identity(device), "status": "running",
    })
    rows = [] if done.empty else done.to_dict("records")
    try:
        for number, sample in enumerate(samples, 1):
            if sample["sample_id"] in done_ids:
                continue
            raw, labels = _load_2d_sample(sample, data_root)
            border_min_size = GT_MIN_SIZE_2D.get(sample["dataset"], 0)
            n_objects = int(len(np.unique(labels)) - 1)
            segmenter.clear_state()
            segmenter.initialize(raw, ndim=2)
            for setting in settings:
                name = setting_name(setting["candidate_threshold"], setting["foreground_threshold"])
                features, predictions, lookup = lookups[name]
                proposals = segmenter.propose(
                    multimasking=True, multimask_scorer="predicted_iou", multimask_selection="deferred",
                    return_multimask_features=True, multimask_feature_schema=SCHEMA, **setting,
                )
                oof = _oof_predictions_for_sample(sample["sample_id"], proposals, features, predictions, lookup)
                records = _configured_records(proposals, {"selection": "eager", "merge": "learned"}, oof["selector"])
                targets = {id(record): _record_target(record, labels) for record in records}
                seeded = {int(labels[int(np.clip(round(r["point"][1]), 0, labels.shape[0] - 1)),
                                     int(np.clip(round(r["point"][0]), 0, labels.shape[1] - 1))]) for r in records}
                seeded.discard(0)
                proposed = {
                    int(labels[int(np.clip(round(r["point"][1]), 0, labels.shape[0] - 1)),
                               int(np.clip(round(r["point"][0]), 0, labels.shape[1] - 1))])
                    for r in records if targets[id(r)] >= 0.5
                }
                proposed.discard(0)
                for threshold, max_overlap, min_size in itertools.product(score_thresholds, max_overlaps, min_sizes):
                    started = time.perf_counter()
                    segmentation, context = segmenter._merge(
                        records, labels.shape, score_threshold=float(threshold), max_overlap=float(max_overlap),
                        min_size=int(min_size), return_context=True, score_filter="selection_score",
                    )
                    select_seconds = time.perf_counter() - started
                    kept = [] if context is None else [
                        context["records"][index] for index in context["matches"].values()
                    ]
                    metrics = compute_metrics(
                        segmentation.astype("uint32"), labels, "sparse", border_min_size=border_min_size,
                    )
                    unmatched = np.unique(unmatched_objects(labels, segmentation))
                    rows.append({
                        "sample_id": sample["sample_id"], "dataset": sample["dataset"], "setting": name,
                        "candidate_threshold": setting["candidate_threshold"],
                        "foreground_threshold": setting["foreground_threshold"],
                        "score_threshold": float(threshold), "max_overlap": float(max_overlap),
                        "min_size": int(min_size),
                        "config_name": f"{name}-t{threshold:g}-mo{max_overlap:g}-ms{min_size}",
                        "n_prompts": len(records), "gt_objects": n_objects, "seeded": len(seeded),
                        "proposed": len(proposed), "scored": _scored_objects(kept, targets, labels),
                        "merged": n_objects - int(np.count_nonzero(unmatched)),
                        "predicted_objects": int(len(np.unique(segmentation)) - 1),
                        "select_seconds": select_seconds, **metrics,
                    })
            _atomic_write_csv(samples_path, pd.DataFrame(rows))
            print(f"[{number}/{len(samples)}] {sample['sample_id']}", flush=True)
    finally:
        segmenter.clear_state()
    table = pd.DataFrame(rows)
    summary = summarize(table)
    _atomic_write_csv(run_dir / "summary.csv", summary)
    metadata = json.load(open(run_dir / "metadata.json"))
    metadata["status"] = "complete"
    _atomic_write_json(run_dir / "metadata.json", metadata)
    top = summary[summary["dataset"] == "__dataset_balanced__"].head(15)
    columns = ["config_name", "msa_mean", "seeded", "proposed", "scored", "merged", "gt_objects"]
    print(top[columns].to_string(index=False))
    print(f"Run directory: {run_dir}")
    return run_dir


def summarize(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sums = ("gt_objects", "seeded", "proposed", "scored", "merged", "n_prompts", "predicted_objects")
    for name, frame in samples.groupby("config_name", sort=False):
        table = frame.groupby("dataset", sort=True).agg(
            n_samples=("sample_id", "count"), msa_mean=("msa", "mean"), select_seconds=("select_seconds", "sum"),
            **{column: (column, "sum") for column in sums},
        ).reset_index()
        table.insert(0, "config_name", name)
        for column in ("candidate_threshold", "foreground_threshold", "score_threshold", "max_overlap", "min_size"):
            table[column] = frame[column].iloc[0]
        rows.append(table)
        rows.append(pd.DataFrame([{
            "config_name": name, "dataset": "__dataset_balanced__", "n_samples": len(frame),
            "msa_mean": float(table["msa_mean"].mean()), "select_seconds": float(table["select_seconds"].sum()),
            **{column: int(table[column].sum()) for column in sums},
            **{column: frame[column].iloc[0] for column in (
                "candidate_threshold", "foreground_threshold", "score_threshold", "max_overlap", "min_size",
            )},
        }]))
    summary = pd.concat(rows, ignore_index=True)
    ranks = summary[summary["dataset"] == "__dataset_balanced__"].sort_values("msa_mean", ascending=False)
    order = {name: index for index, name in enumerate(ranks["config_name"])}
    summary["_order"] = summary["config_name"].map(order)
    return summary.sort_values(["_order", "dataset"]).drop(columns="_order").reset_index(drop=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("extract", "train", "screen", "all"), default="all")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--candidate-threshold", type=float, nargs="*", default=list(DEFAULT_CANDIDATE_THRESHOLDS))
    parser.add_argument("--foreground-threshold", type=float, nargs="*", default=list(DEFAULT_FOREGROUND_THRESHOLDS))
    parser.add_argument("--score-threshold", type=float, nargs="*", default=list(DEFAULT_SCORE_THRESHOLDS))
    parser.add_argument("--max-overlap", type=float, nargs="*", default=list(DEFAULT_MAX_OVERLAPS))
    parser.add_argument("--min-size", type=int, nargs="*", default=list(DEFAULT_MIN_SIZES))
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--artifact", type=Path, default=None, help="Pooled selector artifact for --stage screen.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    manifest_path = args.manifest or _default_manifest_path(args.output_root, "standard", "primary")
    data_root, output_root, manifest_path = _validate_roots(args.data_root, args.output_root, manifest_path)
    manifest = prepare_manifest(data_root, manifest_path, "standard", subset="primary")
    settings = settings_grid(args.candidate_threshold, args.foreground_threshold)
    feature_root = output_root / "multimask_selection" / SCHEMA / "candidate_supply"
    model_root = output_root / "multimask_selection" / "groupwise_v1" / SCHEMA / "candidate_supply" / "models"
    feature_paths = [feature_path(feature_root, setting) for setting in settings]
    artifact = args.artifact
    if args.stage in ("extract", "all"):
        stage_extract(manifest, data_root, feature_root, settings, args.device)
    if args.stage in ("train", "all"):
        artifact = stage_train(feature_paths, model_root, args.device, args.hidden_size)
        print(f"Artifact: {artifact}")
    if args.stage in ("screen", "all"):
        if artifact is None:
            candidates = sorted(model_root.glob(f"{SCHEMA}-groupwise-h{args.hidden_size}-d0p1-regression-pooled*.pt"))
            if not candidates:
                raise SystemExit("No pooled artifact found; run --stage train or pass --artifact.")
            artifact = candidates[-1]
        stage_screen(
            manifest, data_root, output_root, feature_root, model_root, Path(artifact), settings,
            args.score_threshold, args.max_overlap, args.min_size, args.device,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
