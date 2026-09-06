"""Screen compact three-token APG selectors and their learned-score filter threshold.

The primary split is evaluated exclusively with image-level out-of-fold predictions. One deferred
``token_lowres_v1`` proposal pass is shared by every scorer and threshold, so the screen compares
the final merge policies without repeating the image encoder or mask decoder. The winning policy
must still be confirmed with serialized end-to-end timing trials on the holdout split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

EVALUATION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVALUATION_ROOT))

import common  # noqa
from parameter_search import compute_metrics  # noqa
from optimization.benchmark_apg_optimization import (  # noqa
    DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, GT_MIN_SIZE_2D, _atomic_write_csv, _atomic_write_json,
    _content_checksum, _default_manifest_path, _git_revision, _implementation_checksum,
    _load_2d_sample, _validate_roots, prepare_manifest, MANIFEST_SUBSETS,
)
from optimization.screen_apg_multimask import _configured_records, PINNED_PROPOSAL_2D  # noqa


SCHEMA = "token_lowres_v1"
# The primary feature datasets were proposed with the pinned campaign settings; the training_extra dataset
# was extracted by the trainer's plain path, i.e. with the library's per-model defaults. A replay must
# re-propose exactly as its feature dataset was extracted, or the prompt indices do not line up.
PROPOSAL_SETTINGS = {"pinned": PINNED_PROPOSAL_2D, "library": {}}
DEFAULT_THRESHOLDS = tuple(float(value) for value in np.arange(0.15, 0.5001, 0.025))


def _dataset_lookup(path: Path, manifest_checksum: str) -> tuple[np.ndarray, dict]:
    data = np.load(path, allow_pickle=False)
    if str(data["manifest_checksum"]) != manifest_checksum:
        raise ValueError(
            f"Feature dataset {path} was extracted from a different manifest: "
            f"{data['manifest_checksum']} != {manifest_checksum}."
        )
    lookup = {}
    for index, (sample_id, group, alternative) in enumerate(
        zip(data["sample_ids"], data["groups"], data["alternatives"])
    ):
        key = (str(sample_id), int(str(group).rsplit(":", 1)[1]), int(alternative))
        if key in lookup:
            raise ValueError(f"Duplicate feature-dataset key: {key}.")
        lookup[key] = index
    return data["features"].astype("float32", copy=False), lookup


def _indices_for_sample(sample_id: str, proposals: list, lookup: dict) -> np.ndarray:
    indices = []
    for record in proposals:
        key = (sample_id, int(record["prompt_index"]), int(record["multimask_index"]))
        try:
            indices.append(lookup[key])
        except KeyError as error:
            raise ValueError(f"Proposal {key} is missing from the selector dataset.") from error
    return np.asarray(indices, dtype="int64")


def _load_candidates(model_dir: Path, explicit: list[str], n_rows: int) -> dict:
    if explicit:
        paths = {}
        for value in explicit:
            name, separator, path = value.partition("=")
            if not separator or not name or not path:
                raise ValueError(f"Expected NAME=PATH for --oof, got {value!r}.")
            paths[name] = Path(path).resolve(strict=True)
    else:
        paths = {
            path.name.removesuffix("_oof.npy"): path
            for path in sorted(model_dir.glob("*_oof.npy"))
        }
    if not paths:
        raise FileNotFoundError(f"No OOF selector predictions found below {model_dir}.")
    predictions = {}
    for name, path in paths.items():
        values = np.load(path, allow_pickle=False).astype("float32", copy=False)
        if values.shape != (n_rows,):
            raise ValueError(f"OOF predictions for {name!r} have shape {values.shape}, expected {(n_rows,)}.")
        predictions[name] = {"path": path, "values": values}
    return predictions


def _parse_model_name(name: str) -> tuple[str, int]:
    schema, _, remainder = name.partition("-groupwise-h")
    if not remainder:
        return schema, -1
    return schema, int(remainder.partition("-")[0])


def _summarize(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["config_name", "input_schema", "hidden_size", "selection", "score_threshold"]
    for keys, frame in samples.groupby(group_columns, sort=False):
        table = frame.groupby("dataset", sort=True).agg(
            n_samples=("sample_id", "count"), msa_mean=("msa", "mean"),
            selection_seconds=("selection_seconds", "sum"),
        ).reset_index()
        values = dict(zip(group_columns, keys))
        for key, value in values.items():
            table.insert(len(table.columns) - 3, key, value)
        rows.append(table)
        rows.append(pd.DataFrame([{
            **values, "dataset": "__dataset_balanced__", "n_samples": len(frame),
            "msa_mean": float(table["msa_mean"].mean()),
            "selection_seconds": float(table["selection_seconds"].sum()),
        }]))
    summary = pd.concat(rows, ignore_index=True)
    ranking = summary[summary["dataset"] == "__dataset_balanced__"].sort_values(
        ["msa_mean", "selection_seconds"], ascending=[False, True],
    )["config_name"].tolist()
    order = {name: index for index, name in enumerate(ranking)}
    summary["_order"] = summary["config_name"].map(order)
    return summary.sort_values(["_order", "dataset"]).drop(columns="_order").reset_index(drop=True)


def run_screening(
    manifest: dict, data_root: Path, output_root: Path, device: str, feature_dataset: Path,
    candidates: dict, thresholds: tuple[float, ...], selections: tuple[str, ...],
    score_filter: str = "selection_score", subset: str = "primary", proposal_settings: str = "pinned",
) -> tuple[Path, pd.DataFrame]:
    """Replay saved (out-of-fold or leave-one-dataset-out) selector scores through select().

    'score_filter' decides what the threshold applies to: the replayed learned score (the default,
    learned selection and learned filter) or 'predicted_iou' (learned selection only, SAM2's own IoU
    filter), which separates the two effects of a selector.
    """
    feature_rows, lookup = _dataset_lookup(feature_dataset, manifest["manifest_checksum"])
    candidate_data = _load_candidates(candidates["model_dir"], candidates["explicit"], len(feature_rows))
    configs = []
    for name in candidate_data:
        input_schema, hidden_size = _parse_model_name(name)
        for selection in selections:
            for threshold in thresholds:
                configs.append({
                    "name": f"{name}-{selection}-t{threshold:.3f}", "model": name,
                    "input_schema": input_schema, "hidden_size": hidden_size,
                    "threshold": float(threshold), "selection": selection, "merge": "learned",
                })

    identity = {
        "manifest_checksum": manifest["manifest_checksum"],
        "implementation_checksum": _implementation_checksum(),
        "screen_implementation_checksum": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "feature_dataset": hashlib.sha256(feature_dataset.read_bytes()).hexdigest(),
        "oof_predictions": {
            name: hashlib.sha256(item["path"].read_bytes()).hexdigest()
            for name, item in candidate_data.items()
        },
        "thresholds": list(thresholds), "input_schema": SCHEMA,
        "selections": list(selections), "merge": "learned", "prediction_source": "out-of-fold",
        "score_filter": score_filter, "subset": subset, "proposal_settings": proposal_settings,
    }
    checkpoint = common.get_joint_checkpoint("hvit_t", "best")
    checkpoint_id = common.checkpoint_checksum(checkpoint)
    run_dir = output_root / "compact_selector_screening" / "hvit_t" / checkpoint_id / _content_checksum(identity)
    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(run_dir / "metadata.json", {
        **identity, "screening": True, "device": device,
        "git_revision": _git_revision(), "feature_dataset_path": str(feature_dataset),
        "oof_paths": {name: str(item["path"]) for name, item in candidate_data.items()},
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
                return_multimask_features=True, multimask_feature_schema=SCHEMA,
                **PROPOSAL_SETTINGS[proposal_settings],
            )
            indices = _indices_for_sample(sample["sample_id"], proposals, lookup)
            if proposals:
                current = np.stack([record["multimask_features"] for record in proposals])
                if not np.allclose(current, feature_rows[indices], rtol=1e-5, atol=1e-5):
                    raise ValueError(
                        f"Regenerated features differ from the extracted dataset for {sample['sample_id']!r}."
                    )
            configured = {}
            for name, item in candidate_data.items():
                for selection in selections:
                    configured[name, selection] = _configured_records(
                        proposals, {"selection": selection, "merge": "learned"}, item["values"][indices],
                    )
            rows = []
            for config in configs:
                started = time.perf_counter()
                segmentation = segmenter.select(
                    configured[config["model"], config["selection"]], score_filter=score_filter,
                    score_threshold=config["threshold"],
                ).astype("uint32")
                elapsed = time.perf_counter() - started
                metrics = compute_metrics(
                    segmentation, labels, "sparse", border_min_size=GT_MIN_SIZE_2D.get(sample["dataset"], 0),
                )
                rows.append({
                    "sample_id": sample["sample_id"], "dataset": sample["dataset"],
                    "config_name": config["name"], "input_schema": config["input_schema"],
                    "hidden_size": config["hidden_size"], "selection": config["selection"],
                    "score_threshold": config["threshold"],
                    "msa": metrics["msa"], "selection_seconds": elapsed,
                    "predicted_objects": int(segmentation.max()),
                })
            completed = pd.concat([completed, pd.DataFrame(rows)], ignore_index=True)
            _atomic_write_csv(samples_path, completed)
            print(f"[{number}/{len(pending)}] {sample['sample_id']}", flush=True)
    finally:
        segmenter.clear_state()
    summary = _summarize(completed)
    _atomic_write_csv(summary_path, summary)
    balanced = summary[summary["dataset"] == "__dataset_balanced__"].sort_values(
        ["msa_mean", "selection_seconds"], ascending=[False, True],
    )
    winner = balanced.iloc[0].to_dict()
    with open(run_dir / "winner.json", "w") as f:
        json.dump(winner, f, indent=2, sort_keys=True)
        f.write("\n")
    return run_dir, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--feature-dataset", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--oof", action="append", default=[], help="NAME=PATH; repeat for explicit candidates.")
    parser.add_argument("--threshold", action="append", type=float, default=[])
    parser.add_argument("--selection", action="append", choices=("eager", "deferred"), default=[])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--subset", choices=MANIFEST_SUBSETS, default="primary",
                        help="Manifest subset whose images are replayed (the feature dataset must match).")
    parser.add_argument("--score-filter", choices=("selection_score", "predicted_iou"), default="selection_score",
                        help="Apply the thresholds to the replayed learned score or to SAM2's predicted IoU.")
    parser.add_argument("--proposal-settings", choices=tuple(PROPOSAL_SETTINGS), default="pinned",
                        help="Re-propose with the pinned campaign settings or the library defaults (training_extra).")
    args = parser.parse_args()
    manifest_path = args.manifest or _default_manifest_path(args.output_root, "standard", args.subset)
    data_root, output_root, manifest_path = _validate_roots(args.data_root, args.output_root, manifest_path)
    manifest = prepare_manifest(data_root, manifest_path, "standard", subset=args.subset)
    root = output_root / "multimask_selection"
    feature_dataset = (
        args.feature_dataset or root / SCHEMA / "primary_features.npz"
    ).resolve(strict=True)
    model_dir = (
        args.model_dir or root / "groupwise_v1" / SCHEMA / "models"
    ).resolve(strict=True)
    thresholds = tuple(args.threshold) if args.threshold else DEFAULT_THRESHOLDS
    if not thresholds or not all(np.isfinite(thresholds)):
        raise ValueError("At least one finite threshold is required.")
    selections = tuple(args.selection) if args.selection else ("eager",)
    run_dir, summary = run_screening(
        manifest, data_root, output_root, args.device, feature_dataset,
        {"model_dir": model_dir, "explicit": args.oof}, thresholds, selections,
        score_filter=args.score_filter, subset=args.subset, proposal_settings=args.proposal_settings,
    )
    balanced = summary[summary["dataset"] == "__dataset_balanced__"].sort_values(
        ["msa_mean", "selection_seconds"], ascending=[False, True],
    )
    print(balanced.head(20).to_string(index=False))
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
