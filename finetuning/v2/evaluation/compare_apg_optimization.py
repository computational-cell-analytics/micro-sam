"""Compare serialized 2D APG benchmark runs against the optimization acceptance gates.

Repeat ``--baseline-run`` or ``--candidate-run`` for timing trials. Candidate trials with the same
implementation and resolved 2D parameters are grouped, and their per-dataset medians are compared.
All inputs must be complete runs of the same 2D manifest, model and checkpoint.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


EXPECTED_DATASETS = frozenset({"livecell", "tissuenet", "dynamicnuclearnet", "deepbacs", "dic_hepg2"})
BALANCED_ROW = "__dataset_balanced__"


def _read_run(path: Path) -> Tuple[Dict[str, Any], pd.DataFrame]:
    path = path.resolve(strict=True)
    with open(path / "metadata.json") as f:
        metadata = json.load(f)
    if metadata.get("status") != "complete":
        raise RuntimeError(f"Run is not complete: '{path}'.")
    if metadata.get("dimensions") != [2]:
        raise ValueError(f"Expected a 2D-only run, got dimensions={metadata.get('dimensions')} in '{path}'.")
    summary = pd.read_csv(path / "summary.csv")
    summary = summary.loc[summary["dataset"] != BALANCED_ROW].copy()
    datasets = set(summary["dataset"])
    if datasets != EXPECTED_DATASETS:
        raise ValueError(f"Expected datasets {sorted(EXPECTED_DATASETS)}, got {sorted(datasets)} in '{path}'.")
    if summary["dataset"].duplicated().any():
        raise ValueError(f"Dataset rows are not unique in '{path}'.")
    metadata["run_dir"] = str(path)
    return metadata, summary.set_index("dataset").sort_index()


def _validate_compatible(runs: Sequence[Tuple[Dict[str, Any], pd.DataFrame]]) -> None:
    identity_keys = ("manifest_checksum", "checkpoint_checksum", "checkpoint_name", "model_type")
    reference = {key: runs[0][0].get(key) for key in identity_keys}
    for metadata, _ in runs[1:]:
        identity = {key: metadata.get(key) for key in identity_keys}
        if identity != reference:
            raise ValueError(
                "Benchmark identities differ; all comparisons must use the same manifest and checkpoint: "
                f"{reference} != {identity}."
            )


def _group_key(metadata: Dict[str, Any]) -> str:
    identity = {
        "implementation_checksum": metadata["implementation_checksum"],
        "params_2d": metadata["params_2d"],
    }
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def _aggregate(runs: Sequence[Tuple[Dict[str, Any], pd.DataFrame]]) -> Tuple[Dict[str, Any], pd.DataFrame]:
    key = _group_key(runs[0][0])
    if any(_group_key(metadata) != key for metadata, _ in runs):
        raise ValueError("Timing trials in one group must have identical implementations and 2D parameters.")
    metrics = []
    for _, summary in runs:
        columns = ["msa_mean", "total_seconds"]
        if "peak_cuda_memory_bytes" in summary:
            columns.append("peak_cuda_memory_bytes")
        metrics.append(summary[columns])
    aggregated = pd.concat(metrics, keys=range(len(metrics)), names=["trial", "dataset"])
    aggregated = aggregated.groupby("dataset").median(numeric_only=True).sort_index()
    metadata = dict(runs[0][0])
    metadata["trial_ids"] = [run[0].get("trial_id") for run in runs]
    metadata["run_dirs"] = [run[0]["run_dir"] for run in runs]
    metadata["n_trials"] = len(runs)
    return metadata, aggregated


def _compare(
    baseline: Tuple[Dict[str, Any], pd.DataFrame],
    candidate: Tuple[Dict[str, Any], pd.DataFrame],
    target: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    baseline_metadata, baseline_table = baseline
    candidate_metadata, candidate_table = candidate
    rows = []
    for dataset in sorted(EXPECTED_DATASETS):
        base_msa = float(baseline_table.loc[dataset, "msa_mean"])
        candidate_msa = float(candidate_table.loc[dataset, "msa_mean"])
        base_runtime = float(baseline_table.loc[dataset, "total_seconds"])
        candidate_runtime = float(candidate_table.loc[dataset, "total_seconds"])
        rows.append({
            "candidate": candidate_metadata["config_name"],
            "dataset": dataset,
            "baseline_msa": base_msa,
            "candidate_msa": candidate_msa,
            "msa_change": candidate_msa / base_msa - 1.0,
            "baseline_seconds": base_runtime,
            "candidate_seconds": candidate_runtime,
            "runtime_change": candidate_runtime / base_runtime - 1.0,
            "speedup": 1.0 - candidate_runtime / base_runtime,
            "candidate_peak_cuda_memory_bytes": (
                int(candidate_table.loc[dataset, "peak_cuda_memory_bytes"])
                if "peak_cuda_memory_bytes" in candidate_table else None
            ),
        })

    macro_baseline = float(baseline_table["msa_mean"].mean())
    macro_candidate = float(candidate_table["msa_mean"].mean())
    macro_change = macro_candidate / macro_baseline - 1.0
    msa_changes = np.array([row["msa_change"] for row in rows])
    runtime_changes = np.array([row["runtime_change"] for row in rows])
    speedups = np.array([row["speedup"] for row in rows])
    quality_exception = bool(macro_change >= 0.10 and np.all(msa_changes > 0.0))
    if target == "quality":
        checks = {
            "macro_msa_at_least_5_percent": bool(macro_change >= 0.05),
            "at_most_two_datasets_below_minus_5_percent": bool(np.sum(msa_changes < -0.05) <= 2),
            "runtime_cap_or_quality_exception": bool(np.all(runtime_changes <= 0.10) or quality_exception),
        }
    else:
        checks = {
            "every_dataset_quality_loss_at_most_0_5_percent": bool(np.all(msa_changes >= -0.005)),
            "every_dataset_speedup_at_least_5_percent": bool(np.all(speedups >= 0.05)),
        }
    decision = {
        "candidate": candidate_metadata["config_name"],
        "target": target,
        "accepted": bool(all(checks.values())),
        "checks": checks,
        "quality_runtime_exception": quality_exception,
        "macro_baseline_msa": macro_baseline,
        "macro_candidate_msa": macro_candidate,
        "macro_msa_change": macro_change,
        "worst_dataset_msa_change": float(msa_changes.min()),
        "datasets_below_minus_5_percent": int(np.sum(msa_changes < -0.05)),
        "worst_dataset_runtime_change": float(runtime_changes.max()),
        "worst_dataset_speedup": float(speedups.min()),
        "total_runtime_change": (
            float(candidate_table["total_seconds"].sum() / baseline_table["total_seconds"].sum() - 1.0)
        ),
        "baseline": baseline_metadata,
        "candidate_metadata": candidate_metadata,
    }
    return decision, rows


def compare_runs(
    baseline_paths: Sequence[Path], candidate_paths: Sequence[Path], target: str,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    baseline_runs = [_read_run(path) for path in baseline_paths]
    candidate_runs = [_read_run(path) for path in candidate_paths]
    _validate_compatible(baseline_runs + candidate_runs)
    baseline = _aggregate(baseline_runs)
    groups = defaultdict(list)
    for run in candidate_runs:
        groups[_group_key(run[0])].append(run)

    decisions, rows = [], []
    for group in groups.values():
        decision, detail = _compare(baseline, _aggregate(group), target)
        decisions.append(decision)
        rows.extend(detail)
    if target == "quality":
        decisions.sort(key=lambda item: (-item["macro_candidate_msa"], item["worst_dataset_runtime_change"]))
    else:
        decisions.sort(key=lambda item: (
            -item["worst_dataset_speedup"], item["total_runtime_change"], -item["macro_candidate_msa"]
        ))
    return decisions, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", action="append", required=True, type=Path)
    parser.add_argument("--candidate-run", action="append", required=True, type=Path)
    parser.add_argument("--target", required=True, choices=("quality", "efficiency"))
    parser.add_argument("--output", type=Path, help="Optional JSON path; detailed rows use the same stem as CSV.")
    args = parser.parse_args()

    decisions, details = compare_runs(args.baseline_run, args.candidate_run, args.target)
    payload = json.dumps(decisions, indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
        details.to_csv(args.output.with_suffix(".csv"), index=False)
        print(f"Decisions: {args.output}")
        print(f"Details: {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
