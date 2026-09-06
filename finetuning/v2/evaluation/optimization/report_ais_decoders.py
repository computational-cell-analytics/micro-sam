"""Compare the AIS decoder variants of the 2026-09 training campaign on the benchmark's cached runs.

Reads the run directories of the staged checkpoints (`<root>/ais/hvit_t/<checkpoint id>/`), joins the
requested subsets, and reports per (variant, configuration): balanced mSA and the generalization gate against
the baseline model under the library defaults, the mechanism columns of the D2 decomposition as a share of
the ground-truth objects (merged, absorbed, unseeded, background seeds) and the extent figures
(matched IoU, foreground IoU, foreground area ratio). Reader only: not part of the implementation checksum.

    python report_ais_decoders.py --subsets primary training_extra --ndim 2 --output <root>/ais/reports/decoders_dev
    python report_ais_decoders.py --kind apg3d --subsets primary holdout --ndim 3 --configs current-defaults
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import benchmark_ais_optimization as ais  # noqa: E402
from common import checkpoint_checksum  # noqa: E402

DEFAULT_ROOT = Path("/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/apg_optimization")
DEFAULT_STAGED = DEFAULT_ROOT / "ais_decoder_training" / "staged" / "joint_sam2_hvit_t_multi_gpu"
VARIANTS = ("baseline", "contact", "fgcal", "both")
MECHANISMS = ("seeded_merged", "unseeded_absorbed", "gt_with_0_seeds", "seeded_split", "background_seeds")
EXTENT = ("matched_iou", "fg_iou", "fg_area_ratio")


def find_runs(
    root: Path, model_type: str, checkpoint_id: str, manifest_checksums: Sequence[str], config_names: Sequence[str],
    dimensions: List[int], epoch: Optional[str],
) -> Dict[str, List[Tuple[Path, Dict]]]:
    """Complete run directories of one checkpoint, keyed by configuration name; the newest epoch if not given."""
    runs: Dict[str, List[Tuple[Path, Dict]]] = {}
    for metadata_path in sorted((root / ais.CAMPAIGN / model_type / checkpoint_id).glob("*/metadata.json")):
        with open(metadata_path) as f:
            metadata = json.load(f)
        if metadata.get("status") != "complete" or metadata.get("manifest_checksum") not in manifest_checksums:
            continue
        if metadata.get("config_name") not in config_names:
            continue
        if not set(dimensions) <= set(metadata.get("dimensions", [])):
            continue
        if epoch is not None and metadata.get("implementation_checksum") != epoch:
            continue
        runs.setdefault(metadata["config_name"], []).append((metadata_path.parent, metadata))
    # One run per (config, manifest): keep the newest epoch / trial when several exist.
    for name, entries in runs.items():
        by_manifest: Dict[str, Tuple[Path, Dict]] = {}
        for run_dir, metadata in sorted(entries, key=lambda e: e[0].stat().st_mtime):
            by_manifest[metadata["manifest_checksum"]] = (run_dir, metadata)
        runs[name] = list(by_manifest.values())
    return runs


def load_samples(entries: Sequence[Tuple[Path, Dict]], ndim: int, datasets: Optional[Sequence[str]]) -> pd.DataFrame:
    samples = pd.concat([ais.load_run(run_dir)[1] for run_dir, _ in entries], ignore_index=True)
    samples = samples[samples["ndim"] == ndim]
    if datasets:
        samples = samples[samples["dataset"].isin(datasets)]
    return samples.reset_index(drop=True)


def mechanism_table(samples: pd.DataFrame) -> pd.DataFrame:
    """Per dataset: the mechanism counts as a share of the ground-truth objects and the extent means."""
    rows = []
    for dataset, group in samples.groupby("dataset", sort=True):
        gt = float(group["gt_objects"].sum())
        row = {"dataset": dataset, "gt_objects": int(gt), "msa": float(group["msa"].mean())}
        for column in MECHANISMS:
            row[column] = float(group[column].sum() / gt) if column in group and gt else float("nan")
        for column in EXTENT:
            row[column] = float(group[column].mean()) if column in group else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--staged-dir", type=Path, default=DEFAULT_STAGED,
                        help="Directory of the staged <variant>.pt files.")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--baseline-variant", default="baseline")
    parser.add_argument("--production-checkpoint", default=None,
                        help="A joint checkpoint (e.g. the v4 production one) to include as the variant 'production'.")
    parser.add_argument("--configs", nargs="+", default=["current-defaults", "contact-ridge", "contact-mask"])
    parser.add_argument("--baseline-config", default="current-defaults")
    parser.add_argument("--kind", default="v5", choices=ais.KINDS)
    parser.add_argument("--subsets", nargs="+", default=["primary", "training_extra"])
    parser.add_argument("--ndim", type=int, default=2)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--epoch", default=None, help="Implementation checksum to select; default: any (newest).")
    parser.add_argument("--model-type", default="hvit_t")
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/vast-nhr/projects/cidas/cca/data"))
    parser.add_argument("--campaign-root", type=Path, default=DEFAULT_ROOT / "3d_v2")
    parser.add_argument("--output", default=None, help="Prefix of the CSVs to write.")
    args = parser.parse_args()

    manifests = [
        ais.load_campaign_manifest(args.kind, subset, args.output_root, args.data_root, args.campaign_root)
        for subset in args.subsets
    ]
    checksums = [m["manifest_checksum"] for m in manifests]
    dimensions = [args.ndim]

    checkpoints = {}
    for variant in args.variants:
        path = args.staged_dir / f"{variant}.pt"
        if path.exists():
            checkpoints[variant] = checkpoint_checksum(str(path))
        else:
            print(f"[skip] no staged checkpoint for '{variant}' at {path}")
    if args.production_checkpoint:
        checkpoints["production"] = checkpoint_checksum(args.production_checkpoint)

    tables: Dict[Tuple[str, str], pd.DataFrame] = {}
    for variant, checkpoint_id in checkpoints.items():
        runs = find_runs(
            args.output_root, args.model_type, checkpoint_id, checksums, args.configs, dimensions, args.epoch,
        )
        for config_name, entries in runs.items():
            found = {m["manifest_checksum"] for _, m in entries}
            if found != set(checksums):
                print(f"[skip] {variant}/{config_name}: runs for {len(found)}/{len(checksums)} subsets only")
                continue
            tables[(variant, config_name)] = load_samples(entries, args.ndim, args.datasets)
            print(f"[ok] {variant:10s} {config_name:18s} checkpoint {checkpoint_id[:8]} "
                  f"epoch {entries[0][1]['implementation_checksum'][:8]} n={len(tables[(variant, config_name)])}")

    reference = (args.baseline_variant, args.baseline_config)
    if reference not in tables:
        raise SystemExit(f"The reference {reference} has no complete runs; nothing to compare against.")
    baseline_scores = ais.dataset_scores(tables[reference])
    baseline_mechanisms = mechanism_table(tables[reference]).set_index("dataset")

    summary_rows, detail_rows, mechanism_rows = [], [], []
    for (variant, config_name), samples in tables.items():
        scores = ais.dataset_scores(samples)
        verdict = ais.gate_table(baseline_scores, scores)
        mechanisms = mechanism_table(samples).set_index("dataset")
        weights = mechanisms["gt_objects"]
        row = {
            "variant": variant, "config": config_name, "n_samples": int(len(samples)),
            "balanced": verdict["balanced_candidate"], "balanced_gain": verdict["balanced_gain"],
            "n_up": verdict["n_up"], "n_datasets": verdict["n_datasets"], "worst_relative": verdict["worst_relative"],
            "passed": verdict["passed"],
        }
        for column in MECHANISMS:
            # Object-weighted share over the datasets, and its change against the reference in percentage points.
            share = float(np.nansum(mechanisms[column] * weights) / weights.sum())
            reference_share = float(np.nansum(baseline_mechanisms[column] * weights) / weights.sum())
            row[column] = share
            row[f"{column}_delta"] = share - reference_share
        for column in EXTENT:
            row[column] = float(mechanisms[column].mean())
        summary_rows.append(row)
        for dataset in verdict["datasets"]:
            detail_rows.append({
                "variant": variant, "config": config_name, "dataset": dataset,
                "baseline": float(baseline_scores[dataset]), "candidate": float(scores[dataset]),
                "relative": verdict["relative"][dataset],
            })
        for dataset, values in mechanisms.iterrows():
            mechanism_rows.append({"variant": variant, "config": config_name, "dataset": dataset, **values.to_dict()})

    summary = pd.DataFrame(summary_rows).sort_values("balanced", ascending=False).reset_index(drop=True)
    details = pd.DataFrame(detail_rows)
    mechanisms_all = pd.DataFrame(mechanism_rows)

    pd.set_option("display.width", 250)
    shown = summary.copy()
    for column in ("balanced_gain", "worst_relative"):
        shown[column] = shown[column].map(lambda v: "n/a" if v is None or not np.isfinite(v) else f"{100 * v:+.1f}%")
    for column in MECHANISMS:
        shown[column] = shown[column].map(lambda v: f"{100 * v:.1f}%")
        shown[f"{column}_delta"] = shown[f"{column}_delta"].map(lambda v: f"{100 * v:+.1f}")
    print("\nSummary (mechanisms as % of ground-truth objects, deltas in percentage points vs the reference):")
    print(shown.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    pivot = details.pivot_table(index=["variant", "config"], columns="dataset", values="relative")
    print("\nRelative mSA change per dataset vs the reference:")
    print(pivot.to_string(float_format=lambda v: f"{100 * v:+.1f}%"))
    absolute = details.pivot_table(index=["variant", "config"], columns="dataset", values="candidate")
    print("\nmSA per dataset:")
    print(absolute.to_string(float_format=lambda v: f"{v:.4f}"))
    extent = mechanisms_all.pivot_table(index=["variant", "config"], columns="dataset", values="fg_area_ratio")
    print("\nForeground area ratio (fg > threshold / ground truth) per dataset:")
    print(extent.to_string(float_format=lambda v: f"{v:.2f}"))
    lost = mechanisms_all.assign(lost=mechanisms_all["seeded_merged"] + mechanisms_all["unseeded_absorbed"])
    merged = lost.pivot_table(index=["variant", "config"], columns="dataset", values="lost")
    print("\nMerged + absorbed objects (% of ground truth) per dataset:")
    print(merged.to_string(float_format=lambda v: f"{100 * v:.1f}%"))

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        summary.to_csv(f"{args.output}.csv", index=False)
        details.to_csv(f"{args.output}_datasets.csv", index=False)
        mechanisms_all.to_csv(f"{args.output}_mechanisms.csv", index=False)
        print(f"\nwritten {args.output}.csv, _datasets.csv, _mechanisms.csv")


if __name__ == "__main__":
    main()
