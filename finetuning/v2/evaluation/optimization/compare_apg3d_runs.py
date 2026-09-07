"""Compare the 3d benchmark runs of two checkpoints and two configurations by object counts first, mSA second.

Reads the per-crop results of `benchmark_apg_3d.py run` for joint/v2 and joint/v4 geodesic with the volume
defaults and the `points+boxes` refinement, and prints per source: ground-truth objects, matched objects
(IoU 0.5), genuine misses, predicted objects (so false positives are visible as predicted minus matched), mSA
with a per-crop bootstrap interval, propagation passes and seconds. Family, unseen-source and dataset-balanced
macros follow the conventions of `benchmark_apg_3d.summarize`.

Usage:
    python compare_apg3d_runs.py --subset primary
    python compare_apg3d_runs.py --subset holdout --out /some/where/table.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

OPTIMIZATION_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(OPTIMIZATION_ROOT))
sys.path.insert(0, str(OPTIMIZATION_ROOT.parent))

from package_apg3d_cases import CHECKPOINTS, CONFIGS, DEFAULT_OUTPUT_ROOT, load_run  # noqa

COUNTS = ("gt_objects", "merged", "genuine_misses", "predicted_objects", "propagation_passes", "total_seconds")


def per_source(table: pd.DataFrame) -> pd.DataFrame:
    from benchmark_apg_3d import _bootstrap_ci

    rows = []
    for dataset, group in table.groupby("dataset", sort=True):
        low, high = _bootstrap_ci(group["msa"].to_numpy()) if len(group) > 1 else (np.nan, np.nan)
        rows.append({
            "dataset": dataset, "crops": len(group), "family": group["family"].iloc[0],
            "seen": str(group["seen_in_training"].iloc[0]), "msa": float(group["msa"].mean()),
            "msa_low": float(low), "msa_high": float(high), **{key: float(group[key].sum()) for key in COUNTS},
        })
    frame = pd.DataFrame(rows).set_index("dataset")
    frame["matched_fraction"] = frame["merged"] / frame["gt_objects"]
    frame["extra_predictions"] = frame["predicted_objects"] - frame["merged"]
    return frame


def macros(frame: pd.DataFrame) -> Dict[str, float]:
    family = frame.groupby("family")["msa"].mean().mean()
    unseen = frame[frame["seen"] == "False"]["msa"].mean()
    return {
        "family_macro_msa": float(family), "unseen_macro_msa": float(unseen),
        "balanced_msa": float(frame["msa"].mean()),
        "matched_total": float(frame["merged"].sum()), "gt_total": float(frame["gt_objects"].sum()),
        "misses_total": float(frame["genuine_misses"].sum()), "extra_total": float(frame["extra_predictions"].sum()),
        "seconds_total": float(frame["total_seconds"].sum()),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", default="primary")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    pd.set_option("display.width", 260)

    frames, tables = {}, {}
    for checkpoint in CHECKPOINTS:
        for config in CONFIGS:
            _, table = load_run(checkpoint, config, args.subset)
            tables[(checkpoint, config)] = table
            frames[f"{checkpoint}_{config}"] = per_source(table)
    names = list(frames)
    common = None
    for table in tables.values():
        common = set(table.index) if common is None else common & set(table.index)
    per_run = ", ".join(
        f"{name} {len(frames[name].index)} datasets / {int(frames[name]['crops'].sum())} crops" for name in names
    )
    print(f"{args.subset}: crops per run {per_run}; {len(common)} crops in all four runs")

    joined = pd.concat({name: frames[name][["msa", "merged", "gt_objects", "genuine_misses", "extra_predictions"]]
                        for name in names}, axis=1)
    print("\nPer source (mSA | matched objects | ground truth | genuine misses | extra predictions):")
    for name in names:
        block = frames[name]
        print(f"\n== {name}")
        columns = [
            "crops", "msa", "msa_low", "msa_high", "gt_objects", "merged", "matched_fraction", "genuine_misses",
            "extra_predictions", "propagation_passes", "total_seconds",
        ]
        print(block[columns].round(3).to_string())
    print("\nMacros:")
    summary = pd.DataFrame({name: macros(frames[name]) for name in names}).T
    print(summary.round(4).to_string())
    base = frames["v2_defaults"]
    print("\nRelative mSA change per source vs v2 defaults (%):")
    rel = pd.DataFrame({
        name: (frames[name]["msa"] - base["msa"]) / base["msa"] * 100 for name in names if name != "v2_defaults"
    })
    rel["matched v2->v4 defaults"] = frames["v4_defaults"]["merged"] - base["merged"]
    rel["matched v4 defaults->refine"] = frames["v4_refine"]["merged"] - frames["v4_defaults"]["merged"]
    rel["extra v2->v4 defaults"] = frames["v4_defaults"]["extra_predictions"] - base["extra_predictions"]
    print(rel.round(2).to_string())
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        joined.to_csv(args.out)
        summary.to_csv(args.out.with_name(args.out.stem + "_macros.csv"))
        rel.to_csv(args.out.with_name(args.out.stem + "_relative.csv"))
        print(f"Tables: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
