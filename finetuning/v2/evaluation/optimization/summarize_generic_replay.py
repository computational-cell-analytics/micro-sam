"""Join the compact-selector replay screens of the primary and training_extra manifests into one table.

Every screen run directory (``compact_selector_screening/hvit_t/<ckpt>/<hash>/``) carries ``samples.csv``
with one mSA per (image, config) and ``metadata.json`` with the subset and the score filter. This script
takes any number of run directories, stacks their per-image rows, and reports for every candidate score and
threshold the dataset-balanced mSA over all datasets seen, the change against the predicted-IoU baseline
candidate at its best threshold, and the worst per-dataset change, so in-domain (OOF) and out-of-domain
(LODO) variants of the same model can be read side by side.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_runs(run_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for run_dir in run_dirs:
        metadata = json.loads((run_dir / "metadata.json").read_text())
        samples = pd.read_csv(run_dir / "samples.csv")
        samples["subset"] = metadata.get("subset", "primary")
        samples["score_filter"] = metadata.get("score_filter", "selection_score")
        samples["candidate"] = samples["config_name"].str.replace(r"-(eager|deferred)-t[0-9.]+$", "", regex=True)
        frames.append(samples)
    return pd.concat(frames, ignore_index=True)


def summarize(samples: pd.DataFrame, baseline: str = "baseline_predicted_iou") -> pd.DataFrame:
    per_dataset = samples.groupby(["score_filter", "candidate", "score_threshold", "dataset"], sort=False)["msa"].mean()
    table = per_dataset.unstack("dataset")
    n_datasets = table.notna().sum(axis=1)
    balanced = table.mean(axis=1)
    rows = []
    for score_filter, group in table.groupby(level="score_filter"):
        has_baseline = baseline in group.index.get_level_values("candidate")
        base_rows = group.xs(baseline, level="candidate", drop_level=False) if has_baseline else None
        if base_rows is None or base_rows.empty:
            best_base = None
        else:
            best_base_key = base_rows.mean(axis=1).idxmax()
            best_base = base_rows.loc[best_base_key]
        for key, values in group.iterrows():
            record = {
                "score_filter": score_filter, "candidate": key[1], "threshold": key[2],
                "n_datasets": int(n_datasets.loc[key]), "balanced_msa": float(balanced.loc[key]),
            }
            if best_base is not None:
                deltas = (values - best_base) / best_base
                record.update({
                    "baseline_threshold": float(best_base_key[2]), "baseline_balanced_msa": float(best_base.mean()),
                    "balanced_change": float(balanced.loc[key] / best_base.mean() - 1.0),
                    "worst_dataset_change": float(deltas.min()), "worst_dataset": str(deltas.idxmin()),
                    "datasets_improved": int((deltas > 0).sum()), "datasets_below_2pct": int((deltas < -0.02).sum()),
                })
            rows.append(record)
    summary = pd.DataFrame(rows)
    best = summary.sort_values("balanced_msa", ascending=False).drop_duplicates(["score_filter", "candidate"])
    best = best.sort_values(["score_filter", "balanced_msa"], ascending=[True, False]).reset_index(drop=True)
    return best, table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--baseline", default="baseline_predicted_iou")
    args = parser.parse_args()
    samples = load_runs(args.run_dirs)
    best, table = summarize(samples, args.baseline)
    columns = [
        "score_filter", "candidate", "threshold", "n_datasets", "balanced_msa", "balanced_change",
        "worst_dataset_change", "worst_dataset", "datasets_improved", "datasets_below_2pct",
    ]
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_rows", 200):
        print(best[[c for c in columns if c in best.columns]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if args.output:
        best.to_csv(args.output, index=False)
        table.to_csv(args.output.with_name(args.output.stem + "_per_dataset.csv"))
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
