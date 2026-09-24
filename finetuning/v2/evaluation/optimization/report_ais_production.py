"""Compare two tagged AIS production evaluations dataset by dataset.

Reads the result files `evaluate_automatic_segmentation.py` writes
(`<experiment>/results/<dataset>_micro_sam2_<model>_ais_<tag>_ckpt-<checksum>.csv`) for a baseline tag
and a candidate tag, and reports the metric per dataset (mSA, or the CREMI score for the dense EM
datasets), the relative change, the balanced means over the 2d datasets, over the twelve 2d datasets no
tuning ever saw (EXPERIMENTAL_SETUP.md, section 3) and over the 3d datasets, and the generalization gate.

Usage:
    python report_ais_production.py -e <experiment folder> --baseline default_old-defaults \\
        --candidate default_a2-defaults
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

EVALUATION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVALUATION_ROOT))

from common import DATASETS_2D, DATASETS_3D_EM, DATASETS_3D_LM, DATASETS_DENSE, VAL_SPLITS  # noqa
from optimization.benchmark_ais_optimization import GATE  # noqa

# The 2d development corpus of the 2026-09 campaigns; every other 2d dataset is strictly unseen by any tuning.
DEVELOPMENT_2D = (
    "livecell", "tissuenet", "dynamicnuclearnet", "deepbacs", "dic_hepg2",
    "yeaz", "neurips_cellseg", "deepseas", "puma", "covid_if", "tnbc",
)
UNSEEN_2D = tuple(d for d in DATASETS_2D if d not in DEVELOPMENT_2D)


def read_results(experiment: Path, model_type: str, tag: str, checksum: Optional[str]) -> Dict[str, pd.Series]:
    """The result row of every dataset with the given tag, keyed by dataset."""
    results = {}
    pattern = re.compile(
        rf"^(?P<dataset>.+)_micro_sam2_{re.escape(model_type)}_ais_{re.escape(tag)}_ckpt-(?P<ck>[0-9a-f]+)\.csv$"
    )
    for path in sorted((experiment / "results").glob("*.csv")):
        match = pattern.match(path.name)
        if match is None or (checksum is not None and not match.group("ck").startswith(checksum)):
            continue
        table = pd.read_csv(path)
        if len(table) != 1:
            raise ValueError(f"Expected one row in '{path}', got {len(table)}.")
        results[match.group("dataset")] = table.iloc[0]
    return results


def score(row: pd.Series, dataset: str) -> float:
    """mSA, or the negated CREMI score on the dense EM datasets (higher is better either way)."""
    if dataset in DATASETS_DENSE:
        return -float(row["cremi"]) if "cremi" in row else float("nan")
    return float(row["mSA"])


def compare(baseline: Dict[str, pd.Series], candidate: Dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for dataset in sorted(set(baseline) & set(candidate)):
        base, cand = score(baseline[dataset], dataset), score(candidate[dataset], dataset)
        rows.append({
            "dataset": dataset,
            "group": "2d" if dataset in DATASETS_2D else ("3d_lm" if dataset in DATASETS_3D_LM else "3d_em"),
            "unseen_2d": dataset in UNSEEN_2D,
            "has_val_split": dataset in VAL_SPLITS,
            "metric": "-cremi" if dataset in DATASETS_DENSE else "msa",
            "baseline": base, "candidate": cand,
            "relative": cand / base - 1.0 if base else np.nan, "absolute": cand - base,
        })
    return pd.DataFrame(rows)


def gate(table: pd.DataFrame) -> Dict[str, object]:
    if table.empty:
        return {"n": 0}
    relative, absolute = table["relative"].to_numpy(), table["absolute"].to_numpy()
    up = int((absolute > 0).sum())
    violates = (relative < GATE["max_relative_loss"]) & (absolute < GATE["max_absolute_loss"])
    balanced_gain = float(table["candidate"].mean() / table["baseline"].mean() - 1.0)
    return {
        "n": int(len(table)), "n_up": up, "balanced_baseline": float(table["baseline"].mean()),
        "balanced_candidate": float(table["candidate"].mean()), "balanced_gain": balanced_gain,
        "worst_relative": float(np.nanmin(relative)), "loss_limit_ok": bool(not violates.any()),
        "passed": bool(
            up >= len(table) - GATE["max_down"] and not violates.any() and balanced_gain >= GATE["min_balanced_gain"]
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-e", "--experiment_folder", type=Path, required=True)
    parser.add_argument("-m", "--model_type", default="hvit_t")
    parser.add_argument("--baseline", required=True, help="Result tag of the baseline, e.g. default_old-defaults.")
    parser.add_argument("--candidate", required=True, help="Result tag of the candidate, e.g. default_a2-defaults.")
    parser.add_argument("--checksum", default=None, help="Checkpoint checksum prefix the result files must carry.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    baseline = read_results(args.experiment_folder, args.model_type, args.baseline, args.checksum)
    candidate = read_results(args.experiment_folder, args.model_type, args.candidate, args.checksum)
    table = compare(baseline, candidate)
    missing = sorted((set(baseline) ^ set(candidate)))
    if table.empty:
        raise SystemExit(f"No dataset has both tags (baseline {len(baseline)}, candidate {len(candidate)} results).")
    pd.set_option("display.width", 200)
    shown = table.copy()
    shown["relative"] = shown["relative"].map(lambda v: f"{100 * v:+.1f}%")
    print(shown[["dataset", "group", "unseen_2d", "metric", "baseline", "candidate", "relative"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    for name, mask in (
        ("all 2d", table["group"] == "2d"),
        ("2d strictly unseen (out of domain)", (table["group"] == "2d") & table["unseen_2d"]),
        ("2d development", (table["group"] == "2d") & ~table["unseen_2d"]),
        ("3d LM", table["group"] == "3d_lm"),
        ("3d EM (dense, -CREMI)", table["group"] == "3d_em"),
    ):
        verdict = gate(table[mask])
        if verdict["n"]:
            print(f"\n{name}: n {verdict['n']}, up {verdict['n_up']}, balanced {verdict['balanced_baseline']:.4f} -> "
                  f"{verdict['balanced_candidate']:.4f} ({100 * verdict['balanced_gain']:+.1f} %), worst "
                  f"{100 * verdict['worst_relative']:+.1f} %, loss limit ok {verdict['loss_limit_ok']}, "
                  f"gate {verdict['passed']}")
    if missing:
        print(f"\nDatasets with only one of the two tags so far: {missing}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
        print(f"\nTable: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
