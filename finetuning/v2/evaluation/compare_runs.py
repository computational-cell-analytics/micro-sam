"""Collect the v2 and v3 evaluation results into one comparison table per segmentation mode.

Both runs went through the same pipeline, so their result files differ only in the experiment root and
in the checkpoint tag the interactive filenames carry. Writes 'ais.csv', 'apg.csv' and
'interactive.csv', each holding one row per dataset (and per prompt and iteration for the interactive
mode) with the score of both runs, their difference and which run is better.

Usage:
    python compare_runs.py
    python compare_runs.py -o /path/to/output
"""

import argparse
from glob import glob
from pathlib import Path

import pandas as pd

from common import DATASETS_2D, DATASETS_3D_EM
from baselines_common import interactive_result_name

EXPERIMENT_ROOT = Path("/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments")

# The two runs to compare. 'tag' is the method name the interactive filenames use, which encodes the
# joint checkpoint the run was evaluated with.
RUNS = {
    "v2": {
        "results": EXPERIMENT_ROOT / "v2_rerun_evaluation" / "results",
        "apg_3d": EXPERIMENT_ROOT / "v2_rerun_apg_3d",
        "tag": "micro_sam2",
    },
    "v3": {
        "results": EXPERIMENT_ROOT / "v3_joint_evaluation" / "results",
        "apg_3d": EXPERIMENT_ROOT / "v3_apg_3d",
        "tag": "micro_sam2_best_epoch72",
    },
}

DATASETS = ["livecell", "tissuenet", "dynamicnuclearnet", "deepbacs", "gonuclear", "embedseg", "cremi", "snemi"]
PROMPTS = ["box", "point"]
N_ITERATIONS = 8


def dataset_kind(dataset_name: str) -> str:
    """Return '2d LM', '3d LM' or '3d EM' for a dataset."""
    if dataset_name in DATASETS_2D:
        return "2d LM"
    return "3d EM" if dataset_name in DATASETS_3D_EM else "3d LM"


def metric_for(dataset_name: str) -> str:
    """EM neurons are ranked by the CREMI score, everything else by mSA."""
    return "cremi" if dataset_name in DATASETS_3D_EM else "mSA"


def read_score(path, metric: str):
    """Return the metric of a result CSV, or None when the file or the column is missing."""
    path = Path(path)
    if not path.exists():
        return None
    table = pd.read_csv(path)
    return float(table[metric].iloc[0]) if metric in table.columns else None


def comparison_row(dataset_name: str, scores: dict, extra: dict = None) -> dict:
    """Build one row from the per-run scores, with the delta oriented by the metric."""
    metric = metric_for(dataset_name)
    lower_is_better = (metric == "cremi")
    v2, v3 = scores["v2"], scores["v3"]
    delta = None if (v2 is None or v3 is None) else v3 - v2
    if delta is None:
        better = None
    else:
        better = "v3" if ((delta < 0) if lower_is_better else (delta > 0)) else "v2"
    row = {"dataset": dataset_name, "kind": dataset_kind(dataset_name), "metric": metric}
    row.update(extra or {})
    row.update({"v2": v2, "v3": v3, "delta": delta, "better": better})
    return row


def collect_ais(model_type: str) -> pd.DataFrame:
    """The automatic instance segmentation scores, from the tuned post-processing."""
    rows = []
    for dataset_name in DATASETS:
        scores = {
            run: read_score(
                cfg["results"] / f"{dataset_name}_micro_sam2_{model_type}_auto_tuned.csv",
                metric_for(dataset_name),
            ) for run, cfg in RUNS.items()
        }
        rows.append(comparison_row(dataset_name, scores))
    return pd.DataFrame(rows)


def collect_apg(model_type: str) -> pd.DataFrame:
    """The automatic prompt generation scores.

    2d results are written by the baseline script with a parameter digest in the name, 3d results by
    'evaluate_3d.py --mode apg' into a per-dataset folder, so both locations are searched.
    """
    rows = []
    for dataset_name in DATASETS:
        metric = metric_for(dataset_name)
        scores = {}
        for run, cfg in RUNS.items():
            matches = sorted(glob(str(cfg["results"] / f"{dataset_name}_micro_sam2_{model_type}_apg_*.csv")))
            if matches:
                scores[run] = read_score(matches[0], metric)
            else:
                scores[run] = read_score(cfg["apg_3d"] / dataset_name / "results" / "apg_tuned_tuned.csv", metric)
        rows.append(comparison_row(dataset_name, scores))
    return pd.DataFrame(rows)


def collect_interactive(model_type: str) -> pd.DataFrame:
    """The interactive scores for every prompt type and iteration."""
    rows = []
    for dataset_name in DATASETS:
        metric = metric_for(dataset_name)
        ndim = 2 if dataset_kind(dataset_name) == "2d LM" else 3
        for prompt in PROMPTS:
            for iteration in range(N_ITERATIONS):
                scores = {}
                for run, cfg in RUNS.items():
                    name = interactive_result_name(
                        dataset_name, cfg["tag"], model_type, prompt, iteration, ndim=ndim
                    )
                    scores[run] = read_score(cfg["results"] / name, metric)
                rows.append(comparison_row(dataset_name, scores, {"prompt": prompt, "iteration": iteration}))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output_dir", default=".", help="Directory to write the comparison CSVs to.")
    parser.add_argument("-m", "--model_type", default="hvit_t", help="The SAM2 backbone the runs used.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "ais.csv": collect_ais(args.model_type),
        "apg.csv": collect_apg(args.model_type),
        "interactive.csv": collect_interactive(args.model_type),
    }
    for name, table in tables.items():
        path = output_dir / name
        table.to_csv(path, index=False)
        missing = int(table["delta"].isna().sum())
        print(f"Wrote {path} with {len(table)} rows, {missing} incomplete.")
        wins = table["better"].value_counts()
        print(f"{name}: v3 better in {int(wins.get('v3', 0))} of {int(wins.sum())} comparisons.")


if __name__ == "__main__":
    main()
