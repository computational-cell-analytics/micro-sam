"""Check the learned 2d APG configuration on every production 2d dataset, seen and unseen alike.

The learned selector and refinement gate were fitted on five datasets' validation splits. Whether
their gain carries to the other production datasets is the question this script answers: it runs
`evaluate_automatic_segmentation.py --mode apg` per dataset for the control and every candidate
configuration (`--submit` fans the tasks out through the campaign submitter), then `--report`
compares the result files, dataset by dataset and as seen / unseen macros.

The test splits never drive a selection here: the configurations are frozen before this runs.

Usage examples:
    python evaluate_apg_generalization.py tasks --print-only
    python evaluate_apg_generalization.py tasks --name e1_generalization --preset 2d --throttle 12
    python evaluate_apg_generalization.py report
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EVALUATION_ROOT = Path(__file__).resolve().parent.parent
OPTIMIZATION_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALUATION_ROOT))
sys.path.insert(0, str(OPTIMIZATION_ROOT))

import common  # noqa
from submit_optimization_jobs import add_submit_arguments, submit_from_args  # noqa

OUTPUT_ROOT = Path("/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/apg_optimization")
EXPERIMENT_FOLDER = OUTPUT_ROOT / "production_generalization" / "v2_best"
CONFIG_ROOT = OPTIMIZATION_ROOT / "configs"
SELECTOR = (
    OUTPUT_ROOT / "multimask_selection/groupwise_v1/token_lowres_v1/models/"
    "token_lowres_v1-groupwise-h64-d0p1-regression.pt"
)
GATE = (
    OUTPUT_ROOT / "multimask_selection/groupwise_v1/refinement_gate/compact_h64_eager/postmerge_signed/models/"
    "postmerge-gate-mlp-h128x64-d0p1-regression-signed.pt"
)
# The five datasets the selector and gate were fitted on; everything else in DATASETS_2D is unseen.
SEEN = ("livecell", "tissuenet", "dynamicnuclearnet", "deepbacs", "dic_hepg2")
CONFIGS = {
    "registry-defaults": (None, {}),
    "campaign-defaults": (CONFIG_ROOT / "apg_control_campaign_defaults.json", {}),
    "selector-only": (CONFIG_ROOT / "apg_accepted_selector_only.json", {"multimask_scorer_artifact": SELECTOR}),
    "selector-gate15": (
        CONFIG_ROOT / "apg_accepted_selector_gate15.json",
        {"multimask_scorer_artifact": SELECTOR, "refinement_gate_artifact": GATE},
    ),
    # The proposal-side E2 setting on the plain predicted-IoU path (no learned component). It was chosen on
    # the eleven datasets with validation splits (SEEN plus yeaz, neurips_cellseg, puma, tnbc, covid_if,
    # deepseas), so for this configuration only the twelve remaining datasets are strictly unseen.
    "e2-plain-t0p5": (CONFIG_ROOT / "apg_e2_plain_t0p5.json", {}),
}
# The structural, label-free candidates of the 2026-09 generalization campaign (`configs/apg_s_*.json`,
# see APG_2D_GENERALIZATION_CAMPAIGN_PLAN.md). They were screened on the eleven datasets with validation
# splits, so, as for the E2 setting, the twelve remaining datasets are the strictly unseen ones.
CONFIGS.update({
    path.stem[4:].replace("_", "-"): (path, {}) for path in sorted(CONFIG_ROOT.glob("apg_s_*.json"))
})
# Relative loss a dataset may show before it counts as a regression, and the absolute allowance for
# datasets whose baseline is near zero (as in compare_apg_optimization's replacement gate).
LOSS_LIMIT = -0.05
ABSOLUTE_ALLOWANCE = 0.005
MODEL_TYPE = "hvit_t"
CHECKPOINT = "best"


def unseen_datasets() -> List[str]:
    return [dataset for dataset in common.DATASETS_2D if dataset not in SEEN]


def build_tasks(
    experiment_folder: Path = EXPERIMENT_FOLDER, configs: Optional[Sequence[str]] = None,
    datasets: Optional[Sequence[str]] = None, model_type: str = MODEL_TYPE,
) -> List[Tuple[str, str]]:
    script = EVALUATION_ROOT / "evaluate_automatic_segmentation.py"
    tasks = []
    for name in (configs or CONFIGS):
        config_path, artifacts = CONFIGS[name]
        for dataset in (datasets or common.DATASETS_2D):
            args: List[Any] = [
                "python", str(script), "-d", dataset, "-m", model_type, "--mode", "apg",
                "-e", str(experiment_folder), "--skip_tuning", "--result_tag", name,
            ]
            if config_path is not None:
                args.extend(["--apg_params", str(config_path)])
            for flag, path in artifacts.items():
                args.extend([f"--{flag}", str(path)])
            tasks.append((f"e1_{name}_{dataset}", shlex.join(str(arg) for arg in args)))
    return tasks


def _result_path(experiment_folder: Path, dataset: str, name: str, model_type: str) -> Optional[Path]:
    matches = sorted((experiment_folder / "results").glob(
        f"{dataset}_micro_sam2_{model_type}_apg_default_{name}_ckpt-*.csv"
    ))
    return matches[-1] if matches else None


def load_results(experiment_folder: Path = EXPERIMENT_FOLDER, model_type: str = MODEL_TYPE) -> pd.DataFrame:
    rows = []
    for name in CONFIGS:
        for dataset in common.DATASETS_2D:
            path = _result_path(experiment_folder, dataset, name, model_type)
            if path is None:
                continue
            table = pd.read_csv(path)
            metric = "mSA" if "mSA" in table else ("msa" if "msa" in table else None)
            row = {"config": name, "dataset": dataset, "seen": dataset in SEEN, "path": str(path)}
            if metric is not None:
                row["msa"] = float(table[metric].iloc[0])
            for column in ("SA50", "precision", "recall", "Precision", "Recall"):
                if column in table:
                    row[column.lower()] = float(table[column].iloc[0])
            rows.append(row)
    return pd.DataFrame(rows)


def compare_production_results(results: pd.DataFrame, control: str = "registry-defaults") -> Dict[str, Any]:
    """Per-dataset deltas against the control and seen / unseen / all macros per candidate."""
    table = results.pivot(index="dataset", columns="config", values="msa")
    decision: Dict[str, Any] = {"control": control, "candidates": {}}
    if control not in table:
        raise SystemExit(f"No control results for '{control}'.")
    for name in table.columns:
        if name == control:
            continue
        both = table[[control, name]].dropna()
        delta = both[name] - both[control]
        relative = delta / both[control].replace(0, np.nan)
        regressions = [
            dataset for dataset in both.index
            if relative[dataset] < LOSS_LIMIT and delta[dataset] < -ABSOLUTE_ALLOWANCE
        ]
        macros = {}
        for group, members in (("seen", SEEN), ("unseen", unseen_datasets()), ("all", list(common.DATASETS_2D))):
            selected = both.loc[[dataset for dataset in both.index if dataset in members]]
            if selected.empty:
                continue
            macro_control = float(selected[control].mean())
            macro_candidate = float(selected[name].mean())
            macros[group] = {
                "n_datasets": int(len(selected)), "control": macro_control, "candidate": macro_candidate,
                "relative_change": (macro_candidate - macro_control) / macro_control if macro_control else float("nan"),
            }
        unseen = macros.get("unseen", {})
        decision["candidates"][name] = {
            "macros": macros,
            "regressions": regressions,
            "per_dataset": {
                dataset: {"control": float(both.loc[dataset, control]), "candidate": float(both.loc[dataset, name]),
                          "delta": float(delta[dataset]), "relative": float(relative[dataset])}
                for dataset in both.index
            },
            "accepted": bool(unseen and unseen["relative_change"] >= 0.05 and not [
                dataset for dataset in regressions if dataset not in SEEN
            ]),
        }
    return decision


def report(experiment_folder: Path = EXPERIMENT_FOLDER, model_type: str = MODEL_TYPE) -> None:
    results = load_results(experiment_folder, model_type)
    if results.empty:
        raise SystemExit(f"No results under {experiment_folder / 'results'}.")
    results.to_csv(experiment_folder / "generalization_results.csv", index=False)
    decision = compare_production_results(results)
    with open(experiment_folder / "generalization_decision.json", "w") as f:
        json.dump(decision, f, indent=2, sort_keys=True)
    rows = []
    for name, entry in decision["candidates"].items():
        for dataset, values in entry["per_dataset"].items():
            rows.append({"config": name, "dataset": dataset, "seen": dataset in SEEN, **values})
    summary = pd.DataFrame(rows)
    summary.to_csv(experiment_folder / "generalization_summary.csv", index=False)
    print(results.pivot(index="dataset", columns="config", values="msa").round(4).to_string())
    for name, entry in decision["candidates"].items():
        macros = entry["macros"]
        line = ", ".join(
            f"{group}: {values['control']:.4f} -> {values['candidate']:.4f} "
            f"({values['relative_change']:+.2%}, n={values['n_datasets']})"
            for group, values in macros.items()
        )
        print(f"{name}: {line}; regressions {entry['regressions']}; accepted={entry['accepted']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    tasks = subparsers.add_parser("tasks", help="Build (and submit) the evaluation tasks.")
    tasks.add_argument("--configs", nargs="*", default=None, choices=sorted(CONFIGS))
    tasks.add_argument("--datasets", nargs="*", default=None)
    tasks.add_argument("--experiment-folder", type=Path, default=EXPERIMENT_FOLDER)
    tasks.add_argument("--print-only", action="store_true")
    add_submit_arguments(tasks)
    rep = subparsers.add_parser("report", help="Compare the result files.")
    rep.add_argument("--experiment-folder", type=Path, default=EXPERIMENT_FOLDER)
    args = parser.parse_args(argv)
    if args.command == "report":
        report(args.experiment_folder)
        return 0
    task_list = build_tasks(args.experiment_folder, args.configs, args.datasets)
    for tag, command in task_list:
        print(f"{tag}\t{command}")
    if args.print_only:
        return 0
    args.experiment_folder.mkdir(parents=True, exist_ok=True)
    submit_from_args(task_list, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
