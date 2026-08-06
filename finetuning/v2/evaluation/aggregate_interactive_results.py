"""Aggregate the per-iteration interactive result CSVs into one table per prompt type.

Reads '<experiment_folder>/results/<dataset>_<method>_<model><dim>_<prompt><tags>_iter<NN>.csv' and
writes '<experiment_folder>/results_interactive_<prompt>.csv'.

The EM datasets additionally get variation of information, adapted Rand error and the CREMI score,
computed from the stored predictions. mSA rewards matching whole objects, which is the wrong question
for neurite segmentation where merges and splits are what matter. Those columns are empty for the
LM datasets, where mSA is the right metric.

Usage examples:
    python aggregate_interactive_results.py --method micro_sam2 -m hvit_t hvit_s hvit_b
    python aggregate_interactive_results.py --method micro_sam2 -m hvit_t --mask_threshold 2
    python aggregate_interactive_results.py --method micro_sam2 -m hvit_t --min_size_3d 50
"""

import os
import argparse

import numpy as np
import pandas as pd
import imageio.v3 as imageio

from elf.evaluation import cremi_score

from baselines_common import _load_data, interactive_result_name, interactive_run_tag
from submit_all_evaluations import DATASETS_2D, DATASETS_3D_LM, DATASETS_3D_EM

EXPERIMENT_FOLDER = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/v2_joint_evaluation"
DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"
DATASETS_3D = tuple(sorted(set(DATASETS_3D_LM + DATASETS_3D_EM)))
COLUMNS = ["dataset", "dimensionality", "modality", "model", "iteration", "n_prompts",
           "mSA", "SA50", "SA75", "precision", "recall", "f1",
           "vi_split", "vi_merge", "arand", "cremi_score"]


def prediction_path(experiment_folder, dataset, method, model_type, prompt, sample_id, iteration, min_size_3d):
    tag = interactive_run_tag(ndim=3, min_size=min_size_3d)
    return os.path.join(
        experiment_folder, "predictions", f"{method}_{model_type}", f"{dataset}{tag}",
        f"sample_{sample_id:05d}", "interactive_segmentation_3d", f"start_with_{prompt}",
        "without_masks", f"iteration{iteration}", f"{sample_id:05d}.tif",
    )


def em_scores(experiment_folder, dataset, method, model_type, prompt, n_iterations, min_size_3d):
    """Mean (vi_split, vi_merge, arand, cremi) per iteration, or None when predictions are missing."""
    per_iteration = [[] for _ in range(n_iterations)]
    # Mirrors run_sam2_evaluation: sample ids enumerate every loaded volume, empty ones are skipped.
    for sample_id, (_, labels, valid_roi) in enumerate(_load_data(dataset, DATA_ROOT, 3, min_size_3d)):
        if labels.max() == 0:
            continue
        for iteration in range(n_iterations):
            path = prediction_path(
                experiment_folder, dataset, method, model_type, prompt, sample_id, iteration, min_size_3d,
            )
            if not os.path.exists(path):
                return None
            seg = imageio.imread(path)
            if valid_roi is not None:  # Partially annotated, see run_sam2_evaluation.
                seg[~valid_roi] = 0
            per_iteration[iteration].append(
                cremi_score(seg.astype("uint32"), labels.astype("uint32"), ignore_gt=[0])
            )
    if not per_iteration[0]:
        return None
    return [np.mean(np.array(vals), axis=0) for vals in per_iteration]


def collect(experiment_folder, method, model_types, prompt, n_iterations,
            use_masks, mask_threshold, min_size_3d, with_em):
    results_dir = os.path.join(experiment_folder, "results")
    rows = []
    for dataset in DATASETS_2D + DATASETS_3D:
        is_3d = dataset in DATASETS_3D
        is_em = dataset in DATASETS_3D_EM
        for model_type in model_types:
            names = [
                interactive_result_name(
                    dataset, method, model_type, prompt, it, ndim=3 if is_3d else 2,
                    use_masks=use_masks, mask_threshold=mask_threshold,
                    min_size=min_size_3d if is_3d else 0,
                )
                for it in range(n_iterations)
            ]
            paths = [os.path.join(results_dir, name) for name in names]
            if not any(os.path.exists(p) for p in paths):
                continue

            scores = None
            if is_em and with_em:
                scores = em_scores(
                    experiment_folder, dataset, method, model_type, prompt, n_iterations, min_size_3d,
                )

            for iteration, path in enumerate(paths):
                if not os.path.exists(path):
                    continue
                entry = pd.read_csv(path).iloc[0]
                row = {
                    "dataset": dataset,
                    "dimensionality": "3d" if is_3d else "2d",
                    "modality": "em" if is_em else "lm",
                    "model": model_type,
                    "iteration": iteration,
                    # Prompts placed per object: the initial box or click, then one positive and one
                    # negative correction per round. Derived from the protocol, not measured, so it
                    # only holds while both paths use 'IterativePromptGenerator' unchanged.
                    "n_prompts": 1 + 2 * iteration,
                    "mSA": entry["mSA"],
                    "SA50": entry["SA50"],
                    "SA75": entry["SA75"],
                    "precision": entry["Precision"],
                    "recall": entry["Recall"],
                    "f1": entry["F1 Score"],
                }
                if scores is not None:
                    vis, vim, arand, cremi = scores[iteration]
                    row.update({"vi_split": vis, "vi_merge": vim, "arand": arand, "cremi_score": cremi})
                rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def main():
    parser = argparse.ArgumentParser(description="Aggregate interactive evaluation results.")
    parser.add_argument("--method", type=str, default="micro_sam2")
    parser.add_argument("-m", "--model_type", type=str, nargs="+", required=True)
    parser.add_argument("-p", "--prompt_choice", type=str, nargs="+", default=["box", "point"])
    parser.add_argument("-e", "--experiment_folder", type=str, default=EXPERIMENT_FOLDER)
    parser.add_argument("-iter", "--n_iterations", type=int, default=8)
    parser.add_argument("--use_masks", action=argparse.BooleanOptionalAction, default=True,
                        help="The 2d setting the runs were written with, see interactive_result_name.")
    parser.add_argument("--mask_threshold", type=float, default=0.0, help="The 2d threshold the runs used.")
    parser.add_argument("--min_size_3d", type=int, default=0, help="The min_size the 3d runs used, e.g. 50.")
    parser.add_argument("--em_metrics", action=argparse.BooleanOptionalAction, default=True,
                        help="Score the EM datasets with VI, adapted Rand and CREMI. Reads the stored predictions.")
    args = parser.parse_args()

    for prompt in args.prompt_choice:
        table = collect(
            args.experiment_folder, args.method, args.model_type, prompt, args.n_iterations,
            args.use_masks, args.mask_threshold, args.min_size_3d, args.em_metrics,
        )
        save_path = os.path.join(args.experiment_folder, f"results_interactive_{prompt}.csv")
        table.to_csv(save_path, index=False)
        n_em = int(table.cremi_score.notna().sum())
        print(f"{save_path}: {len(table)} rows, {table.dataset.nunique()} datasets, {n_em} rows with EM metrics")


if __name__ == "__main__":
    main()
