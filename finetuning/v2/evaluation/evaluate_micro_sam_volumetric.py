"""Evaluate micro-sam v1 interactive segmentation for 3D volumes.

This script uses the volumetric implementation from micro-sam instead of
slice-by-slice 2D iterative prompting.

Example:
    python evaluate_micro_sam_volumetric.py -d embedseg -e ./exp -m vit_b_lm -p box
    python evaluate_micro_sam_volumetric.py -d cremi -e ./exp -m vit_b_em_organelles -p point
"""

import argparse
import os

import pandas as pd
from tqdm import tqdm

from micro_sam.evaluation.multi_dimensional_segmentation import run_multi_dimensional_segmentation_grid_search

from baselines_common import MAX_EVALUATION_SAMPLES, _load_data
from common import DATA_ROOT, DATASETS_3D, DATASETS_3D_EM, get_data_paths


_MICROSAM_V1_LM_MODEL = "vit_b_lm"
_MICROSAM_V1_EM_MODEL = "vit_b_em_organelles"


def _default_grid_values():
    return {
        "iou_threshold": [0.8],
        "projection": ["mask"],
        "box_extension": [0.025],
    }


def run_micro_sam_volumetric_evaluation(
    dataset_name,
    data_root,
    experiment_folder,
    model_type=None,
    checkpoint=None,
    prompt_choice="box",
    full_grid_search=False,
    store_segmentation=True,
    min_size=0,
):
    if dataset_name not in DATASETS_3D:
        raise ValueError(f"Volumetric micro-sam v1 evaluation is 3D-only; got '{dataset_name}'.")

    if dataset_name in DATASETS_3D_EM:
        raise ValueError(f"Volumetric micro-sam v1 only supports LM datasets (vit_b_lm); got EM dataset '{dataset_name}'.")

    if model_type is None:
        model_type = _MICROSAM_V1_LM_MODEL

    interactive_seg_mode = "points" if prompt_choice == "point" else "box"
    grid_search_values = None if full_grid_search else _default_grid_values()

    n = min(len(get_data_paths(dataset_name, data_root)[0]), MAX_EVALUATION_SAMPLES)
    rows = []
    for sample_id, (raw, labels) in enumerate(tqdm(_load_data(dataset_name, data_root, 3), total=n, desc="micro-sam-3d")):
        sample_name = f"sample_{sample_id:05d}"
        result_dir = os.path.join(
            experiment_folder, "results", f"{dataset_name}_micro-sam_{model_type}_3d_{prompt_choice}", sample_name,
        )
        embedding_path = os.path.join(
            experiment_folder, "embeddings", f"{dataset_name}_micro-sam_{model_type}_3d", sample_name,
        )

        best_params_path = run_multi_dimensional_segmentation_grid_search(
            volume=raw,
            ground_truth=labels,
            model_type=model_type,
            checkpoint_path=checkpoint,
            embedding_path=embedding_path,
            result_dir=result_dir,
            interactive_seg_mode=interactive_seg_mode,
            grid_search_values=grid_search_values,
            min_size=min_size,
            store_segmentation=store_segmentation,
            verbose=False,
        )

        best_params = pd.read_csv(best_params_path)
        best_params.insert(0, "sample_id", sample_id)
        rows.append(best_params)

    if rows:
        summary = pd.concat(rows, ignore_index=True)
        summary_path = os.path.join(
            experiment_folder,
            "results",
            f"{dataset_name}_micro-sam_{model_type}_3d_{prompt_choice}_summary.csv",
        )
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"Stored summary at '{summary_path}'.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate micro-sam v1 volumetric interactive segmentation.")
    parser.add_argument("-d", "--dataset_name", required=True, choices=sorted(DATASETS_3D))
    parser.add_argument("-i", "--input_path", type=str, default=DATA_ROOT)
    parser.add_argument("-e", "--experiment_folder", type=str, required=True)
    parser.add_argument("-m", "--model_type", type=str, default=None)
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("-p", "--prompt_choice", type=str, default="box", choices=("box", "point"))
    parser.add_argument("--full_grid_search", action="store_true")
    parser.add_argument("--no_store_segmentation", action="store_true")
    parser.add_argument("--min_size", type=int, default=0)
    args = parser.parse_args()

    run_micro_sam_volumetric_evaluation(
        dataset_name=args.dataset_name,
        data_root=args.input_path,
        experiment_folder=args.experiment_folder,
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        prompt_choice=args.prompt_choice,
        full_grid_search=args.full_grid_search,
        store_segmentation=not args.no_store_segmentation,
        min_size=args.min_size,
    )


if __name__ == "__main__":
    main()
