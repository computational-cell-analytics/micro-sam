import json
import os
import warnings

from glob import glob
from pathlib import Path

import pandas as pd
from micro_sam.util import get_sam_model
from micro_sam.evaluation import (
    automatic_mask_generation, inference, evaluation,
    default_experiment_settings, get_experiment_setting_name
)
from micro_sam.evaluation.livecell import _get_livecell_paths

DATA_ROOT = "/scratch/projects/nim00007/sam/datasets"
LIVECELL_ROOT = "/scratch/projects/nim00007/data/LiveCELL"
PROMPT_ROOT = "/scratch-grete/projects/nim00007/sam/experiments/prompts"

LM_DATASETS = (
    "covid-if",
    "deepbacs",
    "hpa",
    "livecell",
    "lizard",
    "mouse-embryo",
    "plantseg-ovules",
    "plantseg-root",
    "tissuenet",
)

EM_DATASETS = (
    "cremi",
    "lucchi",
    "mitoem",
    "nuc_mm/mouse",
    "nuc_mm/zebrafish",
    "platy-cell",
    "platy-cuticle",
    "platy-nuclei",
    "snemi",
    "sponge-em",
    "vnc",
)
ALL_DATASETS = EM_DATASETS + LM_DATASETS


###
# Dataset functionality
###


def get_data_paths(dataset, split, max_num_images=None):
    if dataset == "livecell":
        n_val_per_cell_type = None if max_num_images is None else int(max_num_images / 8)
        return _get_livecell_paths(LIVECELL_ROOT, split=split, n_val_per_cell_type=n_val_per_cell_type)

    image_pattern = os.path.join(DATA_ROOT, dataset, split, "image*.tif")
    image_paths = sorted(glob(image_pattern))
    gt_paths = sorted(glob(os.path.join(DATA_ROOT, dataset, split, "label*.tif")))
    assert len(image_paths) == len(gt_paths)
    assert len(image_paths) > 0, image_pattern
    if max_num_images is not None:
        image_paths, gt_paths = image_paths[:max_num_images], gt_paths[:max_num_images]
    return image_paths, gt_paths


###
# Evaluation functionality
###


def get_generalist_predictor(checkpoint, model_type, return_state=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return get_sam_model(
            model_type=model_type, checkpoint_path=checkpoint, return_state=return_state,
        )


def evaluate_checkpoint_for_dataset(
    checkpoint, model_type, dataset, experiment_folder,
    run_default_evaluation, do_amg,
    predictor=None, max_num_val_images=None,
):
    """Evaluate a generalist checkpoint for a given dataset.
    """
    assert run_default_evaluation or do_amg

    prompt_dir = os.path.join(PROMPT_ROOT, dataset)

    if predictor is None:
        predictor = get_generalist_predictor(checkpoint, model_type)
    test_image_paths, test_gt_paths = get_data_paths(dataset, "test")

    embedding_dir = os.path.join(experiment_folder, "test", "embeddings")
    os.makedirs(embedding_dir, exist_ok=True)
    result_dir = os.path.join(experiment_folder, "results")

    results = []
    if run_default_evaluation:
        prompt_settings = default_experiment_settings()
        for setting in prompt_settings:

            setting_name = get_experiment_setting_name(setting)
            prediction_dir = os.path.join(experiment_folder, "test", setting_name)
            os.makedirs(prediction_dir, exist_ok=True)

            inference.run_inference_with_prompts(
                predictor, test_image_paths, test_gt_paths,
                embedding_dir, prediction_dir,
                use_points=setting["use_points"], use_boxes=setting["use_boxes"],
                n_positives=setting["n_positives"], n_negatives=setting["n_negatives"],
                prompt_save_dir=prompt_dir,
            )

            if dataset == "livecell":
                pred_paths = [
                    os.path.join(prediction_dir, os.path.basename(gt_path)) for gt_path in test_gt_paths
                ]
                assert all(os.path.exists(pred_path) for pred_path in pred_paths)
            else:
                pred_paths = sorted(glob(os.path.join(prediction_dir, "*.tif")))
            result_path = os.path.join(result_dir, f"{setting_name}.csv")
            os.makedirs(Path(result_path).parent, exist_ok=True)

            result = evaluation.run_evaluation(test_gt_paths, pred_paths, result_path)
            result.insert(0, "setting", [setting_name])
            results.append(result)

    if do_amg:
        val_embedding_dir = os.path.join(experiment_folder, "val", "embeddings")
        val_result_dir = os.path.join(experiment_folder, "val", "results")
        os.makedirs(val_embedding_dir, exist_ok=True)

        val_image_paths, val_gt_paths = get_data_paths(dataset, "val", max_num_images=max_num_val_images)
        automatic_mask_generation.run_amg_grid_search(
            predictor, val_image_paths, val_gt_paths, val_embedding_dir,
            val_result_dir, verbose_gs=True,
        )

        best_iou_thresh, best_stability_thresh, _ = automatic_mask_generation.evaluate_amg_grid_search(val_result_dir)
        best_settings = {"pred_iou_thresh": best_iou_thresh, "stability_score_thresh": best_stability_thresh}
        gs_result_path = os.path.join(experiment_folder, "best_gs_params.json")
        with open(gs_result_path, "w") as f:
            json.dump(best_settings, f)

        prediction_dir = os.path.join(experiment_folder, "test", "amg")
        os.makedirs(prediction_dir, exist_ok=True)
        automatic_mask_generation.run_amg_inference(
            predictor, test_image_paths, embedding_dir, prediction_dir,
            amg_generate_kwargs=best_settings,
        )

        if dataset == "livecell":
            pred_paths = [
                os.path.join(prediction_dir, os.path.basename(gt_path)) for gt_path in test_gt_paths
            ]
            assert all(os.path.exists(pred_path) for pred_path in pred_paths)
        else:
            pred_paths = sorted(glob(os.path.join(prediction_dir, "*.tif")))

        result_path = os.path.join(result_dir, "amg.csv")
        os.makedirs(Path(result_path).parent, exist_ok=True)

        result = evaluation.run_evaluation(test_gt_paths, pred_paths, result_path)
        result.insert(0, "setting", ["amg"])
        results.append(result)

    results = pd.concat(results)
    results.insert(0, "dataset", [dataset] * results.shape[0])
    return results


def evaluate_checkpoint_for_datasets(
    checkpoint, model_type, experiment_root, datasets,
    run_default_evaluation, do_amg,
    predictor=None, max_num_val_images=None,
):
    if predictor is None:
        predictor = get_generalist_predictor(checkpoint, model_type)

    results = []
    for dataset in datasets:
        experiment_folder = os.path.join(experiment_root, dataset)
        os.makedirs(experiment_folder, exist_ok=True)
        result = evaluate_checkpoint_for_dataset(
            None, None, dataset, experiment_folder,
            run_default_evaluation=run_default_evaluation,
            do_amg=do_amg, predictor=predictor, max_num_val_images=max_num_val_images,
        )
        results.append(result)

    return pd.concat(results)
