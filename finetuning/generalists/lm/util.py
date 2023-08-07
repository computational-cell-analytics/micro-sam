import os
from glob import glob
from pathlib import Path

from micro_sam.evaluation import (
    inference, evaluation,
    default_experiment_settings, get_experiment_setting_name
)

DATA_ROOT = "/scratch/projects/nim00007/sam/ood/LM"
PROMPT_ROOT = "/scratch-grete/projects/nim00007/sam/experiments/prompts"
DATASETS = (
    "covid-if",
    "deepbacs",
    "hpa",
    "lizard",
    "mouse-embryo",
    "plantseg-ovules",
    "plantseg-root",
    "tissuenet",
)


def get_data_paths(dataset, split):
    image_paths = sorted(glob(os.path.join(DATA_ROOT, dataset, split, "image_*.tif")))
    gt_paths = sorted(glob(os.path.join(DATA_ROOT, dataset, split, "labels_*.tif")))
    assert len(image_paths) == len(gt_paths)
    assert len(image_paths) > 0
    return image_paths, gt_paths


def evaluate_checkpoint_for_dataset(
    checkpoint, model_type, dataset, experiment_folder,
    run_default_evaluation, run_amg, predictor=None,
):
    """Evaluate a generalist checkpoint for a given dataset
    """

    prompt_dir = os.path.join(PROMPT_ROOT, dataset)

    if predictor is None:
        predictor = inference.get_predictor(checkpoint, model_type)
    test_image_paths, test_gt_paths = get_data_paths(dataset, "test")

    if run_default_evaluation:
        embedding_dir = os.path.join(experiment_folder, "test", "embeddings")
        os.makedirs(embedding_dir, exist_ok=True)

        result_dir = os.path.join(experiment_folder, "results")

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

            pred_paths = sorted(glob(os.path.join(prediction_dir, "*.tif")))
            result_path = os.path.join(result_dir, f"{setting_name}.csv")
            os.makedirs(Path(result_path).parent, exist_ok=True)

            # TODO return the result and combine all in one table
            evaluation.run_evaluation(test_gt_paths, pred_paths, result_path)

    if run_amg:
        raise NotImplementedError


def evaluate_checkpoint_for_datasets(
    checkpoint, model_type, experiment_root, datasets,
    run_default_evaluation, run_amg, predictor=None,
):
    if predictor is None:
        predictor = inference.get_predictor(checkpoint, model_type)
    for dataset in datasets:
        experiment_folder = os.path.join(experiment_root, dataset)
        os.makedirs(experiment_folder, exist_ok=True)
        evaluate_checkpoint_for_dataset(
            None, None, dataset, experiment_folder,
            run_default_evaluation=run_default_evaluation,
            run_amg=run_amg, predictor=predictor,
        )


def evaluate_checkpoint_for_datasets_slurm(
    checkpoint, model_type, experiment_root, datasets,
    run_default_evaluation, run_amg,
):
    raise NotImplementedError
