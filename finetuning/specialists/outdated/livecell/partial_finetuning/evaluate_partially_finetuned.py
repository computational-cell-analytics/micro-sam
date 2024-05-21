import argparse
import os
import warnings
from subprocess import run

import pandas as pd

from micro_sam.util import get_sam_model
from micro_sam.evaluation import (
    inference,
    evaluation,
    default_experiment_settings,
    get_experiment_setting_name
)
from micro_sam.evaluation.livecell import _get_livecell_paths

DATA_ROOT = "/scratch-grete/projects/nim00007/data/LiveCELL"
EXPERIMENT_ROOT = "/scratch-grete/projects/nim00007/sam/experiments/livecell/partial-finetuning"
PROMPT_DIR = "/scratch-grete/projects/nim00007/sam/experiments/prompts/livecell"
MODELS = {
    "freeze-image_encoder": "/scratch-grete/projects/nim00007/sam/partial_finetuning/checkpoints/livecell_sam-freeze-image_encoder",
    "freeze-image_encoder-mask_decoder": "/scratch-grete/projects/nim00007/sam/partial_finetuning/checkpoints/livecell_sam-freeze-image_encoder-mask_decoder",
    "freeze-image_encoder-prompt_encoder": "/scratch-grete/projects/nim00007/sam/partial_finetuning/checkpoints/livecell_sam-freeze-image_encoder-prompt_encoder",
    "freeze-mask_decoder": "/scratch-grete/projects/nim00007/sam/partial_finetuning/checkpoints/livecell_sam-freeze-mask_decoder",
    "freeze-None": "/scratch-grete/projects/nim00007/sam/partial_finetuning/checkpoints/livecell_sam-freeze-None",
    "freeze-prompt_encoder": "/scratch-grete/projects/nim00007/sam/partial_finetuning/checkpoints/livecell_sam-freeze-prompt_encoder",
    "freeze-prompt_encoder-mask_decoder": "/scratch-grete/projects/nim00007/sam/partial_finetuning/checkpoints/livecell_sam-freeze-prompt_encoder-mask_decoder",
    "vanilla": "/home/nimcpape/.sam_models/sam_vit_b_01ec64.pth",
}


def evaluate_model(model_id):
    model_name = list(MODELS.keys())[model_id]
    print("Evaluating", model_name)

    try:
        checkpoint = os.path.join(MODELS[model_name], "best.pt")
        assert os.path.exists(checkpoint)
    except AssertionError:
        checkpoint = MODELS[model_name]

    print("Evalute", model_name, "from", checkpoint)
    model_type = "vit_b"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictor = get_sam_model(model_type=model_type, checkpoint_path=checkpoint)

    experiment_dir = os.path.join(EXPERIMENT_ROOT, model_name)

    embedding_dir = os.path.join(experiment_dir, "embeddings")
    os.makedirs(embedding_dir, exist_ok=True)

    result_dir = os.path.join(experiment_dir, "results")
    os.makedirs(result_dir, exist_ok=True)

    image_paths, gt_paths = _get_livecell_paths(DATA_ROOT)
    experiment_settings = default_experiment_settings()

    results = []
    for setting in experiment_settings:
        setting_name = get_experiment_setting_name(setting)
        prediction_dir = os.path.join(experiment_dir, setting_name)

        os.makedirs(prediction_dir, exist_ok=True)
        inference.run_inference_with_prompts(
            predictor, image_paths, gt_paths,
            embedding_dir, prediction_dir,
            prompt_save_dir=PROMPT_DIR, **setting
        )

        pred_paths = [os.path.join(prediction_dir, os.path.basename(gt_path)) for gt_path in gt_paths]
        assert len(pred_paths) == len(gt_paths)
        result_path = os.path.join(result_dir, f"{setting_name}.csv")

        if os.path.exists(result_path):
            result = pd.read_csv(result_path)
        else:
            result = evaluation.run_evaluation(gt_paths, pred_paths, result_path)
        result.insert(0, "setting", [setting_name])
        results.append(result)

    results = pd.concat(results)
    return results


def combine_results():
    results = []
    for model_id, model_name in enumerate(MODELS):
        res = evaluate_model(model_id)
        res.insert(0, "frozen", res.shape[0] * [model_name.lstrip("freeze-")])
        results.append(res)
    results = pd.concat(results)
    res_path = os.path.join(EXPERIMENT_ROOT, "partial_finetuning_results.csv")
    results.to_csv(res_path, index=False)


def submit_array_job():
    n_models = len(MODELS)
    cmd = ["sbatch", "-a", f"0-{n_models-1}", "evaluate_partially_finetuned.sbatch"]
    run(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--evaluate", action="store_true")
    args = parser.parse_args()

    if args.evaluate:
        combine_results()
        return

    job_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    if job_id is None:
        submit_array_job()
    else:
        evaluate_model(int(job_id))


if __name__ == "__main__":
    main()
