import argparse
import os

from glob import glob
from subprocess import run

import pandas as pd
from util import evaluate_checkpoint_for_datasets, get_generalist_predictor

CHECKPOINT_ROOT = "/scratch/projects/nim00007/sam/models/LM/generalist/v2"
EXPERIMENT_ROOT = "/scratch/projects/nim00007/sam/experiments/training-evolution"
# We evaluate these three datasets for the training evolution.
# These are chosen based on observations from preliminary experiments.
# - covid-if: out-of-domain dataset that shows the expected improvement (over vanilla).
# - deepbacs: in domain dataset where we see the biggest gap to the specialist.
# - lizard: out-of-domain that is furthest from the training data.
EVAL_DATASETS = ("covid-if", "deepbacs", "lizard")


def evaluate_checkpoint_slurm(model_type, job_id, checkpoints):
    checkpoint = checkpoints[job_id]

    predictor, state = get_generalist_predictor(
        checkpoint, model_type, return_state=True
    )
    epoch = state["epoch"] + 1

    print("Run evaluation for", model_type, "epoch", epoch)
    experiment_root = os.path.join(EXPERIMENT_ROOT, f"{model_type}-epoch-{epoch}")
    result = evaluate_checkpoint_for_datasets(
        None, None, experiment_root, EVAL_DATASETS,
        run_default_evaluation=True, do_amg=False,
        predictor=predictor,
    )

    result.insert(0, "epoch", [epoch] * result.shape[0])
    return result


def evaluate_training_evolution(model_type, checkpoints):
    results = []
    for i in range(len(checkpoints)):
        result = evaluate_checkpoint_slurm(model_type, i, checkpoints)
        results.append(result)
    results = pd.concat(results)
    save_path = os.path.join(EXPERIMENT_ROOT, f"{model_type}.csv")
    results.to_csv(save_path, index=False)


def submit_array_job(model_type, checkpoints):
    n_checkpoints = len(checkpoints)
    cmd = ["sbatch", "-a", f"0-{n_checkpoints-1}", "evaluate_training_evolution.sbatch", model_type]
    run(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type")
    parser.add_argument("-e", "--evaluate", action="store_true")
    args = parser.parse_args()

    checkpoints = sorted(glob(os.path.join(CHECKPOINT_ROOT, args.model_type, "epoch-*.pt")))
    assert len(checkpoints) > 0

    if args.evaluate:
        evaluate_training_evolution(args.model_type, checkpoints)
        return

    job_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    if job_id is None:  # this is the main script that submits slurm jobs
        submit_array_job(args.model_type, checkpoints)
    else:  # we're in a slurm job
        job_id = int(job_id)
        evaluate_checkpoint_slurm(args.model_type, job_id, checkpoints)


if __name__ == "__main__":
    main()
