import argparse
import os

from subprocess import run

import micro_sam.evaluation as evaluation
from util import get_data_paths, ALL_DATASETS

PROMPT_ROOT = "/scratch/projects/nim00007/sam/experiments/prompts"


def precompute_prompts(dataset):
    # everything for livecell has been computed already
    if dataset == "livecell":
        return

    prompt_folder = os.path.join(PROMPT_ROOT, dataset)
    _, gt_paths = get_data_paths(dataset, "test")

    settings = evaluation.default_experiment_settings()
    evaluation.precompute_all_prompts(gt_paths, prompt_folder, settings)


def precompute_prompts_slurm(job_id):
    dataset = ALL_DATASETS[job_id]
    precompute_prompts(dataset)


def submit_array_job():
    n_datasets = len(ALL_DATASETS)
    cmd = ["sbatch", "-a", f"0-{n_datasets-1}", "precompute_prompts.sbatch"]
    run(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset")
    args = parser.parse_args()
    if args.dataset is not None:
        precompute_prompts(args.dataset)
        return

    # this will fail if the dataset is invalid
    job_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)

    if job_id is None:  # this is the main script that submits slurm jobs
        submit_array_job()
    else:  # we're in a slurm job and precompute a setting
        job_id = int(job_id)
        precompute_prompts_slurm(job_id)


if __name__ == "__main__":
    main()
