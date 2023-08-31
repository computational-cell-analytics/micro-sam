import argparse
import os
import pickle

from subprocess import run

import micro_sam.evaluation as evaluation
from util import get_data_paths, ALL_DATASETS, LM_DATASETS
from tqdm import tqdm

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


def _check_prompts(dataset, settings, expected_len):
    prompt_folder = os.path.join(PROMPT_ROOT, dataset)

    def check_prompt_file(prompt_file):
        assert os.path.exists(prompt_file), prompt_file
        with open(prompt_file, "rb") as f:
            prompts = pickle.load(f)
        assert len(prompts) == expected_len, f"{len(prompts)}, {expected_len}"

    for setting in settings:
        pos, neg = setting["n_positives"], setting["n_negatives"]
        prompt_file = os.path.join(prompt_folder, f"points-p{pos}-n{neg}.pkl")
        if pos == 0 and neg == 0:
            prompt_file = os.path.join(prompt_folder, "boxes.pkl")
        check_prompt_file(prompt_file)


def check_prompts_and_datasets():

    def check_dataset(dataset):
        try:
            images, _ = get_data_paths(dataset, "test")
        except AssertionError as e:
            print("Checking test split failed for datasset", dataset, "due to", e)

        if dataset not in LM_DATASETS:
            return len(images)

        try:
            get_data_paths(dataset, "val")
        except AssertionError as e:
            print("Checking val split failed for datasset", dataset, "due to", e)

        return len(images)

    settings = evaluation.default_experiment_settings()
    for ds in tqdm(ALL_DATASETS, desc="Checking datasets"):
        n_images = check_dataset(ds)
        _check_prompts(ds, settings, n_images)
    print("All checks done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset")
    parser.add_argument("--check", "-c", action="store_true")
    args = parser.parse_args()

    if args.check:
        check_prompts_and_datasets()
        return

    if args.dataset is not None:
        precompute_prompts(args.dataset)
        return

    job_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    if job_id is None:  # this is the main script that submits slurm jobs
        submit_array_job()
    else:  # we're in a slurm job and precompute a setting
        job_id = int(job_id)
        precompute_prompts_slurm(job_id)


if __name__ == "__main__":
    main()
