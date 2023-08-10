import argparse
import os
from glob import glob

import pickle
from subprocess import run

import micro_sam.evaluation as evaluation
from tqdm import tqdm

DATA_ROOT = "/scratch/projects/nim00007/sam/ood/EM"
PROMPT_ROOT = "/scratch/projects/nim00007/sam/experiments/prompts"


def get_paths(dataset):
    pattern = os.path.join(DATA_ROOT, dataset, "label_*.tif")
    paths = sorted(glob(pattern))
    assert len(paths) > 0, pattern
    return paths


def precompute_setting(prompt_settings, dataset):
    gt_paths = get_paths(dataset)
    prompt_folder = os.path.join(PROMPT_ROOT, dataset)
    evaluation.precompute_all_prompts(gt_paths, prompt_folder, prompt_settings)


def submit_array_job(prompt_settings, dataset):
    n_settings = len(prompt_settings)
    cmd = ["sbatch", "-a", f"0-{n_settings-1}", "precompute_prompts.sbatch", dataset]
    run(cmd)


def check_settings(dataset, settings, expected_len):
    prompt_folder = os.path.join(PROMPT_ROOT, dataset)

    def check_prompt_file(prompt_file):
        assert os.path.exists(prompt_file), prompt_file
        with open(prompt_file, "rb") as f:
            prompts = pickle.load(f)
        assert len(prompts) == expected_len, f"{len(prompts)}, {expected_len}"

    for setting in tqdm(settings, desc="Check prompt files"):
        pos, neg = setting["n_positives"], setting["n_negatives"]
        prompt_file = os.path.join(prompt_folder, f"points-p{pos}-n{neg}.pkl")
        if pos == 0 and neg == 0:
            prompt_file = os.path.join(prompt_folder, "boxes.pkl")
        check_prompt_file(prompt_file)

    print("All files checked!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("-c", "--check", action="store_true")
    args = parser.parse_args()

    # this will fail if the dataset is invalid
    gt_paths = get_paths(args.dataset)

    settings = evaluation.default_experiment_settings()
    # we may use this as the point setting instead of p2-n4,
    # so we also precompute it
    settings.append(
        {"use_points": True, "use_boxes": False, "n_positives": 4, "n_negatives": 8},  # p4-n8
    )

    if args.check:
        check_settings(args.dataset, settings, len(gt_paths))
        return

    job_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)

    if job_id is None:  # this is the main script that submits slurm jobs
        submit_array_job(settings, args.dataset)
    else:  # we're in a slurm job and precompute a setting
        job_id = int(job_id)
        this_settings = [settings[job_id]]
        precompute_setting(this_settings, args.dataset)


if __name__ == "__main__":
    main()
