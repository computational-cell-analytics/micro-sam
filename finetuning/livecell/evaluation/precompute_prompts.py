import argparse
import os
import pickle
from subprocess import run

import micro_sam.evaluation as evaluation
from tqdm import tqdm
from util import get_paths, PROMPT_FOLDER


def precompute_setting(prompt_settings):
    _, gt_paths = get_paths()
    evaluation.precompute_all_prompts(gt_paths, PROMPT_FOLDER, prompt_settings)


def submit_array_job(prompt_settings, full_settings):
    n_settings = len(prompt_settings)
    cmd = ["sbatch", "-a", f"0-{n_settings-1}", "precompute_prompts.sbatch"]
    if full_settings:
        cmd.append("-f")
    run(cmd)


def check_settings(settings):

    def check_prompt_file(prompt_file):
        assert os.path.exists(prompt_file), prompt_file
        with open(prompt_file, "rb") as f:
            prompts = pickle.load(f)
        assert len(prompts) == 1512, f"{len(prompts)}"

    for setting in tqdm(settings, desc="Check prompt files"):
        pos, neg = setting["n_positives"], setting["n_negatives"]
        prompt_file = os.path.join(PROMPT_FOLDER, f"points-p{pos}-n{neg}.pkl")
        if pos == 0 and neg == 0:
            prompt_file = os.path.join(PROMPT_FOLDER, "boxes.pkl")
        check_prompt_file(prompt_file)

    print("All files checked!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--full_settings", action="store_true")
    parser.add_argument("-c", "--check", action="store_true")
    args = parser.parse_args()

    if args.full_settings:
        settings = evaluation.full_experiment_settings(use_boxes=True)
    else:
        settings = evaluation.default_experiment_settings()

    if args.check:
        check_settings(settings)
        return

    job_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)

    if job_id is None:  # this is the main script that submits slurm jobs
        submit_array_job(settings, args.full_settings)
    else:  # we're in a slurm job and precompute a setting
        job_id = int(job_id)
        this_settings = [settings[job_id]]
        precompute_setting(this_settings)


if __name__ == "__main__":
    main()
