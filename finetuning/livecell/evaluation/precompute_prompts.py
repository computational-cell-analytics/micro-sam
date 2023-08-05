import argparse
import os
from subprocess import run

import micro_sam.evaluation as evaluation
from util import get_paths, EXPERIMENT_ROOT


def precompute_setting(prompt_settings):
    prompt_save_dir = os.path.join(EXPERIMENT_ROOT, "prompts")
    _, gt_paths = get_paths()
    evaluation.precompute_all_prompts(gt_paths, prompt_save_dir, prompt_settings)


def submit_array_job(prompt_settings, full_settings):
    n_settings = len(prompt_settings)
    cmd = ["sbatch", "-a", f"0-{n_settings-1}", "precompute_prompts.sbatch"]
    if full_settings:
        cmd.append("-f")
    run(cmd)


# TODO
def check_settings(settings):
    pass


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
