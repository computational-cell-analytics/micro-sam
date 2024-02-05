import os
import re
import subprocess
from glob import glob


CMD = "python submit_all_evaluation.py "
CHECKPOINT_ROOT = "/scratch/usr/nimanwai/experiments/micro-sam/unetr-decoder-updates/"
EXPERIMENT_ROOT = "/scratch/projects/nim00007/sam/experiments/new_models/test/unetr-decoder-updates"


def run_eval_process(cmd):
    proc = subprocess.Popen(cmd)
    try:
        outs, errs = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        proc.terminate()
        outs, errs = proc.communicate()


def run_specific_experiment(dataset_name, model_type, setup):
    all_checkpoint_dirs = sorted(glob(os.path.join(CHECKPOINT_ROOT, f"{setup}-*")))
    for checkpoint_dir in all_checkpoint_dirs:
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoints", model_type, "lm_generalist_sam", "best.pt")

        experiment_name = checkpoint_dir.split("/")[-1]
        experiment_folder = os.path.join(EXPERIMENT_ROOT, experiment_name, dataset_name, model_type)

        cmd = CMD + f"-d {dataset_name} " + f"-m {model_type} " + "-e generalist "
        cmd += f"--checkpoint_path {checkpoint_path} "
        cmd += f"--experiment_path {experiment_folder}"
        print(f"Running the command: {cmd} \n")
        _cmd = re.split(r"\s", cmd)
        run_eval_process(_cmd)


def run_one_setup(all_dataset_list, all_model_list, setup):
    for dataset_name in all_dataset_list:
        for model_type in all_model_list:
            run_specific_experiment(dataset_name=dataset_name, model_type=model_type, setup=setup)


def for_all_lm(setup):
    assert setup in ["conv-transpose", "bilinear"]

    # let's run for in-domain
    run_one_setup(
        all_dataset_list=["tissuenet", "deepbacs", "plantseg/root", "livecell", "neurips-cell-seg"],
        all_model_list=["vit_t", "vit_b", "vit_l", "vit_h"],
        setup=setup
    )


def main():
    os.chdir("../")
    for_all_lm("conv-transpose")


if __name__ == "__main__":
    main()
