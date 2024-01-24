import os
import re
import subprocess


ROOT = "/scratch/usr/nimanwai"
EXPERIMENT_ROOT = "/scratch/projects/nim00007/sam/experiments/new_models/test/"

CMD = "python submit_experiment_evaluation.py "


def run_eval_process(cmd):
    proc = subprocess.Popen(cmd)
    try:
        outs, errs = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        proc.terminate()
        outs, errs = proc.communicate()


def for_vit_t():
    checkpoint = os.path.join(
        ROOT, "experiments", "test", "micro-sam", "vit_t", "checkpoints", "vit_t", "livecell_sam", "best.pt"
    )
    experiment_folder = os.path.join(EXPERIMENT_ROOT, "vit_t")

    cmd = CMD + "-m vit_t " + f"-c {checkpoint} " + f"-e {experiment_folder}"
    print(f"Running the command: {cmd} \n")

    _cmd = re.split(r"\s", cmd)

    run_eval_process(_cmd)


def for_n_objects():
    raise NotImplementedError


def for_freezing_backbones():
    raise NotImplementedError


def main():
    for_vit_t()


if __name__ == "__main__":
    main()
