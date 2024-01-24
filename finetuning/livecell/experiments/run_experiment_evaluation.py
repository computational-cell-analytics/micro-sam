import os
import re
import itertools
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


def for_n_objects(max_objects=45):
    ckpt_root = os.path.join(ROOT, "experiments", "micro-sam", "n_objects_per_batch")
    exp_root = os.path.join(EXPERIMENT_ROOT, "n_objects_per_batch")
    for i in range(1, max_objects+1):
        checkpoint = os.path.join(ckpt_root, f"{i}", "checkpoints", "vit_b", "livecell_sam", "best.pt")
        experiment_folder = os.path.join(exp_root, f"{i}")

        cmd = CMD + "-m vit_b " + f"-c {checkpoint} " + f"-e {experiment_folder}"
        print(f"Running the command: {cmd} \n")

        _cmd = re.split(r"\s", cmd)

        run_eval_process(_cmd)


def for_freezing_backbones():
    ckpt_root = os.path.join(ROOT, "experiments", "micro-sam", "partial-finetuning")
    exp_root = os.path.join(EXPERIMENT_ROOT, "partial-finetuning")

    # let's get all combinations need for the freezing backbone experiments
    backbone_combinations = ["image_encoder", "prompt_encoder", "mask_decoder"]

    all_combinations = []
    for i in range(len(backbone_combinations)):
        _one_set = itertools.combinations(backbone_combinations, r=i)
        for _per_combination in _one_set:
            if len(_per_combination) == 0:
                all_combinations.append(None)
            else:
                all_combinations.append(list(_per_combination))

    for _setup in all_combinations:
        if isinstance(_setup, list):
            checkpoint = os.path.join(ckpt_root, "freeze-")
            experiment_folder = os.path.join(exp_root, "freeze-")
            for _name in _setup:
                checkpoint += f"{_name}-"
                experiment_folder += f"{_name}-"
            checkpoint = checkpoint[:-1]
            experiment_folder = experiment_folder[:-1]
        else:
            checkpoint = os.path.join(ckpt_root, f"freeze-{_setup}")
            experiment_folder = os.path.join(exp_root, f"freeze-{_setup}")

        checkpoint = os.path.join(checkpoint, "checkpoints", "vit_b", "livecell_sam", "best.pt")

        cmd = CMD + "-m vit_b " + f"-c {checkpoint} " + f"-e {experiment_folder}"
        print(f"Running the command: {cmd} \n")

        _cmd = re.split(r"\s", cmd)

        run_eval_process(_cmd)


def main():
    for_freezing_backbones()


if __name__ == "__main__":
    main()
