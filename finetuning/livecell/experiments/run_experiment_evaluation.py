import os
import re
import itertools
import subprocess


ROOT = "/scratch/usr/nimanwai"
EXPERIMENT_ROOT = "/scratch/projects/nim00007/sam/experiments/new_models/test/"

CMD = "python ../../evaluation/submit_all_evaluation.py "


def run_eval_process(cmd):
    print(f"Running the command: {cmd} \n")

    _cmd = re.split(r"\s", cmd)

    proc = subprocess.Popen(_cmd)
    try:
        outs, errs = proc.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        proc.terminate()
        outs, errs = proc.communicate()


def _submit_scripts(model, checkpoint, experiment_folder, specific_script):
    cmd = CMD + f"-m {model} " + "-d livecell "
    cmd += f"--checkpoint_path {checkpoint} "
    cmd += f"--experiment_path {experiment_folder}"
    if specific_script is not None:
        cmd += f" -s {specific_script}"

    run_eval_process(cmd)


def for_n_objects(model, max_objects=45, specific_script=None):
    ckpt_root = os.path.join(ROOT, "experiments", "micro-sam", "n_objects_per_batch")
    exp_root = os.path.join(EXPERIMENT_ROOT, "n_objects_per_batch")
    for i in range(1, max_objects+1):
        checkpoint = os.path.join(ckpt_root, f"{i}", "freeze-None", "checkpoints", model, "livecell_sam", "best.pt")
        experiment_folder = os.path.join(exp_root, f"{i}")

        _submit_scripts(model, checkpoint, experiment_folder, specific_script)


def for_freezing_backbones(model, specific_script=None):
    ckpt_root = os.path.join(ROOT, "experiments", "micro-sam", "freezing-livecell")
    exp_root = os.path.join(EXPERIMENT_ROOT, "freezing-livecell")

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

        checkpoint = os.path.join(checkpoint, "checkpoints", model, "livecell_sam", "best.pt")

        _submit_scripts(model, checkpoint, experiment_folder, specific_script)


def main():
    for_freezing_backbones("vit_l")
    for_n_objects("vit_b")


if __name__ == "__main__":
    main()
