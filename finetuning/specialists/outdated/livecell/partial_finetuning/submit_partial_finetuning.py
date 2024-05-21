import os
import shutil
import itertools
import subprocess
from glob import glob
from datetime import datetime


def write_batch_script(env_name, out_path, model_type, iterations, freeze=None, save_root=None):
    """Writing scripts with different partial finetuning for micro-sam
    """
    batch_script = f"""#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -t 2-00:00:00
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A gzz0001
#SBATCH --constraint=80gb
#SBATCH --job-name=sam-partial-finetuning

source ~/.bashrc
mamba activate {env_name} \n"""

    # python script
    python_script = "python ../../livecell_finetuning.py "

    _op = out_path[:-3] + "_partial-finetuning.sh"

    # name of the model configuration
    python_script += f"-m {model_type} "

    # iterations to run the experiment for (10k)
    python_script += f"--iterations {iterations} "

    # save root for the checkpoints and logs
    python_script += f"-s {save_root} "

    # freeze different parts of SAM
    python_script += "--freeze "

    if isinstance(freeze, list):
        for _fp in freeze:
            python_script += f"{_fp} "
    else:
        python_script += f"{freeze} "

    # let's add the python script to the bash script
    batch_script += python_script

    with open(_op, "w") as f:
        f.write(batch_script)


def get_batch_script_names(tmp_folder):
    tmp_folder = os.path.expanduser(tmp_folder)
    os.makedirs(tmp_folder, exist_ok=True)

    script_name = "livecell-finetuning"

    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = script_name + dt
    batch_script = os.path.join(tmp_folder, f"{tmp_name}.sh")

    return batch_script


def submit_slurm():
    """Submit python script that needs gpus with given inputs on a slurm node.
    """
    tmp_folder = "./gpu_jobs"

    # parameters to run the inference scripts
    environment_name = "sam"
    model_type = "vit_b"
    iterations = int(1e4)
    backbone_combinations = ["image_encoder", "prompt_encoder", "mask_decoder"]
    # NOTE: overwrite the path below to save the checkpoints to your desired path
    _save_root = "/scratch/usr/nimanwai/experiments/micro-sam/partial-finetuning/"

    all_combinations = []
    for i in range(len(backbone_combinations)):
        _one_set = itertools.combinations(backbone_combinations, r=i)
        for _per_combination in _one_set:
            if len(_per_combination) == 0:
                all_combinations.append(None)
            else:
                all_combinations.append(list(_per_combination))

    for current_setup in all_combinations:
        if isinstance(current_setup, list):
            save_root = os.path.join(_save_root, "freeze-")
            for _name in current_setup:
                save_root += f"{_name}-"
            save_root = save_root[:-1]
        else:
            save_root = os.path.join(_save_root, f"freeze-{current_setup}")

        write_batch_script(
            env_name=environment_name,
            out_path=get_batch_script_names(tmp_folder),
            model_type=model_type,
            iterations=iterations,
            freeze=current_setup,
            save_root=save_root,
        )

    for i, my_script in enumerate(sorted(glob(tmp_folder + "/*"))):
        cmd = ["sbatch", my_script]
        subprocess.run(cmd)


if __name__ == "__main__":
    try:
        shutil.rmtree("./gpu_jobs")
    except FileNotFoundError:
        pass

    submit_slurm()
