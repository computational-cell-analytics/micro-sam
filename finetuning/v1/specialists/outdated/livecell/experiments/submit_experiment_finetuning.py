import os
import shutil
import itertools
import subprocess
from glob import glob
from datetime import datetime


def write_batch_script(
    experiment_name, env_name, out_path, model_type, iterations, n_objects, save_root, freeze=None
):
    """Writing scripts with:
        - different number of objects for finetuning
        - different freezing experiments
    """
    batch_script = f"""#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -t 14-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A gzz0001
#SBATCH --qos=14d
#SBATCH --constraint=80gb
#SBATCH --job-name={experiment_name}

source ~/.bashrc
mamba activate {env_name} \n"""

    # python script
    python_script = "python ../../livecell_finetuning.py "

    _op = out_path[:-3] + "_partial-finetuning.sh"

    # name of the model configuration
    python_script += f"-m {model_type} "

    # iterations to run the experiment for (100k)
    python_script += f"--iterations {iterations} "

    # let's select the number of objects per batch
    python_script += f"--n_objects {n_objects} "

    # freeze different parts of SAM
    if freeze is not None:
        here_freeze = ""
        add_save_root = "freeze-"
        for fp in freeze:
            here_freeze += f"{fp} "
            add_save_root += f"{fp}-"

        python_script += f"--freeze {here_freeze} "
        save_root = os.path.join(save_root, add_save_root[:-1])
    else:
        save_root = os.path.join(save_root, "freeze-None")

    # save root for the checkpoints and logs
    python_script += f"-s {save_root} "

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


def submit_n_objects_slurm():
    """Submit python script that needs gpus with given inputs on a slurm node.
    """
    tmp_folder = "./gpu_jobs"

    # parameters to run the inference scripts
    iterations = int(1e5)
    num_combinations = range(1, 46, 1)
    # NOTE: overwrite the path below to save the checkpoints to your desired path
    _save_root = "/scratch/usr/nimanwai/experiments/micro-sam/n_objects_per_batch/"

    for current_setup in num_combinations:
        write_batch_script(
            experiment_name="sam-n_objects",
            env_name="sam",
            out_path=get_batch_script_names(tmp_folder),
            model_type="vit_b",
            iterations=iterations,
            n_objects=current_setup,
            save_root=os.path.join(_save_root, f"{current_setup}"),
        )

    for my_script in sorted(glob(tmp_folder + "/*")):
        cmd = ["sbatch", my_script]
        subprocess.run(cmd)


def submit_freezing_slurm():
    """Submit python script that needs gpus with given inputs on a slurm node.
    """
    tmp_folder = "./gpu_jobs"

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

    # parameters to run the inference scripts
    iterations = int(1e5)
    # NOTE: overwrite the path below to save the checkpoints to your desired path
    _save_root = "/scratch/usr/nimanwai/experiments/micro-sam/freezing-livecell/"

    for i, current_setup in enumerate(all_combinations):
        write_batch_script(
            experiment_name="sam-freezing",
            env_name="sam",
            out_path=get_batch_script_names(tmp_folder),
            model_type="vit_l",
            iterations=iterations,
            n_objects=25,
            freeze=current_setup,
            save_root=_save_root,
        )

    for my_script in sorted(glob(tmp_folder + "/*")):
        cmd = ["sbatch", my_script]
        subprocess.run(cmd)


if __name__ == "__main__":
    try:
        shutil.rmtree("./gpu_jobs")
    except FileNotFoundError:
        pass

    print("Running experiments for 'n_objects'")
    submit_n_objects_slurm()

    shutil.rmtree("./gpu_jobs")

    print("Running experiments for 'finetuning parts'")
    submit_freezing_slurm()
