import os
import shutil
import subprocess
from glob import glob
from datetime import datetime


def write_batch_script(env_name, out_path, model_type, iterations, n_objects, save_root=None):
    """Writing scripts with different number of objects for finetuning
    """
    batch_script = f"""#!/bin/bash
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH -t 2-00:00:00
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A gzz0001
#SBATCH --constraint=80gb
#SBATCH --job-name=sam-n-objects

source ~/.bashrc
mamba activate {env_name} \n"""

    # python script
    python_script = "python ../../livecell_finetuning.py "

    _op = out_path[:-3] + "_partial-finetuning.sh"

    # name of the model configuration
    python_script += f"-m {model_type} "

    # iterations to run the experiment for (10k)
    python_script += f"--iterations {iterations} "

    # let's select the number of objects per batch
    python_script += f"--n_objects {n_objects} "

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


def submit_slurm():
    """Submit python script that needs gpus with given inputs on a slurm node.
    """
    tmp_folder = "./gpu_jobs"

    # parameters to run the inference scripts
    environment_name = "sam"
    model_type = "vit_b"
    iterations = int(1e4)
    num_combinations = range(1, 46)
    # NOTE: overwrite the path below to save the checkpoints to your desired path
    _save_root = "/scratch/usr/nimanwai/experiments/micro-sam/n_objects_per_batch/"

    for current_setup in num_combinations:
        write_batch_script(
            env_name=environment_name,
            out_path=get_batch_script_names(tmp_folder),
            model_type=model_type,
            iterations=iterations,
            n_objects=current_setup,
            save_root=os.path.join(_save_root, f"{current_setup}"),
        )

    for my_script in sorted(glob(tmp_folder + "/*")):
        cmd = ["sbatch", my_script]
        subprocess.run(cmd)


if __name__ == "__main__":
    try:
        shutil.rmtree("./gpu_jobs")
    except FileNotFoundError:
        pass

    submit_slurm()
