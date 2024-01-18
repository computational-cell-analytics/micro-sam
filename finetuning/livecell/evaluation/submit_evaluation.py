import os
import shutil
import subprocess
from glob import glob
from datetime import datetime


def write_batch_script(env_name, out_path, inference_setup, checkpoint, model_type, experiment_folder, delay=True):
    """Writing scripts with different fold-trainings for micro-sam evaluation
    """
    batch_script = f"""#!/bin/bash
#SBATCH -c 8
#SBATCH --mem 128G
#SBATCH -t 6:00:00
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A nim00007
#SBATCH --job-name={inference_setup}

source ~/.bashrc
mamba activate {env_name} \n"""

    if delay:
        batch_script += "sleep 10m \n"

    # python script
    python_script = f"python {inference_setup}.py "

    _op = out_path[:-3] + f"_{inference_setup}.sh"

    # add the finetuned checkpoint
    python_script += f"-c {checkpoint} "

    # name of the model configuration
    python_script += f"-m {model_type} "

    # experiment folder
    python_script += f"-e {experiment_folder} "

    # let's add the python script to the bash script
    batch_script += python_script

    with open(_op, "w") as f:
        f.write(batch_script)

    # we run the first prompt for iterative once starting with point, and then starting with box (below)
    if inference_setup == "iterative_prompting":
        batch_script += "--box "

        new_path = out_path[:-3] + f"_{inference_setup}_box.sh"
        with open(new_path, "w") as f:
            f.write(batch_script)


def get_batch_script_names(tmp_folder):
    tmp_folder = os.path.expanduser(tmp_folder)
    os.makedirs(tmp_folder, exist_ok=True)

    script_name = "livecell-inference"

    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = script_name + dt
    batch_script = os.path.join(tmp_folder, f"{tmp_name}.sh")

    return batch_script


def submit_slurm():
    """Submit python script that needs gpus with given inputs on a slurm node.
    """
    tmp_folder = "./gpu_jobs"

    # parameters to run the inference scripts
    env_name = "sam"
    model_type = "vit_b"
    checkpoint = f"/scratch/usr/nimanwai/micro-sam/checkpoints/{model_type}/lm_generalist_sam/best.pt"
    experiment_folder = f"/scratch/projects/nim00007/sam/experiments/new_models/generalists/lm/livecell/{model_type}/"

    all_setups = ["precompute_embeddings", "evaluate_amg", "evaluate_instance_segmentation", "iterative_prompting"]
    for current_setup in all_setups:
        write_batch_script(
            env_name=env_name,
            out_path=get_batch_script_names(tmp_folder),
            inference_setup=current_setup,
            checkpoint=checkpoint,
            model_type=model_type,
            experiment_folder=experiment_folder,
            delay=False if current_setup == "precompute_embeddings" else True
            )

    for my_script in glob(tmp_folder + "/*"):
        cmd = ["sbatch", my_script]
        subprocess.run(cmd)


if __name__ == "__main__":
    try:
        shutil.rmtree("./gpu_jobs")
    except FileNotFoundError:
        pass

    submit_slurm()
