import os
import shutil
import subprocess
from glob import glob
from datetime import datetime


def write_batch_script(
    env_name, out_path, inference_setup, checkpoint, model_type, experiment_folder, dataset_name, delay=True
):
    """Writing scripts with different fold-trainings for micro-sam evaluation
    """
    batch_script = f"""#!/bin/bash
#SBATCH -c 8
#SBATCH --mem 128G
#SBATCH -t 12:00:00
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A gzz0001
#SBATCH --job-name={inference_setup}

source ~/.bashrc
mamba activate {env_name} \n"""

    if delay:
        batch_script += "sleep 5m \n"

    # python script
    python_script = f"python {inference_setup}.py "

    _op = out_path[:-3] + f"_{inference_setup}.sh"

    # add the finetuned checkpoint
    python_script += f"-c {checkpoint} "

    # name of the model configuration
    python_script += f"-m {model_type} "

    # experiment folder
    python_script += f"-e {experiment_folder} "

    # IMPORTANT: choice of the dataset
    python_script += f"-d {dataset_name} "

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

    script_name = "em-inference"

    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = script_name + dt
    batch_script = os.path.join(tmp_folder, f"{tmp_name}.sh")

    return batch_script


def submit_slurm():
    """Submit python script that needs gpus with given inputs on a slurm node.
    """
    tmp_folder = "./gpu_jobs"

    # parameters to run the inference scripts
    dataset_name = "lucchi"  # name of the dataset in lower-case
    model_type = "vit_b"
    with_cem = False  # use the models trained with mitolab
    experiment_set = "vanilla"  # infer using generalists or vanilla models

    if with_cem:
        em_name = "with_cem"
    else:
        em_name = "without_cem"

    # let's set the experiment type - either using the generalists or just using vanilla model
    if experiment_set == "generalists":
        checkpoint = "/scratch/usr/nimanwai/micro-sam/checkpoints/"
        checkpoint += f"{model_type}/{em_name}/mito_nuc_em_generalist_sam/best.pt"
    elif experiment_set == "vanilla":
        checkpoint = None
    else:
        raise ValueError("Choose from generalists/vanilla")

    experiment_folder = "/scratch/projects/nim00007/sam/experiments/new_models/"
    experiment_folder += f"{experiment_set}/em/{dataset_name}/mito_nuc_em_generalist_sam/"
    if experiment_set == "generalists":
        experiment_folder += f"{em_name}/"
    experiment_folder += f"{model_type}/"

    # now let's run the experiments
    if experiment_set == "vanilla":
        all_setups = ["precompute_embeddings", "evaluate_amg", "iterative_prompting"]
    else:
        all_setups = ["precompute_embeddings", "evaluate_amg", "evaluate_instance_segmentation", "iterative_prompting"]
    for current_setup in all_setups:
        write_batch_script(
            env_name="sam",
            out_path=get_batch_script_names(tmp_folder),
            inference_setup=current_setup,
            checkpoint=checkpoint,
            model_type=model_type,
            experiment_folder=experiment_folder,
            dataset_name=dataset_name,
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
