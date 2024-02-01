import re
import os
import shutil
import subprocess
from glob import glob
from datetime import datetime


def write_batch_script(
    env_name, out_path, inference_setup, checkpoint, model_type, experiment_folder, dataset_name, delay=None
):
    "Writing scripts with different fold-trainings for micro-sam evaluation"
    batch_script = f"""#!/bin/bash
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 2-00:00:00
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A gzz0001
#SBATCH --job-name={inference_setup}

source ~/.bashrc
mamba activate {env_name} \n"""

    if delay is not None:
        batch_script += f"sleep {delay} \n"

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

    script_name = "micro-sam-inference"

    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = script_name + dt
    batch_script = os.path.join(tmp_folder, f"{tmp_name}.sh")

    return batch_script


def get_checkpoint_path(experiment_set, dataset_name, model_type, region):
    # let's set the experiment type - either using the generalist or just using vanilla model
    if experiment_set == "generalist":
        checkpoint = f"/scratch/usr/nimanwai/micro-sam/checkpoints/{model_type}/"

        if region == "organelles":
            checkpoint += "with_cem/mito_nuc_em_generalist_sam/best.pt"
        elif region == "boundaries":
            checkpoint += "boundaries_em_generalist_sam/best.pt"
        elif region == "lm":
            checkpoint += "lm_generalist_sam/best.pt"
        else:
            raise ValueError("Choose `region` from lm / organelles / boundaries")

    elif experiment_set == "specialist":
        if dataset_name.split("/") > 1:
            # it's the case for plantseg/root, we catch it and convert it to the expected format
            dataset_name = f"{dataset_name[0]}_{dataset_name[1]}"

        checkpoint = f"/scratch/usr/nimanwai/micro-sam/checkpoints/{model_type}/{dataset_name}_sam/best.pt"

    elif experiment_set == "vanilla":
        checkpoint = None

    else:
        raise ValueError("Choose from generalist / vanilla")

    if checkpoint is not None:
        assert os.path.exists(checkpoint)

    return checkpoint


def submit_slurm(args):
    "Submit python script that needs gpus with given inputs on a slurm node."
    tmp_folder = "./gpu_jobs"

    # parameters to run the inference scripts
    dataset_name = args.dataset_name  # name of the dataset in lower-case
    model_type = args.model_type
    experiment_set = args.experiment_set  # infer using generalist or vanilla models
    region = args.roi  # use the organelles model or boundaries model
    make_delay = "10s"  # wait for precomputing the embeddings and later run inference scripts

    checkpoint = get_checkpoint_path(experiment_set, dataset_name, model_type, region)

    modality = region if region == "lm" else "em"

    experiment_folder = "/scratch/projects/nim00007/sam/experiments/new_models/"
    experiment_folder += f"{experiment_set}/{modality}/{dataset_name}/{model_type}/"

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
            delay=None if current_setup == "precompute_embeddings" else make_delay
            )

    # the logic below automates the process of first running the precomputation of embeddings, and only then inference.
    job_id = []
    for i, my_script in enumerate(sorted(glob(tmp_folder + "/*"))):
        cmd = ["sbatch", my_script]

        if i > 0:
            cmd.insert(1, f"--dependency=afterany:{job_id[0]}")

        cmd_out = subprocess.run(cmd, capture_output=True, text=True)
        print(cmd_out.stdout if len(cmd_out.stdout) > 1 else cmd_out.stderr)

        if i == 0:
            job_id.append(re.findall(r'\d+', cmd_out.stdout)[0])


def main(args):
    try:
        shutil.rmtree("./gpu_jobs")
    except FileNotFoundError:
        pass

    submit_slurm(args)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", type=str, required=True)
    parser.add_argument("-m", "--model_type", type=str, required=True)
    parser.add_argument("-e", "--experiment_set", type=str, required=True)
    parser.add_argument("-r", "--roi", type=str, required=True)
    args = parser.parse_args()
    main(args)
