import os
import shutil
import subprocess
from datetime import datetime


def write_batch_script(
    dataset_name, method, model_type, out_path, baseline=False, dry=True,
):
    """Writing scripts to submit multiple evaluations relevant for APG.
    """
    if method == "cellpose":
        if model_type == "cyto3":
            env = "cp3"
        elif model_type == "cpsam":
            env = "cp4"
        else:
            raise ValueError
    else:
        env = "super"

    batch_script = f"""#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p grete-h100:shared
#SBATCH -G H100:1
#SBATCH -A gzz0001
#SBATCH -c 16
#SBATCH --mem 64G
#SBATCH --job-name=apg_evaluation

source ~/.bashrc
micromamba activate {env} \n"""

    # Prep the python script.
    if baseline:
        python_script = "python prepare_baselines.py "
        python_script += f"-d {dataset_name} "
        python_script += f"--method {method} "
        python_script += f"-m {model_type}"
    else:
        python_script = "python run_evaluation.py "
        python_script += f"-d {dataset_name} "
        python_script += f"-m {model_type}"

    # Add the python script to the bash script
    batch_script += python_script

    _op = out_path[:-3] + "_apg.sh"
    with open(_op, "w") as f:
        f.write(batch_script)

    cmd = ["sbatch", _op]
    if not dry:
        subprocess.run(cmd)


def get_batch_script_names(tmp_folder):
    tmp_folder = os.path.expanduser(tmp_folder)
    os.makedirs(tmp_folder, exist_ok=True)

    script_name = "micro-sam++"

    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = script_name + dt
    batch_script = os.path.join(tmp_folder, f"{tmp_name}.sh")

    return batch_script


def submit_slurm(args):
    """Submit python scripts that need slurm resources with given inputs on a slurm node.
    """
    # Other random stuff.
    tmp_folder = "./gpu_jobs"
    generalist_model = args.model_type
    dataset_name = args.dataset_name
    dry = args.dry

    method_combinations = [
        # SAM-based models
        ["amg", "vit_b"],
        ["amg", generalist_model],
        ["ais", generalist_model],
        ["apg", generalist_model],
        # SAM2
        # ["sam2", "hvit_b"],
        # SAM3
        # ["sam3", "cells"]
        # And other external methods.
        ["cellpose", "cyto3"],
        ["cellpose", "cpsam"],
        ["cellsam", "cellsam"],
        # ["instanseg", "brightgield_nuclei"]
        # ["instanseg", "fluoroscence_cells_and_nuclei"]
    ]

    if args.baseline:  # Let's run the baselines.
        for curr_method in method_combinations:
            print(f"Submitting scripts for {dataset_name}")
            method, model_type = curr_method
            write_batch_script(
                dataset_name=dataset_name,
                out_path=get_batch_script_names(tmp_folder),
                method=method,
                model_type=model_type,
                baseline=True,
                dry=dry,
            )
    else:  # Run the APG grid-search script
        print(f"Submitting grid-search script for {dataset_name}")
        write_batch_script(
            dataset_name=dataset_name,
            out_path=get_batch_script_names(tmp_folder),
            method="apg",
            model_type=generalist_model,
            baseline=False,
            dry=dry,
        )


def main(args):
    tmp_dir = "./gpu_jobs"
    if os.path.exists(tmp_dir):
        shutil.rmtree("./gpu_jobs")

    submit_slurm(args)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline", action="store_true", help="Whether to run a baseline script"
    )
    parser.add_argument(
        "-d", "--dataset_name", type=str, default=None, help="The choice of dataset name.",
    )
    parser.add_argument(
        "--method", type=str, default=None, help="The choice of baseline method."
    )
    parser.add_argument(
        "-m", "--model_type", type=str, default=None, help="The choice of model type for baseline / SAM methods."
    )
    parser.add_argument(
        "--dry", action="store_true", help="Whether to submit the scripts to slurm or only store the scripts."
    )
    args = parser.parse_args()
    main(args)
