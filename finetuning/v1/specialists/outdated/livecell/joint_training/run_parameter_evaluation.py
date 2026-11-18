import os
import re
import shutil
import subprocess
from glob import glob
from pathlib import Path
from datetime import datetime


ALL_SCRIPTS = [
    "../../evaluation/precompute_embeddings",
    "../../evaluation/iterative_prompting",
    "../../evaluation/evaluate_amg",
    "../../evaluation/evaluate_instance_segmentation"
]

ROOT = "/scratch/usr/nimanwai/experiments/micro-sam/parameters_ablation/"
DATA_DIR = "/scratch/projects/nim00007/sam/data/livecell"


def write_slurm_scripts(
    inference_setup, env_name, checkpoint, model_type, experiment_folder, out_path
):
    batch_script = f"""#!/bin/bash
#SBATCH -c 8
#SBATCH --mem 16G
#SBATCH -t 2-00:00:00
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A gzz0001
#SBATCH --job-name={Path(inference_setup).stem}

source activate {env_name} \n"""

    # python script
    batch_script += f"python {inference_setup}.py -c {checkpoint} -m {model_type} -e {experiment_folder} -d livecell "

    _op = out_path[:-3] + f"_{Path(inference_setup).stem}.sh"

    with open(_op, "w") as f:
        f.write(batch_script)

    # we run the first prompt for iterative once starting with point, and then starting with box (below)
    if inference_setup.endswith("iterative_prompting"):
        batch_script += "--box "

        new_path = out_path[:-3] + f"_{Path(inference_setup).stem}_box.sh"
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


def run_slurm_scripts(model_type, checkpoint, experiment_folder, scripts=ALL_SCRIPTS):
    tmp_folder = "./gpu_jobs"
    shutil.rmtree(tmp_folder, ignore_errors=True)

    for current_setup in scripts:
        write_slurm_scripts(
            inference_setup=current_setup,
            env_name="mobilesam" if model_type == "vit_t" else "sam",
            checkpoint=checkpoint,
            model_type=model_type,
            experiment_folder=experiment_folder,
            out_path=get_batch_script_names(tmp_folder)
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


def main():
    checkpoint_paths = glob(os.path.join(ROOT, "checkpoints", "**", "best.pt"), recursive=True)
    # everything betweek checkpoints and best.pt is to be stored
    for checkpoint_path in checkpoint_paths:
        print(checkpoint_path)
        name = checkpoint_path[76:-8]
        run_slurm_scripts(
            model_type="vit_b",
            checkpoint=checkpoint_path,
            experiment_folder=os.path.join(ROOT, "results", name)
        )


if __name__ == "__main__":
    main()
