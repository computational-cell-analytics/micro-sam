import os
import shutil
import itertools
import subprocess
from datetime import datetime


def _write_batch_script(script_path, dataset_name, exp_script, save_root, phase, with_sam):
    job_name = exp_script.split("_")[-1] + ("-sam-" if with_sam else "-") + dataset_name

    batch_script = f"""#!/bin/bash
#SBATCH -t 2-00:00:00
#SBATCH --mem 64G
#SBATCH -c 16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p grete-h100:shared
#SBATCH -G H100:1
#SBATCH -A gzz0001
#SBATCH --job-name={job_name}

source activate sam \n"""

    # python script
    script = f"python {exp_script}.py "

    # all other parameters
    script += f"-d {dataset_name} -s {save_root} -p {phase} "

    # whether the model is trained using SAM pretrained weights
    if with_sam:
        script += "--sam "

    # let's combine both the scripts
    batch_script += script

    output_path = script_path[:-3] + f"_{job_name}.sh"
    with open(output_path, "w") as f:
        f.write(batch_script)

    cmd = ["sbatch", output_path]
    subprocess.run(cmd)


def _get_batch_script(tmp_folder):
    tmp_folder = os.path.expanduser(tmp_folder)
    os.makedirs(tmp_folder, exist_ok=True)

    script_name = "ais_benchmarking"

    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = script_name + dt
    batch_script = os.path.join(tmp_folder, f"{tmp_name}.sh")

    return batch_script


def _submit_to_slurm(tmp_folder):
    save_root = "/scratch/share/cidas/cca/models/micro-sam/ais_benchmarking/"
    phase = "predict"  # this can be updated to "train" / "predict" to run the respective scripts.

    scripts = ["train_unet", "train_unetr", "train_semanticsam"]
    datasets = ["livecell", "covid_if-1", "covid_if-2", "covid_if-5", "covid_if-10"]
    sam_combinations = [True, False]

    for (exp_script, dataset_name, with_sam) in itertools.product(scripts, datasets, sam_combinations):
        if exp_script.endswith("_unet") and with_sam:
            continue

        _write_batch_script(
            script_path=_get_batch_script(tmp_folder),
            dataset_name=dataset_name,
            exp_script=exp_script,
            save_root=save_root,
            phase=phase,
            with_sam=with_sam,
        )


def main():
    tmp_folder = "./gpu_jobs"

    try:
        shutil.rmtree(tmp_folder)
    except FileNotFoundError:
        pass

    _submit_to_slurm(tmp_folder)


if __name__ == "__main__":
    main()
