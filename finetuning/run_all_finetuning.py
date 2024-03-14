import os
import shutil
import subprocess
from datetime import datetime


N_OBJECTS = {
    "vit_t": 50,
    "vit_b": 40,
    "vit_l": 30,
    "vit_h": 25
}


def write_batch_script(out_path, _name, env_name, model_type, save_root):
    "Writing scripts with different micro-sam finetunings."
    batch_script = f"""#!/bin/bash
#SBATCH -t 14-00:00:00
#SBATCH --mem 64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A nim00007
#SBATCH -c 16
#SBATCH --qos=14d
#SBATCH --constraint=80gb
#SBATCH --job-name={os.path.split(_name)[-1]}

source activate {env_name} \n"""

    # python script
    python_script = f"python {_name}.py "

    # save root folder
    python_script += f"-s {save_root} "

    # name of the model configuration
    python_script += f"-m {model_type} "

    # choice of the number of objects
    python_script += f"--n_objects {N_OBJECTS[model_type]} "

    # let's add the python script to the bash script
    batch_script += python_script

    _op = out_path[:-3] + f"_{os.path.split(_name)[-1]}.sh"
    with open(_op, "w") as f:
        f.write(batch_script)

    cmd = ["sbatch", _op]
    subprocess.run(cmd)


def get_batch_script_names(tmp_folder):
    tmp_folder = os.path.expanduser(tmp_folder)
    os.makedirs(tmp_folder, exist_ok=True)

    script_name = "micro-sam-finetuning"

    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = script_name + dt
    batch_script = os.path.join(tmp_folder, f"{tmp_name}.sh")

    return batch_script


def submit_slurm(args):
    "Submit python script that needs gpus with given inputs on a slurm node."
    tmp_folder = "./gpu_jobs"

    script_combinations = {
        "livecell_specialist": "livecell_finetuning",
        "deepbacs_specialist": "specialists/training/light_microscopy/deepbacs_finetuning",
        "tissuenet_specialist": "specialists/training/light_microscopy/tissuenet_finetuning",
        "plantseg_root_specialist": "specialists/training/light_microscopy/plantseg_root_finetuning",
        "neurips_cellseg_specialist": "specialists/training/light_microscopy/neurips_cellseg_finetuning",
        "lm_generalist": "generalists/training/light_microscopy/train_lm_generalist",
        "em_mito_nuc_generalist": "generalists/training/electron_microscopy/mito_nuc/train_mito_nuc_em_generalist",
        "em_boundaries_generalist": "generalists/training/electron_microscopy/boundaries/train_boundaries_em_generalist"
    }
    if args.experiment_name is None:
        experiments = list(script_combinations.keys())
    else:
        assert args.experiment_name in list(script_combinations.keys()), \
            f"Choose from {list(script_combinations.keys())}"
        experiments = [args.experiment_name]

    if args.model_type is None:
        models = list(N_OBJECTS.keys())
    else:
        models = [args.model_type]

    for experiment in experiments:
        script_name = script_combinations[experiment]
        print(f"Running for {script_name}")
        for model_type in models:
            write_batch_script(
                out_path=get_batch_script_names(tmp_folder),
                _name=script_name,
                env_name="mobilesam" if model_type == "vit_t" else "sam",
                model_type=model_type,
                save_root=args.save_root
            )


def main(args):
    try:
        shutil.rmtree("./gpu_jobs")
    except FileNotFoundError:
        pass

    submit_slurm(args)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_name", type=str, default=None)
    parser.add_argument("-s", "--save_root", type=str, default="/scratch/usr/nimanwai/micro-sam/")
    parser.add_argument("-m", "--model_type", type=str, default=None)
    args = parser.parse_args()
    main(args)
