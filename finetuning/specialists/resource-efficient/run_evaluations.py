import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

import h5py
import imageio.v3 as imageio


ALL_SCRIPTS = [
    "../../evaluation/precompute_embeddings",
    "../../evaluation/evaluate_amg",
    "../../evaluation/iterative_prompting",
    "../../evaluation/evaluate_instance_segmentation"
]


def process_covid_if(input_path):
    all_image_paths = sorted(glob(os.path.join(input_path, "*")))[13:]

    for image_path in tqdm(all_image_paths):
        image_id = Path(image_path).stem

        image_save_dir = os.path.join(Path(image_path).parent, "slices", "raw")
        label_save_dir = os.path.join(Path(image_path).parent, "slices", "labels")

        os.makedirs(image_save_dir, exist_ok=True)
        os.makedirs(label_save_dir, exist_ok=True)

        with h5py.File(image_path, "r") as f:
            raw = f["raw/serum_IgG/s0"][:]
            labels = f["labels/cells/s0"][:]

            imageio.imwrite(os.path.join(image_save_dir, f"{image_id}.tif"), raw)
            imageio.imwrite(os.path.join(label_save_dir, f"{image_id}.tif"), labels)


def write_slurm_scripts(
    inference_setup, env_name, checkpoint, model_type, experiment_folder, out_path
):
    batch_script = f"""#!/bin/bash
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 2-00:00:00
#SBATCH -p gpu
#SBATCH -G rtx5000:1
#SBATCH --job-name={inference_setup}

source ~/.bashrc
mamba activate {env_name} \n"""

    # python script
    batch_script += f"python {inference_setup}.py -c {checkpoint} -m {model_type} -e {experiment_folder}"

    _op = out_path[:-3] + f"_{Path(inference_setup).stem}.sh"

    with open(_op, "w") as f:
        f.write(batch_script)

    # we run the first prompt for iterative once starting with point, and then starting with box (below)
    if inference_setup.endswith("iterative_prompting"):
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


def run_slurm_scripts(model_type, checkpoint):
    tmp_folder = "./gpu_jobs"

    for current_setup in ALL_SCRIPTS:
        write_slurm_scripts(
            inference_setup=current_setup,
            env_name="mobilesam" if model_type == "vit_t" else "sam",
            checkpoint=checkpoint,
            model_type=model_type,
            experiment_folder=experiment_folder,
            out_path=get_batch_script_names(tmp_folder)
        )


def main(args):
    process_covid_if(
        input_path=args.input_path
    )

    all_checkpoint_paths = glob("/scratch/users/archit/experiments/**/best.pt", recursive=True)
    for checkpoint_path in all_checkpoint_paths:
        print(checkpoint_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, default="/scratch/users/archit/data/covid-if")
    args = parser.parse_args()
    main(args)
