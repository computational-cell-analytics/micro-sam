import os
import re
import shutil
import itertools
import subprocess
from glob import glob
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

import h5py
import imageio.v3 as imageio


ALL_SCRIPTS = [
    "../../evaluation/precompute_embeddings",
    "../../evaluation/iterative_prompting",
    "../../evaluation/evaluate_amg",
    "../../evaluation/evaluate_instance_segmentation"
]

DATA_DIR = "/scratch/projects/nim00007/sam/data/covid_if"
ROOT = "/scratch/share/cidas/cca/experiments/resource-efficient-finetuning/"


def process_covid_if(input_path):
    if os.path.exists(os.path.join(input_path, "slices")):
        return

    all_image_paths = sorted(glob(os.path.join(input_path, "*.h5")))

    # val images
    for image_path in tqdm(all_image_paths[10:13]):
        image_id = Path(image_path).stem

        image_save_dir = os.path.join(Path(image_path).parent, "slices", "val", "raw")
        label_save_dir = os.path.join(Path(image_path).parent, "slices", "val", "labels")

        os.makedirs(image_save_dir, exist_ok=True)
        os.makedirs(label_save_dir, exist_ok=True)

        with h5py.File(image_path, "r") as f:
            raw = f["raw/serum_IgG/s0"][:]
            labels = f["labels/cells/s0"][:]

            imageio.imwrite(os.path.join(image_save_dir, f"{image_id}.tif"), raw)
            imageio.imwrite(os.path.join(label_save_dir, f"{image_id}.tif"), labels)

    # test images
    for image_path in tqdm(all_image_paths[13:]):
        image_id = Path(image_path).stem

        image_save_dir = os.path.join(Path(image_path).parent, "slices", "test", "raw")
        label_save_dir = os.path.join(Path(image_path).parent, "slices", "test", "labels")

        os.makedirs(image_save_dir, exist_ok=True)
        os.makedirs(label_save_dir, exist_ok=True)

        with h5py.File(image_path, "r") as f:
            raw = f["raw/serum_IgG/s0"][:]
            labels = f["labels/cells/s0"][:]

            imageio.imwrite(os.path.join(image_save_dir, f"{image_id}.tif"), raw)
            imageio.imwrite(os.path.join(label_save_dir, f"{image_id}.tif"), labels)


def write_slurm_scripts(
    inference_setup, env_name, checkpoint, model_type, experiment_folder, out_path, lora,
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
    batch_script += f"python {inference_setup}.py -c {checkpoint} -m {model_type} -e {experiment_folder} -d covid_if "

    # Whether the model was trained with LoRA
    # NOTE: We use rank 4 for LoRA.
    if lora:
        batch_script += "--lora_rank 4 "

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


def run_slurm_scripts(
    model_type, checkpoint, experiment_folder, dry, scripts=ALL_SCRIPTS, has_lora=False, freeze_image_encoder=False
):
    tmp_folder = "./gpu_jobs"
    shutil.rmtree(tmp_folder, ignore_errors=True)

    lora_combinations = [True, False]
    for (current_setup, use_lora) in itertools.product(scripts, lora_combinations):
        # for experiments such as vanilla and generalist models.
        if not has_lora == use_lora:
            continue

        # for experiments such as GTX1080, where freezing the image encoder is the way and we do not use LoRA
        if freeze_image_encoder and use_lora:
            continue

        write_slurm_scripts(
            inference_setup=current_setup,
            env_name="mobilesam" if model_type == "vit_t" else "sam",
            checkpoint=checkpoint,
            model_type=model_type,
            experiment_folder=experiment_folder,
            out_path=get_batch_script_names(tmp_folder),
            lora=use_lora,
        )

    if not dry:
        # the logic below automates running the precomputation of embeddings first and then inference.
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
    # preprocess the data
    process_covid_if(input_path=DATA_DIR)

    # results on vanilla models
    # run_slurm_scripts(
    #     model_type="vit_b",
    #     checkpoint=None,
    #     experiment_folder=os.path.join(ROOT, "vanilla", "vit_b"),
    #     scripts=ALL_SCRIPTS[:-1],
    #     dry=args.dry,
    # )

    # results on generalist models
    # vit_b_lm_path = "/scratch/usr/nimanwai/micro-sam/checkpoints/vit_b/lm_generalist_sam/best.pt"
    # run_slurm_scripts(
    #     model_type="vit_b",
    #     checkpoint=vit_b_lm_path,
    #     experiment_folder=os.path.join(ROOT, "generalist", "vit_b"),
    #     dry=args.dry,
    # )

    # results on resource-efficient finetuned checkpoints
    all_checkpoint_paths = glob(os.path.join(ROOT, "**", "best.pt"), recursive=True)

    for checkpoint_path in all_checkpoint_paths:
        # NOTE: We run the inference only for `vit_b` models. Remove this to run for `vit_t` models as well.
        _searcher = checkpoint_path.find("vit_b")
        if _searcher == -1:  # i.e. `vit_b` keyword was not found.
            continue

        _searcher2 = checkpoint_path.find("freeze-image_encoder")
        freeze_image_encoder = False
        if _searcher2 != -1:
            freeze_image_encoder = True

        _searcher3 = checkpoint_path.find("lora-finetuning")
        has_lora = False
        if _searcher3 != -1:
            has_lora = True

        experiment_folder = os.path.join("/", *checkpoint_path.split("/")[:-4])
        run_slurm_scripts(
            model_type=checkpoint_path.split("/")[-3],
            checkpoint=checkpoint_path,
            experiment_folder=experiment_folder,
            has_lora=has_lora,
            dry=args.dry,
            freeze_image_encoder=freeze_image_encoder,
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    main(args)
