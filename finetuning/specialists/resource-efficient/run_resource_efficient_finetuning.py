import os
import shutil
import itertools
import subprocess
from datetime import datetime


def base_slurm_script(env_name, partition, cpu_mem, cpu_cores, gpu_name=None):
    assert partition in ["grete:shared", "gpu", "medium"]
    if gpu_name is not None:
        assert gpu_name in ["GTX1080", "RTX5000", "V100", "A100"]

    base_script = f"""#!/bin/bash
#SBATCH -c {cpu_cores}
#SBATCH --mem {cpu_mem}
#SBATCH -p {partition}
#SBATCH -t 2-00:00:00
#SBATCH --job-name micro-sam-resource-efficient-finetuning
"""
    if gpu_name is not None:
        base_script += f"#SBATCH -G {gpu_name}:1 \n"

    if partition.startswith("grete"):
        base_script += "#SBATCH -A gzz0001 \n"

    base_script += "\n" + "source ~/.bashrc" + "\n" + "mamba activate {env_name}" + "\n"

    return base_script


def write_batch_sript(
    env_name, partition, cpu_mem, cpu_cores, gpu_name, input_path, save_root,
    model_type, n_objects, n_images, script_name, freeze, lora, dry,
):
    assert model_type in ["vit_t", "vit_b", "vit_t_lm", "vit_b_lm"]

    "Writing scripts for resource-efficient trainings for micro-sam finetuning on Covid-IF."
    batch_script = base_slurm_script(
        env_name=env_name,
        partition=partition,
        cpu_mem=cpu_mem,
        cpu_cores=cpu_cores,
        gpu_name=gpu_name
    )

    python_script = "python covid_if_finetuning.py "

    # add parameters to the python script
    python_script += f"-i {input_path} "  # path to the covid-if data
    python_script += f"-m {model_type} "  # choice of vit
    python_script += f"--n_objects {n_objects} "  # number of objects per batch for finetuning
    python_script += f"--n_images {n_images} "  # number of images we train for

    # Whether to use LoRA-based finetuning
    # NOTE: We use rank as 4 for LoRA.
    if lora:
        python_script += "--lora_rank 4 "

    if gpu_name is not None:
        resource_name = f"{gpu_name}"
    else:
        resource_name = f"cpu_{cpu_mem}-mem_{cpu_cores}-cores"

    # Updating the path where the model checkpoints and logs will be saved.
    updated_save_root = os.path.join(
        save_root,
        resource_name,
        model_type,
        "lora-finetuning" if lora else "full-finetuning",
        "freeze-None" if freeze is None else f"freeze-{freeze}",
        f"{n_images}-images"
    )
    if save_root is not None:
        python_script += f"-s {updated_save_root} "  # path to save model checkpoints and logs

    # Whether to freeze a certain part of the SAM model.
    if freeze is not None:
        python_script += f"--freeze {freeze} "

    # let's add the python script to the bash script
    batch_script += python_script

    with open(script_name, "w") as f:
        f.write(batch_script)

    if not dry:
        cmd = ["sbatch", script_name]
        subprocess.run(cmd)


def get_batch_script_names(tmp_folder):
    tmp_folder = os.path.expanduser(tmp_folder)
    os.makedirs(tmp_folder, exist_ok=True)
    script_name = "micro-sam-finetuning"
    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = script_name + dt
    batch_script = os.path.join(tmp_folder, f"{tmp_name}.sh")
    return batch_script


def main(args):
    tmp_folder = "./gpu_jobs"
    model_type = args.model_type

    all_n_images = [1, 2, 5, 10]
    use_lora = [False, True]

    for (n_images, lora) in itertools.product(all_n_images, use_lora):
        # We cannot use LoRA and freeze the image encoder at the same time.
        if lora and args.freeze == "image_encoder":
            continue

        write_batch_sript(
            env_name="mobilesam" if model_type[:5] == "vit_t" else "sam",
            partition=args.partition,
            cpu_mem=args.mem,
            cpu_cores=args.cpu_cores,
            gpu_name=args.gpu_name,
            input_path=args.input_path,
            save_root=args.save_root,
            model_type=model_type,
            n_objects=args.n_objects,
            n_images=n_images,
            script_name=get_batch_script_names(tmp_folder),
            freeze=args.freeze,
            lora=lora,
            dry=args.dry,
        )


if __name__ == "__main__":
    try:
        shutil.rmtree("./gpu_jobs")
    except FileNotFoundError:
        pass

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, required=True, help="Path to Covid-IF dataset.")
    parser.add_argument("-s", "--save_root", type=str, default=None, help="Path to save checkpoints.")
    parser.add_argument("-m", "--model_type", type=str, required=True, help="Choice of image encoder in SAM")
    parser.add_argument("--n_objects", type=int, required=True, help="The number of objects (instances) per batch.")
    parser.add_argument("--freeze", type=str, default=None, help="Which parts of the model to freeze for finetuning.")

    parser.add_argument("--partition", type=str, required=True, help="Name of the partition for running the job.")
    parser.add_argument("--mem", type=str, required=True, help="Amount of cpu memory.")
    parser.add_argument("-c", "--cpu_cores", type=int, required=True, help="Number of cpu cores.")
    parser.add_argument("-G", "--gpu_name", type=str, default=None, help="The GPU resources used for finetuning.")

    parser.add_argument("--dry", action="store_true", help="Whether to avoid submitting the configured scripts.")

    args = parser.parse_args()
    main(args)
