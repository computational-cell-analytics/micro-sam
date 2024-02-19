import os
import shutil
import datetime
import subprocess

import common


def write_batch_sript(
    tier_choice, env_name, resource_name, input_path, save_root, model_type, n_objects, n_samples, script_name
):
    "Writing scripts for resource-efficient trainings for micro-sam finetuning on Covid-IF."
    batch_script = common.base_slurm_script(
        tier_choice=tier_choice,
        env_name=env_name,
        resource_name=resource_name
    )

    python_script = "python covid_if_finetuning.py "

    # add parameters to the python script
    python_script += f"-i {input_path} "  # path to the covid-if data
    python_script += f"-m {model_type} "  # choice of vit
    python_script += f"--n_objects {n_objects} "  # number of objects per batch for finetuning
    python_script += f"--n_samples {n_samples} "  # number of samples we train for
    if save_root is not None:
        python_script += f"-s {save_root} "  # path to save the model checkpoints and logs

    # let's add the python script to the bash script
    batch_script += python_script

    print(batch_script)

    with open(script_name, "w") as f:
        f.write(batch_script)

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
    model_type = args.model_type

    script_name = "<NAME>.sh"

    write_batch_sript(
        tier_choice=args.tier,
        env_name="mobilesam" if model_type == "vit_t" else "sam",
        resource_name=args.resource_name,
        input_path=args.input_path,
        save_root=args.save_root,
        model_type=model_type,
        n_objects=args.n_objects,
        n_samples=args.n_samples,
        script_name=script_name,
    )


if __name__ == "__main__":
    try:
        shutil.rmtree("./gpu_jobs")
    except FileNotFoundError:
        pass

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--tier", type=int, required=True, help="The different tiers of resource-efficient finetuning."
    )
    parser.add_argument(
        "-r", "--resource_name", type=str, required=True, help="The hardware resources used for finetuning."
    )
    parser.add_argument("-i", "--input_path", type=str, required=True, help="Path to Covid-IF dataset.")
    parser.add_argument("-s", "--save_root", type=str, default=None, help="Path to save checkpoints.")
    parser.add_argument("-m", "--model_type", type=str, required=True, help="Choice of image encoder in SAM")
    parser.add_argument("--n_objects", type=int, required=True, help="The number of objects (instances) per batch.")
    parser.add_argument("--n_samples", type=int, required=True, help="The number of training samples.")
    args = parser.parse_args()
    main(args)
