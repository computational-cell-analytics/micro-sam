import os
import subprocess
from datetime import datetime


ROOT = "/scratch/usr/nimanwai/test/3d_segmentation"
CKPT_ROOT = "/scratch/usr/nimanwai/micro-sam/checkpoints"


def submit_3d_segmentation(
    dataset_name, model_type, outpath, species=None, do_interactive=False, do_auto=False
):
    batch_script = """#!/bin/bash
#SBATCH --job-name=micro-sam-3d
#SBATCH -t 14-00:00:00
#SBATCH --mem 64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A nim00007
#SBATCH -c 16
#SBATCH --constraint=80gb
#SBATCH --qos=14d

source activate sam \n"""

    # python script
    python_script = f"python evaluate_{dataset_name}.py "

    # experiment folder
    experiment_folder = os.path.join(
        ROOT, dataset_name if species is None else f"{dataset_name}_{species}", model_type
    )
    python_script += f"-e {experiment_folder} "

    if species is not None:
        # let's add species
        python_script += f"--species {species} "

    if do_interactive:
        python_script += "--int "

    if do_auto and len(model_type) > 5:  # only perform ais with finetuned models
        python_script += "--ais "

    assert do_auto + do_interactive > 0, "Please choose one of interactive or automatic instance segmentation."

    # model name
    python_script += f"-m {model_type[:5]} "

    if model_type == "vit_b_lm":
        python_script += f"-c {CKPT_ROOT}/{model_type[:5]}/lm_generalist_sam/best.pt "
    elif model_type == "vit_b_em_organelles":
        python_script += f"-c {CKPT_ROOT}/{model_type[:5]}/mito_nuc_em_generalist_sam/best.pt "
    else:
        assert len(model_type) == 5

    # let's add the python script to the bash script
    batch_script += python_script

    with open(outpath, "w") as f:
        f.write(batch_script)

    cmd = ["sbatch", outpath]
    subprocess.run(cmd)


def get_batch_script_name(tmp_folder):
    tmp_folder = os.path.expanduser(tmp_folder)
    os.makedirs(tmp_folder, exist_ok=True)
    script_name = "micro-sam-3d"
    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = script_name + dt
    batch_script = os.path.join(tmp_folder, f"{tmp_name}.sh")
    return batch_script


def submit_all_scripts():
    tmp_folder = "./gpu_jobs"

    # for plantseg ovules
    submit_3d_segmentation(
        dataset_name="plantseg", model_type="vit_b",
        outpath=get_batch_script_name(tmp_folder),
        species="ovules", do_interactive=True, do_auto=True
    )
    submit_3d_segmentation(
        dataset_name="plantseg", model_type="vit_b_lm",
        outpath=get_batch_script_name(tmp_folder),
        species="ovules", do_interactive=True, do_auto=True
    )

    # for plantseg root
    submit_3d_segmentation(
        dataset_name="plantseg", model_type="vit_b",
        outpath=get_batch_script_name(tmp_folder),
        species="root", do_interactive=True, do_auto=True
    )
    submit_3d_segmentation(
        dataset_name="plantseg", model_type="vit_b_lm",
        outpath=get_batch_script_name(tmp_folder),
        species="root", do_interactive=True, do_auto=True
    )

    # for lucchi
    submit_3d_segmentation(
        dataset_name="lucchi", model_type="vit_b",
        outpath=get_batch_script_name(tmp_folder),
        do_interactive=True, do_auto=True
    )
    submit_3d_segmentation(
        dataset_name="lucchi", model_type="vit_b_em_organelles",
        outpath=get_batch_script_name(tmp_folder),
        do_interactive=True, do_auto=True
    )

    # for mitoem rat
    submit_3d_segmentation(
        dataset_name="mitoem", model_type="vit_b",
        outpath=get_batch_script_name(tmp_folder),
        species="rat", do_interactive=True, do_auto=True
    )
    submit_3d_segmentation(
        dataset_name="mitoem", model_type="vit_b_em_organelles",
        outpath=get_batch_script_name(tmp_folder),
        species="rat", do_interactive=True, do_auto=True
    )

    # for mitoem human
    submit_3d_segmentation(
        dataset_name="mitoem", model_type="vit_b",
        outpath=get_batch_script_name(tmp_folder),
        species="human", do_interactive=True, do_auto=True
    )
    submit_3d_segmentation(
        dataset_name="mitoem", model_type="vit_b_em_organelles",
        outpath=get_batch_script_name(tmp_folder),
        species="human", do_interactive=True, do_auto=True
    )


def main():
    try:
        import shutil
        shutil.rmtree("./gpu_jobs")
    except FileNotFoundError:
        pass

    submit_all_scripts()


if __name__ == "__main__":
    main()
