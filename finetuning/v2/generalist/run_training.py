import os
import shutil
import subprocess
from datetime import datetime


# Epochs per model: uniform 150, fits comfortably under the 96h qos for every model size.
# Overrunning is harmless anyway: ReduceLROnPlateau doesn't depend on n_epochs, and
# checkpoints are written every epoch, so hitting the wall clock only loses the final one.
EPOCHS = {
    "hvit_t": 150,
    "hvit_s": 150,
    "hvit_b": 150,
    "hvit_l": 150,
}

SCRIPT = "/mnt/vast-nhr/home/archit/u16934/micro-sam/finetuning/v2/generalist/train_joint.py"


def write_batch_script(out_path, model_type, n_epochs, dataset_choice, distance_type, save_root, dry):
    "Writing the multi-node sbatch script for one joint SAM2 training run (2 nodes x 4 H100 = 8 GPUs)."
    # IB-supported NCCL comms across nodes.
    batch_script = rf"""#!/bin/bash
#SBATCH --job-name=micro-sam2_joint_{model_type}
#SBATCH -t 4-00:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH -p kisski-h100
#SBATCH --gpus-per-node=H100:4
#SBATCH --cpus-per-task 32
#SBATCH --mem 384G
#SBATCH --qos=96h
#SBATCH --constraint=inet

source ~/.bashrc
micromamba activate super

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SAVE_ROOT={save_root}

GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29500

export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=WARN

srun --cpu-bind=none bash -c "torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --node_rank=\$SLURM_NODEID \
    {SCRIPT} --model_type {model_type} --n_epochs {n_epochs} --dataset_choice {dataset_choice} \
    --distance_type {distance_type}"
"""

    _op = out_path[:-3] + f"_{model_type}.sh"
    with open(_op, "w") as f:
        f.write(batch_script)

    cmd = ["sbatch", _op]
    if not dry:
        subprocess.run(cmd)


def get_batch_script_names(tmp_folder):
    tmp_folder = os.path.expanduser(tmp_folder)
    os.makedirs(tmp_folder, exist_ok=True)

    script_name = "joint-sam2-multi-node"

    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    tmp_name = script_name + dt
    batch_script = os.path.join(tmp_folder, f"{tmp_name}.sh")

    return batch_script


def submit_slurm(args):
    "Submit the joint SAM2 multi-node training jobs to slurm."
    tmp_folder = "./gpu_jobs"

    models = list(EPOCHS.keys()) if args.model_type is None else [args.model_type]

    for model_type in models:
        print(f"Submitting joint training for {model_type}")
        write_batch_script(
            out_path=get_batch_script_names(tmp_folder),
            model_type=model_type,
            n_epochs=EPOCHS[model_type],
            dataset_choice=args.dataset_choice,
            distance_type=args.distance_type,
            save_root=args.save_root,
            dry=args.dry,
        )


def main(args):
    tmp_dir = "./gpu_jobs"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    submit_slurm(args)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_type", type=str, default=None, choices=list(EPOCHS.keys()),
        help="The choice of model type. Submits all four models if not specified.",
    )
    parser.add_argument(
        "-s", "--save_root", type=str, default="/mnt/vast-nhr/projects/cidas/cca/models/micro_sam2/joint/v3",
        help="The path where to store the model checkpoints and logs.",
    )
    parser.add_argument(
        "--dataset_choice", type=str, default="all", choices=["lm", "em", "hp", "all"],
        help="The choice of datasets for joint training.",
    )
    parser.add_argument(
        "--distance_type", type=str, default="geodesic", choices=["geodesic", "directed"],
        help="The regression target of the automatic branch.",
    )
    parser.add_argument(
        "--dry", action="store_true", help="Whether to only write the sbatch scripts without submitting them."
    )
    args = parser.parse_args()

    main(args)
