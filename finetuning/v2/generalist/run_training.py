import os
import shutil
import argparse
import subprocess
from datetime import datetime


# Epochs per model. One epoch takes 53 min for hvit_t and 57 min for hvit_l, so these fit the 96 h qos.
EPOCHS = {
    "hvit_t": 94,
    "hvit_s": 92,
    "hvit_b": 90,
    "hvit_l": 90,
}

# GDR_LEVEL=LOC is mandatory with IB: GPUDirect RDMA fails on these nodes with IBV_WC_LOC_PROT_ERR.
NCCL_ENV = {
    True: {
        "NCCL_IB_DISABLE": "0",
        "NCCL_SOCKET_IFNAME": "ib0",
        "NCCL_NET_GDR_LEVEL": "LOC",
        "NCCL_DEBUG": "WARN",
    },
    False: {
        "NCCL_IB_DISABLE": "1",
        "NCCL_SOCKET_IFNAME": "ib0",
        "NCCL_DEBUG": "WARN",
    },
}

SCRIPT = "/mnt/vast-kisski/home/archit/u28048/micro-sam/finetuning/v2/generalist/train_joint.py"
PARTITION = "kisski-h100"
GPU_TYPE = "H100"
SAVE_ROOT = "/mnt/vast-nhr/projects/cidas/cca/models/micro_sam2/joint/v5"


def write_batch_script(
    out_path, model_type, n_epochs, dataset_choice, save_root, reservation, enable_ib, dry, tag,
    with_boundaries=False, boundary_dice_weight=1.0,
):
    """Write the sbatch script for one joint SAM2 training run on 2 nodes x 4 H100, and submit it."""
    nccl_block = "\n".join(f"export {key}={value}" for key, value in NCCL_ENV[enable_ib].items())
    if not 0.0 <= boundary_dice_weight <= 1.0:
        raise ValueError("boundary_dice_weight must be between zero and one.")
    boundary_args = f" --with_boundaries --boundary_dice_weight {boundary_dice_weight}" if with_boundaries else ""

    batch_script = rf"""#!/bin/bash
#SBATCH --job-name=μSAM2_joint_{model_type}{"_" + tag if tag else ""}
#SBATCH -t 4-00:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH -p {PARTITION}
#SBATCH --exclude=ggpu241
#SBATCH --gpus-per-node={GPU_TYPE}:4
#SBATCH --cpus-per-task 192
# 0 gives the job all memory of the node. Host memory grows over the epochs.
#SBATCH --mem 0
#SBATCH --qos=96h
#SBATCH --constraint=inet

source ~/.bashrc
micromamba activate super

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SAVE_ROOT={save_root}
# One malloc arena per loader worker. A per-thread arena never shrinks, and host memory then grows every epoch.
export MALLOC_ARENA_MAX=2
# A fixed trim threshold also stops glibc from moving the large distance transform arrays out of mmap.
export MALLOC_TRIM_THRESHOLD_=134217728
# 4 ranks x 8 workers x 6 threads = 192 threads, one per CPU. Fewer workers keep fewer copies of the file handles.
export LABEL_TRAFO_THREADS=6
export N_WORKERS=8
export RUN_TAG={tag}
# The compile cache must be node-local. A cache on the shared filesystem blocks on its file locks.
export TORCHINDUCTOR_CACHE_DIR=/local/jobs/${{USER}}_${{SLURM_JOB_ID}}/inductor

GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29500

{nccl_block}

srun --cpu-bind=none bash -c "torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --node_rank=\$SLURM_NODEID \
    {SCRIPT} --model_type {model_type} --n_epochs {n_epochs} --dataset_choice {dataset_choice} --compile{boundary_args}"
"""
    if not tag:
        batch_script = batch_script.replace("export RUN_TAG=\n", "")
    if reservation:
        batch_script = batch_script.replace(
            f"#SBATCH -p {PARTITION}\n", f"#SBATCH -p {PARTITION}\n#SBATCH --reservation={reservation}\n"
        )

    script_path = out_path[:-3] + f"_{model_type}.sh"
    with open(script_path, "w") as f:
        f.write(batch_script)

    if not dry:
        subprocess.run(["sbatch", script_path])


def get_batch_script_names(tmp_folder):
    """Return a unique path for a new sbatch script in the given folder."""
    tmp_folder = os.path.expanduser(tmp_folder)
    os.makedirs(tmp_folder, exist_ok=True)
    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    return os.path.join(tmp_folder, f"joint-sam2-multi-node{dt}.sh")


def submit_slurm(args):
    """Submit the joint SAM2 multi-node training jobs to slurm."""
    tmp_folder = "./gpu_jobs"
    models = list(EPOCHS.keys()) if args.model_type is None else [args.model_type]
    save_root = os.path.join(os.path.dirname(SAVE_ROOT), args.tag) if args.tag else SAVE_ROOT

    for model_type in models:
        print(f"Submitting joint training for {model_type}")
        write_batch_script(
            out_path=get_batch_script_names(tmp_folder),
            model_type=model_type,
            n_epochs=EPOCHS[model_type],
            dataset_choice=args.dataset_choice,
            save_root=os.path.abspath(args.save_root or save_root),
            reservation=args.reservation,
            enable_ib=args.enable_ib == "yes",
            dry=args.dry,
            tag=args.tag,
            with_boundaries=args.with_boundaries,
            boundary_dice_weight=args.boundary_dice_weight,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_type", type=str, default=None, choices=list(EPOCHS.keys()),
        help="The model type. Submits all four models if not given.",
    )
    parser.add_argument(
        "--dataset_choice", type=str, default="all", choices=["lm", "em", "hp", "all"],
        help="The datasets for the joint training.",
    )
    parser.add_argument("-r", "--reservation", type=str, default=None, help="The slurm reservation to submit under.")
    parser.add_argument(
        "-s", "--save_root", type=str, default=None,
        help="Where to save checkpoints and logs. Defaults to the shared v5 folder, or to its '<tag>' sibling.",
    )
    parser.add_argument("--tag", type=str, default=None, help="Run tag, e.g. 'v5a', added to the run and job names.")
    parser.add_argument("--enable_ib", type=str, default="yes", choices=["yes", "no"], help="Use IB, not sockets.")
    parser.add_argument(
        "--with_boundaries", action="store_true",
        help="Train with the additional object-boundary channel. Disabled by default.",
    )
    parser.add_argument(
        "--boundary_dice_weight", type=float, default=1.0,
        help="Boundary Dice weight between 0 and 1: 1 selects Dice only, 0 selects BCE only. "
             "Used with --with_boundaries.",
    )
    parser.add_argument("--dry", action="store_true", help="Write the sbatch scripts but do not submit them.")
    args = parser.parse_args()
    if not 0.0 <= args.boundary_dice_weight <= 1.0:
        parser.error("--boundary_dice_weight must be between zero and one.")

    tmp_dir = "./gpu_jobs"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    submit_slurm(args)


if __name__ == "__main__":
    main()
