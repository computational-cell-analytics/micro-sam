import os
import shutil
import subprocess
from datetime import datetime


# Epochs per model: uniform 150, fits comfortably under the 96h qos for every model size.
# Overrunning is harmless anyway: ReduceLROnPlateau doesn't depend on n_epochs, and
# checkpoints are written every epoch, so hitting the wall clock only loses the final one.
EPOCHS = {
    "hvit_t": 180,  # currently at 77h for 150 epochs
    "hvit_s": 150,
    "hvit_b": 150,
    "hvit_l": 150,
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
SAVE_ROOT = "/mnt/vast-nhr/projects/cidas/cca/models/micro_sam2/joint/v4"


def write_batch_script(
    out_path, model_type, n_epochs, dataset_choice, save_root, reservation, enable_ib, use_compile, dry
):
    "Writing the multi-node sbatch script for one joint SAM2 training run (2 nodes x 4 H100 = 8 GPUs)."
    nccl_block = "\n".join(f"export {key}={value}" for key, value in NCCL_ENV[enable_ib].items())
    compile_flag = " --compile" if use_compile else ""

    batch_script = rf"""#!/bin/bash
#SBATCH --job-name=μSAM2_joint_{model_type}
#SBATCH -t 4-00:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH -p {PARTITION}
#SBATCH --gpus-per-node={GPU_TYPE}:4
#SBATCH --cpus-per-task 32
#SBATCH --mem 384G
#SBATCH --qos=96h
#SBATCH --constraint=inet

source ~/.bashrc
micromamba activate super

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SAVE_ROOT={save_root}
# The torch.compile cache must be node-local. A cache on the shared filesystem blocks the compile on its file locks.
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
    {SCRIPT} --model_type {model_type} --n_epochs {n_epochs} --dataset_choice {dataset_choice}{compile_flag}"
"""
    if reservation:
        batch_script = batch_script.replace(
            f"#SBATCH -p {PARTITION}\n", f"#SBATCH -p {PARTITION}\n#SBATCH --reservation={reservation}\n"
        )

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
            save_root=SAVE_ROOT,
            reservation=args.reservation,
            enable_ib=args.enable_ib == "yes",
            use_compile=args.compile,
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
        "--dataset_choice", type=str, default="all", choices=["lm", "em", "hp", "all"],
        help="The choice of datasets for joint training.",
    )
    parser.add_argument(
        "-r", "--reservation", type=str, default=None, help="Slurm reservation to submit under, if any."
    )
    parser.add_argument(
        "--enable_ib", type=str, default="yes", choices=["yes", "no"], help="Use IB verbs instead of sockets."
    )
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True,
        help="Compile the encoder, the decoder and the loss with torch.compile. The first iterations take longer.",
    )
    parser.add_argument(
        "--dry", action="store_true", help="Whether to only write the sbatch scripts without submitting them."
    )
    args = parser.parse_args()

    main(args)
