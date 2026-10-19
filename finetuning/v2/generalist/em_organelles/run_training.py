import os
import shutil
import argparse
import subprocess
from datetime import datetime


# The training script of each domain.
SCRIPT_DIR = "/mnt/vast-kisski/home/archit/u28048/micro-sam/finetuning/v2/generalist/em_organelles"
SCRIPTS = {
    "em_mito": os.path.join(SCRIPT_DIR, "train_em_mito.py"),
    "em_nucleus": os.path.join(SCRIPT_DIR, "train_em_nucleus.py"),
}
# The epochs keep a margin to the 96h walltime; one epoch takes about 30-38 min on 2 x 4 H100.
EPOCHS = {
    "em_mito": {"hvit_t": 140, "hvit_s": 140, "hvit_b": 150, "hvit_l": 150},
    "em_nucleus": {"hvit_t": 160, "hvit_s": 160, "hvit_b": 190, "hvit_l": 190},
}
MODEL_TYPES = ("hvit_t", "hvit_s", "hvit_b", "hvit_l")
BOUNDARY_DICE_WEIGHT = 0.5
ENABLE_IB = True

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

PARTITION = "kisski-h100"
GPU_TYPE = "H100"


def write_batch_script(out_path, domain, model_type, n_epochs, save_root, reservation, dry, tag):
    """Write the sbatch script for one training run of a domain on 2 nodes x 4 H100, and submit it."""
    nccl_block = "\n".join(f"export {key}={value}" for key, value in NCCL_ENV[ENABLE_IB].items())

    batch_script = rf"""#!/bin/bash
#SBATCH --job-name=μSAM2_{domain}_{model_type}{"_" + tag if tag else ""}
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
# A requeued job restarts from epoch 0 and overwrites the checkpoints; a failed job must be resumed by hand.
#SBATCH --no-requeue

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
export RUN_TAG={tag or ""}

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
    {SCRIPTS[domain]} --model_type {model_type} --n_epochs {n_epochs} --compile \
    --boundary_dice_weight {BOUNDARY_DICE_WEIGHT}"
"""
    if not tag:
        batch_script = batch_script.replace("export RUN_TAG=\n", "")
    if reservation:
        batch_script = batch_script.replace(
            f"#SBATCH -p {PARTITION}\n", f"#SBATCH -p {PARTITION}\n#SBATCH --reservation={reservation}\n"
        )

    script_path = out_path[:-3] + f"_{domain}_{model_type}.sh"
    with open(script_path, "w") as f:
        f.write(batch_script)

    if not dry:
        subprocess.run(["sbatch", script_path])


def get_batch_script_names(tmp_folder):
    """Return a unique path for a new sbatch script in the given folder."""
    tmp_folder = os.path.expanduser(tmp_folder)
    os.makedirs(tmp_folder, exist_ok=True)
    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    return os.path.join(tmp_folder, f"organelle-sam2-multi-node{dt}.sh")


def submit_slurm(args):
    """Submit the multi-node training jobs of one domain to slurm."""
    tmp_folder = "./gpu_jobs"
    models = MODEL_TYPES if args.model_type is None else [args.model_type]
    for model_type in models:
        print(f"Submitting {args.domain} training for {model_type}")
        write_batch_script(
            out_path=get_batch_script_names(tmp_folder),
            domain=args.domain,
            model_type=model_type,
            n_epochs=EPOCHS[args.domain][model_type],
            save_root=os.path.abspath(args.save_root),
            reservation=args.reservation,
            dry=args.dry,
            tag=args.tag,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--domain", required=True, choices=list(SCRIPTS), help="The training domain.")
    parser.add_argument(
        "-m", "--model_type", type=str, default=None, choices=MODEL_TYPES,
        help="The model type. Submits all four models if not given.",
    )
    parser.add_argument("-r", "--reservation", type=str, default=None, help="The slurm reservation to submit under.")
    parser.add_argument(
        "-s", "--save_root", type=str, default=os.getcwd(),
        help="Where to save checkpoints and logs. Defaults to the current working directory.",
    )
    parser.add_argument("--tag", type=str, default=None, help="Run tag, added to the run and job names.")
    parser.add_argument("--dry", action="store_true", help="Write the sbatch scripts but do not submit them.")
    args = parser.parse_args()

    tmp_dir = "./gpu_jobs"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    submit_slurm(args)


if __name__ == "__main__":
    main()
