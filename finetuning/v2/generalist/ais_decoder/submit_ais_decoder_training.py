"""Submit the AIS decoder trainings as single-GPU SLURM jobs on grete:shared.

    python submit_ais_decoder_training.py --variants baseline contact fgcal --iterations 30000 --batch-size 8 --dry
    python submit_ais_decoder_training.py --variants both --iterations 30000 --batch-size 8 --after 1234 1235 1236

Writes <save-root>/jobs/<timestamp>_<variant>.sh and a submit.json with the job ids. `--after` makes the job
start only after the listed jobs have started (SLURM 'after' dependency), which is how the fourth model waits
for the other three.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ais_decoder_lib as lib  # noqa: E402

REPOSITORY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
TRAIN_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_ais_decoder.py")

TEMPLATE = """#!/bin/bash
#SBATCH --job-name=ais_decoder_{variant}
#SBATCH -p {partition}
#SBATCH -G {gres}
#SBATCH -c {cpus}
#SBATCH --mem={mem}
#SBATCH -t {time}
#SBATCH --constraint=inet
#SBATCH -A {account}
#SBATCH -o {log_dir}/ais_decoder_{variant}_%j.out
#SBATCH -e {log_dir}/ais_decoder_{variant}_%j.err
{dependency}
set -eo pipefail
source ~/.bashrc
set -u
micromamba activate {env}
cd {repository}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
nvidia-smi --query-gpu=name,memory.total --format=csv
python {script} --variant {variant} --iterations {iterations} --batch-size {batch_size} \\
    --n-workers {n_workers} --val-workers {val_workers} --lr {lr} --epoch-scale {epoch_scale} \\
    --save-root {save_root} {extra}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", required=True, choices=sorted(lib.VARIANTS))
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-workers", type=int, default=12)
    parser.add_argument("--val-workers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epoch-scale", type=float, default=1.0)
    parser.add_argument("--save-root", default=lib.CAMPAIGN_ROOT)
    parser.add_argument("--partition", default="grete:shared")
    parser.add_argument("--gres", default="A100:1")
    parser.add_argument("--cpus", type=int, default=16)
    parser.add_argument("--mem", default="64G")
    parser.add_argument("--time", default="12:00:00")
    parser.add_argument("--account", default="nim00007")
    parser.add_argument("--env", default="new-stack")
    parser.add_argument("--after", nargs="*", default=None, help="Job ids this job waits for (start after they start).")
    parser.add_argument("--extra", default="", help="Extra arguments for train_ais_decoder.py, verbatim.")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    job_dir = os.path.join(args.save_root, "jobs")
    log_dir = os.path.join(args.save_root, "logs", "slurm")
    os.makedirs(job_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    dependency = f"#SBATCH --dependency=after:{':'.join(args.after)}" if args.after else ""
    submitted = {}
    for variant in args.variants:
        script = TEMPLATE.format(
            variant=variant, partition=args.partition, gres=args.gres, cpus=args.cpus, mem=args.mem, time=args.time,
            account=args.account, log_dir=log_dir, dependency=dependency, env=args.env, repository=REPOSITORY_ROOT,
            script=TRAIN_SCRIPT, iterations=args.iterations, batch_size=args.batch_size, n_workers=args.n_workers,
            val_workers=args.val_workers, lr=args.lr, epoch_scale=args.epoch_scale, save_root=args.save_root,
            extra=args.extra,
        )
        script_path = os.path.join(job_dir, f"{stamp}_{variant}.sh")
        with open(script_path, "w") as f:
            f.write(script)
        if args.dry:
            print(f"--- {script_path} ---\n{script}")
            continue
        job_id = subprocess.check_output(["sbatch", "--parsable", script_path], text=True).strip().split(";")[0]
        submitted[variant] = job_id
        print(f"submitted {variant}: job {job_id} ({script_path})")
    if submitted:
        record = {"timestamp": stamp, "argv": sys.argv, "jobs": submitted}
        with open(os.path.join(job_dir, f"{stamp}_submit.json"), "w") as f:
            json.dump(record, f, indent=2)


if __name__ == "__main__":
    main()
