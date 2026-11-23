"""Submit the calibration and the training of a decoder to a H100 (default: the shared queue of grete).

Every job runs from a frozen copy of this folder (<root>/jobs/code_<timestamp>), so that later edits cannot change
a pending or running job. The sbatch script is kept next to it.

    python submit_decoder_training.py calibrate [--z_slices 14 8 16 24 32 --time 04:00:00]
    python submit_decoder_training.py train --batch_size_2d <B> --z_slice <Z> [--n_jobs 2]
    python submit_decoder_training.py train ... --resume [--after <job id>]

'calibrate' checks the data (check_data.py) and then runs calibrate_decoder.py. 'train' submits a chain of
'--n_jobs' jobs: the first one starts the training (or resumes it with '--resume'), each further job resumes from
the latest checkpoint after the previous job ended. A job that finds a finished training exits immediately.
Use '--dry-run' to only write the scripts.
"""

import argparse
import os
import shutil
import subprocess
import time

import common

HEADER = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH -p {partition}
#SBATCH -G H100:1
#SBATCH -c {cpus}
#SBATCH --mem={mem}
#SBATCH -t {time}
#SBATCH --constraint=inet
#SBATCH -A {account}
#SBATCH -o {root}/logs/slurm/{job_name}_%j.out
#SBATCH -e {root}/logs/slurm/{job_name}_%j.err
{extra}
set -eo pipefail
source ~/.bashrc
set -u
micromamba activate {env}
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=134217728
export DECODER_SAVE_ROOT={root}
cd {code}
nvidia-smi --query-gpu=name,memory.total --format=csv
"""


def freeze_code(root):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    code = os.path.join(root, "jobs", f"code_{stamp}")
    shutil.copytree(os.path.dirname(os.path.abspath(__file__)), code, ignore=shutil.ignore_patterns("__pycache__"))
    return code, stamp


def submit(
    root, code, job_name, commands, time_limit, cpus, mem, extra="", dry_run=False,
    partition="grete-h100:shared", account="nim00007", env="new-stack",
):
    os.makedirs(os.path.join(root, "logs", "slurm"), exist_ok=True)
    script = HEADER.format(
        job_name=job_name, root=root, code=code, time=time_limit, cpus=cpus, mem=mem, extra=extra,
        partition=partition, account=account, env=env,
    ) + "\n".join(commands) + "\n"
    script_path = os.path.join(root, "jobs", f"{job_name}_{os.path.basename(code)[len('code_'):]}.sh")
    with open(script_path, "w") as f:
        f.write(script)
    if dry_run:
        print(f"Dry run, wrote {script_path}:\n{script}")
        return None
    output = subprocess.run(["sbatch", "--parsable", script_path], capture_output=True, text=True, check=True)
    job_id = output.stdout.strip().split(";")[0]
    print(f"Submitted {job_name} as job {job_id} ({script_path})")
    return job_id


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["calibrate", "train"])
    parser.add_argument("--model_type", default="hvit_t_cells")
    parser.add_argument("--decoder", default="hiera_u")
    parser.add_argument("--batch_size_2d", type=int, default=None)
    parser.add_argument("--z_slice", type=int, default=None)
    parser.add_argument("--z_slices", type=int, nargs="+", default=[14],
                        help="The numbers of slices for the data check of the calibration.")
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--scheduler_patience", type=int, default=None,
                        help="The patience of ReduceLROnPlateau for the training. Default: see common.py.")
    parser.add_argument("--partition", default="grete-h100:shared", help="E.g. 'kisski-h100'.")
    parser.add_argument("--account", default="nim00007")
    parser.add_argument("--env", default="new-stack", help="The micromamba environment.")
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--label_trafo_threads", type=int, default=6)
    parser.add_argument("--cpus", type=int, default=48)
    parser.add_argument("--mem", default="250G")
    parser.add_argument("--time", default=None, help="The time limit. Default: 2 h (calibrate), 48 h (train).")
    parser.add_argument("--qos", default=None, help="The QoS. Default: '2h' for calibrate (starts earlier).")
    parser.add_argument("--n_jobs", type=int, default=1, help="The number of chained training jobs.")
    parser.add_argument("--resume", action="store_true", help="Let the first training job resume.")
    parser.add_argument("--after", default=None, help="A job that has to end before the first job starts.")
    parser.add_argument("--skip_data_check", action="store_true")
    parser.add_argument("--root", default=common.SAVE_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.root, "jobs"), exist_ok=True)
    code, _ = freeze_code(args.root)
    workers = f"--n_workers {args.n_workers} --label_trafo_threads {args.label_trafo_threads}"
    model = f"--model_type {args.model_type} --decoder {args.decoder}"
    resources = {
        "cpus": args.cpus, "mem": args.mem, "dry_run": args.dry_run,
        "partition": args.partition, "account": args.account, "env": args.env,
    }

    if args.mode == "calibrate":
        commands = []
        if not args.skip_data_check:
            z_slices = " ".join(str(z) for z in args.z_slices)
            commands.append(
                f"python check_data.py --z_slices {z_slices} --only_3d_after_first --n_workers {args.cpus - 4}"
            )
        settings = ""
        if args.batch_size_2d is not None and args.z_slice is not None:
            settings = f" --batch_size_2d {args.batch_size_2d} --z_slice {args.z_slice}"
            commands.append(f"python calibrate_decoder.py loop {model} {workers}{settings}")
        else:
            commands.append(f"python calibrate_decoder.py both {model} {workers}")
        qos = "2h" if (args.qos is None and args.time is None) else args.qos
        submit(args.root, code, f"deccal_{args.decoder}_{args.model_type}", commands, args.time or "02:00:00",
               extra="" if qos is None else f"#SBATCH --qos={qos}", **resources)
        return

    if args.batch_size_2d is None or args.z_slice is None:
        raise ValueError("The training needs --batch_size_2d and --z_slice, see the calibration.")
    command = (
        f"python train_decoder.py {model} {workers} --batch_size_2d {args.batch_size_2d} --z_slice {args.z_slice}"
    )
    if args.n_epochs is not None:
        command += f" --n_epochs {args.n_epochs}"
    if args.scheduler_patience is not None:
        command += f" --scheduler_patience {args.scheduler_patience}"
    previous = args.after
    for i in range(args.n_jobs):
        resume = args.resume or i > 0
        # The next job has to start after a time out as well, hence 'afterany'.
        extra = "" if previous is None else f"#SBATCH --dependency=afterany:{previous}"
        if args.qos is not None:
            extra += f"\n#SBATCH --qos={args.qos}"
        previous = submit(
            args.root, code, f"dectrain_{args.decoder}_{args.model_type}_{i}",
            [command + (" --resume" if resume else "")], args.time or "48:00:00", extra=extra, **resources,
        )
        if previous is None:  # Dry run.
            previous = "<previous job>"
        time.sleep(1.1)  # The script names contain the time stamp of the code copy; keep the jobs apart.


if __name__ == "__main__":
    main()
