"""Submit the calibration and the training jobs of the decoder architecture evaluation.

Every job runs from a frozen copy of this folder (<root>/jobs/code_<timestamp>), so later edits cannot change a
pending or running job, and its sbatch script is kept next to it.

    python submit_decoder_arch.py calibrate --arms A B C
    python summarize_calibration.py                       # -> the iteration count N
    python submit_decoder_arch.py train --arm A --seed 0 --iterations N
    python submit_decoder_arch.py evaluate --run armA_current_seed0 [--after <training job>]

'evaluate' chains three jobs with afterok: predict-val (GPU), tune (CPU slice), test (GPU), see
evaluate_decoder_arch.py. The validation crops must exist (evaluate_decoder_arch.py place-val-crops).
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decoder_arch_lib as lib  # noqa: E402

HEADER = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH -p {partition}
#SBATCH -G {gres}
#SBATCH -c {cpus}
#SBATCH --mem={mem}
#SBATCH -t {time}
#SBATCH --constraint=inet
#SBATCH -A nim00007
{qos}
#SBATCH -o {root}/logs/slurm/{job_name}_%j.out
#SBATCH -e {root}/logs/slurm/{job_name}_%j.err
{dependency}
set -eo pipefail
source ~/.bashrc
set -u
micromamba activate new-stack
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd {code}
nvidia-smi --query-gpu=name,memory.total --format=csv
"""


def freeze_code(root):
    """Copy this folder and the evaluation harness files that eval_lib imports to <root>/jobs/code_<timestamp>."""
    import eval_lib
    stamp = time.strftime("%Y%m%d_%H%M%S")
    code = os.path.join(root, "jobs", f"code_{stamp}")
    here = os.path.dirname(os.path.abspath(__file__))
    shutil.copytree(here, code, ignore=shutil.ignore_patterns("__pycache__", "harness"))
    os.makedirs(os.path.join(code, "harness"))
    for name in eval_lib.HARNESS_FILES:
        shutil.copy2(os.path.join(eval_lib.EVALUATION_DIR, name), os.path.join(code, "harness", name))
    return code, stamp


# The resources of the evaluation stages: (partition, gres, cpus, mem, time, qos). Estimates from A0 at epoch 40:
# predict-val ~10 min on an A100; tune ~18 CPU-hours (2D 60 s per image, 3D ~40 min per crop), ~1.2 h on the 16
# workers of the usual CPU preset plus the edge rounds; test ~460 3D crops at ~5 s prediction each plus ~3000 2D
# images, ~1-1.5 h.
EVALUATION_STAGES = {
    "predict-val": ("grete:shared", "A100:1", 8, "64G", "01:00:00", "2h"),
    # 2D + 3D tuning took 1.6-2.5 h in step 1 and timed out twice at 2 h, so it gets 4 h (no '2h' QoS).
    "tune": ("grete:preemptible", "1g.10gb:1", 16, "64G", "04:00:00", None),
    "test": ("grete:shared", "A100:1", 16, "128G", "02:00:00", "2h"),
}


def submit(root, job_name, body, args, code, stamp, after=None):
    after = args.after if after is None else after
    dependency = f"#SBATCH --dependency=afterok:{after}" if after else ""
    script = HEADER.format(
        job_name=job_name, partition=args.partition, gres=args.gres, cpus=args.cpus, mem=args.mem, time=args.time,
        root=root, code=code, dependency=dependency, qos=f"#SBATCH --qos={args.qos}" if args.qos else "",
    ) + body
    path = os.path.join(root, "jobs", f"{job_name}_{stamp}.sh")
    with open(path, "w") as f:
        f.write(script)
    if args.dry_run:
        print(script)
        return f"<{job_name}>"
    out = subprocess.check_output(["sbatch", "--parsable", path], text=True).strip()
    print(f"submitted {job_name}: job {out} ({path})")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["calibrate", "train", "evaluate"])
    parser.add_argument("--run", help="evaluate: the run, e.g. armA_current_seed0")
    parser.add_argument("--stages", nargs="+", default=list(EVALUATION_STAGES), choices=list(EVALUATION_STAGES))
    parser.add_argument("--arms", nargs="+", default=list(lib.STEP1_ARMS), help="calibrate: the arms to time")
    parser.add_argument("--calibration-iterations", type=int, default=200)
    parser.add_argument("--arm", choices=sorted(lib.ARMS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--partition", default="grete:shared")
    parser.add_argument("--gres", default="A100:1")
    parser.add_argument("--cpus", type=int, default=16)
    parser.add_argument("--mem", default="64G")
    parser.add_argument("--time", default=None)
    parser.add_argument("--after", default=None, help="afterok dependency (job id)")
    parser.add_argument("--qos", default=None, help="QoS; calibrate defaults to '2h' (jobs up to 2 h start sooner)")
    parser.add_argument("--root", default=lib.EXPERIMENT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.makedirs(os.path.join(args.root, "jobs"), exist_ok=True)
    os.makedirs(os.path.join(args.root, "logs", "slurm"), exist_ok=True)
    code, stamp = freeze_code(args.root)
    workers = f"--n-workers {args.cpus - 2} --val-workers 4"

    if args.mode == "calibrate":
        args.time = args.time or "01:30:00"
        args.qos = args.qos or "2h"
        body = "".join(
            f"python train_decoder_arch.py --arm {arm} --calibrate {args.calibration_iterations} {workers}\n"
            for arm in args.arms
        ) + "python summarize_calibration.py\n"  # N from the step-1 arms; prints step-2 hours
        submit(args.root, "decarch_calibrate", body, args, code, stamp)
    elif args.mode == "evaluate":
        if args.run is None:
            raise SystemExit("evaluate needs --run")
        import eval_lib
        if not os.path.exists(eval_lib.VAL_CROPS_PATH):
            message = f"{eval_lib.VAL_CROPS_PATH} is missing: run 'evaluate_decoder_arch.py place-val-crops'."
            if not args.dry_run:
                raise SystemExit(message)
            print(f"WARNING: {message}")
        after = args.after
        for stage in args.stages:
            args.partition, args.gres, args.cpus, args.mem, args.time, args.qos = EVALUATION_STAGES[stage]
            options = {"predict-val": "", "tune": f" --workers {args.cpus}", "test": f" --workers {args.cpus - 2}"}
            body = f"python evaluate_decoder_arch.py {stage} --run {args.run}{options[stage]}\n"
            after = submit(args.root, f"decarch_{stage}_{args.run}", body, args, code, stamp, after=after or "")
    else:
        if args.arm is None or args.iterations is None:
            raise SystemExit("train needs --arm and --iterations")
        args.time = args.time or "08:00:00"
        body = (f"python train_decoder_arch.py --arm {args.arm} --seed {args.seed} --iterations {args.iterations} "
                f"{workers}\n")
        submit(args.root, f"decarch_{args.arm}{args.seed}", body, args, code, stamp)


if __name__ == "__main__":
    main()
