"""Submit the APG cross-dataset stages to slurm, with a dependency-gated merge per dataset."""

import argparse
import subprocess
from pathlib import Path
from datetime import datetime

SHARED = Path(__file__).resolve().parent
SCRIPT = SHARED / "apg_crossdataset.py"

ACCOUNT = "nim00007"
PARTITION = "grete:preemptible"
TIME_LIMIT = "04:00:00"


def write_script(job_folder, tag, command, cpus, memory, dependency=None):
    path = job_folder / f"{tag}.sh"
    dep = f"#SBATCH --dependency=afterok:{dependency}\n" if dependency else ""
    path.write_text(f"""#!/bin/bash
#SBATCH -c {cpus}
#SBATCH --mem {memory}
#SBATCH -t {TIME_LIMIT}
#SBATCH -p {PARTITION}
#SBATCH -G 1
#SBATCH -A {ACCOUNT}
#SBATCH --job-name={tag}
#SBATCH --constraint=inet
#SBATCH -o {job_folder}/logs/{tag}_%j.out
#SBATCH -e {job_folder}/logs/{tag}_%j.err
{dep}
source ~/.bashrc
micromamba activate super

python {SCRIPT} {command}
""")
    return path


def submit(path, dry):
    if dry:
        print(f"  [dry] {path}")
        return None
    result = subprocess.run(["sbatch", str(path)], capture_output=True, text=True)
    output = (result.stdout or result.stderr).strip()
    print(f"  {output}")
    return output.split()[-1] if output.startswith("Submitted batch job") else None


def submit_tune(args, job_folder, dataset):
    """Shards of the val grid-search, then a merge gated on all of them."""
    ids = []
    suffix = f" --round {args.round}" if args.stage == "tune" else ""
    prefix = "tune" + args.round if args.stage == "tune" else "ais"
    for shard in range(args.n_shards):
        tag = f"{prefix}_{dataset}_s{shard}"
        command = (
            f"--stage {args.stage} -d {dataset}{suffix} --shard {shard} --n_shards {args.n_shards}"
        )
        path = write_script(job_folder, tag, command, args.cpus, args.memory)
        job_id = submit(path, args.dry)
        if job_id:
            ids.append(job_id)

    if args.dry or len(ids) != args.n_shards:
        print(f"  not submitting the merge for {dataset}")
        return None
    command = f"--stage {args.stage} -d {dataset}{suffix} --n_shards {args.n_shards} --merge"
    path = write_script(
        job_folder, f"{prefix}_{dataset}_merge", command, args.cpus, args.memory,
        dependency=":".join(ids),
    )
    return submit(path, args.dry)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="tune", choices=["tune", "ais_tune", "box", "mcs", "test"])
    parser.add_argument("-d", "--datasets", nargs="+", default=["livecell", "deepbacs", "dynamicnuclearnet", "dsb"])
    parser.add_argument("--methods", nargs="+", default=["apg_v2", "ais_v2", "ais_v1", "apg_v1"])
    parser.add_argument("--n_shards", type=int, default=15)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--memory", default="32G")
    parser.add_argument("--round", default="1", choices=["1", "wide"])
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    job_folder = SHARED / "gpu_jobs_apg_crossdataset" / datetime.now().strftime("%Y%m%d_%H%M%S")
    (job_folder / "logs").mkdir(parents=True, exist_ok=True)
    print(f"job folder: {job_folder}")

    for dataset in args.datasets:
        print(f"{dataset}:")
        if args.stage in ("tune", "ais_tune"):
            submit_tune(args, job_folder, dataset)
        elif args.stage in ("box", "mcs"):
            command = f"--stage {args.stage} -d {dataset}"
            tag = f"{args.stage}_{dataset}"
            submit(write_script(job_folder, tag, command, args.cpus, args.memory), args.dry)
        else:
            for method in args.methods:
                ids = []
                for shard in range(args.n_shards):
                    tag = f"test_{dataset}_{method}_s{shard}"
                    command = (
                        f"--stage test -d {dataset} --method {method}"
                        f" --shard {shard} --n_shards {args.n_shards}"
                    )
                    job_id = submit(write_script(job_folder, tag, command, args.cpus, args.memory), args.dry)
                    if job_id:
                        ids.append(job_id)
                if args.dry or len(ids) != args.n_shards:
                    continue
                command = f"--stage test -d {dataset} --method {method} --n_shards {args.n_shards} --merge"
                path = write_script(
                    job_folder, f"test_{dataset}_{method}_merge", command, args.cpus, args.memory,
                    dependency=":".join(ids),
                )
                submit(path, args.dry)


if __name__ == "__main__":
    main()
