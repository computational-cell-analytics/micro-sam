"""Submit the parameter search and the test of a decoder checkpoint as a few Slurm jobs.

One job per setting runs the whole pipeline of the setting on one small GPU slice: the predictions of the tuning
data, the parameter search on the CPUs of the job and the test with the parameters it found (the commands of
tune_decoder.py and evaluate_decoder.py). The 2d and 3d settings get a second job, which takes tasks of the same
search and scores the other half of the test samples. A further job computes the reference results (micro-sam2 hvit_t,
library defaults) that have no result with the current scoring, see `evaluate_decoder.py reference`.
With '--reference_search', one more job per setting searches the parameters of the reference model itself (the
registry decoder of hvit_t_cells, same tuning data, grid and stages) and tests it with them. These results are named
'tuned' and lie next to the ones with the library defaults.

    python submit_checkpoint_evaluation.py --checkpoint <root>/checkpoints/hiera_u_hvit_t_cells/best.pt [--dry-run]
    python submit_checkpoint_evaluation.py --checkpoint <staged checkpoint> --data dense --reference_search dense
    # Afterwards:
    python evaluate_decoder.py reference summary
    python evaluate_decoder.py report -e <root>/evaluation/<staged checkpoint> [--reference_parameters tuned]

The jobs run from a copy of the code that includes the packages micro_sam and torch_em, so another branch can be
checked out in their working trees while the jobs run. No job waits for the result of another one, and a job that
is preempted and started again continues: stored predictions, finished sweeps and the result rows of scored samples
are kept. The sizes of the jobs (JOBS) come from
the searches of the epoch 25 and epoch 29 checkpoints: about 1200 - 1400 (2d, with the worker jobs), 970 (3d, 105
crops) and 17 (dense) core hours, 1.6 - 2.9 GB per running task, and 3 (2d), 7 (3d) and 3 (dense) GPU hours for the
test.
"""

import argparse
import json
import os
import shutil
import subprocess
import time

import evaluate_decoder
import tune_decoder

# The CPUs and the memory of the jobs of a setting, and the jobs that share its search and test.
JOBS = {
    "2d": {"cpus": 32, "mem": "48G", "n_jobs": 2},
    "3d": {"cpus": 32, "mem": "64G", "n_jobs": 2},
    "dense": {"cpus": 16, "mem": "40G", "n_jobs": 1},
    "reference": {"cpus": 8, "mem": "32G", "n_jobs": 1},
    "reference_search": {"cpus": 16, "mem": "40G", "n_jobs": 1},
}
# The registry model of the reference, see `tune_decoder.REGISTRY_MODELS`.
REFERENCE_MODEL = "hvit_t_cells"
# A job that only takes tasks of a search ends after this time without an open task (the search job is gone).
WORKER_MAX_IDLE = 3 * 3600

HEADER = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH -p {partition}
#SBATCH -G {gres}
#SBATCH -c {cpus}
#SBATCH --mem={mem}
#SBATCH -t {time}
#SBATCH --constraint=inet
#SBATCH -A nim00007
#SBATCH --requeue
#SBATCH -o {logs}/{job_name}_%j.out
#SBATCH -e {logs}/{job_name}_%j.err
set -eo pipefail
source ~/.bashrc
set -u
micromamba activate new-stack
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd {code}
# The copies of micro_sam and torch_em in the code folder, see `evaluate_decoder.freeze_code`.
export PYTHONPATH={code}${{PYTHONPATH:+:$PYTHONPATH}}
{commands}
"""


def stage(checkpoint_path, experiment_folder):
    """Stage the checkpoint and copy it into the experiment folder, so that no job has to stage it."""
    staged, checksum = tune_decoder.resolve_checkpoint(checkpoint_path)
    folder = os.path.join(experiment_folder, "checkpoints")
    os.makedirs(folder, exist_ok=True)
    copy = os.path.join(folder, os.path.basename(staged))
    if not os.path.exists(copy) or not os.path.exists(copy + ".json"):
        shutil.copyfile(staged, copy + ".tmp")
        os.replace(copy + ".tmp", copy)
        # The sidecar file marks a staged copy and holds the checksum of the training checkpoint.
        shutil.copyfile(staged + ".json", copy + ".json")
    return copy, checksum


def setting_commands(checkpoint, experiment_folder, data, datasets, job_index):
    """The commands of one job of a setting: predictions, search (or tasks of the search) and test."""
    cpus, n_jobs = JOBS[data]["cpus"], JOBS[data]["n_jobs"]
    session = tune_decoder.session_folder(checkpoint, data)
    result = os.path.join(session, "best_parameters.json")
    session_args = f"--checkpoint {checkpoint} --data {data} -e {session}"
    if job_index == 0:
        commands = [
            f"python tune_decoder.py predict {session_args}",
            # A job that is started again during its test does not search again.
            f"[ -f {result} ] || python tune_decoder.py search {session_args} --threads {cpus} "
            f"--slots {cpus // 2} --retry_failed",
        ]
    else:
        commands = [
            f"[ -f {result} ] || python tune_decoder.py worker -e {session} --threads {cpus} --slots {cpus // 2} "
            f"--max_idle {WORKER_MAX_IDLE}",
            f'[ -f {result} ] || {{ echo "The search has no result: {result}"; exit 1; }}',
        ]
    # The shards store a row per sample, so a job that is started again continues. A job scores its own shard
    # first and then the others, which only fills in what another job left open.
    for shard in [(job_index + offset) % n_jobs for offset in range(n_jobs)]:
        commands.append(
            f"python evaluate_decoder.py run --checkpoint {checkpoint} -e {experiment_folder} --params {result} "
            f"--n_shards {n_jobs} --shard_index {shard} --skip_failed --datasets {' '.join(datasets)}"
        )
    return commands


def reference_commands(evaluation_common):
    """The commands that compute the reference results without a result with the current scoring."""
    datasets = [
        dataset for dataset, *_ in evaluate_decoder.evaluation_datasets()
        if evaluate_decoder.is_recomputed(dataset, evaluation_common) and not (
            (path := evaluate_decoder._reference_result(dataset))
            and evaluation_common.result_is_current(path, dataset)
        )
    ]
    if not datasets:
        return datasets, []
    _, decoder_path = evaluation_common.registry_weight_paths("hvit_t_cells")
    command = (
        f"python evaluation/evaluate_automatic_segmentation.py -e {evaluate_decoder.REFERENCE_ROOT} "
        f"-m {evaluate_decoder.DEFAULT_MODEL_TYPE} --mode ais --skip_tuning -c {decoder_path} "
        '-d "$dataset" --n_shards 1 --shard_index 0'
    )
    return datasets, [
        f"for dataset in {' '.join(datasets)}; do",
        f'    {command} || echo "FAILED $dataset"',
        "done",
    ]


def reference_search_commands(evaluation_common, data, datasets):
    """The commands that search the parameters of the reference model for one setting and test it with them.

    The search is a session of tune_decoder.py for the staged registry decoder. The test is the command of the
    reference results with the result of the search instead of the library defaults.
    """
    cpus = JOBS["reference_search"]["cpus"]
    staged, _ = tune_decoder.resolve_checkpoint(REFERENCE_MODEL)
    session = tune_decoder.session_folder(staged, data)
    result = os.path.join(session, "best_parameters.json")
    session_args = f"--checkpoint {staged} --data {data} -e {session}"
    _, decoder_path = evaluation_common.registry_weight_paths(REFERENCE_MODEL)
    command = (
        f"python evaluation/evaluate_automatic_segmentation.py -e {evaluate_decoder.REFERENCE_ROOT} "
        f"-m {evaluate_decoder.DEFAULT_MODEL_TYPE} --mode ais -c {decoder_path} --params_file {result} "
        '-d "$dataset" --n_shards 1 --shard_index 0'
    )
    return [
        f"python tune_decoder.py predict {session_args}",
        f"[ -f {result} ] || python tune_decoder.py search {session_args} --threads {cpus} "
        f"--slots {cpus // 2} --retry_failed",
        f"for dataset in {' '.join(datasets)}; do",
        f'    {command} || echo "FAILED $dataset"',
        "done",
    ]


def sbatch(script_path, dependency=None, dry_run=False):
    command = ["sbatch", "--parsable"] + ([f"--dependency={dependency}"] if dependency else [])
    if dry_run:
        print("  " + " ".join(command + [script_path]))
        return "DRYRUN"
    return subprocess.run(command + [script_path], capture_output=True, text=True, check=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="The training checkpoint, or its staged copy.")
    parser.add_argument("--data", nargs="+", default=list(tune_decoder.SETTINGS), choices=tune_decoder.SETTINGS)
    parser.add_argument("--skip_reference", action="store_true", help="Do not submit the job of the reference.")
    parser.add_argument(
        "--reference_search", nargs="+", default=[], choices=tune_decoder.SETTINGS,
        help="Also search the parameters of the reference model for these settings and test it with them.",
    )
    parser.add_argument("--partition", default="grete:preemptible")
    parser.add_argument("--gres", default="1g.10gb:1")
    parser.add_argument("--time", default="2-00:00:00")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run", help="Only write the scripts.")
    args = parser.parse_args()

    _, evaluation_common = evaluate_decoder.import_evaluation()
    name = os.path.basename(tune_decoder.resolve_checkpoint(args.checkpoint)[0])[:-len(".pt")]
    experiment_folder = os.path.join(evaluate_decoder.SAVE_ROOT, "evaluation", name)
    checkpoint, checksum = stage(args.checkpoint, experiment_folder)
    code = evaluate_decoder.freeze_code(experiment_folder)
    suffix = "epoch" + name.rsplit("_epoch", 1)[-1]
    print(f"Checkpoint {checkpoint} ({checksum}), code {code}")

    def write(job_name, key, logs, commands):
        os.makedirs(logs, exist_ok=True)
        path = os.path.join(code, f"{job_name}.sbatch")
        with open(path, "w") as f:
            f.write(HEADER.format(
                job_name=job_name, partition=args.partition, gres=args.gres, cpus=JOBS[key]["cpus"],
                mem=JOBS[key]["mem"], time=args.time, logs=logs, code=code, commands="\n".join(commands),
            ))
        return path

    def setting_datasets(data):
        return [
            dataset for dataset, *_ in evaluate_decoder.evaluation_datasets()
            if evaluate_decoder.data_type(dataset, evaluation_common) == data
        ]

    jobs = {}
    logs = os.path.join(experiment_folder, "logs", "slurm")
    for data in args.data:
        datasets = setting_datasets(data)
        first = None
        for job_index in range(JOBS[data]["n_jobs"]):
            job_name = f"deceval_{data}_{suffix}" + (f"_{job_index}" if JOBS[data]["n_jobs"] > 1 else "")
            commands = setting_commands(checkpoint, experiment_folder, data, datasets, job_index)
            path = write(job_name, data, logs, commands)
            # A job that takes tasks of a search starts once the job of the search has started.
            jobs[job_name] = sbatch(path, dependency=f"after:{first}" if job_index else None, dry_run=args.dry_run)
            first = first or jobs[job_name]
            print(f"{job_name}: job {jobs[job_name]}, {len(datasets)} test datasets")

    reference_logs = os.path.join(evaluate_decoder.REFERENCE_ROOT, "logs", "slurm")
    for data in args.reference_search:
        datasets = setting_datasets(data)
        job_name = f"decref_search_hvit_t_{data}"
        path = write(job_name, "reference_search", reference_logs, reference_search_commands(
            evaluation_common, data, datasets
        ))
        jobs[job_name] = sbatch(path, dry_run=args.dry_run)
        print(f"{job_name}: job {jobs[job_name]}, {len(datasets)} test datasets")

    if not args.skip_reference:
        datasets, commands = reference_commands(evaluation_common)
        if commands:
            path = write("decref_hvit_t", "reference", reference_logs, commands)
            jobs["decref_hvit_t"] = sbatch(path, dry_run=args.dry_run)
            print(f"decref_hvit_t: job {jobs['decref_hvit_t']}, {len(datasets)} datasets: {datasets}")
        else:
            print("Every recomputed dataset has a current reference result.")

    if not args.dry_run:
        with open(os.path.join(experiment_folder, f"jobs_{time.strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
            json.dump({"checkpoint": checkpoint, "checksum": checksum, "code": code, "jobs": jobs}, f, indent=1)


if __name__ == "__main__":
    main()
