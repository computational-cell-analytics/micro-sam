"""Search the AIS post-processing parameters for a decoder trained on a frozen uSAM2 encoder.

The search is the one of 'finetuning/v2/evaluation/parameter_search.py' (`search`): the broad grid on the tuning data
of every in-domain dataset, rounds past the grid edges, the boundary stage and, for the sparse settings, the
'min_size' stage that gives the best 'min_size' per dataset. It runs once per setting: '2d' (flow), '3d' (flow) and
'dense' (multicut), see TUNING_DATA.md. 'parameter_search.py run' loads registry models; this script predicts with
the decoder of a training checkpoint instead, in blocks of 14 slices for volumes like its evaluation
(evaluate_decoder.py), and scores from the cached predictions on CPUs.

    # The three sessions as Slurm jobs: per setting a GPU job that predicts and a CPU job that searches.
    python tune_decoder.py submit --checkpoint <root>/checkpoints/hiera_u_hvit_t_cells/best.pt [--dry-run]
    # Only the search jobs, if the predictions were stored with 'predict' before (they take minutes on one GPU).
    python tune_decoder.py submit --checkpoint <staged checkpoint> --skip_predict

    # The steps of one session, here.
    python tune_decoder.py predict --checkpoint <staged checkpoint> --data 3d -e <session folder>
    python tune_decoder.py search --checkpoint <staged checkpoint> --data 3d -e <session folder>
    # Further (small) jobs that take tasks of a running search, so that it needs no full node.
    python tune_decoder.py worker -e <session folder> --threads 16

A session writes '<session folder>/best_parameters.json' with the best parameters over the datasets and the
'min_size' per dataset, which 'evaluate_decoder.py run --params' reads. Finished sweeps are kept, so a search that is
started again continues. The sessions of a checkpoint are in '<DECODER_SAVE_ROOT>/tuning/<staged checkpoint>/<data>'.

The same search runs for the decoder of a registry model with its name as checkpoint (REGISTRY_MODELS), e.g. for the
reference of the decoder evaluation. It predicts with the library defaults, like the reference results, and
'evaluate_automatic_segmentation.py --params_file' reads its result.
    python tune_decoder.py predict --checkpoint hvit_t_cells --data dense
    python tune_decoder.py search --checkpoint hvit_t_cells --data dense
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import warnings

import evaluate_decoder
import tune_set

HERE = os.path.dirname(os.path.abspath(__file__))
TUNING_ROOT = os.path.join(evaluate_decoder.SAVE_ROOT, "tuning")
# The name of the model in the search. Its first six characters name the folders of the sweeps, like the result files
# of the evaluation.
MODEL = evaluate_decoder.DEFAULT_MODEL_TYPE
SETTINGS = ("2d", "3d", "dense")
# Written next to the cached predictions of a dataset once all of them are there.
COMPLETE = "complete.json"
# The registry models whose decoder can be searched like a checkpoint: pass the name as '--checkpoint'. The reference
# of the decoder evaluation is hvit_t_cells (`evaluate_decoder.REFERENCE_ROOT`).
REGISTRY_MODELS = ("hvit_t_cells",)
# With this variable set, the search runs on small grids, to check the pipeline.
CHECK_ENV = "TUNE_DECODER_CHECK"
CHECK_GRIDS = {
    "sparse": {
        "foreground_threshold": [0.5], "density_threshold": [10.0, 30.0], "min_size": [25, 50], "sigma": [0.5],
        "n_iter": [50], "dt": [1.0], "foreground_weight": [0.5, 0.9],
    },
    "dense": {"beta": [0.4, 0.5], "density_threshold": [3.0], "sigma": [0.5], "n_iter": [25, 50], "dt": [0.5]},
}


def import_search():
    """The modules 'parameter_search' and its 'common', with the small grids of a check if CHECK_ENV is set."""
    search, search_common = tune_set.import_search()
    if os.environ.get(CHECK_ENV):
        search.BROAD_GRIDS_2D["sparse"] = CHECK_GRIDS["sparse"]
        search.BROAD_GRIDS_3D["sparse_3d"] = CHECK_GRIDS["sparse"]
        search.BROAD_GRIDS_3D["dense"] = CHECK_GRIDS["dense"]
        search.BOUNDARY_TOP.update({"sparse": 2, "dense": 1})
    return search, search_common


def stage_registry_decoder(model_name):
    """Stage the decoder of a registry model, so that a session can search its parameters like those of a checkpoint
    of the decoder training.

    The copy is named '<model>_registry_<checksum>.pt'. Its sidecar file names the registry model, which
    `load_model` reads. The checksum is the one of the decoder file, i.e. the checkpoint id of its results in
    `evaluate_automatic_segmentation.py -c <decoder>`.
    """
    _, search_common = import_search()
    _, decoder_path = search_common.registry_weight_paths(model_name)
    checksum = search_common.checkpoint_checksum(decoder_path)
    folder = os.path.join(TUNING_ROOT, "checkpoints")
    os.makedirs(folder, exist_ok=True)
    staged = os.path.join(folder, f"{model_name}_registry_{checksum[:8]}.pt")
    if not os.path.exists(staged):
        shutil.copyfile(decoder_path, f"{staged}.{os.getpid()}.tmp")
        os.replace(f"{staged}.{os.getpid()}.tmp", staged)
    if not os.path.exists(staged + ".json"):
        with open(f"{staged}.json.{os.getpid()}.tmp", "w") as f:
            json.dump({
                "source": decoder_path, "checksum": checksum, "registry_model": model_name,
                "staged": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)
        os.replace(f"{staged}.json.{os.getpid()}.tmp", staged + ".json")
    return staged, checksum


def registry_model(staged):
    """The registry model a staged checkpoint is the decoder of, or None for a checkpoint of the decoder training."""
    with open(staged + ".json") as f:
        return json.load(f).get("registry_model")


def load_model(staged, device):
    """The model of a staged checkpoint and the arguments of its prediction.

    A checkpoint of the decoder training predicts volumes in the z blocks of the test evaluation
    (evaluate_decoder.py). The decoder of a registry model predicts like its reference results do
    (`evaluate_automatic_segmentation.py`, library defaults).
    """
    import torch

    name = registry_model(staged)
    if name is None:
        model = evaluate_decoder.load_decoder_model(torch.load(staged, map_location="cpu", weights_only=True), device)
        return model, tune_set.HIERA_U_TILING
    _, search_common = import_search()
    return search_common.build_model("ais", name, device, 3, checkpoint_path=staged), None


def resolve_checkpoint(checkpoint_path):
    """The staged copy of a checkpoint and the checksum of the training checkpoint it was made from.

    'checkpoint_path' is a training checkpoint, a staged copy, or the name of a registry model in REGISTRY_MODELS,
    whose decoder is staged (`stage_registry_decoder`).
    """
    if checkpoint_path in REGISTRY_MODELS:
        return stage_registry_decoder(checkpoint_path)
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(checkpoint_path + ".json"):
        staged, checksum, _ = evaluate_decoder.stage_checkpoint(checkpoint_path, TUNING_ROOT)
        return staged, checksum
    with open(checkpoint_path + ".json") as f:
        return checkpoint_path, json.load(f)["checksum"]


def session_folder(staged, data):
    return os.path.join(TUNING_ROOT, os.path.basename(staged)[:-len(".pt")], data)


def make_context(search, search_common, args, checksum):
    """The context of `parameter_search.search` for one session."""
    datasets = search.search_datasets(args.datasets, [args.data], "cells")
    if not datasets[args.data]:
        raise SystemExit(f"No tuning dataset for '{args.data}'.")
    return {
        "experiment_folder": os.path.abspath(args.experiment_folder), "data_root": search_common.DATA_ROOT,
        "datasets": datasets, "n_samples": args.n_samples, "edge_rounds": getattr(args, "edge_rounds", 2),
        "crop_3d": search.PIPELINE_CROP_3D, "n_threads": getattr(args, "threads", 2),
        # The identity is given, so the search does not look for the weights of a registry model.
        "identities": {MODEL: checksum}, "lock": threading.Lock(), "weights": {},
        "boundary_top": getattr(args, "boundary_top", None), "min_size_stage": True,
    }


def cache_dir(experiment_folder, checksum, dataset):
    return os.path.join(experiment_folder, "predictions", MODEL, checksum, dataset)


def tuning_crops_id(search_common, dataset):
    """The pinned tuning crops of a dataset (None without pinned crops), which its stored predictions belong to."""
    if dataset not in search_common.TUNING_CROPS:
        return None
    shape, samples = search_common.TUNING_CROPS[dataset]
    return {
        "crop_shape": list(shape),
        "starts": {name: [list(start) for start, _ in crops] for name, crops in samples.items()},
    }


def check_predictions(search_common, marker, dataset):
    """Stop if the stored predictions of a dataset belong to other tuning crops than the current ones.

    The predictions are stored by the index of their sample, so those of other crops would be scored against the
    wrong labels. A marker from before 2026-11-23 does not name its crops; for volumes it is compared by the number
    of predictions.
    """
    with open(marker) as f:
        stored = json.load(f)
    if "crops" in stored:
        current = stored["crops"] == tuning_crops_id(search_common, dataset)
    else:
        n_samples = search_common.n_samples(dataset, search_common.DATA_ROOT, split="val")
        current = dataset not in search_common.DATASETS_3D or stored["n_predictions"] == n_samples
    if not current:
        raise SystemExit(
            f"The stored predictions of '{dataset}' belong to other tuning crops ({os.path.dirname(marker)}). "
            "Use a new session folder (-e), or remove these predictions and the sweeps of the dataset in the session."
        )


def predict_command(args):
    """Predict the tuning samples of every dataset of a setting and store the predictions for the sweeps."""
    import torch

    search, search_common = import_search()
    evaluation, _ = evaluate_decoder.import_evaluation()
    staged, checksum = resolve_checkpoint(args.checkpoint)
    args.experiment_folder = args.experiment_folder or session_folder(staged, args.data)
    context = make_context(search, search_common, args, checksum)

    # A sweep skips a sample that fails with a warning. Here that would leave a hole in the stored predictions,
    # e.g. after running out of GPU memory, so it is an error.
    warnings.filterwarnings("error", message="Skipping validation sample")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", torch.cuda.get_device_name() if device == "cuda" else "CPU")
    print(f"Checkpoint {staged} ({checksum}).")
    model, predict_kwargs = None, None
    for dataset in context["datasets"][args.data]:
        marker = os.path.join(cache_dir(context["experiment_folder"], checksum, dataset), COMPLETE)
        if os.path.exists(marker):
            check_predictions(search_common, marker, dataset)
            print(f"{dataset}: predictions are complete.")
            continue
        if model is None:
            model, predict_kwargs = load_model(staged, device)
        search_common.check_data_download(dataset, search_common.DATA_ROOT, download=False, split="val")
        options = search.sample_options(context, args.data, dataset)
        t0 = time.time()
        search.tune_parameters(
            model, "ais", dataset, search_common.DATA_ROOT, MODEL,
            os.path.join(context["experiment_folder"], "tuning", args.data, "cache", "ais"), device,
            crop_shape=options["crop"], checkpoint_id=checksum, grid_preset="broad", subsample=options["subsample"],
            prediction_cache=os.path.join(context["experiment_folder"], "predictions"), cache_only=True,
            # The tiling and the padding of small images of the test evaluation, see `load_model`.
            predict_kwargs=predict_kwargs, min_shape=evaluation.TRAINING_SCALE_MIN_SHAPE_2D.get(dataset),
        )
        n_predictions = len([name for name in os.listdir(os.path.dirname(marker)) if name.endswith(".npy")])
        if n_predictions == 0:
            raise RuntimeError(f"No prediction was written for '{dataset}'.")
        with open(marker, "w") as f:
            json.dump({
                "n_predictions": n_predictions, "subsample": options["subsample"], "checksum": checksum,
                "crops": tuning_crops_id(search_common, dataset),
            }, f)
        print(f"{dataset}: {n_predictions} predictions ({time.time() - t0:.0f} s)", flush=True)


def sweep_command(args):
    """One task of the queue: sweep a grid or a list of combos on one dataset, from the cached predictions."""
    search, search_common = import_search()
    _, checksum = resolve_checkpoint(args.checkpoint)
    marker = os.path.join(cache_dir(args.experiment_folder, checksum, args.dataset), COMPLETE)
    # A sweep skips the samples it cannot load a prediction for, so an incomplete cache would go unnoticed.
    if not os.path.exists(marker):
        raise SystemExit(f"The predictions of '{args.dataset}' are not complete ({marker}). Run 'predict' first.")
    check_predictions(search_common, marker, args.dataset)
    if args.cache_only:
        return
    search.tune_parameters(
        None, "ais", args.dataset, search_common.DATA_ROOT, MODEL, os.path.join(args.tuning_root, "ais"), "cpu",
        n_threads=args.n_threads, crop_shape=tuple(args.crop_3d) if args.crop_3d else None, checkpoint_id=checksum,
        shard_index=args.shard_index, num_shards=args.num_shards, grid_preset="broad", subsample=args.subsample,
        prediction_cache=os.path.join(args.experiment_folder, "predictions"), params_file=args.params_file,
    )


def make_runner(search, context, staged, threads, slots, max_shards):
    """The queue of `parameter_search.SlurmRunner`, with worker slots in this process instead of Slurm job arrays.

    The tasks call the 'sweep' command of this script, which scores from the cached predictions.
    """

    class QueueRunner(search.SlurmRunner):
        def __init__(self):
            super().__init__(context, "", 0, threads, slots, max_shards)
            self.workers = []

        def command(self, sweep, shard_index=0, num_shards=1, cache_only=False):
            command = [
                sys.executable, os.path.abspath(__file__), "sweep", "-d", sweep["dataset"],
                "-e", context["experiment_folder"], "--checkpoint", staged, "--tuning_root", sweep["root"],
            ]
            if sweep["params_file"]:
                command += ["--params_file", sweep["params_file"]]
            if sweep["crop"]:
                command += ["--crop_3d", *map(str, sweep["crop"])]
            if sweep["subsample"]:
                command += ["--subsample", str(sweep["subsample"])]
            if cache_only:
                command += ["--cache_only"]
            if num_shards > 1:
                command += ["--num_shards", str(num_shards), "--shard_index", str(shard_index)]
            return command

        def submit_workers(self):
            # A slot ends when it finds no open task, and a later stage adds tasks. Each slot is a thread of its
            # own that is started again here: slots that share one worker thread are not, as long as one of them
            # is left, and then the next stage runs on that single slot.
            self.workers = [worker for worker in self.workers if worker.is_alive()]
            for _ in range(max(1, slots) - len(self.workers)):
                worker = threading.Thread(
                    target=search.worker_loop, args=(str(self.queue_dir), max(1, threads // max(1, slots))),
                    daemon=True,
                )
                worker.start()
                self.workers.append(worker)

    return QueueRunner()


def search_command(args):
    if args.check:
        os.environ[CHECK_ENV] = "1"  # Also for the tasks, which are processes of their own.
    search, search_common = import_search()
    staged, checksum = resolve_checkpoint(args.checkpoint)
    args.experiment_folder = args.experiment_folder or session_folder(staged, args.data)
    context = make_context(search, search_common, args, checksum)
    print(f"Checkpoint {staged} ({checksum}), {args.data}: {context['datasets'][args.data]}", flush=True)

    claims = os.path.join(context["experiment_folder"], "queue", "claims")
    if args.retry_failed and os.path.isdir(claims):
        for name in os.listdir(claims):
            if os.path.exists(os.path.join(claims, name, "failed")):
                os.rename(os.path.join(claims, name), os.path.join(claims, f"{name}.failed.{time.time():.0f}"))

    runner = make_runner(search, context, staged, args.threads, args.slots, args.max_shards)
    summary = search.search(context, runner, MODEL, "ais", args.data)
    path = os.path.join(context["experiment_folder"], "best_parameters.json")
    with open(path + ".tmp", "w") as f:
        json.dump({
            "checkpoint": staged, "checksum": checksum, "data": args.data, "params": summary["params"],
            "min_size_per_dataset": summary.get("min_size_per_dataset", {}),
            "datasets": context["datasets"][args.data], "check": bool(os.environ.get(CHECK_ENV)),
            "n_samples": args.n_samples, "summary": summary,
        }, f, indent=1)
    os.replace(path + ".tmp", path)
    print(f"Written to {path}")


def worker_command(args):
    """Take tasks from the queue of a session that a 'search' job drives, so that a search is spread over several
    small jobs. The worker waits between the stages and ends when the result is written, when its time limit is
    close, or when no task was started or closed for 'max_idle' seconds: the search job is gone, or the open tasks
    wait for tasks that failed.
    """
    search, _ = import_search()
    queue = os.path.join(args.experiment_folder, "queue")
    result = os.path.join(args.experiment_folder, "best_parameters.json")

    def queue_state():
        """The number of tasks of the queue and of those that are done or failed."""
        tasks_path = os.path.join(queue, "tasks.json")
        n_tasks = 0
        if os.path.exists(tasks_path):
            with open(tasks_path) as f:
                n_tasks = len(json.load(f))
        claims = os.path.join(queue, "claims")
        n_closed = sum(
            any(os.path.exists(os.path.join(claims, str(index), state)) for state in ("done", "failed"))
            for index in range(n_tasks)
        )
        return n_tasks, n_closed

    state, last_change = None, time.time()
    while not os.path.exists(result) and search.seconds_left() > search.MIN_SECONDS_LEFT:
        n_tasks, n_closed = queue_state()
        if n_closed < n_tasks:
            # Returns when no task is open or can run (failed tasks count as closed), or the time limit is close.
            search.run_worker(queue, args.threads, args.slots)
        new_state = queue_state()
        if new_state != state:
            state, last_change = new_state, time.time()
        elif time.time() - last_change > args.max_idle:
            print(f"No task was started or closed for {args.max_idle} s, the worker ends.", flush=True)
            break
        time.sleep(30)


HEADER = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH -p {partition}
#SBATCH -c {cpus}
#SBATCH --mem={mem}
#SBATCH -t {time}
#SBATCH -A nim00007
#SBATCH -o {root}/logs/{job_name}_%j.out
#SBATCH -e {root}/logs/{job_name}_%j.err
{extra}
set -eo pipefail
source ~/.bashrc
set -u
micromamba activate new-stack
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd {code}
{command}
"""


def _sbatch(script_path, dependency=None, dry_run=False):
    command = ["sbatch", "--parsable"] + ([f"--dependency=afterok:{dependency}"] if dependency else [])
    if dry_run:
        print(" ".join(command + [script_path]))
        return "DRYRUN"
    return subprocess.run(
        command + [script_path], capture_output=True, text=True, check=True
    ).stdout.strip().split(";")[0]


def submit_command(args):
    """Submit the sessions: per setting a GPU job that predicts and a CPU job that searches, from a frozen copy of
    the code. The search job starts after the prediction job ('afterok').
    """
    staged, checksum = resolve_checkpoint(args.checkpoint)
    name = os.path.basename(staged)[:-len(".pt")]
    root = os.path.join(TUNING_ROOT, name)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    code = os.path.join(root, "jobs", f"code_{stamp}")
    shutil.copytree(HERE, code, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(
        evaluate_decoder.evaluation_dir(), os.path.join(code, "evaluation"),
        ignore=shutil.ignore_patterns("__pycache__", "gpu_jobs", "optimization"),
    )
    os.makedirs(os.path.join(root, "logs"), exist_ok=True)

    jobs = {}
    for data in args.data:
        folder = os.path.join(root, data)
        common_args = f"--checkpoint {staged} --data {data} -e {folder}"
        scripts = {
            "predict": HEADER.format(
                job_name=f"dectune_predict_{data}", partition=args.gpu_partition, cpus=8, mem="64G", time=args.gpu_time,
                root=root, code=code, extra=f"#SBATCH -G {args.gres}\n#SBATCH --constraint=inet",
                command=f"python tune_decoder.py predict {common_args}",
            ),
            "search": HEADER.format(
                job_name=f"dectune_search_{data}", partition=args.cpu_partition, cpus=args.cpus, mem=args.mem,
                time=args.cpu_time, root=root, code=code, extra="",
                command=(
                    f"python tune_decoder.py search {common_args} --threads {args.cpus} --slots {args.slots} "
                    f"--max_shards {args.max_shards} --retry_failed"
                ),
            ),
        }
        for step, script in scripts.items():
            with open(os.path.join(code, f"{step}_{data}.sbatch"), "w") as f:
                f.write(script)
        predict_job = None
        if not args.skip_predict:
            predict_job = _sbatch(os.path.join(code, f"predict_{data}.sbatch"), dry_run=args.dry_run)
        search_job = _sbatch(os.path.join(code, f"search_{data}.sbatch"), dependency=predict_job, dry_run=args.dry_run)
        jobs[data] = {
            "predict": predict_job, "search": search_job, "folder": folder,
            "best_parameters": os.path.join(folder, "best_parameters.json"),
        }
        print(f"{data}: predict {predict_job}, search {search_job} -> {jobs[data]['best_parameters']}")
    if not args.dry_run:
        with open(os.path.join(root, f"jobs_{stamp}.json"), "w") as f:
            json.dump({"checkpoint": staged, "checksum": checksum, "code": code, "jobs": jobs}, f, indent=1)
    print(f"Checkpoint {staged} ({checksum}), code {code}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_session(sub):
        sub.add_argument("--checkpoint", required=True,
                         help="The training checkpoint, its staged copy, or a registry model (REGISTRY_MODELS).")
        sub.add_argument("--data", required=True, choices=SETTINGS, help="The setting of the session.")
        sub.add_argument("-e", "--experiment_folder", default=None,
                         help="The folder of the session. Default: <DECODER_SAVE_ROOT>/tuning/<checkpoint>/<data>.")
        sub.add_argument("--datasets", nargs="+", default=None,
                         help="Datasets of the setting to tune on. Default: its in-domain datasets.")
        sub.add_argument("--n_samples", type=int, default=None, help="At most this many samples per dataset.")

    sub = commands.add_parser("predict", help="Store the predictions of the tuning data of one setting (GPU).")
    add_session(sub)

    sub = commands.add_parser("search", help="Run the search of one setting from the stored predictions (CPU).")
    add_session(sub)
    sub.add_argument("--edge_rounds", type=int, default=2, help="Rounds that sweep past the grid edges.")
    sub.add_argument("--boundary_top", type=int, default=None,
                     help="The best combos that the boundary stage starts from. 0 skips the stage.")
    sub.add_argument("--threads", type=int, default=os.cpu_count(), help="The CPU threads for all tasks together.")
    sub.add_argument("--slots", type=int, default=None, help="The tasks that run at once. Default: threads / 2.")
    sub.add_argument("--max_shards", type=int, default=16, help="Split the sweep of a dataset into this many tasks.")
    sub.add_argument("--retry_failed", action="store_true", help="Run the failed tasks of an earlier run again.")
    sub.add_argument("--check", action="store_true", help="Small grids, to check the pipeline.")

    sub = commands.add_parser("sweep", help="One task of a search.")
    sub.add_argument("-d", "--dataset", required=True)
    sub.add_argument("-e", "--experiment_folder", required=True)
    sub.add_argument("--checkpoint", required=True)
    sub.add_argument("--tuning_root", required=True)
    sub.add_argument("--params_file", default=None)
    sub.add_argument("--crop_3d", type=int, nargs=3, default=None)
    sub.add_argument("--subsample", type=int, default=None)
    sub.add_argument("--cache_only", action="store_true")
    sub.add_argument("--num_shards", type=int, default=1)
    sub.add_argument("--shard_index", type=int, default=0)
    sub.add_argument("--n_threads", type=int, default=2)

    sub = commands.add_parser("worker", help="Take tasks of a running search, as a further (small) job.")
    sub.add_argument("-e", "--experiment_folder", required=True, help="The folder of the session.")
    sub.add_argument("--threads", type=int, default=os.cpu_count(), help="The CPU threads for all tasks together.")
    sub.add_argument("--slots", type=int, default=None, help="The tasks that run at once. Default: threads / 2.")
    sub.add_argument("--max_idle", type=int, default=1800,
                     help="End after this many seconds in which no task was started or closed.")

    sub = commands.add_parser("submit", help="Submit the sessions as Slurm jobs.")
    sub.add_argument("--checkpoint", required=True,
                     help="The training checkpoint, its staged copy, or a registry model (REGISTRY_MODELS).")
    sub.add_argument("--data", nargs="+", default=list(SETTINGS), choices=SETTINGS)
    sub.add_argument("--gpu_partition", default="grete:shared")
    sub.add_argument("--gres", default="A100:1")
    sub.add_argument("--gpu_time", default="04:00:00")
    sub.add_argument("--cpu_partition", default="standard96s:shared")
    sub.add_argument("--cpus", type=int, default=192, help="The CPUs of a search job (a node has 192 threads).")
    sub.add_argument("--mem", default="400G")
    sub.add_argument("--cpu_time", default="48:00:00")
    sub.add_argument("--slots", type=int, default=96, help="The tasks a search job runs at once.")
    sub.add_argument("--skip_predict", action="store_true",
                     help="Submit only the search jobs: the predictions are stored already.")
    sub.add_argument("--max_shards", type=int, default=16)
    sub.add_argument("--dry-run", action="store_true", dest="dry_run", help="Only write the scripts.")

    args = parser.parse_args()
    if args.command in ("search", "worker") and args.slots is None:
        args.slots = max(1, args.threads // 2)
    {
        "predict": predict_command, "search": search_command, "sweep": sweep_command, "worker": worker_command,
        "submit": submit_command,
    }[args.command](args)


if __name__ == "__main__":
    main()
