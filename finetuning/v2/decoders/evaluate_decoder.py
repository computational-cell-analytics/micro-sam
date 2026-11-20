"""Automatic instance segmentation (AIS) evaluation of a decoder trained on a frozen uSAM2 encoder.

The decoder is scored on the test data of the automatic segmentation panels of Figures 2 and 3 of the micro-sam2 paper
(Extended Data Figure 1 uses the same datasets), with the evaluation pipeline of
'finetuning/v2/evaluation/evaluate_automatic_segmentation.py': the same test samples and crops, padding, upscaling,
tiling of large images, postprocessing and metrics. The reference is micro-sam2 hvit_t, i.e. the registry models
'hvit_t_cells' and 'hvit_t_cells_decoder' with the library defaults, whose results are read from the registry
evaluation of the paper (REFERENCE_FOLDERS).

The decoder model ('FrozenEncoderHybridUNETR') takes and returns the same as the UniSAM2 model of the reference:
three channels in [0, 1], resized to 1024 pixels inside the model, and the five output channels foreground,
three directed distances and boundaries. The only difference in the pipeline is the z-context of volumes: the
decoder was trained on stacks of 14 slices and processes blocks of 10 + 2 * 2 slices (Z_TILE, Z_HALO), the
reference blocks of 4 + 2 * 2 slices.

TODO: the postprocessing uses the library defaults of hvit_t_cells (see DEFAULT_MODEL_TYPE), which were tuned for the
previous decoder. Replace them with parameters that are optimized for the new decoder.

Usage:

    # Score the first two samples of every dataset with the current best checkpoint.
    python evaluate_decoder.py run --checkpoint <root>/checkpoints/hiera_u_hvit_t_cells/best.pt --n_samples 2 \\
        --skip_failed -e <root>/evaluation/smoke
    # Compare with the reference on the same samples.
    python evaluate_decoder.py report -e <root>/evaluation/smoke --n_samples 2
    # The full evaluation, as a Slurm array over datasets and shards.
    python evaluate_decoder.py submit --checkpoint <root>/checkpoints/hiera_u_hvit_t_cells/best.pt [--dry-run]

The decoder of the checkpoint is copied to '<experiment>/checkpoints/' first, since the training overwrites 'best.pt'.
The checksum of the training checkpoint names the result files ('<dataset>_micro_sam2_hvit_t_ais_default_ckpt-
<checksum>.csv'), as in the reference.
"""

import argparse
import ast
import functools
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback

# Without expandable segments, the blocks of 14 slices do not fit into 10 GB of GPU memory (fragmentation).
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

HERE = os.path.dirname(os.path.abspath(__file__))
SAVE_ROOT = os.environ.get(
    "DECODER_SAVE_ROOT", "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/decoder_training"
)
DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"

# The model type that names the result files and selects the postprocessing parameters. 'hvit_t' uses the defaults of
# the cells models, i.e. those of 'hvit_t_cells' (micro_sam._model_settings.DEFAULT_POSTPROCESSING).
# TODO: these AIS defaults were tuned for the previous UNETR decoder. Replace them with parameters that are optimized
# for the new decoder.
DEFAULT_MODEL_TYPE = "hvit_t"

# The z tile and halo of volumes: blocks of 10 + 2 * 2 = 14 slices, the number of slices of the training. The y and x
# tiles are the ones of the reference (384 + 2 * 64 = 512 pixels, the size of the training patches).
Z_TILE, Z_HALO = 10, 2

# The results of the reference, micro-sam2 hvit_t with the registry models and the library defaults. A result in a
# later folder replaces the result of the same dataset in an earlier one, as in the tables of the paper figures
# ('collect_automatic_results.py' in experiments/baselines_2d/control).
EXPERIMENTS = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments"
REFERENCE_FOLDERS = [
    f"{EXPERIMENTS}/v6_cells_registry_evaluation/results",
    f"{EXPERIMENTS}/v6_cells_registry_evaluation_crops32/results",
    f"{EXPERIMENTS}/v6_cells_registry_evaluation_isotropic/results",
]

# The samples per array task of the full evaluation.
SAMPLES_PER_TASK = {2: 250, 3: 8}


def _find_dir(*candidates):
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise RuntimeError(f"None of these folders exists: {candidates}.")


def _repo_root():
    import micro_sam
    return os.path.dirname(os.path.dirname(os.path.abspath(micro_sam.__file__)))


def evaluation_dir():
    """The evaluation code: the frozen copy of a Slurm job, or the one of the repository."""
    return _find_dir(os.path.join(HERE, "evaluation"), os.path.join(_repo_root(), "finetuning", "v2", "evaluation"))


@functools.lru_cache(maxsize=None)
def import_evaluation():
    """Import 'evaluate_automatic_segmentation' and its 'common' module.

    The evaluation folder has a module 'common', like this folder. It is imported here under its own name and removed
    from 'sys.modules' afterwards, so that it does not shadow the 'common' module of this folder. The modules are
    imported once.

    Returns:
        The modules 'evaluate_automatic_segmentation' and its 'common'.
    """
    folder = evaluation_dir()
    saved = sys.modules.pop("common", None)
    sys.path.insert(0, folder)
    try:
        import evaluate_automatic_segmentation as evaluation
        evaluation_common = sys.modules["common"]
    finally:
        sys.path.remove(folder)
        sys.modules.pop("common", None)
        if saved is not None:
            sys.modules["common"] = saved
    return evaluation, evaluation_common


def panel_datasets():
    """The datasets of the automatic segmentation panels of Figures 2 and 3.

    They are read from 'PANEL_DATASETS' of the plotting scripts, which set the in-domain and out-of-domain datasets of
    the figures. Extended Data Figure 1 uses the same datasets.

    Returns:
        A list of (dataset, figure, panel, domain).
    """
    folder = _find_dir(
        os.path.join(HERE, "for_paper"), os.path.join(_repo_root(), "scripts", "plotting", "v2", "for_paper")
    )
    datasets = []
    for figure in ("figure2", "figure3"):
        with open(os.path.join(folder, f"plots_for_{figure}.py")) as f:
            tree = ast.parse(f.read())
        panels = next(
            ast.literal_eval(node.value) for node in tree.body
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "PANEL_DATASETS"
        )
        for panel, domains in panels.items():
            for domain, names in domains.items():
                datasets.extend((name, figure, panel, domain) for name in names)
    return datasets


# The modules whose objects a training checkpoint may build. Everything else, e.g. the datasets and augmentations
# that torch_em stores with the loaders in 'init', is replaced by a stub.
_LOADABLE_MODULES = ("torch", "collections", "builtins", "__builtin__", "copyreg", "numpy", "_codecs")


class _Stub:
    """Stands in for an object of the training set-up that the evaluation does not need."""
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _StubUnpickler(__import__("pickle").Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in _LOADABLE_MODULES:
            return super().find_class(module, name)
        return _Stub


class _StubPickleModule:
    """A pickle module for torch.load that does not rebuild the training datasets of a checkpoint."""
    Unpickler = _StubUnpickler

    @staticmethod
    def load(file, **kwargs):
        return _StubUnpickler(file, **kwargs).load()


# The entries of a training checkpoint that the evaluation keeps.
_EVALUATION_KEYS = ("epoch", "best_metric", "decoder_state", "decoder_config", "encoder_model_type")


def load_training_checkpoint(checkpoint_path):
    """Load the entries of a decoder training checkpoint that the evaluation needs.

    torch_em stores the datasets of the loaders in the checkpoints ('init'), and loading them in full rebuilds all
    datasets of the training, which takes minutes and many GB. They are replaced by stubs here.
    """
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, pickle_module=_StubPickleModule)
    return {key: checkpoint[key] for key in _EVALUATION_KEYS if key in checkpoint}


def stage_checkpoint(checkpoint_path, experiment_folder):
    """Copy the decoder of a training checkpoint into the experiment folder and return the copy, its id and content.

    The training overwrites 'best.pt' and 'latest.pt', so the evaluation reads a fixed copy. The copy keeps only
    the entries in _EVALUATION_KEYS. Its id is the checksum of the training checkpoint it was made from, so a
    checkpoint is staged once.
    """
    import torch

    _, evaluation_common = import_evaluation()
    folder = os.path.join(experiment_folder, "checkpoints")
    os.makedirs(folder, exist_ok=True)
    checkpoint_path = os.path.abspath(checkpoint_path)
    if os.path.exists(checkpoint_path + ".json"):  # A staged copy: it keeps the id of its training checkpoint.
        with open(checkpoint_path + ".json") as f:
            checksum = json.load(f)["checksum"]
        staged = os.path.join(folder, os.path.basename(checkpoint_path))
        if staged != checkpoint_path:
            shutil.copyfile(checkpoint_path + ".json", staged + ".json")
            shutil.copyfile(checkpoint_path, staged + ".tmp")
            os.replace(staged + ".tmp", staged)
        return staged, checksum, torch.load(staged, map_location="cpu", weights_only=True)

    # Read one copy, so that the checksum and the weights belong to the same file even if the training replaces it.
    tmp_path = os.path.join(folder, f".staging_{os.getpid()}.pt")
    shutil.copyfile(checkpoint_path, tmp_path)
    try:
        checksum = evaluation_common.checkpoint_checksum(tmp_path)
        checkpoint = load_training_checkpoint(tmp_path)
    finally:
        os.remove(tmp_path)
    run = os.path.basename(os.path.dirname(checkpoint_path))
    staged = os.path.join(folder, f"{run}_epoch{checkpoint['epoch'] + 1}_{checksum[:8]}.pt")
    if not os.path.exists(staged):
        torch.save(checkpoint, staged + ".tmp")
        os.replace(staged + ".tmp", staged)
    with open(staged + ".json", "w") as f:
        json.dump({
            "source": checkpoint_path, "checksum": checksum, "epochs_finished": checkpoint["epoch"] + 1,
            "best_metric": float(checkpoint.get("best_metric", float("nan"))),
            "encoder_model_type": checkpoint["encoder_model_type"], "decoder_config": checkpoint["decoder_config"],
            "staged": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    return staged, checksum, checkpoint


def load_decoder_model(checkpoint, device):
    """Build the model with the frozen encoder and the trained decoder of a checkpoint.

    Args:
        checkpoint: The loaded checkpoint of the decoder training, see `DecoderTrainer`.
        device: The device for the model.

    Returns:
        The model in evaluation mode.
    """
    from micro_sam.v2.training.decoder_training import FrozenEncoderHybridUNETR

    config = checkpoint["decoder_config"]
    model = FrozenEncoderHybridUNETR(
        model_type=checkpoint["encoder_model_type"], decoder=config["decoder"],
        decoder_kwargs=config.get("decoder_kwargs"), out_channels=config["out_channels"],
    )
    model.load_decoder_state(checkpoint["decoder_state"])
    model.model_type = DEFAULT_MODEL_TYPE
    return model.to(device).eval()


def default_experiment_folder(checkpoint_path):
    return os.path.join(SAVE_ROOT, "evaluation", os.path.basename(os.path.dirname(os.path.abspath(checkpoint_path))))


def _is_out_of_memory(error):
    import torch
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def _result_tag(z_tile, z_halo):
    """The tag of the result files for a z-context other than the default one."""
    return None if (z_tile, z_halo) == (Z_TILE, Z_HALO) else f"z{z_tile}h{z_halo}"


def run(args):
    import torch

    evaluation, evaluation_common = import_evaluation()
    experiment_folder = args.experiment_folder or default_experiment_folder(args.checkpoint)
    staged, checksum, checkpoint = stage_checkpoint(args.checkpoint, experiment_folder)
    print(f"Checkpoint {staged} (epoch {checkpoint['epoch'] + 1}, checksum {checksum}).")

    datasets = args.datasets or [name for name, *_ in panel_datasets()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", torch.cuda.get_device_name() if device == "cuda" else "CPU")
    model = load_decoder_model(checkpoint, device)
    del checkpoint

    shard = None if args.n_shards is None else (args.shard_index, args.n_shards)
    failed_path = os.path.join(experiment_folder, "failed.json")
    failed = {}
    if os.path.exists(failed_path):
        with open(failed_path) as f:
            failed = json.load(f)

    for dataset in datasets:
        ndim = 3 if dataset in evaluation_common.DATASETS_3D else 2
        predict_kwargs = (
            {"tile_shape_3d": (args.z_tile, 384, 384), "halo_3d": (args.z_halo, 64, 64)} if ndim == 3 else None
        )
        key = f"{dataset}_n{args.n_samples}" if args.n_samples else dataset
        start = time.time()
        try:
            evaluation_common.check_data_download(dataset, DATA_ROOT, download=False)
            # params=None: the postprocessing uses the defaults of DEFAULT_MODEL_TYPE.
            # TODO: replace them with parameters optimized for the new decoder.
            evaluation.run_evaluation(
                model, "ais", dataset, DATA_ROOT, experiment_folder, DEFAULT_MODEL_TYPE, None, device,
                args.n_samples, checkpoint_id=checksum, tuned=False,
                result_tag=_result_tag(args.z_tile, args.z_halo), shard=shard,
                min_shape=evaluation.TRAINING_SCALE_MIN_SHAPE_2D.get(dataset), predict_kwargs=predict_kwargs,
            )
            failed.pop(key, None)
        except Exception as e:
            if not args.skip_failed:
                raise
            reason = "out of memory" if _is_out_of_memory(e) else f"{type(e).__name__}: {e}"
            print(f"FAILED {dataset}: {reason}")
            traceback.print_exc()
            failed[key] = {"reason": reason, "checksum": checksum, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            if device == "cuda":
                torch.cuda.empty_cache()
        print(f"{dataset}: {time.time() - start:.1f} s", flush=True)
        if failed or os.path.exists(failed_path):
            with open(failed_path, "w") as f:
                json.dump(failed, f, indent=2)


def _metric_column(dataset, evaluation_common):
    if dataset in evaluation_common.DATASETS_DENSE:
        return "cremi"
    return "SA50" if dataset in evaluation_common.DATASETS_SA50 else "mSA"


def _reference_result(dataset):
    found = None
    for folder in REFERENCE_FOLDERS:
        paths = glob.glob(os.path.join(folder, f"{dataset}_micro_sam2_hvit_t_ais_default_ckpt-*.csv"))
        if len(paths) > 1:
            raise RuntimeError(f"More than one reference result for '{dataset}' in {folder}: {paths}.")
        if paths:
            found = paths[0]
    return found


def _sample_rows(result_path, n_samples=None, sample_ids=None):
    """The per-sample rows of the first 'n_samples' samples, without the samples that have no labels.

    'sample_ids' maps the samples to the positions of the stored rows, for a reference written before test
    volumes moved into tuning (see `common.legacy_test_sample_ids`). None means the positions are unchanged.
    """
    import pandas as pd

    # A dataset can have fewer samples than 'n_samples'.
    n_stored = len(glob.glob(os.path.join(result_path[:-4] + "_samples", "sample_*.csv")))
    if sample_ids is None:
        sample_ids = range(n_stored)
    if n_samples is not None:
        sample_ids = sample_ids[:n_samples]
    rows = []
    for index in sample_ids:
        path = os.path.join(result_path[:-4] + "_samples", f"sample_{index:04d}.csv")
        if not os.path.exists(path):
            raise RuntimeError(f"The reference has no row for sample {index}: {path}.")
        row = pd.read_csv(path)
        if "skipped" in row.columns and bool(row["skipped"].iloc[0]):
            continue
        rows.append(row)
    return pd.concat(rows, ignore_index=True) if rows else None


def _new_result(experiment_folder, dataset, tag, checkpoint_id):
    paths = glob.glob(os.path.join(
        experiment_folder, "results", f"{dataset}_micro_sam2_{DEFAULT_MODEL_TYPE}_ais_default{tag}_ckpt-*.csv"
    ))
    if checkpoint_id is not None:
        paths = [path for path in paths if path.endswith(f"_ckpt-{checkpoint_id}.csv")]
    if len(paths) > 1:
        raise RuntimeError(f"Results of several checkpoints for '{dataset}', pass --checkpoint_id: {paths}.")
    return paths[0] if paths else None


def report(args):
    """Compare the decoder with the reference, per dataset and as means per panel and domain."""
    import pandas as pd

    _, evaluation_common = import_evaluation()
    result_tag = _result_tag(args.z_tile, args.z_halo)
    tag = (f"_{result_tag}" if result_tag else "") + (f"_n{args.n_samples}" if args.n_samples else "")

    rows = []
    for dataset, figure, panel, domain in panel_datasets():
        column = _metric_column(dataset, evaluation_common)
        row = {
            "dataset": dataset, "figure": figure, "panel": panel, "domain": domain,
            "metric": column.lower(), "new": None, "reference": None, "n_samples": None,
        }
        new_path = _new_result(args.experiment_folder, dataset, tag, args.checkpoint_id)
        if new_path is not None:
            row["new"] = float(pd.read_csv(new_path)[column].iloc[0])
        reference_path = _reference_result(dataset)
        # The reference predates the move of test volumes into tuning, so its rows are matched by sample.
        legacy_ids = evaluation_common.legacy_test_sample_ids(dataset, DATA_ROOT)
        if reference_path is not None:
            if args.n_samples or legacy_ids is not None:
                samples = _sample_rows(reference_path, args.n_samples, legacy_ids)
                if samples is not None:
                    row["reference"], row["n_samples"] = float(samples[column].mean()), len(samples)
            else:
                row["reference"] = float(pd.read_csv(reference_path)[column].iloc[0])
        rows.append(row)

    table = pd.DataFrame(rows)
    table["delta"] = table["new"] - table["reference"]
    path = os.path.join(args.experiment_folder, f"report{tag}.csv")
    table.to_csv(path, index=False)

    pd.set_option("display.width", 200)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    both = table.dropna(subset=["new", "reference"])
    # Per metric, so that the datasets scored by SA50 (common.DATASETS_SA50) do not enter a mean of mSA values.
    groups = ["figure", "panel", "domain", "metric"]
    means = both.groupby(groups)[["new", "reference", "delta"]].mean()
    means["n_datasets"] = both.groupby(groups).size()
    print("\nMeans over the datasets with both results (msa, sa50: higher is better, cremi: lower is better):")
    print(means.to_string(float_format=lambda value: f"{value:.4f}"))
    missing = table[table["new"].isna()]["dataset"].tolist()
    if missing:
        print(f"\nNo result for {len(missing)} dataset(s): {missing}")
    print(f"\nWritten to {path}")


SBATCH = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH -p {partition}
#SBATCH -G {gres}
#SBATCH -c {cpus}
#SBATCH --mem={mem}
#SBATCH -t {time}
#SBATCH --constraint=inet
#SBATCH -A nim00007
#SBATCH --array=1-{n_tasks}%{max_parallel}
#SBATCH -o {experiment}/logs/slurm/{job_name}_%A_%a.out
#SBATCH -e {experiment}/logs/slurm/{job_name}_%A_%a.err
set -eo pipefail
source ~/.bashrc
set -u
micromamba activate new-stack
export PYTHONUNBUFFERED=1 PYTORCH_ALLOC_CONF=expandable_segments:True
cd {code}
ARGS=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {tasks})
python evaluate_decoder.py run --checkpoint {checkpoint} -e {experiment} --skip_failed $ARGS
"""


def submit(args):
    """Submit the full evaluation as a Slurm array over datasets and shards, from a frozen copy of the code."""
    _, evaluation_common = import_evaluation()
    experiment_folder = args.experiment_folder or default_experiment_folder(args.checkpoint)
    staged, checksum, _ = stage_checkpoint(args.checkpoint, experiment_folder)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    code = os.path.join(experiment_folder, "jobs", f"code_{stamp}")
    shutil.copytree(HERE, code, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(evaluation_dir(), os.path.join(code, "evaluation"), ignore=shutil.ignore_patterns("__pycache__"))
    for_paper = os.path.join(_repo_root(), "scripts", "plotting", "v2", "for_paper")
    os.makedirs(os.path.join(code, "for_paper"))
    for figure in ("figure2", "figure3"):
        shutil.copy(os.path.join(for_paper, f"plots_for_{figure}.py"), os.path.join(code, "for_paper"))

    tasks = []
    datasets = args.datasets or [name for name, *_ in panel_datasets()]
    z_args = f"--z_tile {args.z_tile} --z_halo {args.z_halo}"
    for dataset in datasets:
        ndim = 3 if dataset in evaluation_common.DATASETS_3D else 2
        n_shards = max(1, math.ceil(evaluation_common.n_samples(dataset, DATA_ROOT) / SAMPLES_PER_TASK[ndim]))
        tasks.extend(
            f"--datasets {dataset} --n_shards {n_shards} --shard_index {index} {z_args}" for index in range(n_shards)
        )
    tasks_path = os.path.join(code, "tasks.txt")
    with open(tasks_path, "w") as f:
        f.write("\n".join(tasks) + "\n")

    job_name = f"deceval_{os.path.basename(staged)[:-3]}"
    script = SBATCH.format(
        job_name=job_name, partition=args.partition, gres=args.gres, cpus=args.cpus, mem=args.mem, time=args.time,
        n_tasks=len(tasks), max_parallel=args.max_parallel, experiment=experiment_folder, code=code,
        tasks=tasks_path, checkpoint=staged,
    )
    script_path = os.path.join(code, "submit.sbatch")
    with open(script_path, "w") as f:
        f.write(script)
    os.makedirs(os.path.join(experiment_folder, "logs", "slurm"), exist_ok=True)
    print(f"{len(tasks)} tasks for {len(datasets)} datasets, checkpoint {staged} ({checksum}).")
    print(f"Script: {script_path}")
    if args.dry_run:
        return
    print(subprocess.run(["sbatch", script_path], capture_output=True, text=True, check=True).stdout.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def add_common(sub, checkpoint_required):
        sub.add_argument("--checkpoint", required=checkpoint_required, help="The checkpoint of the decoder training.")
        sub.add_argument("-e", "--experiment_folder", default=None,
                         help="The output folder. Default: <DECODER_SAVE_ROOT>/evaluation/<training run>.")
        sub.add_argument("--datasets", nargs="+", default=None,
                         help="The datasets to evaluate. Default: the panel datasets of Figures 2 and 3.")
        sub.add_argument("--z_tile", type=int, default=Z_TILE, help="The z tile of volumes.")
        sub.add_argument("--z_halo", type=int, default=Z_HALO, help="The z halo of volumes.")

    sub = commands.add_parser("run", help="Evaluate a checkpoint here.")
    add_common(sub, True)
    sub.add_argument("--n_samples", type=int, default=None, help="Score only the first N samples, for a check.")
    sub.add_argument("--n_shards", type=int, default=None, help="Split the samples over this many array tasks.")
    sub.add_argument("--shard_index", type=int, default=None, help="The shard this array task scores.")
    sub.add_argument("--skip_failed", action="store_true",
                     help="Continue with the next dataset after an error, e.g. out of GPU memory (see failed.json).")

    sub = commands.add_parser("report", help="Compare the results with the reference.")
    sub.add_argument("-e", "--experiment_folder", required=True)
    sub.add_argument("--n_samples", type=int, default=None,
                     help="The '--n_samples' of the run; the reference is then the mean over the same samples.")
    sub.add_argument("--checkpoint_id", default=None, help="The checksum of the checkpoint, if there are several.")
    sub.add_argument("--z_tile", type=int, default=Z_TILE)
    sub.add_argument("--z_halo", type=int, default=Z_HALO)

    sub = commands.add_parser("submit", help="Submit the full evaluation as a Slurm array.")
    add_common(sub, True)
    sub.add_argument("--partition", default="grete:shared")
    sub.add_argument("--gres", default="A100:1")
    sub.add_argument("--cpus", type=int, default=8)
    sub.add_argument("--mem", default="64G")
    sub.add_argument("--time", default="04:00:00")
    sub.add_argument("--max_parallel", type=int, default=16)
    sub.add_argument("--dry-run", action="store_true", dest="dry_run", help="Only write the script and the tasks.")

    args = parser.parse_args()
    if args.command == "run":
        if (args.n_shards is None) != (args.shard_index is None):
            parser.error("Pass both --n_shards and --shard_index, or neither.")
        run(args)
    elif args.command == "report":
        report(args)
    else:
        submit(args)


if __name__ == "__main__":
    main()
