"""Generate and submit every Slurm job of the v2 evaluation.

Four controls pick what runs:
    --data              which datasets, by name or with --all_datasets
    --modality          lm or em (or both)
    --segmentation_type automatic or interactive
    --segmentation_mode ais or apg, for micro-sam2

A run without --method evaluates micro-sam2 itself, through evaluate_automatic_segmentation.py or
evaluate_interactive_segmentation.py. With --method it evaluates the baselines instead, through
evaluate_automatic_baselines.py or evaluate_interactive_baselines.py. Datasets a method cannot
handle are dropped from its selection rather than submitted and failed.

Usage examples:
    # micro-sam2 automatic prompt generation on every LM dataset, for three backbones.
    python submit_all_evaluations.py --all_datasets --modality lm \\
        --segmentation_type automatic --segmentation_mode apg -m hvit_t hvit_s hvit_b

    # micro-sam2 interactive with box prompts, every dataset.
    python submit_all_evaluations.py --all_datasets --segmentation_type interactive -m hvit_t -p box

    # Two interactive baselines on the EM datasets.
    python submit_all_evaluations.py --all_datasets --modality em \\
        --segmentation_type interactive --method nninteractive sam2
"""

import os
import re
import math
import shlex
import argparse
import subprocess
from pathlib import Path
from typing import Optional
from datetime import datetime

from common import (
    DATA_ROOT, DATASETS_2D, DATASETS_3D_LM, DATASETS_3D_EM, MODEL_TYPES,
)

EVAL_ROOT = Path(__file__).resolve().parent
SCRIPTS = {
    ("automatic", "micro_sam2"): EVAL_ROOT / "evaluate_automatic_segmentation.py",
    ("automatic", "baseline"): EVAL_ROOT / "evaluate_automatic_baselines.py",
    ("interactive", "micro_sam2"): EVAL_ROOT / "evaluate_interactive_segmentation.py",
    ("interactive", "baseline"): EVAL_ROOT / "evaluate_interactive_baselines.py",
}

# Written into every job script, so a queued job sees the configuration the submission had rather
# than whatever the environment holds when it starts.
PINNED_ENV_VARS = ("MICRO_SAM2_JOINT_CHECKPOINT_ROOT", "MICRO_SAM2_JOINT_EXPORT_ROOT")

DATASETS_LM = tuple(DATASETS_2D + DATASETS_3D_LM)
DATASETS_EM = tuple(DATASETS_3D_EM)
DATASETS = tuple(sorted(set(DATASETS_LM + DATASETS_EM)))
DATASETS_3D = tuple(sorted(set(DATASETS_3D_LM + DATASETS_3D_EM)))

SEGMENTATION_MODES = ("ais", "apg")
AUTOMATIC_METHODS = ("cellpose", "stardist", "cellsam", "microsam_ais", "microsam_apg", "segneuron")
INTERACTIVE_METHODS = ("nninteractive", "sam3", "sam", "sam2", "micro-sam", "microsam_vol")

# Interactive 'sam2' is the pretrained backbone of the very engine micro-sam2 finetunes, so it runs
# through the same script rather than a second copy of the same inference.
SHARED_ENGINE_METHODS = {"sam2"}

# What each method can actually be run on. A method that is absent runs on everything.
METHOD_SUPPORT = {
    ("automatic", "cellsam"): {"ndim": (2,)},
    ("automatic", "microsam_ais"): {"modality": ("lm",)},
    ("automatic", "microsam_apg"): {"modality": ("lm",)},
    ("automatic", "segneuron"): {"modality": ("em",), "ndim": (3,)},
    ("interactive", "sam"): {"ndim": (2,)},
    ("interactive", "micro-sam"): {"ndim": (2,)},
    ("interactive", "nninteractive"): {"ndim": (3,)},
    ("interactive", "microsam_vol"): {"ndim": (3,), "modality": ("lm",)},
}

# Methods whose packages do not live in the default environment. The names are per machine, so
# --env overrides them and a missing one is reported before anything is submitted.
METHOD_ENV = {"cellpose": "cp3", "stardist": "sd"}
DEFAULT_ENV = "super"

# Slurm resources per job. Only the grete partitions are available. 'grete:preemptible' is usually
# free and starts within minutes, where the shared pools queue for days. It is MIG only, so the GPU
# needs a type. Preemption is safe because predictions persist and jobs are requeued. Note that
# 'grete-h100:shared' offers no MIG slice; override --gpu with '1' to use it.
PARTITION = "grete:preemptible"
CPUS = 4
TIME_LIMIT = "08:00:00"

# A 2d job peaks at about 3 GiB, so the smallest slice covers it. A volume is tiled through the
# encoder and the decoder and overruns that slice, so a 3d job takes the largest one instead.
GPU_2D, GPU_3D = "1g.10gb:1", "3g.40gb:1"
MEMORY_2D, MEMORY_3D = "16G", "64G"

EXPERIMENT_FOLDER = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/v2_joint_evaluation"


def env_exports() -> str:
    """Return 'export' lines pinning the run configuration, or an empty string if none is set."""
    return "".join(f"export {name}={shlex.quote(os.environ[name])}\n" for name in PINNED_ENV_VARS if name in os.environ)


def sanitize(name: str) -> str:
    """Turn a name into something Slurm and a filesystem both accept."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def resolve_env(args: argparse.Namespace, method: Optional[str]) -> str:
    """The conda environment one job activates."""
    return args.env or METHOD_ENV.get(method, DEFAULT_ENV)


def available_envs() -> set:
    """The conda environments micromamba knows about, or an empty set if it cannot be asked."""
    try:
        listing = subprocess.run(["micromamba", "env", "list"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return set()
    names = set()
    for line in listing.stdout.splitlines()[1:]:
        fields = line.split()
        if fields and not fields[0].startswith("#"):
            names.add(fields[0].lstrip("*").strip() or fields[0])
    return names


def warn_missing_envs(envs: set) -> None:
    """Report the environments a submission needs but this machine does not have.

    A job that activates a missing environment still starts, and then dies on the first import, so
    the failure is much cheaper to see here.
    """
    known = available_envs()
    if not known:
        return
    missing = sorted(env for env in envs if env not in known)
    if missing:
        print(f"Warning: these environments do not exist here: {', '.join(missing)}. "
              f"Their jobs will fail on import. Override with --env.")


def modality_of(dataset_name: str) -> str:
    """The modality a dataset belongs to, 'lm' or 'em'."""
    return "em" if dataset_name in DATASETS_EM else "lm"


def ndim_of(dataset_name: str) -> int:
    """The spatial dimensionality of a dataset, 2 or 3."""
    return 3 if dataset_name in DATASETS_3D else 2


def select_datasets(args: argparse.Namespace, method: Optional[str], mode: Optional[str]) -> tuple:
    """The datasets one job group runs on, after the modality, dimensionality and method filters.

    A method that cannot handle a dataset drops it here rather than being submitted and failing on
    the node, so a broad selection stays usable without listing exceptions by hand.
    """
    datasets = tuple(args.data) if args.data else DATASETS

    if args.modality != "all":
        datasets = tuple(d for d in datasets if modality_of(d) == args.modality)
    if args.ndim is not None:
        datasets = tuple(d for d in datasets if ndim_of(d) == args.ndim)

    support = METHOD_SUPPORT.get((args.segmentation_type, method), {})
    if "modality" in support:
        datasets = tuple(d for d in datasets if modality_of(d) in support["modality"])
    if "ndim" in support:
        datasets = tuple(d for d in datasets if ndim_of(d) in support["ndim"])
    return datasets


def uses_shared_engine(args: argparse.Namespace, method: Optional[str]) -> bool:
    """Whether a method runs through the micro-sam2 script rather than the baseline one."""
    return args.segmentation_type == "interactive" and method in SHARED_ENGINE_METHODS


def build_command(
    args: argparse.Namespace, dataset_name: str, model_type: Optional[str],
    method: Optional[str], mode: Optional[str],
) -> list:
    """The python command one dataset of one job runs."""
    shared_engine = uses_shared_engine(args, method)
    script = SCRIPTS[(args.segmentation_type, "baseline" if (method and not shared_engine) else "micro_sam2")]
    command = [
        "python", str(script),
        "-d", dataset_name,
        "-i", args.data_root,
        "-e", args.experiment_folder,
    ]
    if method and not shared_engine:
        command.extend(["--method", method])
    if shared_engine:
        command.extend(["--weights", "pretrained"])
    if model_type is not None:
        command.extend(["-m", model_type])
    if args.checkpoint is not None:
        command.extend(["-c", args.checkpoint])

    if method is None:
        command.extend(["--joint_checkpoint", args.joint_checkpoint])
        if args.segmentation_type == "automatic":
            command.extend(["--mode", mode])
            if args.skip_tuning:
                command.append("--skip_tuning")
            if args.tuning_root is not None:
                command.extend(["--tuning_root", args.tuning_root])

    if args.segmentation_type == "interactive":
        command.extend(["-p", args.prompt_choice, "-iter", str(args.n_iterations)])
        if args.min_size:
            command.extend(["--min_size", str(args.min_size)])
        if shared_engine and model_type is None:
            command.extend(["-m", "hvit_t"])

    return command


def job_tag(
    args: argparse.Namespace, datasets: tuple, model_type: Optional[str],
    method: Optional[str], mode: Optional[str], chunk_index: int,
) -> str:
    """The name of one job, which also names its script and its logs."""
    # A batched job spans several datasets, so it is named after its chunk instead.
    name = datasets[0] if len(datasets) == 1 else f"chunk{chunk_index:02d}"
    parts = [args.segmentation_type, name, method or f"micro_sam2_{mode or 'interactive'}"]
    if args.segmentation_type == "interactive":
        parts.append(args.prompt_choice)
    if model_type is not None:
        parts.append(model_type)
    return "_".join(sanitize(part) for part in parts)


def write_batch_script(
    args: argparse.Namespace, job_folder: Path, datasets: tuple, model_type: Optional[str],
    method: Optional[str], mode: Optional[str], chunk_index: int,
) -> Path:
    """Write the Slurm script of one job and return its path."""
    tag = job_tag(args, datasets, model_type, method, mode, chunk_index)
    script_path = job_folder / f"{tag}.sh"
    env = DEFAULT_ENV if uses_shared_engine(args, method) else resolve_env(args, method)
    qos_line = f"\n#SBATCH --qos={args.qos}" if args.qos is not None else ""
    is_3d = any(ndim_of(dataset) == 3 for dataset in datasets)
    gpu = args.gpu or (GPU_3D if is_3d else GPU_2D)
    memory = args.memory or (MEMORY_3D if is_3d else MEMORY_2D)
    # The datasets of a chunk run sequentially, and one failure must not skip the rest.
    commands = "\n".join(
        " ".join(build_command(args, dataset, model_type, method, mode)) for dataset in datasets
    )

    batch_script = f"""#!/bin/bash
#SBATCH -c {CPUS}
#SBATCH --mem {memory}
#SBATCH -t {args.time_limit}
#SBATCH -p {args.partition}
#SBATCH -G {gpu}
#SBATCH --job-name={tag}
#SBATCH --requeue{qos_line}
#SBATCH --constraint=inet
#SBATCH -o {job_folder}/logs/{tag}_%j.out
#SBATCH -e {job_folder}/logs/{tag}_%j.err

source ~/.bashrc
micromamba activate {env}
{env_exports()}
{commands}
"""

    with open(script_path, "w") as f:
        f.write(batch_script)
    return script_path


def submit_job(script: Path) -> None:
    """Hand one script to sbatch and print what it said."""
    result = subprocess.run(["sbatch", str(script)], capture_output=True, text=True)
    print(result.stdout.strip() if result.stdout else result.stderr.strip())


def chunked(datasets: tuple, datasets_per_job: int) -> list:
    """Split the datasets over jobs, striding rather than slicing.

    A strided split spreads the few slow datasets over the jobs instead of landing them in one.
    """
    n_chunks = max(1, math.ceil(len(datasets) / datasets_per_job))
    return [datasets[i::n_chunks] for i in range(n_chunks)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--data", nargs="+", default=None, choices=DATASETS,
                        help="Datasets to evaluate. Required unless --all_datasets is set.")
    parser.add_argument("--all_datasets", action="store_true", help="Evaluate every dataset of the selection.")
    parser.add_argument("--modality", default="all", choices=("lm", "em", "all"), help="Restrict to a modality.")
    parser.add_argument("--ndim", type=int, default=None, choices=(2, 3), help="Restrict to a dimensionality.")
    parser.add_argument("--segmentation_type", required=True, choices=("automatic", "interactive"))
    parser.add_argument("--segmentation_mode", nargs="+", default=["ais"], choices=SEGMENTATION_MODES,
                        help="The micro-sam2 automatic modes to run. One job group per mode.")
    parser.add_argument("--method", nargs="+", default=None,
                        help="Evaluate these baselines instead of micro-sam2. One job group per method.")
    parser.add_argument("-m", "--model_type", nargs="+", default=None,
                        help="Model types to evaluate. One job group per model type.")
    parser.add_argument("-e", "--experiment_folder", type=str, default=EXPERIMENT_FOLDER)
    parser.add_argument("--data_root", type=str, default=DATA_ROOT, help="The root the data lives in.")
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="Override the default checkpoint.")
    parser.add_argument("--joint_checkpoint", type=str, default="best",
                        help="Name of the joint trainer checkpoint the micro-sam2 weights are taken from, "
                             "without the '.pt' suffix, e.g. 'best' or the name of a frozen copy.")
    parser.add_argument("--skip_tuning", action="store_true", help="Evaluate micro-sam2 with the library defaults.")
    parser.add_argument("--tuning_root", type=str, default=None, help="Where parameter_search.py wrote its sweeps.")
    parser.add_argument("-p", "--prompt_choice", type=str, default="box", choices=("box", "point"))
    parser.add_argument("-iter", "--n_iterations", type=int, default=8, help="Iterative prompting rounds.")
    parser.add_argument("--min_size", type=int, default=0,
                        help="Drop ground-truth objects below this many pixels. The right value is dataset specific.")
    parser.add_argument("--datasets_per_job", type=int, default=1,
                        help="Datasets per Slurm job. Batching trades queue slots for walltime.")
    parser.add_argument("--partition", type=str, default=PARTITION, help="Slurm partition(s) to submit to.")
    parser.add_argument("--gpu", type=str, default=None,
                        help=f"Slurm GPU spec. Defaults to {GPU_2D} for 2d jobs and {GPU_3D} for 3d ones.")
    parser.add_argument("--memory", default=None,
                        help=f"Memory per job. Defaults to {MEMORY_2D} for 2d jobs and {MEMORY_3D} for 3d ones.")
    parser.add_argument("--time_limit", type=str, default=TIME_LIMIT, help="Slurm time limit per job.")
    parser.add_argument("--qos", type=str, default=None, help="Slurm QoS. Use '2h' with --time_limit 02:00:00.")
    parser.add_argument("--env", type=str, default=None,
                        help=f"Conda environment every job activates. Defaults to {DEFAULT_ENV}, "
                             "or the method-specific one in METHOD_ENV.")
    parser.add_argument("--dry", action="store_true", help="Only write the Slurm scripts; do not submit them.")
    args = parser.parse_args()

    if args.all_datasets and args.data is not None:
        raise ValueError("--all_datasets and -d/--data are mutually exclusive.")
    if not args.all_datasets and args.data is None:
        raise ValueError("Either -d/--data or --all_datasets must be given.")
    if args.checkpoint is not None and args.model_type is not None and len(args.model_type) > 1:
        raise ValueError("An explicit -c/--checkpoint cannot be shared by several model types.")

    valid_methods = AUTOMATIC_METHODS if args.segmentation_type == "automatic" else INTERACTIVE_METHODS
    for method in args.method or ():
        if method not in valid_methods:
            raise ValueError(f"'{method}' is not a {args.segmentation_type} method. Choose from {valid_methods}.")
    if args.method is None and args.model_type is None:
        args.model_type = list(MODEL_TYPES)

    # micro-sam2 runs one group per automatic mode; a baseline has no mode of its own.
    methods = args.method or [None]
    modes = args.segmentation_mode if (args.method is None and args.segmentation_type == "automatic") else [None]
    model_types = args.model_type or [None]

    job_folder = EVAL_ROOT / "gpu_jobs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    (job_folder / "logs").mkdir(parents=True, exist_ok=True)

    scripts = []
    for method in methods:
        for mode in modes:
            datasets = select_datasets(args, method, mode)
            if not datasets:
                print(f"Nothing to run for method={method}, mode={mode}: the selection is empty.")
                continue
            for model_type in model_types:
                for chunk_index, chunk in enumerate(chunked(datasets, args.datasets_per_job)):
                    scripts.append(
                        write_batch_script(args, job_folder, chunk, model_type, method, mode, chunk_index)
                    )

    print(f"Wrote {len(scripts)} Slurm scripts to '{job_folder}'.")
    warn_missing_envs({resolve_env(args, method) for method in methods if not uses_shared_engine(args, method)})
    if args.dry:
        return
    for script in scripts:
        submit_job(script)


if __name__ == "__main__":
    main()
