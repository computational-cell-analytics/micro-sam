"""Generate and submit the Slurm jobs for the v2 evaluation.

Usage examples:
    # The jointly finetuned models on all LM and EM datasets, interactive with box prompts.
    python submit_all_evaluations.py --segmentation_mode interactive --method micro_sam2 \
        -m hvit_t hvit_s hvit_b --all_datasets -p box

    # The same models, automatic segmentation, restricted to the LM datasets.
    python submit_all_evaluations.py --segmentation_mode automatic --method micro_sam2 \
        -m hvit_t hvit_s hvit_b --all_datasets --lm_only
"""

import os
import re
import sys
import json
import math
import shlex
import argparse
import subprocess
from pathlib import Path
from typing import Optional
from datetime import datetime

# Written into every job script, so a queued job sees the configuration the submission had rather than
# whatever the environment holds when it starts. The sample caps are global, so lift them per dataset.
PINNED_ENV_VARS = (
    "MICRO_SAM2_JOINT_CHECKPOINT_ROOT",
    "MICRO_SAM2_JOINT_EXPORT_ROOT",
    "MICRO_SAM_EVAL_MAX_SAMPLES",
    "MICRO_SAM_LIVECELL_PER_CELL_TYPE",
)


def env_exports() -> str:
    """Return 'export' lines pinning the run configuration, or an empty string if none is set."""
    return "".join(f"export {name}={shlex.quote(os.environ[name])}\n" for name in PINNED_ENV_VARS if name in os.environ)


EVAL_ROOT = Path(__file__).resolve().parent
AUTOMATIC_SCRIPT = EVAL_ROOT / "evaluate_automatic_baselines.py"
INTERACTIVE_SCRIPT = EVAL_ROOT / "evaluate_interactive_baselines.py"
VOLUMETRIC_SCRIPT = EVAL_ROOT / "evaluate_micro_sam_volumetric.py"

DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"

# Slurm resources per job. The account only has access to the grete partitions. The jobs peak at
# ~3 GiB VRAM for interactive segmentation, and the automatic encoder batch adapts to the free VRAM,
# so any GPU works and both the H100 and the A100 pool are used. The time limit is kept close to the
# observed worst case, since an oversized request makes a job ineligible for backfill.
# 'grete:preemptible' is usually free and starts within minutes, where the shared pools queue for
# days. It is MIG only, so the GPU needs a type. A 1g.10gb slice covers interactive segmentation,
# which peaks at ~3 GiB. Preemption is safe because predictions persist and jobs are requeued.
# Note that 'grete-h100:shared' does not offer this slice; override --gpu with '1' to use it.
PARTITION = "grete:preemptible"
ACCOUNT = "nim00007"
GPU = "1g.10gb:1"
CPUS = 4
MEMORY = "16G"
TIME_LIMIT = "08:00:00"

# Keep these in sync with the argparse choices in the evaluation scripts.
AUTOMATIC_METHODS = (
    "cellpose",
    "stardist",
    "cellsam",
    "sam",
    "sam2",
    "micro_sam2",
    "micro_sam2_apg",
    "microsam_ais",
    "microsam_apg",
    "segneuron",
)

INTERACTIVE_METHODS = (
    "nninteractive",
    "sam3",
    "sam",
    "sam2",
    "micro-sam",
    "micro_sam2",
    "microsam_vol",
)
PROMPT_CHOICES = ("box", "point")


DATASETS_2D = (
    "livecell",
    "arvidsson", "bitdepth_nucseg", "cellbindb", "cellpose_data",
    "covid_if", "cvz_fluo", "deepbacs", "deepseas", "dic_hepg2", "dsb",
    "dynamicnuclearnet", "hpa", "microbeseg", "neurips_cellseg", "omnipose",
    "segpc", "tissuenet", "usiigaci", "vicar", "yeaz",
)
DATASETS_3D_LM = (
    "blastospim", "cartocell", "celegans_atlas", "cellseg_3d", "embedseg",
    "gonuclear", "mouse_embryo", "nis3d", "plantseg", "pnas_arabidopsis",
)
DATASETS_3D_EM = ("platynereis_nuclei", "cremi", "snemi", "humanneurons")
DATASETS = tuple(sorted(set(DATASETS_2D + DATASETS_3D_LM + DATASETS_3D_EM)))

# Automatic methods that cannot run on EM datasets.
_MICROSAM_V1_METHODS = ("sam", "microsam_ais", "microsam_apg")
# Automatic methods that are 2D-only. Volumetric APG has its own script, see submit_apg_3d.py.
_2D_ONLY_AUTO_METHODS = ("cellsam", "sam2", "micro_sam2_apg")
# Interactive methods that are 2D-only.
_SAM_V1_INTERACTIVE_METHODS = ("sam", "micro-sam", "sam3")
# Interactive methods that are 3D-only.
_3D_ONLY_INTERACTIVE_METHODS = ("microsam_vol", "nninteractive")

# Methods that require a non-default conda environment.
_METHOD_ENV = {
    "cellpose": "cp3",
    "stardist": "sd",
}


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def _tuned_params(grid_search_root: str, dataset_name: str, model_type: str) -> str:
    """Return the best parameter combination of a grid search as a JSON string."""
    sys.path.insert(0, str(EVAL_ROOT))
    from common import read_tuned_params

    return json.dumps(read_tuned_params(grid_search_root, dataset_name, model_type))


def _command(args: argparse.Namespace, dataset_name: str, model_type: Optional[str]) -> list[str]:
    if args.method == "microsam_vol":
        command = [
            "python", str(VOLUMETRIC_SCRIPT),
            "-d", dataset_name,
            "-i", DATA_ROOT,
            "-e", args.experiment_folder,
            "-p", args.prompt_choice,
        ]
        if model_type is not None:
            command.extend(["-m", model_type])
        if args.checkpoint is not None:
            command.extend(["-c", args.checkpoint])
        return command

    script = AUTOMATIC_SCRIPT if args.segmentation_mode == "automatic" else INTERACTIVE_SCRIPT
    command = [
        "python", str(script),
        "-d", dataset_name,
        "-i", DATA_ROOT,
        "-e", args.experiment_folder,
        "--method", args.method,
    ]

    if model_type is not None:
        command.extend(["-m", model_type])
    if args.checkpoint is not None:
        command.extend(["-c", args.checkpoint])
    if args.method in ("micro_sam2", "micro_sam2_apg"):
        command.extend(["--joint_checkpoint", args.joint_checkpoint])
    if args.grid_search_root is not None:
        # APG is tuned over its own grid and takes its parameters on a different flag, which is why
        # the AIS and the APG sweeps have to be given separate roots.
        flag = "--apg_params" if args.method == "micro_sam2_apg" else "--postprocessing_params"
        command.extend([flag, shlex.quote(_tuned_params(args.grid_search_root, dataset_name, model_type))])

    if args.segmentation_mode == "interactive":
        command.extend(["-p", args.prompt_choice, "-iter", str(args.n_iterations)])
        if args.ndim is not None:
            command.extend(["--ndim", str(args.ndim)])
        if args.use_masks is not None:
            command.append("--use_masks" if args.use_masks else "--no-use_masks")
        if args.min_size:
            command.extend(["--min_size", str(args.min_size)])

    return command


def _job_tag(
    args: argparse.Namespace, datasets: tuple, model_type: Optional[str], chunk_index: int
) -> str:
    # A batched job spans several datasets, so it is named after its chunk instead.
    name = datasets[0] if len(datasets) == 1 else f"chunk{chunk_index:02d}"
    parts = [args.segmentation_mode, name, args.method]
    if args.segmentation_mode == "interactive":
        parts.append(args.prompt_choice)
    if model_type is not None:
        parts.append(model_type)
    return "_".join(_sanitize(part) for part in parts)


def _write_batch_script(
    args: argparse.Namespace, job_folder: Path, datasets: tuple, model_type: Optional[str], chunk_index: int = 0
) -> Path:
    tag = _job_tag(args, datasets, model_type, chunk_index)
    script_path = job_folder / f"{tag}.sh"
    env = _METHOD_ENV.get(args.method, "super")
    qos_line = f"\n#SBATCH --qos={args.qos}" if args.qos is not None else ""
    # The datasets of a chunk run sequentially, and one failure must not skip the rest.
    commands = "\n".join(" ".join(_command(args, dataset, model_type)) for dataset in datasets)

    batch_script = f"""#!/bin/bash
#SBATCH -c {CPUS}
#SBATCH --mem {args.memory}
#SBATCH -t {args.time_limit}
#SBATCH -p {args.partition}
#SBATCH -G {args.gpu}
#SBATCH -A {args.account}
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


def _submit_job(script: Path) -> None:
    result = subprocess.run(["sbatch", str(script)], capture_output=True, text=True)
    print(result.stdout.strip() if result.stdout else result.stderr.strip())


def _validate_args(args: argparse.Namespace) -> None:
    methods = AUTOMATIC_METHODS if args.segmentation_mode == "automatic" else INTERACTIVE_METHODS
    if args.method not in methods:
        raise ValueError(f"Method {args.method!r} is not valid for {args.segmentation_mode!r} segmentation.")


def _select_datasets(args: argparse.Namespace) -> tuple:
    if not args.all_datasets:
        return tuple(args.dataset_name)

    if args.ndim == 2:
        datasets = DATASETS_2D
    elif args.ndim == 3:
        datasets = tuple(sorted(set(DATASETS_3D_LM + DATASETS_3D_EM)))
    else:
        datasets = DATASETS

    if args.lm_only or args.method in _MICROSAM_V1_METHODS:
        datasets = tuple(d for d in datasets if d not in DATASETS_3D_EM)
    if args.segmentation_mode == "automatic" and args.method in _2D_ONLY_AUTO_METHODS:
        datasets = tuple(d for d in datasets if d in DATASETS_2D)
    if args.segmentation_mode == "interactive" and args.method in _SAM_V1_INTERACTIVE_METHODS:
        datasets = tuple(d for d in datasets if d in DATASETS_2D)
    if args.segmentation_mode == "interactive" and args.method in _3D_ONLY_INTERACTIVE_METHODS:
        datasets = tuple(d for d in datasets if d in set(DATASETS_3D_LM))
    return datasets


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentation_mode", required=True, choices=("automatic", "interactive"))
    parser.add_argument("-d", "--dataset_name", nargs="+", default=None, choices=DATASETS,
                        help="Datasets to evaluate. Required unless --all-datasets is set.")
    parser.add_argument("--all_datasets", "--all-datasets", action="store_true",
                        help="Submit one job per dataset. -d is ignored when this flag is set.")
    parser.add_argument("--lm_only", action="store_true", help="Skip the 3d EM datasets.")
    parser.add_argument(
        "-e", "--experiment_folder", type=str,
        default="/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/v2_joint_evaluation",
    )
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("-m", "--model_type", type=str, nargs="+", default=None,
                        help="Model types to evaluate. One job is submitted per model type.")
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("--joint_checkpoint", type=str, default="best",
                        help="Name of the joint trainer checkpoint the micro_sam2 weights are taken from, "
                             "without the '.pt' suffix, e.g. 'best' or the name of a frozen copy.")
    parser.add_argument("--grid_search_root", type=str, default=None,
                        help="Run automatic segmentation with the best parameters found under this root.")
    parser.add_argument("--datasets_per_job", type=int, default=1,
                        help="Datasets per Slurm job. Batching trades queue slots for walltime.")
    parser.add_argument("--time_limit", type=str, default=TIME_LIMIT, help="Slurm time limit per job.")
    parser.add_argument("--partition", type=str, default=PARTITION, help="Slurm partition(s) to submit to.")
    parser.add_argument("--account", default=ACCOUNT, help="Slurm account to charge the jobs to.")
    parser.add_argument("--memory", default=MEMORY, help="Memory per job.")
    parser.add_argument(
        "--gpu", type=str, default=GPU,
        help="Slurm GPU spec. MIG partitions need a type, e.g. '1g.10gb:1' on grete:preemptible."
    )
    parser.add_argument(
        "--min_size", type=int, default=0,
        help="Drop ground-truth objects below this many pixels. The right value is dataset specific."
    )
    parser.add_argument("--qos", type=str, default=None, help="Slurm QoS. Use '2h' with --time_limit 02:00:00.")
    parser.add_argument("-p", "--prompt_choice", type=str, default="box", choices=PROMPT_CHOICES)
    parser.add_argument("-iter", "--n_iterations", type=int, default=8)
    parser.add_argument("--ndim", type=int, default=None, choices=(2, 3))
    parser.add_argument(
        "--use_masks", action=argparse.BooleanOptionalAction, default=None,
        help="Override the mask-prompt default of the interactive evaluation script (SAM/SAM2 2D only)."
    )
    parser.add_argument("--dry", action="store_true", help="Only write the Slurm scripts; do not submit them.")
    args = parser.parse_args(argv)

    if args.all_datasets and args.dataset_name is not None:
        raise ValueError("--all-datasets and -d/--dataset_name are mutually exclusive.")
    if not args.all_datasets and args.dataset_name is None:
        raise ValueError("Either -d/--dataset_name or --all-datasets must be specified.")
    if args.checkpoint is not None and args.model_type is not None and len(args.model_type) > 1:
        raise ValueError("An explicit -c/--checkpoint cannot be shared by multiple model types.")

    _validate_args(args)

    job_folder = EVAL_ROOT / "gpu_jobs_v2_eval" / datetime.now().strftime("%Y%m%d_%H%M%S")
    (job_folder / "logs").mkdir(parents=True, exist_ok=True)

    model_types = args.model_type if args.model_type else [None]
    datasets = _select_datasets(args)
    # Strided rather than contiguous chunks, so the few slow datasets spread over the jobs instead
    # of landing in the same one.
    n_chunks = math.ceil(len(datasets) / args.datasets_per_job)
    chunks = [datasets[i::n_chunks] for i in range(n_chunks)]
    scripts = [
        _write_batch_script(args, job_folder, chunk, model_type, chunk_index)
        for model_type in model_types for chunk_index, chunk in enumerate(chunks)
    ]
    print(f"Wrote {len(scripts)} Slurm scripts to '{job_folder}'.")

    if args.dry:
        return
    for script in scripts:
        _submit_job(script)


if __name__ == "__main__":
    main()
