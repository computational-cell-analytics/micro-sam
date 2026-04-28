"""Generate a Slurm job for one v2 baseline evaluation command."""

import argparse
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional


EVAL_ROOT = Path(__file__).resolve().parent
AUTOMATIC_SCRIPT = EVAL_ROOT / "evaluate_automatic_baselines.py"
INTERACTIVE_SCRIPT = EVAL_ROOT / "evaluate_interactive_baselines.py"
VOLUMETRIC_SCRIPT = EVAL_ROOT / "evaluate_micro_sam_volumetric.py"

DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"
MICROSAMV2_AUTOMATIC_CHECKPOINT = (
    "/mnt/vast-nhr/projects/cidas/cca/models/micro_sam2/automatic/v1/checkpoints/unisam2-both/best.pt"
)
MICROSAMV2_INTERACTIVE_CHECKPOINT = (
    "/mnt/vast-nhr/projects/cidas/cca/models/micro_sam2/interactive/v1/checkpoints/checkpoint.pt"
)

# Keep these in sync with the argparse choices in the evaluation scripts.
AUTOMATIC_METHODS = (
    "cellpose",
    "stardist",
    "cellsam",
    "sam",
    "sam2",
    "micro_sam2",
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
DATASETS_3D_EM = ("lucchi", "platynereis_nuclei", "cremi", "snemi", "humanneurons")
DATASETS = tuple(sorted(set(DATASETS_2D + DATASETS_3D_LM + DATASETS_3D_EM)))

# Automatic methods that cannot run on EM datasets.
_MICROSAM_V1_METHODS = ("microsam_ais", "microsam_apg")
# Interactive methods that are 2D-only (SAM v1 raises ValueError for 3D).
_SAM_V1_INTERACTIVE_METHODS = ("sam", "micro-sam")
# Interactive methods that are 3D-only.
_3D_ONLY_INTERACTIVE_METHODS = ("microsam_vol",)


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def _command_from_args(args: argparse.Namespace) -> list[str]:
    if args.method == "microsam_vol":
        command = [
            "python", str(VOLUMETRIC_SCRIPT),
            "-d", args.dataset_name,
            "-i", DATA_ROOT,
            "-e", args.experiment_folder,
            "-p", args.prompt_choice,
        ]
        if args.model_type is not None:
            command.extend(["-m", args.model_type])
        if args.checkpoint is not None:
            command.extend(["-c", args.checkpoint])
        return command

    script = AUTOMATIC_SCRIPT if args.segmentation_mode == "automatic" else INTERACTIVE_SCRIPT
    checkpoint = _checkpoint_from_args(args)
    command = [
        "python", str(script),
        "-d", args.dataset_name,
        "-i", DATA_ROOT,
        "-e", args.experiment_folder,
        "--method", args.method,
    ]

    if args.model_type is not None:
        command.extend(["-m", args.model_type])
    if checkpoint is not None:
        command.extend(["-c", checkpoint])

    if args.segmentation_mode == "interactive":
        command.extend(["-p", args.prompt_choice, "-iter", str(args.n_iterations)])
        if args.ndim is not None:
            command.extend(["--ndim", str(args.ndim)])
        if args.use_masks:
            command.append("--use_masks")

    return command


def _checkpoint_from_args(args: argparse.Namespace) -> Optional[str]:
    if args.checkpoint is not None:
        return args.checkpoint
    if args.method != "micro_sam2":
        return None
    if args.segmentation_mode == "automatic":
        return MICROSAMV2_AUTOMATIC_CHECKPOINT
    return MICROSAMV2_INTERACTIVE_CHECKPOINT


def _job_tag(args: argparse.Namespace) -> str:
    parts = [args.segmentation_mode, args.dataset_name, args.method]
    if args.segmentation_mode == "interactive":
        parts.append(args.prompt_choice)
    if args.model_type is not None:
        parts.append(args.model_type)
    return "_".join(_sanitize(part) for part in parts)


def _write_batch_script(args: argparse.Namespace) -> Path:
    job_folder = Path("gpu_jobs_v2_eval")
    log_folder = Path("gpu_jobs_v2_eval/logs")
    job_folder.mkdir(parents=True, exist_ok=True)
    log_folder.mkdir(parents=True, exist_ok=True)

    tag = _job_tag(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_path = job_folder / f"{timestamp}_{tag}.sh"
    command = _command_from_args(args)

    batch_script = f"""#!/bin/bash
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 24:00:00
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A nim00007
#SBATCH --job-name=micro_sam2-eval

source ~/.bashrc
micromamba activate super

{" ".join(command)}
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


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentation_mode", required=True, choices=("automatic", "interactive"))
    parser.add_argument("-d", "--dataset_name", default=None, choices=DATASETS,
                        help="Dataset to evaluate. Required unless --all-datasets is set.")
    parser.add_argument("--all_datasets", "--all-datasets", action="store_true",
                        help="Submit one job per dataset. -d is ignored when this flag is set.")
    parser.add_argument(
        "-e", "--experiment_folder", type=str,
        default="/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/v0_evaluation",
    )
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("-m", "--model_type", type=str, default=None)
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("-p", "--prompt_choice", type=str, default="box", choices=PROMPT_CHOICES)
    parser.add_argument("-iter", "--n_iterations", type=int, default=8)
    parser.add_argument("--ndim", type=int, default=None, choices=(2, 3))
    parser.add_argument("--use_masks", action="store_true",
                        help="Pass --use_masks to the interactive evaluation script (SAM/SAM2 2D only).")
    parser.add_argument("--dry", action="store_true", help="Only write the Slurm script; do not submit it.")
    args = parser.parse_args(argv)

    job_folder = Path("gpu_jobs_v2_eval")
    if job_folder.exists():
        shutil.rmtree(job_folder)

    if args.all_datasets and args.dataset_name is not None:
        raise ValueError("--all-datasets and -d/--dataset_name are mutually exclusive.")
    if not args.all_datasets and args.dataset_name is None:
        raise ValueError("Either -d/--dataset_name or --all-datasets must be specified.")

    _validate_args(args)

    if args.all_datasets:
        if args.ndim == 2:
            datasets = DATASETS_2D
        elif args.ndim == 3:
            datasets = tuple(sorted(set(DATASETS_3D_LM + DATASETS_3D_EM)))
        else:
            datasets = DATASETS
        if args.method in _MICROSAM_V1_METHODS:
            datasets = tuple(d for d in datasets if d not in DATASETS_3D_EM)
        if args.segmentation_mode == "interactive" and args.method in _SAM_V1_INTERACTIVE_METHODS:
            datasets = tuple(d for d in datasets if d in DATASETS_2D)
        if args.segmentation_mode == "interactive" and args.method in _3D_ONLY_INTERACTIVE_METHODS:
            datasets = tuple(d for d in datasets if d in set(DATASETS_3D_LM))
        for dataset in datasets:
            args.dataset_name = dataset
            script = _write_batch_script(args)
            print(f"Wrote Slurm script to '{script}'.")
            if not args.dry:
                _submit_job(script)
    else:
        script = _write_batch_script(args)
        print(f"Wrote Slurm script to '{script}'.")
        if not args.dry:
            _submit_job(script)


if __name__ == "__main__":
    main()
