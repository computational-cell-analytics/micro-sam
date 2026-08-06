"""Generate and submit the Slurm jobs for the automatic-segmentation grid search.

One job per (model type, dataset) writes '<output_root>/<model_type>/<dataset>.csv'. The best row of
each CSV is picked up by submit_all_evaluations.py via '--grid_search_root'.

Usage examples:
    # Every dataset for the jointly finetuned models.
    python submit_grid_search.py -m hvit_t hvit_s hvit_b --all_datasets

    # A single dataset, without submitting.
    python submit_grid_search.py -m hvit_b -d cremi --dry
"""

import argparse
import subprocess
from pathlib import Path
from typing import Optional
from datetime import datetime


EVAL_ROOT = Path(__file__).resolve().parent
GRID_SEARCH_SCRIPT = EVAL_ROOT / "grid_search_automatic_cells.py"

OUTPUT_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/grid-search-joint-v2"

MODEL_TYPES = ("hvit_t", "hvit_s", "hvit_b", "hvit_l")

# Keep these in sync with the dataset lists in common.py.
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

# The grid search is postprocessing bound, so it gets more cores than the evaluation jobs.
PARTITION = "grete-h100:shared,grete:shared"
ACCOUNT = "nim00007"
GPU = "1"
CPUS = 8
MEMORY = "32G"
TIME_LIMIT = "12:00:00"


def _command(args: argparse.Namespace, model_type: str, dataset_name: str) -> list[str]:
    return [
        "python", str(GRID_SEARCH_SCRIPT),
        "-d", dataset_name,
        "-m", model_type,
        "-o", str(Path(args.output_root) / model_type),
        "--n_threads", str(CPUS),
    ]


def _write_batch_script(args: argparse.Namespace, job_folder: Path, model_type: str, dataset_name: str) -> Path:
    tag = f"grid_search_{dataset_name}_{model_type}"
    script_path = job_folder / f"{tag}.sh"
    command = _command(args, model_type, dataset_name)

    batch_script = f"""#!/bin/bash
#SBATCH -c {CPUS}
#SBATCH --mem {MEMORY}
#SBATCH -t {TIME_LIMIT}
#SBATCH -p {PARTITION}
#SBATCH -G {GPU}
#SBATCH -A {ACCOUNT}
#SBATCH --job-name={tag}
#SBATCH --constraint=inet
#SBATCH -o {job_folder}/logs/{tag}_%j.out
#SBATCH -e {job_folder}/logs/{tag}_%j.err

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


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", nargs="+", required=True, choices=MODEL_TYPES,
                        help="Model types to tune. One job is submitted per model type and dataset.")
    parser.add_argument("-d", "--dataset_name", nargs="+", default=None, choices=DATASETS,
                        help="Datasets to tune on. Required unless --all_datasets is set.")
    parser.add_argument("--all_datasets", action="store_true", help="Tune on every dataset.")
    parser.add_argument("--lm_only", action="store_true", help="Skip the 3d EM datasets.")
    parser.add_argument("-o", "--output_root", default=OUTPUT_ROOT, help="Root directory for the result CSVs.")
    parser.add_argument("--dry", action="store_true", help="Only write the Slurm scripts; do not submit them.")
    args = parser.parse_args(argv)

    if args.all_datasets and args.dataset_name is not None:
        raise ValueError("--all_datasets and -d/--dataset_name are mutually exclusive.")
    if not args.all_datasets and args.dataset_name is None:
        raise ValueError("Either -d/--dataset_name or --all_datasets must be specified.")

    datasets = args.dataset_name if args.dataset_name else DATASETS
    if args.lm_only:
        datasets = tuple(d for d in datasets if d not in DATASETS_3D_EM)

    job_folder = EVAL_ROOT / "gpu_jobs_grid_search" / datetime.now().strftime("%Y%m%d_%H%M%S")
    (job_folder / "logs").mkdir(parents=True, exist_ok=True)

    scripts = [
        _write_batch_script(args, job_folder, model_type, dataset_name)
        for model_type in args.model_type for dataset_name in datasets
    ]
    print(f"Wrote {len(scripts)} Slurm scripts to '{job_folder}'.")

    if args.dry:
        return
    for script in scripts:
        _submit_job(script)


if __name__ == "__main__":
    main()
