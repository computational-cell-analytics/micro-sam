"""Generate and submit the Slurm jobs for the automatic-segmentation grid search.

One job per (model type, dataset) writes '<output_root>/<model_type>/<dataset>.csv'. The best row of
each CSV is picked up by submit_all_evaluations.py via '--grid_search_root'.

Usage examples:
    # Every dataset for the jointly finetuned models.
    python submit_grid_search.py -m hvit_t hvit_s hvit_b --all_datasets

    # A single dataset, without submitting.
    python submit_grid_search.py -m hvit_b -d cremi --dry
"""

import os
import shlex
import argparse
import subprocess
from pathlib import Path
from typing import Optional
from datetime import datetime


EVAL_ROOT = Path(__file__).resolve().parent
GRID_SEARCH_SCRIPT = EVAL_ROOT / "grid_search_automatic_cells.py"

# Written into every job script, so a queued job sees the configuration the submission had rather than
# whatever the environment holds when it starts.
PINNED_ENV_VARS = (
    "MICRO_SAM2_JOINT_CHECKPOINT_ROOT",
    "MICRO_SAM2_JOINT_EXPORT_ROOT",
)


def env_exports() -> str:
    """Return 'export' lines pinning the run configuration, or an empty string if none is set."""
    return "".join(f"export {name}={shlex.quote(os.environ[name])}\n" for name in PINNED_ENV_VARS if name in os.environ)


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
# 'grete:preemptible' is MIG only, so the GPU needs a type. Preemption is safe: a finished sweep
# writes its CSV, and a requeued job restarts the one it did not finish.
PARTITION = "grete:preemptible"
GPU = "1g.10gb:1"
CPUS = 8
MEMORY = "32G"
TIME_LIMIT = "12:00:00"


def _command(
    args: argparse.Namespace, model_type: str, dataset_name: str,
    shard: Optional[int] = None, merge: bool = False,
) -> list[str]:
    command = [
        "python", str(GRID_SEARCH_SCRIPT),
        "-d", dataset_name,
        "-m", model_type,
        "-o", str(Path(args.output_root) / model_type),
        "--split", args.split,
        "--tune", args.tune,
        "--joint_checkpoint", args.joint_checkpoint,
        "--livecell_per_celltype", str(args.livecell_per_celltype),
        "--n_threads", str(args.cpus),
    ]
    if args.n_images is not None:
        command.extend(["-n", str(args.n_images)])
    if args.n_shards > 1:
        command.extend(["--n_shards", str(args.n_shards)])
        command.extend(["--merge"] if merge else ["--shard", str(shard)])
    return command


def _write_batch_script(
    args: argparse.Namespace, job_folder: Path, tag: str, command: list[str],
    heavy: bool = True, dependency: Optional[str] = None,
) -> Path:
    """Write one Slurm script. 'heavy' sizes a scoring job; a merge job only reads CSVs.

    The GPU is requested either way, because the gpu partitions reject a job without one.
    """
    script_path = job_folder / f"{tag}.sh"
    directives = [
        f"#SBATCH -c {args.cpus if heavy else 2}",
        f"#SBATCH --mem {args.memory}",
        f"#SBATCH -t {TIME_LIMIT if heavy else '00:30:00'}",
        f"#SBATCH -p {args.partition}",
        f"#SBATCH -G {args.gpu}",
        f"#SBATCH --job-name={tag}",
        "#SBATCH --requeue",
        "#SBATCH --constraint=inet",
        f"#SBATCH -o {job_folder}/logs/{tag}_%j.out",
        f"#SBATCH -e {job_folder}/logs/{tag}_%j.err",
    ]
    if dependency is not None:
        directives.append(f"#SBATCH --dependency=afterok:{dependency}")

    batch_script = "#!/bin/bash\n{}\n\nsource ~/.bashrc\nmicromamba activate super\n{}\n{}\n".format(
        "\n".join(directives), env_exports(), " ".join(command)
    )
    with open(script_path, "w") as f:
        f.write(batch_script)
    return script_path


def _submit_job(script: Path) -> Optional[str]:
    result = subprocess.run(["sbatch", str(script)], capture_output=True, text=True)
    output = result.stdout.strip() if result.stdout else result.stderr.strip()
    print(output)
    return output.split()[-1] if output.startswith("Submitted batch job") else None


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", nargs="+", required=True, choices=MODEL_TYPES,
                        help="Model types to tune. One job is submitted per model type and dataset.")
    parser.add_argument("-d", "--dataset_name", nargs="+", default=None, choices=DATASETS,
                        help="Datasets to tune on. Required unless --all_datasets is set.")
    parser.add_argument("--all_datasets", action="store_true", help="Tune on every dataset.")
    parser.add_argument("--lm_only", action="store_true", help="Skip the 3d EM datasets.")
    parser.add_argument("-o", "--output_root", default=OUTPUT_ROOT, help="Root directory for the result CSVs.")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Split to tune on.")
    parser.add_argument("--tune", default="ais", choices=["ais", "apg"],
                        help="What to tune: the AIS postprocessing or the automatic prompt generation.")
    parser.add_argument("--joint_checkpoint", default="best",
                        help="Name of the joint trainer checkpoint to tune, without the '.pt' suffix.")
    parser.add_argument("-n", "--n_images", type=int, default=None, help="Cap images per 2d dataset.")
    parser.add_argument("--partition", default=PARTITION, help="Slurm partition(s) to submit to.")
    parser.add_argument("--gpu", default=GPU,
                        help="Slurm GPU spec. MIG partitions need a type, e.g. '1g.10gb:1' on grete:preemptible.")
    parser.add_argument("--cpus", type=int, default=CPUS, help="Cores per job, also the postprocessing thread count.")
    parser.add_argument("--memory", default=MEMORY, help="Memory per job.")
    parser.add_argument("--livecell_per_celltype", type=int, default=25, help="Images per livecell cell type.")
    parser.add_argument("--n_shards", type=int, default=1, help="Split each sweep over this many jobs.")
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

    # One job per shard, plus a CPU-only merge job that Slurm starts once every shard of that
    # (model, dataset) has finished successfully.
    plans = []
    for model_type in args.model_type:
        for dataset_name in datasets:
            base = f"grid_search_{dataset_name}_{model_type}"
            if args.n_shards == 1:
                shards = [(base, _command(args, model_type, dataset_name))]
            else:
                shards = [
                    (f"{base}_shard{i}", _command(args, model_type, dataset_name, shard=i))
                    for i in range(args.n_shards)
                ]
            merge = None
            if args.n_shards > 1:
                merge = (f"{base}_merge", _command(args, model_type, dataset_name, merge=True))
            plans.append((shards, merge))

    n_scripts = sum(len(shards) + (merge is not None) for shards, merge in plans)
    print(f"Writing {n_scripts} Slurm scripts to '{job_folder}'.")

    if args.dry:
        for shards, merge in plans:
            for tag, command in shards:
                _write_batch_script(args, job_folder, tag, command)
            if merge is not None:
                _write_batch_script(args, job_folder, merge[0], merge[1], heavy=False, dependency="PENDING")
        return

    for shards, merge in plans:
        job_ids = []
        for tag, command in shards:
            job_id = _submit_job(_write_batch_script(args, job_folder, tag, command))
            if job_id is not None:
                job_ids.append(job_id)
        if merge is None:
            continue
        if len(job_ids) != len(shards):
            print(f"Not all shards of '{merge[0]}' were submitted, skipping the merge job.")
            continue
        script = _write_batch_script(args, job_folder, merge[0], merge[1], heavy=False, dependency=":".join(job_ids))
        _submit_job(script)


if __name__ == "__main__":
    main()
