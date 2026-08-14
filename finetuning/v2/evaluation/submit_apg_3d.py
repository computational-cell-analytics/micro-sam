"""Generate and submit the Slurm jobs for the volumetric automatic prompt generation evaluation.

One job per dataset runs 'evaluate_3d.py --mode apg', which evaluates AIS and every APG variant from
a single initialization of each volume and writes one CSV per variant.

Usage examples:
    # The default ablation on the 3d LM datasets.
    python submit_apg_3d.py --lm_only

    # A candidate threshold sweep on two datasets, without submitting.
    python submit_apg_3d.py -d gonuclear embedseg --sweep candidate_threshold 0.5 1.5 4.0 --dry
"""

import os
import json
import shlex
import argparse
import subprocess
from pathlib import Path
from typing import Optional
from datetime import datetime

EVAL_ROOT = Path(__file__).resolve().parent
EVALUATION_SCRIPT = EVAL_ROOT / "evaluate_3d.py"

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


OUTPUT_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/apg-3d"

DATASETS_3D_LM = (
    "blastospim", "cartocell", "celegans_atlas", "cellseg_3d", "embedseg",
    "gonuclear", "mouse_embryo", "nis3d", "plantseg", "pnas_arabidopsis",
)
DATASETS_3D_EM = ("platynereis_nuclei", "cremi", "snemi", "humanneurons")
DATASETS = DATASETS_3D_LM + DATASETS_3D_EM

# The refinement round ablated into the mechanisms it combines, so a change can be attributed.
DEFAULT_VARIANTS = {
    "round1": {},
    "control": {"refine_prompts": True, "max_positive_prompts": 1, "max_negative_prompts": 0},
    "negatives": {"refine_prompts": True, "max_positive_prompts": 1},
    "positives": {"refine_prompts": True, "max_negative_prompts": 0},
    "round2": {"refine_prompts": True},
}

# These two hold 250 and 124 volumes where every other dataset holds at most 14, so uncapped they
# take the whole sweep and time out.
VOLUME_CAP = {"blastospim": 20, "pnas_arabidopsis": 20}

PARTITION = "grete:preemptible"
ACCOUNT = "nim00007"
GPU = "1"
CPUS = 8
MEMORY = "64G"
TIME_LIMIT = "04:00:00"


def _tuned_params(grid_search_root: str, dataset_name: str, model_type: str) -> dict:
    """The best combination of a grid search, read the same way the 2d submission reads it."""
    import sys
    sys.path.insert(0, str(EVAL_ROOT))
    from common import read_tuned_params

    return read_tuned_params(grid_search_root, dataset_name, model_type)


def _variants(args: argparse.Namespace, dataset_name: str) -> dict:
    """The variants to evaluate: the tuned combination, an explicit set, a sweep, or the ablation."""
    if args.apg_grid_search_root is not None:
        return {"tuned": _tuned_params(args.apg_grid_search_root, dataset_name, args.model_type)}
    if args.variants is not None:
        return json.loads(args.variants)
    if args.sweep is None:
        return DEFAULT_VARIANTS
    name, *values = args.sweep
    variants = {"round1": {}}
    for value in values:
        variants[f"{name}_{value}"] = {name: json.loads(value)}
    return variants


def _command(args: argparse.Namespace, dataset_name: str, variants: dict) -> list[str]:
    command = [
        "python", str(EVALUATION_SCRIPT),
        "-d", dataset_name,
        "-m", args.model_type,
        "-e", str(Path(args.output_root) / dataset_name),
        "--mode", "apg",
        "--crop_depth", str(args.crop_depth),
        "--tag", args.tag,
        "--joint_checkpoint", args.joint_checkpoint,
        "--apg_params", json.dumps(variants),
    ]
    # The AIS reference is post-processed with its own tuned parameters, so the two columns of the
    # result differ only in how the shared prediction is turned into instances.
    if args.ais_grid_search_root is not None:
        command.extend(["--ais_params", json.dumps(
            _tuned_params(args.ais_grid_search_root, dataset_name, args.model_type)
        )])
    cap = VOLUME_CAP.get(dataset_name)
    if cap is not None:
        command.extend(["--n_volumes", str(cap)])
    return command


def _write_batch_script(args: argparse.Namespace, job_folder: Path, tag: str, command: list[str]) -> Path:
    script_path = job_folder / f"{tag}.sh"
    directives = [
        f"#SBATCH -c {args.cpus}",
        f"#SBATCH --mem {args.memory}",
        f"#SBATCH -t {args.time_limit}",
        f"#SBATCH -p {args.partition}",
        f"#SBATCH -G {args.gpu}",
        f"#SBATCH -A {args.account}",
        f"#SBATCH --job-name={tag}",
        "#SBATCH --constraint=inet",
        *([f"#SBATCH --exclude={args.exclude}"] if args.exclude else []),
        f"#SBATCH -o {job_folder}/logs/{tag}_%j.out",
        f"#SBATCH -e {job_folder}/logs/{tag}_%j.err",
    ]
    # The command holds a json argument, so it is quoted rather than joined verbatim.
    call = " ".join(f"'{part}'" if part.startswith("{") else part for part in command)
    batch_script = "#!/bin/bash\n{}\n\nsource ~/.bashrc\nmicromamba activate super\n{}\n{}\n".format(
        "\n".join(directives), env_exports(), call
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
    parser.add_argument("-d", "--dataset_name", nargs="+", default=None, choices=DATASETS,
                        help="Datasets to evaluate. Defaults to every 3d dataset.")
    parser.add_argument("--lm_only", action="store_true", help="Skip the 3d EM datasets.")
    parser.add_argument("-m", "--model_type", default="hvit_t", help="SAM2 backbone of the joint model.")
    parser.add_argument("-o", "--output_root", default=OUTPUT_ROOT, help="Root directory for the results.")
    parser.add_argument("--crop_depth", type=int, default=8, help="Number of slices each volume is cropped to.")
    parser.add_argument("--tag", default="apg", help="Prefix of the result files.")
    parser.add_argument("--sweep", nargs="+", default=None,
                        help="Sweep one 'generate' parameter, e.g. --sweep candidate_threshold 0.5 1.5.")
    parser.add_argument("--variants", default=None,
                        help="JSON mapping a variant name to its 'generate' overrides. Overrides --sweep.")
    parser.add_argument("--apg_grid_search_root", default=None,
                        help="Evaluate the prompt generation with the best parameters found under this root.")
    parser.add_argument("--ais_grid_search_root", default=None,
                        help="Post-process the AIS reference with the best parameters found under this root.")
    parser.add_argument("--joint_checkpoint", default="best",
                        help="Name of the joint trainer checkpoint, without the '.pt' suffix.")
    parser.add_argument("--partition", default=PARTITION, help="Slurm partition(s) to submit to.")
    parser.add_argument("--account", default=ACCOUNT, help="Slurm account to charge the jobs to.")
    parser.add_argument("--gpu", default=GPU,
                        help="Slurm GPU request. '1' takes any, including a MIG slice too small for "
                             "a large backbone or a deep volume; name a type, e.g. 'A100:1', to avoid that.")
    parser.add_argument("--exclude", default=None, help="Nodes to keep the jobs off, e.g. a faulty one.")
    parser.add_argument("--cpus", type=int, default=CPUS, help="Cores per job.")
    parser.add_argument("--memory", default=MEMORY, help="Memory per job.")
    parser.add_argument("--time_limit", default=TIME_LIMIT, help="Time limit per job.")
    parser.add_argument("--dry", action="store_true", help="Only write the Slurm scripts; do not submit them.")
    args = parser.parse_args(argv)

    datasets = args.dataset_name if args.dataset_name else DATASETS
    if args.lm_only:
        datasets = tuple(dataset for dataset in datasets if dataset not in DATASETS_3D_EM)

    job_folder = EVAL_ROOT / "gpu_jobs_apg_3d" / datetime.now().strftime("%Y%m%d_%H%M%S")
    (job_folder / "logs").mkdir(parents=True, exist_ok=True)

    print(f"Submitting {len(datasets)} jobs to '{job_folder}'.")
    for dataset_name in datasets:
        # Resolved per dataset, since tuned parameters are selected per dataset.
        variants = _variants(args, dataset_name)
        print(f"{dataset_name}: variants {list(variants)}")
        script = _write_batch_script(
            args, job_folder, f"apg3d_{dataset_name}", _command(args, dataset_name, variants)
        )
        if not args.dry:
            _submit_job(script)


if __name__ == "__main__":
    main()
