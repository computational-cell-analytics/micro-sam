"""Report which interactive evaluations are complete and optionally resubmit the rest.

Slurm requeues preempted jobs by itself, and they resume from the predictions already on disk.
This covers the other cases: jobs that failed, were cancelled, or never ran.

Usage examples:
    python check_evaluation_status.py --method micro_sam2 -m hvit_t hvit_s hvit_b --all_datasets --ndim 2
    python check_evaluation_status.py --method micro_sam2 -m hvit_t --all_datasets --ndim 2 --resubmit
"""

import os
import argparse
import subprocess

from baselines_common import interactive_result_name
from submit_all_evaluations import DATASETS_2D, DATASETS_3D_LM, DATASETS_3D_EM, EVAL_ROOT

DATASETS_3D = tuple(sorted(set(DATASETS_3D_LM + DATASETS_3D_EM)))

EXPERIMENT_FOLDER = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/v2_joint_evaluation"


def result_paths(experiment_folder, dataset, method, model_type, prompt, n_iterations, args):
    results_dir = os.path.join(experiment_folder, "results")
    ndim = 3 if dataset in DATASETS_3D else 2
    return [
        os.path.join(results_dir, interactive_result_name(
            dataset, method, model_type, prompt, it, ndim=ndim,
            use_masks=args.use_masks, mask_threshold=args.mask_threshold, min_size=args.min_size,
        ))
        for it in range(n_iterations)
    ]


def scan(args, datasets):
    """Return (complete, partial, missing) as lists of (dataset, model_type, prompt)."""
    complete, partial, missing = [], [], []
    for dataset in datasets:
        for model_type in args.model_type:
            for prompt in args.prompt_choice:
                paths = result_paths(
                    args.experiment_folder, dataset, args.method, model_type,
                    prompt, args.n_iterations, args,
                )
                found = sum(os.path.exists(p) for p in paths)
                entry = (dataset, model_type, prompt)
                if found == len(paths):
                    complete.append(entry)
                elif found == 0:
                    missing.append(entry)
                else:
                    partial.append(entry)
    return complete, partial, missing


def resubmit(args, entries):
    for dataset, model_type, prompt in entries:
        command = [
            "python", str(EVAL_ROOT / "submit_all_evaluations.py"),
            "--segmentation_mode", "interactive", "--method", args.method,
            "-m", model_type, "-d", dataset, "-p", prompt,
            "-e", args.experiment_folder, "--time_limit", args.time_limit,
            "--partition", args.partition, "--gpu", args.gpu,
        ]
        if args.min_size:
            command.extend(["--min_size", str(args.min_size)])
        subprocess.run(command, cwd=str(EVAL_ROOT), check=True)


def main():
    parser = argparse.ArgumentParser(description="Check and resubmit interactive evaluation jobs.")
    parser.add_argument("--method", type=str, default="micro_sam2")
    parser.add_argument("-m", "--model_type", type=str, nargs="+", required=True)
    parser.add_argument("-d", "--dataset_name", type=str, nargs="+", default=None)
    parser.add_argument("--all_datasets", "--all-datasets", action="store_true")
    parser.add_argument("--ndim", type=int, default=None, choices=(2, 3))
    parser.add_argument("-p", "--prompt_choice", type=str, nargs="+", default=["box", "point"])
    parser.add_argument("-e", "--experiment_folder", type=str, default=EXPERIMENT_FOLDER)
    parser.add_argument("-iter", "--n_iterations", type=int, default=8)
    parser.add_argument("--use_masks", action=argparse.BooleanOptionalAction, default=True,
                        help="The setting the runs were written with, see interactive_result_name.")
    parser.add_argument("--mask_threshold", type=float, default=0.0)
    parser.add_argument("--min_size", type=int, default=0)
    parser.add_argument("--resubmit", action="store_true", help="Submit the missing and partial runs.")
    parser.add_argument("--partition", type=str, default="grete:preemptible")
    parser.add_argument("--gpu", type=str, default="1g.10gb:1")
    parser.add_argument("--time_limit", type=str, default="04:00:00")
    args = parser.parse_args()

    if args.all_datasets:
        if args.ndim == 2:
            datasets = DATASETS_2D
        elif args.ndim == 3:
            datasets = tuple(sorted(set(DATASETS_3D_LM + DATASETS_3D_EM)))
        else:
            datasets = tuple(sorted(set(DATASETS_2D + DATASETS_3D_LM + DATASETS_3D_EM)))
    else:
        if not args.dataset_name:
            raise ValueError("Pass -d or --all_datasets.")
        datasets = tuple(args.dataset_name)

    complete, partial, missing = scan(args, datasets)
    total = len(complete) + len(partial) + len(missing)
    print(f"complete: {len(complete)}/{total}   partial: {len(partial)}   missing: {len(missing)}")

    for label, entries in (("partial", partial), ("missing", missing)):
        for dataset, model_type, prompt in entries:
            print(f"  {label:>8}: {dataset} {model_type} {prompt}")

    if args.resubmit and (partial or missing):
        print(f"\nResubmitting {len(partial) + len(missing)} runs.")
        resubmit(args, partial + missing)


if __name__ == "__main__":
    main()
