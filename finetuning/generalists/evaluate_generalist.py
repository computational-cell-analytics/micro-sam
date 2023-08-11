import argparse
import os
from subprocess import run

from util import evaluate_checkpoint_for_dataset, ALL_DATASETS, EM_DATASETS, LM_DATASETS

EXPERIMENT_ROOT = "/scratch/projects/nim00007/sam/experiments/generalists"
CHECKPOINTS = {
    "vit_b": "/home/nimcpape/.sam_models/sam_vit_b_01ec64.pth",
    "vit_h": "/home/nimcpape/.sam_models/sam_vit_h_4b8939.pth",
}


def submit_array_job(model_name, datasets, amg):
    n_datasets = len(datasets)
    cmd = ["sbatch", "-a", f"0-{n_datasets-1}", "evaluate_generalist.sbatch", model_name, "--datasets"]
    cmd.extend(datasets)
    if amg:
        cmd.append("--amg")
    run(cmd)


def evaluate_dataset_slurm(model_name, dataset, run_amg):
    max_num_val_images = None
    if run_amg:
        if dataset in EM_DATASETS:
            run_amg = False
        else:
            run_amg = True
            max_num_val_images = 100

    is_custom_model = model_name not in ("vit_h", "vit_b")
    checkpoint = CHECKPOINTS[model_name]
    model_type = model_name[:5]

    experiment_folder = os.path.join(EXPERIMENT_ROOT, model_name, dataset)
    evaluate_checkpoint_for_dataset(
        checkpoint, model_type, dataset, experiment_folder,
        run_default_evaluation=True, run_amg=run_amg,
        is_custom_model=is_custom_model,
        max_num_val_images=max_num_val_images,
    )


def _get_datasets(lm, em):
    assert lm or em
    datasets = []
    if lm:
        datasets.extend(LM_DATASETS)
    if em:
        datasets.extend(EM_DATASETS)
    return datasets


# evaluation on slurm
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    parser.add_argument("--lm", action="store_true")
    parser.add_argument("--em", action="store_true")
    parser.add_argument("--amg", action="store_true")
    parser.add_argument("--datasets", nargs="+")
    args = parser.parse_args()

    datasets = args.datasets
    if datasets is None or len(datasets) == 0:
        datasets = _get_datasets(args.lm, args.em)
    assert all(ds in ALL_DATASETS for ds in datasets)

    job_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    if job_id is None:  # this is the main script that submits slurm jobs
        submit_array_job(args.model_name, datasets, args.amg)
    else:  # we're in a slurm job and precompute a setting
        job_id = int(job_id)
        evaluate_dataset_slurm(args.model_name, datasets[job_id], args.amg)


if __name__ == "__main__":
    main()
