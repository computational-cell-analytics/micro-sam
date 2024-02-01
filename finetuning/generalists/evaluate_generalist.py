import argparse
import os
from subprocess import run

from util import evaluate_checkpoint_for_dataset, ALL_DATASETS, EM_DATASETS, LM_DATASETS
from micro_sam.evaluation import default_experiment_settings, get_experiment_setting_name

EXPERIMENT_ROOT = "/scratch/projects/nim00007/sam/experiments/generalists"
CHECKPOINTS = {
    # Vanilla models
    "vit_b": "/home/nimcpape/.sam_models/sam_vit_b_01ec64.pth",
    "vit_h": "/home/nimcpape/.sam_models/sam_vit_h_4b8939.pth",
    # Generalist LM models
    "vit_b_lm": "/scratch/projects/nim00007/sam/models/LM/generalist/v2/vit_b/best.pt",
    "vit_h_lm": "/scratch/projects/nim00007/sam/models/LM/generalist/v2/vit_h/best.pt",
    # Generalist EM models
    "vit_b_em": "/scratch/projects/nim00007/sam/models/EM/generalist/v2/vit_b/best.pt",
    "vit_h_em": "/scratch/projects/nim00007/sam/models/EM/generalist/v2/vit_h/best.pt",
    # Specialist Models (we don't add livecell, because these results are all computed already)
    "vit_b_tissuenet": "/scratch/projects/nim00007/sam/models/LM/TissueNet/vit_b/best.pt",
    "vit_h_tissuenet": "/scratch/projects/nim00007/sam/models/LM/TissueNet/vit_h/best.pt",
    "vit_b_deepbacs": "/scratch/projects/nim00007/sam/models/LM/DeepBacs/vit_b/best.pt",
    "vit_h_deepbacs": "/scratch/projects/nim00007/sam/models/LM/DeepBacs/vit_h/best.pt",
}


def submit_array_job(model_name, datasets):
    n_datasets = len(datasets)
    cmd = ["sbatch", "-a", f"0-{n_datasets-1}", "evaluate_generalist.sbatch", model_name, "--datasets"]
    cmd.extend(datasets)
    run(cmd)


def evaluate_dataset_slurm(model_name, dataset):
    if dataset in EM_DATASETS:
        do_amg = False
        max_num_val_images = None
    else:
        do_amg = True
        max_num_val_images = 64

    is_custom_model = model_name not in ("vit_h", "vit_b")
    checkpoint = CHECKPOINTS[model_name]
    model_type = model_name[:5]

    experiment_folder = os.path.join(EXPERIMENT_ROOT, model_name, dataset)
    evaluate_checkpoint_for_dataset(
        checkpoint, model_type, dataset, experiment_folder,
        run_default_evaluation=True, do_amg=do_amg,
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


def check_computation(model_name, datasets):
    prompt_settings = default_experiment_settings()
    for ds in datasets:
        experiment_folder = os.path.join(EXPERIMENT_ROOT, model_name, ds)
        for setting in prompt_settings:
            setting_name = get_experiment_setting_name(setting)
            expected_path = os.path.join(experiment_folder, "results", f"{setting_name}.csv")
            if not os.path.exists(expected_path):
                print("Missing results for:", expected_path)
        if ds in LM_DATASETS:
            expected_path = os.path.join(experiment_folder, "results", "amg.csv")
            if not os.path.exists(expected_path):
                print("Missing results for:", expected_path)
    print("All checks_run")


# evaluation on slurm
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    parser.add_argument("--check", "-c", action="store_true")
    parser.add_argument("--lm", action="store_true")
    parser.add_argument("--em", action="store_true")
    parser.add_argument("--datasets", nargs="+")
    args = parser.parse_args()

    datasets = args.datasets
    if datasets is None or len(datasets) == 0:
        datasets = _get_datasets(args.lm, args.em)
    assert all(ds in ALL_DATASETS for ds in datasets)

    if args.check:
        check_computation(args.model_name, datasets)
        return

    job_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)
    if job_id is None:  # this is the main script that submits slurm jobs
        submit_array_job(args.model_name, datasets)
    else:  # we're in a slurm job
        job_id = int(job_id)
        evaluate_dataset_slurm(args.model_name, datasets[job_id])


if __name__ == "__main__":
    main()
