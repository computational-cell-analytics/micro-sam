import argparse
import os
from glob import glob
from subprocess import run

import imageio.v3 as imageio

from tqdm import tqdm

DATA_ROOT = "/scratch/projects/nim00007/sam/datasets"
EXP_ROOT = "/scratch/projects/nim00007/sam/experiments/cellpose"

DATASETS = (
    "covid-if",
    "deepbacs",
    "hpa",
    "livecell",
    "lizard",
    "mouse-embryo",
    "plantseg-ovules",
    "plantseg-root",
    "tissuenet",
)


def load_cellpose_model():
    from cellpose import models

    device, gpu = models.assign_device(True, True)
    model = models.Cellpose(gpu=gpu, model_type="cyto", device=device)
    return model


def run_cellpose_segmentation(datasets, job_id):
    dataset = datasets[job_id]
    experiment_folder = os.path.join(EXP_ROOT, dataset)

    prediction_folder = os.path.join(experiment_folder, "predictions")
    os.makedirs(prediction_folder, exist_ok=True)

    image_paths = sorted(glob(os.path.join(DATA_ROOT, dataset, "test", "image*.tif")))
    model = load_cellpose_model()

    for path in tqdm(image_paths, desc=f"Segmenting {dataset} with cellpose"):
        fname = os.path.basename(path)
        out_path = os.path.join(prediction_folder, fname)
        if os.path.exists(out_path):
            continue
        image = imageio.imread(path)
        if image.ndim == 3:
            assert image.shape[-1] == 3
            image = image.mean(axis=-1)
        assert image.ndim == 2
        seg = model.eval(image, diameter=None, flow_threshold=None, channels=[0, 0])[0]
        assert seg.shape == image.shape
        imageio.imwrite(out_path, seg, compression=5)


def submit_array_job(datasets):
    n_datasets = len(datasets)
    cmd = ["sbatch", "-a", f"0-{n_datasets-1}", "cellpose_baseline.sbatch"]
    run(cmd)


def evaluate_dataset(dataset):
    from micro_sam.evaluation.evaluation import run_evaluation

    gt_paths = sorted(glob(os.path.join(DATA_ROOT, dataset, "test", "label*.tif")))
    experiment_folder = os.path.join(EXP_ROOT, dataset)
    pred_paths = sorted(glob(os.path.join(experiment_folder, "predictions", "*.tif")))
    assert len(gt_paths) == len(pred_paths), f"{len(gt_paths)}, {len(pred_paths)}"
    result_path = os.path.join(experiment_folder, "cellpose.csv")
    run_evaluation(gt_paths, pred_paths, result_path)


def evaluate_segmentations(datasets):
    for dataset in datasets:
        # we skip livecell, which has already been processed by cellpose
        if dataset == "livecell":
            continue
        evaluate_dataset(dataset)


def check_results(datasets):
    for ds in datasets:
        # we skip livecell, which has already been processed by cellpose
        if ds == "livecell":
            continue
        result_path = os.path.join(EXP_ROOT, ds, "cellpose.csv")
        if not os.path.exists(result_path):
            print("Cellpose results missing for", ds)
    print("All checks passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment", "-s", action="store_true")
    parser.add_argument("--evaluate", "-e", action="store_true")
    parser.add_argument("--check", "-c", action="store_true")
    parser.add_argument("--datasets", nargs="+")
    args = parser.parse_args()

    job_id = os.environ.get("SLURM_ARRAY_TASK_ID", None)

    if args.datasets is None:
        datasets = DATASETS
    else:
        datasets = args.datasets
        assert all(ds in DATASETS for ds in datasets)

    if job_id is not None:
        run_cellpose_segmentation(datasets, int(job_id))
    elif args.segment:
        submit_array_job(datasets)
    elif args.evaluate:
        evaluate_segmentations(datasets)
    elif args.check:
        check_results(datasets)
    else:
        raise ValueError("Doing nothing")


if __name__ == "__main__":
    main()
