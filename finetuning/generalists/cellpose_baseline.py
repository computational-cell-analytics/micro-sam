import os
from glob import glob
from tqdm import tqdm

import imageio.v3 as imageio

from micro_sam.evaluation.evaluation import run_evaluation

DATA_ROOT = "/scratch/projects/nim00007/sam/data"
EXPERIMENT_ROOT = "/scratch/projects/nim00007/sam/experiments/benchmarking/cellpose"

DATASETS = {
    "covid_if": [
        os.path.join(DATA_ROOT, "covid_if", "slices", "test", "raw", "*.tif"),
        os.path.join(DATA_ROOT, "covid_if", "slices", "test", "labels", "*.tif")
    ]
}


def load_cellpose_model(model_type="cyto"):
    from cellpose import models

    device, gpu = models.assign_device(True, True)
    model = models.Cellpose(gpu=gpu, model_type=model_type, device=device)
    return model


def run_cellpose_segmentation(dataset, model_type):
    prediction_folder = os.path.join(EXPERIMENT_ROOT, dataset, "predictions")
    os.makedirs(prediction_folder, exist_ok=True)

    image_paths = sorted(glob(DATASETS[dataset][0]))
    model = load_cellpose_model(model_type)

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


def evaluate_dataset(dataset):
    gt_paths = sorted(glob(DATASETS[dataset][1]))
    experiment_folder = os.path.join(EXPERIMENT_ROOT, dataset)
    pred_paths = sorted(glob(os.path.join(experiment_folder, "predictions", "*.tif")))
    assert len(gt_paths) == len(pred_paths), f"{len(gt_paths)}, {len(pred_paths)}"
    result_path = os.path.join(experiment_folder, "cellpose.csv")
    run_evaluation(gt_paths, pred_paths, result_path)


def main():
    datasets = DATASETS.keys()
    for dataset in datasets:
        run_cellpose_segmentation(dataset, "cyto")
        evaluate_dataset(dataset)


if __name__ == "__main__":
    main()
