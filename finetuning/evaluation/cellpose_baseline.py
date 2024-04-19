import os
import time
from tqdm import tqdm

import numpy as np
import pandas as pd
import imageio.v3 as imageio

from micro_sam.evaluation.evaluation import run_evaluation

# from util import get_paths   # for hlrn
from util import get_pred_paths 


# EXPERIMENT_ROOT = "/scratch/projects/nim00007/sam/experiments/benchmarking/cellpose"  # for hlrn
EXPERIMENT_ROOT = "/scratch/users/archit/experiments/cellpose/"  # for scc

LM_DATASETS = [
    # in-domain (LM)
    "tissuenet/one_chan", "tissuenet/multi_chan", "deepbacs", "plantseg/root", "livecell",
    "neurips-cell-seg/all", "neurips-cell-seg/tuning", "neurips-cell-seg/self",
    # out-of-domain (LM)
    "covid_if", "plantseg/ovules", "hpa", "lizard", "mouse-embryo", "dynamicnuclearnet", "pannuke"
]

FOR_MULTICHAN = ["tissuenet/multi_chan"]

# Time benchmarks for:
#   - LIVECell dataset with "livecell" speclalist model (to stay consistent with our time benchmarking setup)

# GPU (RTX5000)
#       - Run 1: 0.286 s (0.102)
#       - Run 2: 0.267 s (0.053)
#       - Run 3: 0.271 s (0.054)
#       - Run 4: 0.266 s (0.053)
#       - Run 5: 0.267 s (0.052)

# medium partition (64GB CPU mem)
#       - Run 1: 1.387 s (0.391)
#       - Run 2: 1.361 s (0.389)
#       - Run 3: 1.771 s (0.502)
#       - Run 4: 1.371 s (0.387)
#       - Run 5: 1.616 s (0.394)


# override for SCC
def get_paths(dataset_name, split):
    from micro_sam.evaluation.livecell import _get_livecell_paths
    image_paths, gt_paths = _get_livecell_paths(
        input_folder="/scratch/users/archit/data/livecell", split=split
    )
    return sorted(image_paths), sorted(gt_paths)


def load_cellpose_model(model_type):
    from cellpose import models
    device, gpu = models.assign_device(True, True)

    if model_type in ["cyto", "cyto2", "cyto3", "nuclei"]:
        model = models.Cellpose(gpu=gpu, model_type=model_type, device=device)
    elif model_type in ["livecell", "tissuenet", "livecell_cp3"]:
        model = models.CellposeModel(gpu=gpu, model_type=model_type, device=device)
    else:
        raise ValueError(model_type)

    return model


def run_cellpose_segmentation(dataset, model_type):
    prediction_folder = os.path.join(EXPERIMENT_ROOT, dataset, model_type, "predictions")
    os.makedirs(prediction_folder, exist_ok=True)

    image_paths, _ = get_paths(dataset, split="test")
    model = load_cellpose_model(model_type)

    time_per_image = []
    for path in tqdm(image_paths, desc=f"Segmenting {dataset} with cellpose ({model_type})"):
        fname = os.path.basename(path)
        out_path = os.path.join(prediction_folder, fname)
        if os.path.exists(out_path):
            continue
        image = imageio.imread(path)
        channels = [0, 0]  # it's assumed to use one-channel, unless overwritten by logic below

        if image.ndim == 3:
            assert image.shape[-1] == 3
            if dataset in FOR_MULTICHAN:
                channels = [2, 3]
            else:
                image = image.mean(axis=-1)

        start_time = time.time()

        seg = model.eval(image, diameter=None, flow_threshold=None, channels=channels)[0]

        end_time = time.time()
        time_per_image.append(end_time - start_time)

        assert seg.shape == image.shape[:2]
        imageio.imwrite(out_path, seg, compression=5)

    n_images = len(image_paths)
    print(f"The mean time over {n_images} images is:", np.mean(time_per_image), f"({np.std(time_per_image)})")

    return prediction_folder


def evaluate_dataset(prediction_folder, dataset, model_type):
    _, gt_paths = get_paths(dataset, split="test")
    pred_paths = get_pred_paths(prediction_folder)
    assert len(gt_paths) == len(pred_paths), f"{len(gt_paths)}, {len(pred_paths)}"
    result_path = os.path.join(EXPERIMENT_ROOT, dataset, "results", f"cellpose-{model_type}.csv")
    if os.path.exists(result_path):
        print(pd.read_csv(result_path))
        print(f"Results are already saved at {result_path}")
        return result_path

    results = run_evaluation(gt_paths, pred_paths, result_path)
    print(results)
    print(f"Results are saved at {result_path}")


def run_cellpose_baseline(datasets, model_types):
    if isinstance(datasets, str):
        datasets = [datasets]

    if isinstance(model_types, str):
        model_types = [model_types]

    for dataset in datasets:
        for model_type in model_types:
            prediction_folder = run_cellpose_segmentation(dataset, model_type)
            evaluate_dataset(prediction_folder, dataset, model_type)


def main(args):
    if args.dataset is None:
        datasets = LM_DATASETS
    else:
        datasets = args.dataset
        assert datasets in LM_DATASETS

    if args.model_type is None:
        model_types = ["cyto", "cyto2", "cyto3", "livecell", "tissuenet"]
    else:
        model_types = args.model_type

    run_cellpose_baseline(datasets, model_types)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default=None)
    parser.add_argument("-m", "--model_type", type=str, default=None)
    args = parser.parse_args()
    main(args)
