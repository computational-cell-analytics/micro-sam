import os
from tqdm import tqdm
from glob import glob

import z5py
import numpy as np
import pandas as pd
import imageio.v3 as imageio

from elf.evaluation import mean_segmentation_accuracy

from micro_sam.evaluation.evaluation import run_evaluation

from util import get_paths, get_pred_paths


EXPERIMENT_ROOT = "/scratch/projects/nim00007/sam/experiments/benchmarking/cellpose"

LM_DATASETS = [
    # in-domain (LM)
    "tissuenet", "deepbacs", "plantseg/root", "livecell",
    "neurips-cell-seg/all", "neurips-cell-seg/tuning", "neurips-cell-seg/self",
    # out-of-domain (LM)
    "covid_if", "plantseg/ovules", "hpa", "lizard", "mouse-embryo", "ctc/hela_samples", "dynamicnuclearnet", "pannuke"
]


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

    for path in tqdm(image_paths, desc=f"Segmenting {dataset} with cellpose ({model_type})"):
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


def test_cellpose(model_type, view=False):
    # data_dir = "/scratch/projects/nim00007/sam/data/tissuenet/test"
    data_dir = "/media/anwai/ANWAI/data/tissuenet/test"
    all_tissunet_paths = sorted(glob(os.path.join(data_dir, "*.zarr")))

    model = load_cellpose_model(model_type)

    msa13_list, msa23_list = [], []
    for chosen_vol_path in tqdm(all_tissunet_paths):
        with z5py.File(chosen_vol_path, "a", use_zarr_format=True) as f:
            raw = f["raw/rgb"][:]
            labels = f["labels/cell"][:]
            prediction13 = model.eval(raw, diameter=None, flow_threshold=None, channels=[1, 3])[0]
            prediction23 = model.eval(raw, diameter=None, flow_threshold=None, channels=[2, 3])[0]

            msa13_list.append(mean_segmentation_accuracy(prediction13, labels))
            msa23_list.append(mean_segmentation_accuracy(prediction23, labels))

            if view:
                import napari
                v = napari.Viewer()
                v.add_image(raw)
                v.add_labels(labels)
                v.add_labels(prediction13)
                v.add_labels(prediction23)
                napari.run()

    print("mSA score for inference at channel [1, 3]:", np.mean(msa13_list))
    print("mSA score for inference at channel [2, 3]:", np.mean(msa23_list))

    # RESULTS (tissuenet specialist):
    # mSA score for inference at channel [1, 3]: 0.28846337471846234
    # mSA score for inference at channel [2, 3]: 0.4309667789626076 (!!!)

    # RESULTS (cyto2 specialist):
    # mSA score for inference at channel [1, 3]: 0.013506463934033066
    # mSA score for inference at channel [2, 3]: 0.14076119450800934 (!!!)


def main(args):
    if args.dataset is None:
        datasets = LM_DATASETS
    else:
        datasets = args.dataset
        assert datasets in LM_DATASETS

    if args.model_type is None:
        model_types = ["cyto", "cyto2", "cyto3", "nuclei", "livecell", "tissuenet"]
    else:
        model_types = args.model_type

    if args.custom:
        test_cellpose(model_types, view=False)
    else:
        run_cellpose_baseline(datasets, model_types)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default=None)
    parser.add_argument("-m", "--model_type", type=str, default=None)
    parser.add_argument("--custom", action="store_true")
    args = parser.parse_args()
    main(args)
