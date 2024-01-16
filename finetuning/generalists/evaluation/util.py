import os
from glob import glob

from micro_sam.evaluation import get_predictor


DATASETS = {
    "lucchi": [None, None]
}


def get_model(model_type, ckpt):
    predictor = get_predictor(ckpt, model_type)
    return predictor


def get_paths(dataset_name):
    assert dataset_name in DATASETS
    image_dir, gt_dir = DATASETS[dataset_name]
    image_paths = glob(os.path.join(image_dir, "*.tif"))
    gt_paths = glob(os.path.join(gt_dir, "*.tif"))
    return image_paths, gt_paths


def get_pred_paths(prediction_folder):
    pred_paths = sorted(glob(os.path.join(prediction_folder, "*.tif")))
    return pred_paths


def download_em_dataset(dataset_name):
    assert dataset_name in DATASETS

    # TODO: downloading the dataset using the dataloaders (for all splits - especially val and test split)
    raise NotImplementedError
