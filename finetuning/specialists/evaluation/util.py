import os
import argparse
from glob import glob

from torch_em.data import datasets

from micro_sam.evaluation import get_predictor


ROOT = "/scratch/projects/nim00007/sam/data/"
EXPERIMENT_ROOT = "/scratch/projects/nim00007/sam/experiments/new_models"

VANILLA_MODELS = {
    "vit_b": "/scratch-grete/projects/nim00007/sam/models/new_models/vanilla/sam_vit_b_01ec64.pth",
    "vit_l": "/scratch-grete/projects/nim00007/sam/models/new_models/vanilla/sam_vit_l_0b3195.pth",
    "vit_h": "/scratch-grete/projects/nim00007/sam/models/new_models/vanilla/sam_vit_h_4b8939.pth"
}

DATASETS = {
    "covid_if": {
        "val": [
            os.path.join(ROOT, "covid_if", "slices", "val", "raw", "*"),
            os.path.join(ROOT, "covid_if", "slices", "val", "labels", "*")
        ],
        "test": [
            os.path.join(ROOT, "covid_if", "slices", "test", "raw", "*"),
            os.path.join(ROOT, "covid_if", "slices", "test", "labels", "*")
        ]
    }
}


def get_model(model_type, ckpt):
    if ckpt is None:
        ckpt = VANILLA_MODELS[model_type]
    predictor = get_predictor(ckpt, model_type)
    return predictor


def get_paths(dataset_name, split="test"):
    assert dataset_name in DATASETS
    image_dir, gt_dir = DATASETS[dataset_name][split]
    image_paths = sorted(glob(os.path.join(image_dir)))
    gt_paths = sorted(glob(os.path.join(gt_dir)))
    return image_paths, gt_paths


def get_pred_paths(prediction_folder):
    pred_paths = sorted(glob(os.path.join(prediction_folder, "*.tif")))
    return pred_paths


def download_lm_dataset(path):
    # covid-if
    datasets.get_covid_if_dataset(os.path.join(path, "covid_if"), patch_shape=(1, 512, 512), download=True)


#
# PARSER FOR ALL THE REQUIRED ARGUMENTS
#


def get_default_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, required=True,
        help="Provide the model type to initialize the predictor"
    )
    parser.add_argument("-c", "--checkpoint", type=none_or_str, required=True, default=None)
    parser.add_argument("-e", "--experiment_folder", type=str, required=True)
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("--box", action="store_true", help="If passed, starts with first prompt as box")
    args = parser.parse_args()
    return args


def none_or_str(value):
    if value == 'None':
        return None
    return value
