import os
import argparse
import warnings
from glob import glob

from torch_em.data import datasets

from micro_sam.evaluation import get_predictor

ROOT = "/scratch/projects/nim00007/sam/data/"

DATASETS = {
    "lucchi": {
        "val": [
            os.path.join(ROOT, "lucchi", "slices", "raw", "lucchi_train_*"),
            os.path.join(ROOT, "lucchi", "slices", "labels", "lucchi_train_*")
        ],
        "test": [
            os.path.join(ROOT, "lucchi", "slices", "raw", "lucchi_test_*"),
            os.path.join(ROOT, "lucchi", "slices", "labels", "lucchi_test_*")
        ]
    },
    "snemi": {
        "val": [
            os.path.join(ROOT, "snemi", "slices", "val", "raw", "*"),
            os.path.join(ROOT, "snemi", "slices", "val", "labels", "*")
        ],
        "test": [
            os.path.join(ROOT, "snemi", "slices", "test", "raw", "*"),
            os.path.join(ROOT, "snemi", "slices", "test", "labels", "*")
        ]
    },
    "nuc_mm": {
        "mouse": {
            "val": [
                os.path.join(ROOT, "nuc_mm", "slices", "mouse", "raw", "nuc_mm_val_*"),
                os.path.join(ROOT, "nuc_mm", "slices", "mouse", "labels", "nuc_mm_val_*")
            ],
            "test": [
                os.path.join(ROOT, "nuc_mm", "slices", "mouse", "raw", "nuc_mm_train_*"),
                os.path.join(ROOT, "nuc_mm", "slices", "mouse", "labels", "nuc_mm_train_*")
            ]
        },
        "zebrafish": {
            "val": [
                os.path.join(ROOT, "nuc_mm", "slices", "zebrafish", "raw", "nuc_mm_val_*"),
                os.path.join(ROOT, "nuc_mm", "slices", "zebrafish", "labels", "nuc_mm_val_*")
            ],
            "test": [
                os.path.join(ROOT, "nuc_mm", "slices", "zebrafish", "raw", "nuc_mm_train_*"),
                os.path.join(ROOT, "nuc_mm", "slices", "zebrafish", "labels", "nuc_mm_train_*")
            ]
        }

    }
}


VANILLA_MODELS = {
    "vit_b": "/scratch-grete/projects/nim00007/sam/models/new_models/vanilla/sam_vit_b_01ec64.pth",
    "vit_l": "/scratch-grete/projects/nim00007/sam/models/new_models/vanilla/sam_vit_l_0b3195.pth",
    "vit_h": "/scratch-grete/projects/nim00007/sam/models/new_models/vanilla/sam_vit_h_4b8939.pth"
}


def get_model(model_type, ckpt):
    if ckpt is None:
        ckpt = VANILLA_MODELS[model_type]
    predictor = get_predictor(ckpt, model_type)
    return predictor


def get_paths(dataset_name, split="test", species=None):
    assert dataset_name in DATASETS
    if species is None:
        image_dir, gt_dir = DATASETS[dataset_name][split]
    else:
        image_dir, gt_dir = DATASETS[dataset_name][species][split]

    image_paths = sorted(glob(os.path.join(image_dir)))
    gt_paths = sorted(glob(os.path.join(gt_dir)))
    return image_paths, gt_paths


def get_pred_paths(prediction_folder):
    pred_paths = sorted(glob(os.path.join(prediction_folder, "*.tif")))
    return pred_paths


def download_em_dataset(path):
    # lucchi
    datasets.get_lucchi_dataset(os.path.join(path, "lucchi"), split="train", patch_shape=(1, 512, 512), download=True)
    datasets.get_lucchi_dataset(os.path.join(path, "lucchi"), split="test", patch_shape=(1, 512, 512), download=True)

    # snemi
    datasets.get_snemi_dataset(os.path.join(path, "snemi"), patch_shape=(1, 512, 512), sample="train", download=True)
    try:
        datasets.get_snemi_dataset(os.path.join(path, "snemi"), patch_shape=(1, 512, 512), sample="test", download=True)
    except KeyError:
        warnings.warn("SNEMI's test set does not have labels. We download it in one place anyways.")

    # nuc_mm
    datasets.get_nuc_mm_dataset(
        os.path.join(path, "nuc_mm"), sample="mouse", split="train", patch_shape=(1, 192, 192), download=True
    )
    datasets.get_nuc_mm_dataset(
        os.path.join(path, "nuc_mm"), sample="zebrafish", split="train", patch_shape=(1, 64, 64), download=True
    )

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
    parser.add_argument("--species", type=str, default=None)
    parser.add_argument("--box", action="store_true", help="If passed, starts with first prompt as box")
    args = parser.parse_args()
    return args


def none_or_str(value):
    if value == 'None':
        return None
    return value
