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
    "nuc-mm/mouse": {
        "val": [
            os.path.join(ROOT, "nuc_mm", "slices", "mouse", "raw", "nuc_mm_val_*"),
            os.path.join(ROOT, "nuc_mm", "slices", "mouse", "labels", "nuc_mm_val_*")
        ],
        "test": [
            os.path.join(ROOT, "nuc_mm", "slices", "mouse", "raw", "nuc_mm_train_*"),
            os.path.join(ROOT, "nuc_mm", "slices", "mouse", "labels", "nuc_mm_train_*")
        ]
    },
    "nuc-mm/zebrafish": {
        "val": [
            os.path.join(ROOT, "nuc_mm", "slices", "zebrafish", "raw", "nuc_mm_val_*"),
            os.path.join(ROOT, "nuc_mm", "slices", "zebrafish", "labels", "nuc_mm_val_*")
        ],
        "test": [
            os.path.join(ROOT, "nuc_mm", "slices", "zebrafish", "raw", "nuc_mm_train_*"),
            os.path.join(ROOT, "nuc_mm", "slices", "zebrafish", "labels", "nuc_mm_train_*")
        ]
    },
    "platynereis/cilia": {
        "val": [
            os.path.join(ROOT, "platynereis", "slices", "cilia", "raw", "platy_cilia_val_*"),
            os.path.join(ROOT, "platynereis", "slices", "cilia", "labels", "platy_cilia_val_*")
        ],
        "test": [
            os.path.join(ROOT, "platynereis", "slices", "cilia", "raw", "platy_cilia_test_*"),
            os.path.join(ROOT, "platynereis", "slices", "cilia", "labels", "platy_cilia_test_*")
        ]
    },
    "platynereis/nuclei": {
        "val": [
            os.path.join(ROOT, "platynereis", "slices", "nuclei", "raw", "platy_nuclei_val_*"),
            os.path.join(ROOT, "platynereis", "slices", "nuclei", "labels", "platy_nuclei_val_*")
        ],
        "test": [
            os.path.join(ROOT, "platynereis", "slices", "nuclei", "raw", "platy_nuclei_test_*"),
            os.path.join(ROOT, "platynereis", "slices", "nuclei", "labels", "platy_nuclei_test_*")
        ]
    },
    "platynereis/cells": {
        "val": [
            os.path.join(ROOT, "platynereis", "slices", "cells", "raw", "platy_cells_val_*"),
            os.path.join(ROOT, "platynereis", "slices", "cells", "labels", "platy_cells_val_*")
        ],
        "test": [
            os.path.join(ROOT, "platynereis", "slices", "cells", "raw", "platy_cells_test_*"),
            os.path.join(ROOT, "platynereis", "slices", "cells", "labels", "platy_cells_test_*")
        ]
    },
    "mitoem/rat": {
        "val": [
            os.path.join(ROOT, "mitoem", "slices", "rat", "val", "raw", "*"),
            os.path.join(ROOT, "mitoem", "slices", "rat", "val", "labels", "*")
        ],
        "test": [
            os.path.join(ROOT, "mitoem", "slices", "rat", "test", "raw", "*"),
            os.path.join(ROOT, "mitoem", "slices", "rat", "test", "labels", "*")
        ]
    },
    "mitoem/human": {
        "val": [
            os.path.join(ROOT, "mitoem", "slices", "human", "val", "raw", "*"),
            os.path.join(ROOT, "mitoem", "slices", "human", "val", "labels", "*")
        ],
        "test": [
            os.path.join(ROOT, "mitoem", "slices", "human", "test", "raw", "*"),
            os.path.join(ROOT, "mitoem", "slices", "human", "test", "labels", "*")
        ]
    },
    "mitolab/c_elegans": {
        "val": [
            os.path.join(ROOT, "mitolab", "slices", "c_elegans", "val", "raw", "*"),
            os.path.join(ROOT, "mitolab", "slices", "c_elegans", "val", "labels", "*")
        ],
        "test": [
            os.path.join(ROOT, "mitolab", "slices", "c_elegans", "test", "raw", "*"),
            os.path.join(ROOT, "mitolab", "slices", "c_elegans", "test", "labels", "*")
        ]
    },
    "mitolab/fly_brain": {
        "val": [
            os.path.join(ROOT, "mitolab", "slices", "fly_brain", "val", "raw", "*"),
            os.path.join(ROOT, "mitolab", "slices", "fly_brain", "val", "labels", "*")
        ],
        "test": [
            os.path.join(ROOT, "mitolab", "slices", "fly_brain", "test", "raw", "*"),
            os.path.join(ROOT, "mitolab", "slices", "fly_brain", "test", "labels", "*")
        ]
    },
    "mitolab/glycolytic_muscle": {
        "val": [
            os.path.join(ROOT, "mitolab", "slices", "glycolytic_muscle", "val", "raw", "*"),
            os.path.join(ROOT, "mitolab", "slices", "glycolytic_muscle", "val", "labels", "*")
        ],
        "test": [
            os.path.join(ROOT, "mitolab", "slices", "glycolytic_muscle", "test", "raw", "*"),
            os.path.join(ROOT, "mitolab", "slices", "glycolytic_muscle", "test", "labels", "*")
        ]
    },
    "mitolab/hela_cell": {
        "val": [
            os.path.join(ROOT, "mitolab", "slices", "hela_cell", "val", "raw", "*"),
            os.path.join(ROOT, "mitolab", "slices", "hela_cell", "val", "labels", "*")
        ],
        "test": [
            os.path.join(ROOT, "mitolab", "slices", "hela_cell", "test", "raw", "*"),
            os.path.join(ROOT, "mitolab", "slices", "hela_cell", "test", "labels", "*")
        ]
    },
    "mitolab/lucchi_pp": {
        "val": [
            os.path.join(ROOT, "mitolab", "slices", "lucchi_pp", "val", "raw", "*"),
            os.path.join(ROOT, "mitolab", "slices", "lucchi_pp", "val", "labels", "*")
        ],
        "test": [
            os.path.join(ROOT, "mitolab", "slices", "lucchi_pp", "test", "raw", "*"),
            os.path.join(ROOT, "mitolab", "slices", "lucchi_pp", "test", "labels", "*")
        ]
    },
    "mitolab/salivary_gland": {
        "val": [
            os.path.join(ROOT, "mitolab", "slices", "salivary_gland", "val", "raw", "*"),
            os.path.join(ROOT, "mitolab", "slices", "salivary_gland", "val", "labels", "*")
        ],
        "test": [
            os.path.join(ROOT, "mitolab", "slices", "salivary_gland", "test", "raw", "*"),
            os.path.join(ROOT, "mitolab", "slices", "salivary_gland", "test", "labels", "*")
        ]
    },
    "mitolab/tem": {
        "val": [
            os.path.join(ROOT, "mitolab", "slices", "tem", "val", "raw", "*"),
            os.path.join(ROOT, "mitolab", "slices", "tem", "val", "labels", "*")
        ],
        "test": [
            os.path.join(ROOT, "mitolab", "slices", "tem", "test", "raw", "*"),
            os.path.join(ROOT, "mitolab", "slices", "tem", "test", "labels", "*")
        ]
    },
    "axondeepseg": {
        "val": [
            os.path.join(ROOT, "axondeepseg", "slices", "val", "raw", "*"),
            os.path.join(ROOT, "axondeepseg", "slices", "val", "labels", "*")
        ],
        "test": [
            os.path.join(ROOT, "axondeepseg", "slices", "test", "raw", "*"),
            os.path.join(ROOT, "axondeepseg", "slices", "test", "labels", "*")
        ]
    },
    "cremi": {
        "val": [
            os.path.join(ROOT, "cremi", "slices", "raw", "cremi_val_*"),
            os.path.join(ROOT, "cremi", "slices", "labels", "cremi_val*")
        ],
        "test": [
            os.path.join(ROOT, "cremi", "slices", "raw", "cremi_test_*"),
            os.path.join(ROOT, "cremi", "slices", "labels", "cremi_test_*")
        ]
    },
    "uro_cell": {
        "val": [
            os.path.join(ROOT, "uro_cell", "slices", "val", "raw", "*"),
            os.path.join(ROOT, "uro_cell", "slices", "val", "labels", "*")
        ],
        "test": [
            os.path.join(ROOT, "uro_cell", "slices", "test", "raw", "*"),
            os.path.join(ROOT, "uro_cell", "slices", "test", "labels", "*")
        ]
    },
    "sponge_em": {
        "val": [
            os.path.join(ROOT, "sponge_em", "slices", "val", "raw", "*"),
            os.path.join(ROOT, "sponge_em", "slices", "val", "labels", "*")
        ],
        "test": [
            os.path.join(ROOT, "sponge_em", "slices", "test", "raw", "*"),
            os.path.join(ROOT, "sponge_em", "slices", "test", "labels", "*")
        ]
    },
    "isbi": {
        "val": [
            os.path.join(ROOT, "isbi", "slices", "val", "raw", "*"),
            os.path.join(ROOT, "isbi", "slices", "val", "labels", "*")
        ],
        "test": [
            os.path.join(ROOT, "isbi", "slices", "test", "raw", "*"),
            os.path.join(ROOT, "isbi", "slices", "test", "labels", "*")
        ]
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


def get_paths(dataset_name, split):
    assert dataset_name in DATASETS
    image_dir, gt_dir = DATASETS[dataset_name][split]
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

    # platy-cilia
    datasets.get_platynereis_cilia_dataset(os.path.join(path, "platynereis"), patch_shape=(1, 512, 512), download=True)
    datasets.get_platynereis_nuclei_dataset(os.path.join(path, "platynereis"), patch_shape=(1, 512, 512), download=True)
    datasets.get_platynereis_cell_dataset(os.path.join(path, "platynereis"), patch_shape=(1, 512, 512), download=True)

    # mitoem
    datasets.get_mitoem_dataset(
        os.path.join(path, "mitoem"), splits="val", patch_shape=(1, 512, 512), download=True
    )

    # mitolab
    print("MitoLab benchmark datasets need to downloaded separately. See `datasets.cem.get_benchmark_datasets`")

    # uro-cell
    datasets.get_uro_cell_dataset(
        os.path.join(path, "uro_cell"), target="mito", patch_shape=(1, 512, 512), download=True
    )

    # sponge-em
    datasets.get_sponge_em_dataset(
        os.path.join(path, "sponge_em"), mode="instances", patch_shape=(1, 512, 512), download=True
    )

    # isbi
    datasets.get_isbi_dataset(os.path.join(path, "isbi"), patch_shape=(1, 512, 512), download=True)

    # axondeepseg
    datasets.get_axondeepseg_dataset(
        os.path.join(path, "axondeepseg"), name="tem", patch_shape=(1, 512, 512), download=True
    )

    # cremi
    datasets.get_cremi_dataset(os.path.join(path, "cremi"), patch_shape=(1, 512, 512), download=True)

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
