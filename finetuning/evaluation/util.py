import os
import argparse
import warnings
from glob import glob

from torch_em.data import datasets

from micro_sam.evaluation import get_predictor
from micro_sam.evaluation.livecell import _get_livecell_paths


ROOT = "/scratch/projects/nim00007/sam/data/"

EXPERIMENT_ROOT = "/scratch/projects/nim00007/sam/experiments/new_models"

VANILLA_MODELS = {
    "vit_t": "/scratch-grete/projects/nim00007/sam/models/new_models/vanilla/vit_t_mobile_sam.pth",
    "vit_b": "/scratch-grete/projects/nim00007/sam/models/new_models/vanilla/sam_vit_b_01ec64.pth",
    "vit_l": "/scratch-grete/projects/nim00007/sam/models/new_models/vanilla/sam_vit_l_0b3195.pth",
    "vit_h": "/scratch-grete/projects/nim00007/sam/models/new_models/vanilla/sam_vit_h_4b8939.pth"
}


FILE_SPECS = {
    "lucchi": {"val": "lucchi_train_*", "test": "lucchi_test_*"},
    "nuc_mm/mouse": {"val": "nuc_mm_val_*", "test": "nuc_mm_train_*"},
    "nuc_mm/zebrafish": {"val": "nuc_mm_val_*", "test": "nuc_mm_train_*"},
    "platynereis/cilia": {"val": "platy_cilia_val_*", "test": "platy_cilia_test_*"},
    "platynereis/nuclei": {"val": "platy_nuclei_val_*", "test": "platy_nuclei_test_*"},
    "platynereis/cells": {"val": "platy_cells_val_*", "test": "platy_cells_test_*"},
    "cremi": {"val": "cremi_val_*", "test": "cremi_test_*"}
}

# good spot to track all datasets we use atm
DATASETS = [
    # in-domain (LM)
    "tissuenet", "deepbacs", "plantseg/root", "livecell", "neurips-cell-seg",
    # out-of-domain (LM)
    "covid_if", "plantseg/ovules", "hpa", "lizard", "mouse-embryo", "ctc/hela_samples",
    # organelles (EM)
    #   - in-domain
    "mitoem/rat", "mitoem/human", "platynereis/nuclei",
    #   - out-of-domain
    "mitolab/c_elegans", "mitolab/fly_brain", "mitolab/glycolytic_muscle", "mitolab/hela_cell",
    "mitolab/lucchi_pp", "mitolab/salivary_gland", "mitolab/tem", "lucchi", "nuc_mm/mouse",
    "nuc_mm/zebrafish", "uro_cell", "sponge_em", "platynereis/cilia",
    # boundaries - EM
    #   - in-domain
    "cremi", "platynereis/cells",
    #   - out-of-domain
    "axondeepseg", "snemi", "isbi"
]


def get_dataset_paths(dataset_name, split_choice):
    # let's check if we have a particular naming logic to save the images
    try:
        file_search_specs = FILE_SPECS[dataset_name][split_choice]
        is_explicit_split = False
    except KeyError:
        file_search_specs = "*"
        is_explicit_split = True

    # if the datasets have different modalities/species, let's make use of it
    split_names = dataset_name.split("/")
    if len(split_names) > 1:
        assert len(split_names) <= 2
        dataset_name = [split_names[0], "slices", split_names[1]]
    else:
        dataset_name = [*split_names, "slices"]

    # if there is an explicit val/test split made, let's look at them
    if is_explicit_split:
        dataset_name.append(split_choice)

    raw_dir = os.path.join(ROOT, *dataset_name, "raw", file_search_specs)
    labels_dir = os.path.join(ROOT, *dataset_name, "labels", file_search_specs)

    return raw_dir, labels_dir


def get_model(model_type, ckpt):
    if ckpt is None:
        ckpt = VANILLA_MODELS[model_type]
    predictor = get_predictor(ckpt, model_type)
    return predictor


def get_paths(dataset_name, split):
    assert dataset_name in DATASETS, dataset_name

    if dataset_name == "livecell":
        image_paths, gt_paths = _get_livecell_paths(input_folder=os.path.join(ROOT, "livecell"), split=split)
        return sorted(image_paths), sorted(gt_paths)

    image_dir, gt_dir = get_dataset_paths(dataset_name, split)
    image_paths = sorted(glob(os.path.join(image_dir)))
    gt_paths = sorted(glob(os.path.join(gt_dir)))
    return image_paths, gt_paths


def get_pred_paths(prediction_folder):
    pred_paths = sorted(glob(os.path.join(prediction_folder, "*")))
    return pred_paths


def download_all_datasets(path):
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

    # covid-if
    datasets.get_covid_if_dataset(os.path.join(path, "covid_if"), patch_shape=(1, 512, 512), download=True)

    # tissuenet: data cannot be downloaded automatically. please download from here - https://datasets.deepcell.org/data

    # deepbacs
    datasets.get_deepbacs_dataset(os.path.join(path, "deepbacs"), split="train", patch_shape=(256, 256), download=True)
    datasets.get_deepbacs_dataset(os.path.join(path, "deepbacs"), split="val", patch_shape=(256, 256), download=True)
    datasets.get_deepbacs_dataset(os.path.join(path, "deepbacs"), split="test", patch_shape=(256, 256), download=True)

    # plantseg root
    datasets.get_plantseg_dataset(
        os.path.join(path, "plantseg"), name="root", split="train", patch_shape=(1, 512, 512), download=True
    )
    datasets.get_plantseg_dataset(
        os.path.join(path, "plantseg"), name="root", split="val", patch_shape=(1, 512, 512), download=True
    )
    datasets.get_plantseg_dataset(
        os.path.join(path, "plantseg"), name="root", split="test", patch_shape=(1, 512, 512), download=True
    )

    # hpa
    datasets.get_hpa_segmentation_dataset(
        os.path.join(path, "hpa"), split="train", patch_shape=(512, 512), download=True
    )
    datasets.get_hpa_segmentation_dataset(
        os.path.join(path, "hpa"), split="val", patch_shape=(512, 512), download=True
    )

    # lizard: see `torch_em.data.datasets.get_lizard_dataset` for details to download the dataset

    # mouse embryo
    datasets.get_mouse_embryo_dataset(
        os.path.join(path, "mouse-embryo"), name="nuclei", split="train", patch_shape=(1, 512, 512), download=True
    )
    datasets.get_mouse_embryo_dataset(
        os.path.join(path, "mouse-embryo"), name="nuclei", split="val", patch_shape=(1, 512, 512), download=True
    )

    # plantseg ovules
    datasets.get_plantseg_dataset(
        os.path.join(path, "plantseg"), name="ovules", split="train", patch_shape=(1, 512, 512), download=True
    )
    datasets.get_plantseg_dataset(
        os.path.join(path, "plantseg"), name="ovules", split="val", patch_shape=(1, 512, 512), download=True
    )
    datasets.get_plantseg_dataset(
        os.path.join(path, "plantseg"), name="ovules", split="test", patch_shape=(1, 512, 512), download=True
    )

#
# PARSER FOR ALL THE REQUIRED ARGUMENTS
#


def get_default_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Provide the model type to initialize the predictor"
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
