import os
import argparse
import warnings
from glob import glob
from typing import Union, List, Optional

from torch_em.data import datasets

from micro_sam.evaluation import get_predictor, instance_segmentation
from micro_sam.instance_segmentation import (AutomaticMaskGenerator,
                                             load_instance_segmentation_with_decoder_from_checkpoint)

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


def get_paths(dataset_name, split="test"):
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


#
# AMG FUNCTION (will move it to `micro_sam.evaluation.util` after testing)
#


def run_amg(
    checkpoint: Union[str, os.PathLike],
    model_type: str,
    experiment_folder: Union[str, os.PathLike],
    val_image_paths: List[Union[str, os.PathLike]],
    val_gt_paths: List[Union[str, os.PathLike]],
    test_image_paths: List[Union[str, os.PathLike]],
    iou_thresh_values: Optional[List[float]] = None,
    stability_score_values: Optional[List[float]] = None,
) -> str:
    embedding_folder = os.path.join(experiment_folder, "embeddings")  # where the precomputed embeddings are saved
    os.makedirs(embedding_folder, exist_ok=True)

    predictor = get_predictor(checkpoint, model_type)
    amg = AutomaticMaskGenerator(predictor)
    amg_prefix = "amg"

    # where the predictions are saved
    prediction_folder = os.path.join(experiment_folder, amg_prefix, "inference")
    os.makedirs(prediction_folder, exist_ok=True)

    # where the grid-search results are saved
    gs_result_folder = os.path.join(experiment_folder, amg_prefix, "grid_search")
    os.makedirs(gs_result_folder, exist_ok=True)

    grid_search_values = instance_segmentation.default_grid_search_values_amg(
        iou_thresh_values=iou_thresh_values,
        stability_score_values=stability_score_values,
    )

    instance_segmentation.run_instance_segmentation_grid_search_and_inference(
        amg, grid_search_values,
        val_image_paths, val_gt_paths, test_image_paths,
        embedding_folder, prediction_folder, gs_result_folder,
    )
    return prediction_folder


#
# INSTANCE SEGMENTATION FUNCTION (will move this to `micro_sam.evaluation.util` after testing)
#


def run_instance_segmentation_with_decoder(
    checkpoint: Union[str, os.PathLike],
    model_type: str,
    experiment_folder: Union[str, os.PathLike],
    val_image_paths: List[Union[str, os.PathLike]],
    val_gt_paths: List[Union[str, os.PathLike]],
    test_image_paths: List[Union[str, os.PathLike]],
) -> str:
    embedding_folder = os.path.join(experiment_folder, "embeddings")  # where the precomputed embeddings are saved
    os.makedirs(embedding_folder, exist_ok=True)

    segmenter = load_instance_segmentation_with_decoder_from_checkpoint(
        checkpoint, model_type,
    )
    seg_prefix = "instance_segmentation_with_decoder"

    # where the predictions are saved
    prediction_folder = os.path.join(experiment_folder, seg_prefix, "inference")
    os.makedirs(prediction_folder, exist_ok=True)

    # where the grid-search results are saved
    gs_result_folder = os.path.join(experiment_folder, seg_prefix, "grid_search")
    os.makedirs(gs_result_folder, exist_ok=True)

    grid_search_values = instance_segmentation.default_grid_search_values_instance_segmentation_with_decoder()

    instance_segmentation.run_instance_segmentation_grid_search_and_inference(
        segmenter, grid_search_values,
        val_image_paths, val_gt_paths, test_image_paths,
        embedding_dir=embedding_folder, prediction_dir=prediction_folder,
        result_dir=gs_result_folder,
    )
    return prediction_folder


# PARSER FOR ALL THE REQUIRED ARGUMENTS


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
