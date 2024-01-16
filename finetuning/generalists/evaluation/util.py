import os
from glob import glob
from typing import Union, List, Optional

from micro_sam.evaluation import get_predictor
from micro_sam import instance_segmentation, inference
from micro_sam.instance_segmentation import AutomaticMaskGenerator


DATASETS = {
    "lucchi": {
        "val": [None, None],
        "test": [None, None]
    }
}


def get_model(model_type, ckpt):
    predictor = get_predictor(ckpt, model_type)
    return predictor


def get_paths(dataset_name, split="test"):
    assert dataset_name in DATASETS
    image_dir, gt_dir = DATASETS[dataset_name][split]
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

    predictor = inference.get_predictor(checkpoint, model_type)
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
