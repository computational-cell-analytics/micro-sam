import os
from typing import Union, Optional, List

from . import instance_segmentation, inference
from ..instance_segmentation import AutomaticMaskGenerator


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
