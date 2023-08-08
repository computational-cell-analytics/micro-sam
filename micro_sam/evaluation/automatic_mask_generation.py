import os
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import imageio.v3 as imageio
import numpy as np
import pandas as pd

from elf.evaluation import mean_segmentation_accuracy
from segment_anything import SamPredictor
from tqdm import tqdm

from .. import instance_segmentation
from .. import util


def _get_range_of_search_values(input_vals, step):
    if isinstance(input_vals, list):
        search_range = np.arange(input_vals[0], input_vals[1] + step, step)
        search_range = [round(e, 2) for e in search_range]
    else:
        search_range = [input_vals]
    return search_range


def _grid_search(
    amg, gt, image_name, iou_thresh_values, stability_score_values, result_path, amg_generate_kwargs, verbose,
):
    net_list = []
    gs_combinations = [(r1, r2) for r1 in iou_thresh_values for r2 in stability_score_values]

    for iou_thresh, stability_thresh in tqdm(gs_combinations, disable=not verbose):
        masks = amg.generate(
            pred_iou_thresh=iou_thresh, stability_score_thresh=stability_thresh, **amg_generate_kwargs
        )
        instance_labels = instance_segmentation.mask_data_to_segmentation(masks, gt.shape, with_background=True)
        m_sas, sas = mean_segmentation_accuracy(instance_labels, gt, return_accuracies=True)  # type: ignore

        result_dict = {
            "image_name": image_name,
            "pred_iou_thresh": iou_thresh,
            "stability_score_thresh": stability_thresh,
            "mSA": m_sas,
            "SA50": sas[0],
            "SA75": sas[5]
        }
        tmp_df = pd.DataFrame([result_dict])
        net_list.append(tmp_df)

    img_gs_df = pd.concat(net_list)
    img_gs_df.to_csv(result_path, index=False)


# ideally we would generalize the parameters that GS runs over
def run_amg_grid_search(
    predictor: SamPredictor,
    image_paths: List[Union[str, os.PathLike]],
    gt_paths: List[Union[str, os.PathLike]],
    embedding_dir: Union[str, os.PathLike],
    result_dir: Union[str, os.PathLike],
    iou_thresh_values: Optional[List[float]] = None,
    stability_score_values: Optional[List[float]] = None,
    amg_kwargs: Optional[Dict[str, Any]] = None,
    amg_generate_kwargs: Optional[Dict[str, Any]] = None,
    AMG: instance_segmentation.AMGBase = instance_segmentation.AutomaticMaskGenerator,
    verbose_gs: bool = False,
) -> None:
    """

    Args:
    """
    assert len(image_paths) == len(gt_paths)
    amg_kwargs = {} if amg_kwargs is None else amg_kwargs
    amg_generate_kwargs = {} if amg_generate_kwargs is None else amg_generate_kwargs
    if "pred_iou_thresh" in amg_generate_kwargs or "stability_score_thresh" in amg_generate_kwargs:
        raise ValueError("The threshold parameters are optimized in the grid-search. You must not pass them as kwargs.")

    if iou_thresh_values is None:
        iou_thresh_values = _get_range_of_search_values([0.6, 0.9], step=0.025)
    if stability_score_values is None:
        stability_score_values = _get_range_of_search_values([0.6, 0.95], step=0.025)

    os.makedirs(result_dir, exist_ok=True)
    amg = AMG(predictor, **amg_kwargs)

    for image_path, gt_path in tqdm(
        zip(image_paths, gt_paths), desc="Run grid search for AMG", total=len(image_paths)
    ):
        image_name = Path(image_path).stem
        result_path = os.path.join(result_dir, f"{image_name}.csv")

        # We skip images for which the grid search was done already.
        if os.path.exists(result_path):
            continue

        assert os.path.exists(image_path), image_path
        assert os.path.exists(gt_path), gt_path

        image = imageio.imread(image_path)
        gt = imageio.imread(gt_path)

        embedding_path = os.path.join(embedding_dir, f"{image_name[:-4]}.zarr")
        image_embeddings = util.precompute_image_embeddings(predictor, image, embedding_path)
        amg.initialize(image, image_embeddings)

        _grid_search(
            amg, gt, image_name,
            iou_thresh_values, stability_score_values,
            result_path, amg_generate_kwargs, verbose=verbose_gs,
        )


def run_amg_inference(
    predictor: SamPredictor,
    image_paths: List[Union[str, os.PathLike]],
    embedding_dir: Union[str, os.PathLike],
    prediction_dir: Union[str, os.PathLike],
    amg_kwargs: Optional[Dict[str, Any]] = None,
    amg_generate_kwargs: Optional[Dict[str, Any]] = None,
    AMG: instance_segmentation.AMGBase = instance_segmentation.AutomaticMaskGenerator,
) -> None:
    """

    Args:
    """
    amg_kwargs = {} if amg_kwargs is None else amg_kwargs
    amg_generate_kwargs = {} if amg_generate_kwargs is None else amg_generate_kwargs

    amg = AMG(predictor, **amg_kwargs)

    for image_path in tqdm(image_paths, desc="Run inference for automatic mask generation"):
        image_name = os.path.basename(image_path)

        # We skip the images that already have been segmented.
        prediction_path = os.path.join(prediction_dir, image_name)
        if os.path.exists(prediction_path):
            continue

        assert os.path.exists(image_path), image_path
        image = imageio.imread(image_path)

        embedding_path = os.path.join(embedding_dir, f"{image_name[:-4]}.zarr")
        image_embeddings = util.precompute_image_embeddings(predictor, image, embedding_path)

        amg.initialize(image, image_embeddings)
        masks = amg.generate(**amg_generate_kwargs)
        instances = instance_segmentation.mask_data_to_segmentation(masks, image.shape, with_background=True)

        # It's important to compress here, otherwise the predictions would take up a lot of space.
        imageio.imwrite(prediction_path, instances, compression=5)


def evaluate_amg_grid_search(result_dir: Union[str, os.PathLike], criterion: str = "mSA") -> Tuple[float, float, float]:
    """

    Args:
    """

    # load all the grid search results
    gs_files = glob(os.path.join(result_dir, "*.csv"))
    gs_result = pd.concat([pd.read_csv(gs_file) for gs_file in gs_files])

    # contain only the relevant columns and group by the gridsearch columns
    gs_col1 = "pred_iou_thresh"
    gs_col2 = "stability_score_thresh"
    gs_result = gs_result[[gs_col1, gs_col2, criterion]]

    # compute the mean over the grouped columns
    grouped_result = gs_result.groupby([gs_col1, gs_col2]).mean()

    # find the best grouped result and return the corresponding thresholds
    best_score = grouped_result.max().values[0]
    best_result = grouped_result.idxmax()
    best_iou_thresh, best_stability_score = best_result.values[0]
    return best_iou_thresh, best_stability_score, best_score


def run_amg_grid_search_and_inference(
    predictor: SamPredictor,
    val_image_paths: List[Union[str, os.PathLike]],
    val_gt_paths: List[Union[str, os.PathLike]],
    test_image_paths: List[Union[str, os.PathLike]],
    embedding_dir: Union[str, os.PathLike],
    prediction_dir: Union[str, os.PathLike],
    result_dir: Union[str, os.PathLike],
    iou_thresh_values: Optional[List[float]] = None,
    stability_score_values: Optional[List[float]] = None,
    amg_kwargs: Optional[Dict[str, Any]] = None,
    amg_generate_kwargs: Optional[Dict[str, Any]] = None,
    AMG: instance_segmentation.AMGBase = instance_segmentation.AutomaticMaskGenerator,
    verbose_gs: bool = True,
) -> None:
    """

    Args:
    """
    run_amg_grid_search(
        predictor, val_image_paths, val_gt_paths, embedding_dir, result_dir,
        iou_thresh_values=iou_thresh_values, stability_score_values=stability_score_values,
        amg_kwargs=amg_kwargs, amg_generate_kwargs=amg_generate_kwargs, AMG=AMG, verbose_gs=verbose_gs,
    )

    amg_generate_kwargs = {} if amg_generate_kwargs is None else amg_generate_kwargs
    best_iou_thresh, best_stability_score, best_msa = evaluate_amg_grid_search(result_dir)
    print(
        "Best grid-search result:", best_msa,
        f"@ iou_thresh = {best_iou_thresh}, stability_score = {best_stability_score}"
    )
    amg_generate_kwargs["pred_iou_thresh"] = best_iou_thresh
    amg_generate_kwargs["stability_score_thresh"] = best_stability_score

    run_amg_inference(
        predictor, test_image_paths, embedding_dir, prediction_dir, amg_kwargs, amg_generate_kwargs, AMG
    )
