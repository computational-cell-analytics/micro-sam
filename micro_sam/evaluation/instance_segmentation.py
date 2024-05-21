"""Inference and evaluation for the automatic instance segmentation functionality.
"""

import os
from glob import glob
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import imageio.v3 as imageio
import numpy as np
import pandas as pd

from elf.evaluation import mean_segmentation_accuracy
from elf.io import open_file
from tqdm import tqdm

from ..instance_segmentation import AMGBase, InstanceSegmentationWithDecoder, mask_data_to_segmentation
from .. import util


def _get_range_of_search_values(input_vals, step):
    if isinstance(input_vals, list):
        search_range = np.arange(input_vals[0], input_vals[1] + step, step)
        search_range = [round(e, 3) for e in search_range]
    else:
        search_range = [input_vals]
    return search_range


def default_grid_search_values_amg(
    iou_thresh_values: Optional[List[float]] = None,
    stability_score_values: Optional[List[float]] = None,
) -> Dict[str, List[float]]:
    """Default grid-search parameter for AMG-based instance segmentation.

    Return grid search values for the two most important parameters:
    - `pred_iou_thresh`, the threshold for keeping objects according to the IoU predicted by the model.
    - `stability_score_thresh`, the theshold for keepong objects according to their stability.

    Args:
        iou_thresh_values: The values for `pred_iou_thresh` used in the gridsearch.
            By default values in the range from 0.6 to 0.9 with a stepsize of 0.025 will be used.
        stability_score_values: The values for `stability_score_thresh` used in the gridsearch.
            By default values in the range from 0.6 to 0.9 with a stepsize of 0.025 will be used.

    Returns:
        The values for grid search.
    """
    if iou_thresh_values is None:
        iou_thresh_values = _get_range_of_search_values([0.6, 0.9], step=0.025)
    if stability_score_values is None:
        stability_score_values = _get_range_of_search_values([0.6, 0.95], step=0.025)
    return {
        "pred_iou_thresh": iou_thresh_values,
        "stability_score_thresh": stability_score_values,
    }


def default_grid_search_values_instance_segmentation_with_decoder(
    center_distance_threshold_values: Optional[List[float]] = None,
    boundary_distance_threshold_values: Optional[List[float]] = None,
    distance_smoothing_values: Optional[List[float]] = None,
    min_size_values: Optional[List[float]] = None,
) -> Dict[str, List[float]]:
    """Default grid-search parameter for decoder-based instance segmentation.

    Args:
        center_distance_threshold_values: The values for `center_distance_threshold` used in the gridsearch.
            By default values in the range from 0.3 to 0.7 with a stepsize of 0.1 will be used.
        boundary_distance_threshold_values: The values for `boundary_distance_threshold` used in the gridsearch.
            By default values in the range from 0.3 to 0.7 with a stepsize of 0.1 will be used.
        distance_smoothing_values: The values for `distance_smoothing` used in the gridsearch.
            By default values in the range from 1.0 to 2.0 with a stepsize of 0.1 will be used.
        min_size_values: The values for `min_size` used in the gridsearch.
            By default the values 50, 100 and 200  are used.

    Returns:
        The values for grid search.
    """
    if center_distance_threshold_values is None:
        center_distance_threshold_values = _get_range_of_search_values(
            [0.3, 0.7], step=0.1
        )
    if boundary_distance_threshold_values is None:
        boundary_distance_threshold_values = _get_range_of_search_values(
            [0.3, 0.7], step=0.1
        )
    if distance_smoothing_values is None:
        distance_smoothing_values = _get_range_of_search_values(
            [1.0, 2.0], step=0.2
        )
    if min_size_values is None:
        min_size_values = [50, 100, 200]
    return {
        "center_distance_threshold": center_distance_threshold_values,
        "boundary_distance_threshold": boundary_distance_threshold_values,
        "distance_smoothing": distance_smoothing_values,
        "min_size": min_size_values,
    }


def _grid_search_iteration(
    segmenter: Union[AMGBase, InstanceSegmentationWithDecoder],
    gs_combinations: List[Dict],
    gt: np.ndarray,
    image_name: str,
    fixed_generate_kwargs: Dict[str, Any],
    result_path: Optional[Union[str, os.PathLike]],
    verbose: bool = False,
) -> pd.DataFrame:
    net_list = []
    for gs_kwargs in tqdm(gs_combinations, disable=not verbose):
        generate_kwargs = gs_kwargs | fixed_generate_kwargs
        masks = segmenter.generate(**generate_kwargs)

        min_object_size = generate_kwargs.get("min_mask_region_area", 0)
        if len(masks) == 0:
            instance_labels = np.zeros(gt.shape, dtype="uint32")
        else:
            instance_labels = mask_data_to_segmentation(masks, with_background=True, min_object_size=min_object_size)
        m_sas, sas = mean_segmentation_accuracy(instance_labels, gt, return_accuracies=True)  # type: ignore

        result_dict = {"image_name": image_name, "mSA": m_sas, "SA50": sas[0], "SA75": sas[5]}
        result_dict.update(gs_kwargs)
        tmp_df = pd.DataFrame([result_dict])
        net_list.append(tmp_df)

    img_gs_df = pd.concat(net_list)
    img_gs_df.to_csv(result_path, index=False)

    return img_gs_df


def _load_image(path, key, roi):
    if key is None:
        im = imageio.imread(path)
        if roi is not None:
            im = im[roi]
        return im
    with open_file(path, "r") as f:
        im = f[key][:] if roi is None else f[key][roi]
    return im


def run_instance_segmentation_grid_search(
    segmenter: Union[AMGBase, InstanceSegmentationWithDecoder],
    grid_search_values: Dict[str, List],
    image_paths: List[Union[str, os.PathLike]],
    gt_paths: List[Union[str, os.PathLike]],
    result_dir: Union[str, os.PathLike],
    embedding_dir: Optional[Union[str, os.PathLike]],
    fixed_generate_kwargs: Optional[Dict[str, Any]] = None,
    verbose_gs: bool = False,
    image_key: Optional[str] = None,
    gt_key: Optional[str] = None,
    rois: Optional[Tuple[slice, ...]] = None,
) -> None:
    """Run grid search for automatic mask generation.

    The parameters and their respective value ranges for the grid search are specified via the
    'grid_search_values' argument. For example, to run a grid search over the parameters 'pred_iou_thresh'
    and 'stability_score_thresh', you can pass the following:
    ```
    grid_search_values = {
        "pred_iou_thresh": [0.6, 0.7, 0.8, 0.9],
        "stability_score_thresh": [0.6, 0.7, 0.8, 0.9],
    }
    ```
    All combinations of the parameters will be checked.

    You can use the functions `default_grid_search_values_instance_segmentation_with_decoder`
    or `default_grid_search_values_amg` to get the default grid search parameters for the two
    respective instance segmentation methods.

    Args:
        segmenter: The class implementing the instance segmentation functionality.
        grid_search_values: The grid search values for parameters of the `generate` function.
        image_paths: The input images for the grid search.
        gt_paths: The ground-truth segmentation for the grid search.
        result_dir: Folder to cache the evaluation results per image.
        embedding_dir: Folder to cache the image embeddings.
        fixed_generate_kwargs: Fixed keyword arguments for the `generate` method of the segmenter.
        verbose_gs: Whether to run the grid-search for individual images in a verbose mode.
        image_key: Key for loading the image data from a more complex file format like HDF5.
            If not given a simple image format like tif is assumed.
        gt_key: Key for loading the ground-truth data from a more complex file format like HDF5.
            If not given a simple image format like tif is assumed.
        rois: Region of interests to resetrict the evaluation to.
    """
    verbose_embeddings = False

    assert len(image_paths) == len(gt_paths)
    fixed_generate_kwargs = {} if fixed_generate_kwargs is None else fixed_generate_kwargs

    duplicate_params = [gs_param for gs_param in grid_search_values.keys() if gs_param in fixed_generate_kwargs]
    if duplicate_params:
        raise ValueError(
            "You may not pass duplicate parameters in 'grid_search_values' and 'fixed_generate_kwargs'."
            f"The parameters {duplicate_params} are duplicated."
        )

    # Compute all combinations of grid search values.
    gs_combinations = product(*grid_search_values.values())
    # Map each combination back to a valid kwarg input.
    gs_combinations = [
        {k: v for k, v in zip(grid_search_values.keys(), vals)} for vals in gs_combinations
    ]

    os.makedirs(result_dir, exist_ok=True)
    predictor = getattr(segmenter, "_predictor", None)

    for i, (image_path, gt_path) in tqdm(
        enumerate(zip(image_paths, gt_paths)), desc="Run instance segmentation grid-search", total=len(image_paths)
    ):
        image_name = Path(image_path).stem
        result_path = os.path.join(result_dir, f"{image_name}.csv")

        # We skip images for which the grid search was done already.
        if os.path.exists(result_path):
            continue

        assert os.path.exists(image_path), image_path
        assert os.path.exists(gt_path), gt_path

        image = _load_image(image_path, image_key, roi=None if rois is None else rois[i])
        gt = _load_image(gt_path, gt_key, roi=None if rois is None else rois[i])

        if embedding_dir is None:
            segmenter.initialize(image)
        else:
            assert predictor is not None
            embedding_path = os.path.join(embedding_dir, f"{os.path.splitext(image_name)[0]}.zarr")
            image_embeddings = util.precompute_image_embeddings(
                predictor, image, embedding_path, ndim=2, verbose=verbose_embeddings
            )
            segmenter.initialize(image, image_embeddings)

        _grid_search_iteration(
            segmenter, gs_combinations, gt, image_name,
            fixed_generate_kwargs=fixed_generate_kwargs, result_path=result_path, verbose=verbose_gs,
        )


def run_instance_segmentation_inference(
    segmenter: Union[AMGBase, InstanceSegmentationWithDecoder],
    image_paths: List[Union[str, os.PathLike]],
    embedding_dir: Union[str, os.PathLike],
    prediction_dir: Union[str, os.PathLike],
    generate_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """Run inference for automatic mask generation.

    Args:
        segmenter: The class implementing the instance segmentation functionality.
        image_paths: The input images.
        embedding_dir: Folder to cache the image embeddings.
        prediction_dir: Folder to save the predictions.
        generate_kwargs: The keyword arguments for the `generate` method of the segmenter.
    """

    verbose_embeddings = False

    generate_kwargs = {} if generate_kwargs is None else generate_kwargs
    predictor = segmenter._predictor
    min_object_size = generate_kwargs.get("min_mask_region_area", 0)

    for image_path in tqdm(image_paths, desc="Run inference for automatic mask generation"):
        image_name = os.path.basename(image_path)

        # We skip the images that already have been segmented.
        prediction_path = os.path.join(prediction_dir, image_name)
        if os.path.exists(prediction_path):
            continue

        assert os.path.exists(image_path), image_path
        image = imageio.imread(image_path)

        embedding_path = os.path.join(embedding_dir, f"{os.path.splitext(image_name)[0]}.zarr")
        image_embeddings = util.precompute_image_embeddings(
            predictor, image, embedding_path, ndim=2, verbose=verbose_embeddings
        )

        segmenter.initialize(image, image_embeddings)
        masks = segmenter.generate(**generate_kwargs)

        if len(masks) == 0:  # the instance segmentation can have no masks, hence we just save empty labels
            if isinstance(segmenter, InstanceSegmentationWithDecoder):
                this_shape = segmenter._foreground.shape
            elif isinstance(segmenter, AMGBase):
                this_shape = segmenter._original_size
            else:
                this_shape = image.shape[-2:]

            instances = np.zeros(this_shape, dtype="uint32")
        else:
            instances = mask_data_to_segmentation(masks, with_background=True, min_object_size=min_object_size)

        # It's important to compress here, otherwise the predictions would take up a lot of space.
        imageio.imwrite(prediction_path, instances, compression=5)


def evaluate_instance_segmentation_grid_search(
    result_dir: Union[str, os.PathLike],
    grid_search_parameters: List[str],
    criterion: str = "mSA"
) -> Tuple[Dict[str, Any], float]:
    """Evaluate gridsearch results.

    Args:
        result_dir: The folder with the gridsearch results.
        grid_search_parameters: The names for the gridsearch parameters.
        criterion: The metric to use for determining the best parameters.

    Returns:
        The best parameter setting.
        The evaluation score for the best setting.
    """

    # Load all the grid search results.
    gs_files = glob(os.path.join(result_dir, "*.csv"))
    gs_result = pd.concat([pd.read_csv(gs_file) for gs_file in gs_files])

    # Retrieve only the relevant columns and group by the gridsearch columns.
    gs_result = gs_result[grid_search_parameters + [criterion]].reset_index()

    # Compute the mean over the grouped columns.
    grouped_result = gs_result.groupby(grid_search_parameters).mean().reset_index()

    # Find the best score and corresponding parameters.
    best_score, best_idx = grouped_result[criterion].max(), grouped_result[criterion].idxmax()
    best_params = grouped_result.iloc[best_idx]
    assert np.isclose(best_params[criterion], best_score)
    best_kwargs = {k: v for k, v in zip(grid_search_parameters, best_params)}

    return best_kwargs, best_score


def save_grid_search_best_params(best_kwargs, best_msa, grid_search_result_dir=None):
    # saving the best parameters estimated from grid-search in the `results` folder
    param_df = pd.DataFrame.from_dict([best_kwargs])
    res_df = pd.DataFrame.from_dict([{"best_msa": best_msa}])
    best_param_df = pd.merge(res_df, param_df, left_index=True, right_index=True)

    path_name = "grid_search_params_amg.csv" if "pred_iou_thresh" and "stability_score_thresh" in best_kwargs \
        else "grid_search_params_instance_segmentation_with_decoder.csv"

    if grid_search_result_dir is not None:
        os.makedirs(os.path.join(grid_search_result_dir, "results"), exist_ok=True)
        res_path = os.path.join(grid_search_result_dir, "results", path_name)
    else:
        res_path = path_name

    best_param_df.to_csv(res_path)


def run_instance_segmentation_grid_search_and_inference(
    segmenter: Union[AMGBase, InstanceSegmentationWithDecoder],
    grid_search_values: Dict[str, List],
    val_image_paths: List[Union[str, os.PathLike]],
    val_gt_paths: List[Union[str, os.PathLike]],
    test_image_paths: List[Union[str, os.PathLike]],
    embedding_dir: Union[str, os.PathLike],
    prediction_dir: Union[str, os.PathLike],
    result_dir: Union[str, os.PathLike],
    fixed_generate_kwargs: Optional[Dict[str, Any]] = None,
    verbose_gs: bool = True,
) -> None:
    """Run grid search and inference for automatic mask generation.

    Please refer to the documentation of `run_instance_segmentation_grid_search`
    for details on how to specify the grid search parameters.

    Args:
        segmenter: The class implementing the instance segmentation functionality.
        grid_search_values: The grid search values for parameters of the `generate` function.
        val_image_paths: The input images for the grid search.
        val_gt_paths: The ground-truth segmentation for the grid search.
        test_image_paths: The input images for inference.
        embedding_dir: Folder to cache the image embeddings.
        prediction_dir: Folder to save the predictions.
        result_dir: Folder to cache the evaluation results per image.
        fixed_generate_kwargs: Fixed keyword arguments for the `generate` method of the segmenter.
        verbose_gs: Whether to run the gridsearch for individual images in a verbose mode.
    """
    run_instance_segmentation_grid_search(
        segmenter, grid_search_values, val_image_paths, val_gt_paths,
        result_dir=result_dir, embedding_dir=embedding_dir,
        fixed_generate_kwargs=fixed_generate_kwargs, verbose_gs=verbose_gs,
    )

    best_kwargs, best_msa = evaluate_instance_segmentation_grid_search(result_dir, list(grid_search_values.keys()))
    best_param_str = ", ".join(f"{k} = {v}" for k, v in best_kwargs.items())
    print("Best grid-search result:", best_msa, "with parmeters:\n", best_param_str)
    print()

    save_grid_search_best_params(best_kwargs, best_msa, Path(embedding_dir).parent)

    generate_kwargs = {} if fixed_generate_kwargs is None else fixed_generate_kwargs
    generate_kwargs.update(best_kwargs)

    run_instance_segmentation_inference(
        segmenter, test_image_paths, embedding_dir, prediction_dir, generate_kwargs
    )
