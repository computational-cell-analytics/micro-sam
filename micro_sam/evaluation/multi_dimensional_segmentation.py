import os
from tqdm import tqdm
from itertools import product
from typing import Union, Tuple, Optional, List, Dict, Literal

import numpy as np
import pandas as pd
from math import floor
import imageio.v3 as imageio

import torch

from elf.evaluation import mean_segmentation_accuracy, dice_score

from .. import util
from ..inference import batched_inference
from ..prompt_generators import PointAndBoxPromptGenerator
from ..multi_dimensional_segmentation import segment_mask_in_volume
from ..evaluation.instance_segmentation import _get_range_of_search_values, evaluate_instance_segmentation_grid_search


def default_grid_search_values_multi_dimensional_segmentation(
    iou_threshold_values: Optional[List[float]] = None,
    projection_method_values: Optional[Union[str, dict]] = None,
    box_extension_values: Optional[Union[float, int]] = None
) -> Dict[str, List]:
    """Default grid-search parameters for multi-dimensional prompt-based instance segmentation.

    Args:
        iou_threshold_values: The values for `iou_threshold` used in the grid-search.
            By default values in the range from 0.5 to 0.9 with a stepsize of 0.1 will be used.
        projection_method_values: The values for `projection` method used in the grid-search.
            By default the values `mask`, `points`, `box`, `points_and_mask` and `single_point` are used.
        box_extension_values: The values for `box_extension` used in the grid-search.
            By default values in the range from 0 to 0.25 with a stepsize of 0.025 will be used.

    Returns:
        The values for grid search.
    """
    if iou_threshold_values is None:
        iou_threshold_values = _get_range_of_search_values([0.5, 0.9], step=0.1)

    if projection_method_values is None:
        projection_method_values = [
            "mask", "points", "box", "points_and_mask", "single_point"
        ]

    if box_extension_values is None:
        box_extension_values = _get_range_of_search_values([0, 0.25], step=0.025)

    return {
        "iou_threshold": iou_threshold_values,
        "projection": projection_method_values,
        "box_extension": box_extension_values
    }


@torch.no_grad()
def segment_slices_from_ground_truth(
    volume: np.ndarray,
    ground_truth: np.ndarray,
    model_type: str,
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    embedding_path: Optional[Union[str, os.PathLike]] = None,
    save_path: Optional[Union[str, os.PathLike]] = None,
    iou_threshold: float = 0.8,
    projection: Union[str, dict] = "mask",
    box_extension: Union[float, int] = 0.025,
    device: Union[str, torch.device] = None,
    interactive_seg_mode: str = "box",
    verbose: bool = False,
    return_segmentation: bool = False,
    min_size: int = 0,
    evaluation_metric: Literal["sa", "dice"] = "sa",
) -> Union[Dict, Tuple[Dict, np.ndarray]]:
    """Segment all objects in a volume by prompt-based segmentation in one slice per object.

    This function first segments each object in the respective specified slice using interactive
    (prompt-based) segmentation functionality. Then it segments the particular object in the
    remaining slices in the volume.

    Args:
        volume: The input volume.
        ground_truth: The label volume with instance segmentations.
        model_type: Choice of segment anything model.
        checkpoint_path: Path to the model checkpoint.
        embedding_path: Path to cache the computed embeddings.
        save_path: Path to store the segmentations.
        iou_threshold: The criterion to decide whether to link the objects in the consecutive slice's segmentation.
        projection: The projection (prompting) method to generate prompts for consecutive slices.
        box_extension: Extension factor for increasing the box size after projection.
        device: The selected device for computation.
        interactive_seg_mode: Method for guiding prompt-based instance segmentation.
        verbose: Whether to get the trace for projected segmentations.
        return_segmentation: Whether to return the segmented volume.
        min_size: The minimal size for evaluating an object in the ground-truth.
            The size is measured within the central slice.
        evaluation_metric: The choice of supported metric to evaluate predictions.

    Returns:
        A dictionary of results with all desired metrics.
        Optional segmentation result (controlled by `return_segmentation` argument).
    """
    assert volume.ndim == 3

    predictor = util.get_sam_model(model_type=model_type, checkpoint_path=checkpoint_path, device=device)

    # Compute the image embeddings
    embeddings = util.precompute_image_embeddings(
        predictor=predictor, input_=volume, save_path=embedding_path, ndim=3, verbose=verbose,
    )

    # Compute instance ids (without the background)
    label_ids = np.unique(ground_truth)[1:]
    assert len(label_ids) > 0, "There are no objects to perform volumetric segmentation."

    # Create an empty volume to store incoming segmentations
    final_segmentation = np.zeros_like(ground_truth)

    _segmentation_completed = False
    if save_path is not None and os.path.exists(save_path):
        _segmentation_completed = True  # We avoid rerunning the segmentation if it is completed.

    skipped_label_ids = []
    for label_id in tqdm(label_ids, desc="Segmenting per object in the volume", disable=not verbose):
        # Binary label volume per instance (also referred to as object)
        this_seg = (ground_truth == label_id).astype("int")

        # Let's search the slices where we have the current object
        slice_range = np.where(this_seg)[0]

        # Choose the middle slice of the current object for prompt-based segmentation
        slice_range = (slice_range.min(), slice_range.max())
        slice_choice = floor(np.mean(slice_range))
        this_slice_seg = this_seg[slice_choice]
        if min_size > 0 and this_slice_seg.sum() < min_size:
            skipped_label_ids.append(label_id)
            continue

        if _segmentation_completed:
            continue

        if verbose:
            print(f"The object with id {label_id} lies in slice range: {slice_range}")

        # Prompts for segmentation for the current slice
        if interactive_seg_mode == "points":
            _get_points, _get_box = True, False
        elif interactive_seg_mode == "box":
            _get_points, _get_box = False, True
        else:
            raise ValueError(
                f"The provided interactive prompting '{interactive_seg_mode}' for the first slice isn't supported. "
                "Please choose from 'box' / 'points'."
            )

        prompt_generator = PointAndBoxPromptGenerator(
            n_positive_points=1 if _get_points else 0,
            n_negative_points=1 if _get_points else 0,
            dilation_strength=10,
            get_point_prompts=_get_points,
            get_box_prompts=_get_box
        )
        _, box_coords = util.get_centers_and_bounding_boxes(this_slice_seg)
        point_prompts, point_labels, box_prompts, _ = prompt_generator(
            segmentation=torch.from_numpy(this_slice_seg)[None, None].to(torch.float32),
            bbox_coordinates=[box_coords[1]],
        )

        # Prompt-based segmentation on middle slice of the current object
        output_slice = batched_inference(
            predictor=predictor,
            image=volume[slice_choice],
            batch_size=1,
            boxes=box_prompts.numpy() if isinstance(box_prompts, torch.Tensor) else box_prompts,
            points=point_prompts.numpy() if isinstance(point_prompts, torch.Tensor) else point_prompts,
            point_labels=point_labels.numpy() if isinstance(point_labels, torch.Tensor) else point_labels,
            verbose_embeddings=verbose,
        )
        output_seg = np.zeros_like(ground_truth)
        output_seg[slice_choice][output_slice == 1] = 1

        # Segment the object in the entire volume with the specified segmented slice
        this_seg, _ = segment_mask_in_volume(
            segmentation=output_seg,
            predictor=predictor,
            image_embeddings=embeddings,
            segmented_slices=np.array(slice_choice),
            stop_lower=False, stop_upper=False,
            iou_threshold=iou_threshold,
            projection=projection,
            box_extension=box_extension,
            verbose=verbose,
        )

        # Store the entire segmented object
        final_segmentation[this_seg == 1] = label_id

    # Save the volumetric segmentation
    if save_path is not None:
        if _segmentation_completed:
            final_segmentation = imageio.imread(save_path)
        else:
            imageio.imwrite(save_path, final_segmentation, compression="zlib")

    # Evaluate the volumetric segmentation
    if skipped_label_ids:
        curr_gt = ground_truth.copy()
        curr_gt[np.isin(curr_gt, skipped_label_ids)] = 0
    else:
        curr_gt = ground_truth

    if evaluation_metric == "sa":
        msa, sa = mean_segmentation_accuracy(
            segmentation=final_segmentation, groundtruth=curr_gt, return_accuracies=True
        )
        results = {"mSA": msa, "SA50": sa[0], "SA75": sa[5]}

    elif evaluation_metric == "dice":
        # Calculate overall dice score (by binarizing all labels).
        dice = dice_score(segmentation=final_segmentation, groundtruth=curr_gt)
        results = {"Dice": dice}

    elif evaluation_metric == "dice_per_class":
        # Calculate dice per class.
        dice = [
            dice_score(segmentation=(final_segmentation == i), groundtruth=(curr_gt == i))
            for i in np.unique(curr_gt)[1:]
        ]
        dice = np.mean(dice)
        results = {"Dice": dice}

    else:
        raise ValueError(
            f"'{evaluation_metric}' is not a supported evaluation metrics. "
            "Please choose 'sa' / 'dice' / 'dice_per_class'."
        )

    if return_segmentation:
        return results, final_segmentation
    else:
        return results


def _get_best_parameters_from_grid_search_combinations(
    result_dir, best_params_path, grid_search_values, evaluation_metric,
):
    if os.path.exists(best_params_path):
        print("The best parameters are already saved at:", best_params_path)
        return

    criterion = "mSA" if evaluation_metric == "sa" else "Dice"
    best_kwargs, best_metric = evaluate_instance_segmentation_grid_search(
        result_dir=result_dir, grid_search_parameters=list(grid_search_values.keys()), criterion=criterion,
    )

    # let's save the best parameters
    best_kwargs[criterion] = best_metric
    best_param_df = pd.DataFrame.from_dict([best_kwargs])
    best_param_df.to_csv(best_params_path)

    best_param_str = ", ".join(f"{k} = {v}" for k, v in best_kwargs.items())
    print("Best grid-search result:", best_metric, "with parmeters:\n", best_param_str)


def run_multi_dimensional_segmentation_grid_search(
    volume: np.ndarray,
    ground_truth: np.ndarray,
    model_type: str,
    checkpoint_path: Union[str, os.PathLike],
    embedding_path: Optional[Union[str, os.PathLike]],
    result_dir: Union[str, os.PathLike],
    interactive_seg_mode: str = "box",
    verbose: bool = False,
    grid_search_values: Optional[Dict[str, List]] = None,
    min_size: int = 0,
    evaluation_metric: Literal["sa", "dice"] = "sa",
    store_segmentation: bool = False,
) -> str:
    """Run grid search for prompt-based multi-dimensional instance segmentation.

    The parameters and their respective value ranges for the grid search are specified via the
    `grid_search_values` argument. For example, to run a grid search over the parameters `iou_threshold`,
    `projection` and `box_extension`, you can pass the following:
    ```python
    grid_search_values = {
        "iou_threshold": [0.5, 0.6, 0.7, 0.8, 0.9],
        "projection": ["mask", "box", "points"],
        "box_extension": [0, 0.1, 0.2, 0.3, 0.4, 0,5],
    }
    ```
    All combinations of the parameters will be checked.
    If passed None, the function `default_grid_search_values_multi_dimensional_segmentation` is used
    to get the default grid search parameters for the instance segmentation method.

    Args:
        volume: The input volume.
        ground_truth: The label volume with instance segmentations.
        model_type: Choice of segment anything model.
        checkpoint_path: Path to the model checkpoint.
        embedding_path: Path to cache the computed embeddings.
        result_dir: Path to save the grid search results.
        interactive_seg_mode: Method for guiding prompt-based instance segmentation.
        verbose: Whether to get the trace for projected segmentations.
        grid_search_values: The grid search values for parameters of the `segment_slices_from_ground_truth` function.
        min_size: The minimal size for evaluating an object in the ground-truth.
            The size is measured within the central slice.
        evaluation_metric: The choice of metric for evaluating predictions.
        store_segmentation: Whether to store the segmentations per grid-search locally.

    Returns:
        Filepath where the best parameters are saved.
    """
    if grid_search_values is None:
        grid_search_values = default_grid_search_values_multi_dimensional_segmentation()

    assert len(grid_search_values.keys()) == 3, "There must be three grid-search parameters. See above for details."

    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, "all_grid_search_results.csv")
    best_params_path = os.path.join(result_dir, "grid_search_params_multi_dimensional_segmentation.csv")
    if os.path.exists(result_path):
        _get_best_parameters_from_grid_search_combinations(
            result_dir, best_params_path, grid_search_values, evaluation_metric
        )
        return best_params_path

    # Compute all combinations of grid search values.
    gs_combinations = product(*grid_search_values.values())

    # Map each combination back to a valid kwarg input.
    gs_combinations = [
        {k: v for k, v in zip(grid_search_values.keys(), vals)} for vals in gs_combinations
    ]

    prediction_dir = os.path.join(result_dir, "predictions")
    os.makedirs(prediction_dir, exist_ok=True)

    net_list = []
    for i, gs_kwargs in tqdm(
        enumerate(gs_combinations),
        total=len(gs_combinations),
        desc="Run grid-search for multi-dimensional segmentation",
    ):
        results, segmentation = segment_slices_from_ground_truth(
            volume=volume,
            ground_truth=ground_truth,
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            embedding_path=embedding_path,
            interactive_seg_mode=interactive_seg_mode,
            verbose=verbose,
            return_segmentation=True,
            min_size=min_size,
            evaluation_metric=evaluation_metric,
            **gs_kwargs
        )

        # Store the segmentations for each grid search combination, if desired by the user.
        if store_segmentation:
            imageio.imwrite(
                os.path.join(prediction_dir, f"grid_search_result_{i:05}.tif"), segmentation, compression="zlib"
            )

        result_dict = {**results, **gs_kwargs}
        tmp_df = pd.DataFrame([result_dict])
        net_list.append(tmp_df)

    res_df = pd.concat(net_list, ignore_index=True)
    res_df.to_csv(result_path)

    _get_best_parameters_from_grid_search_combinations(
        result_dir, best_params_path, grid_search_values, evaluation_metric
    )
    print("The best grid-search parameters have been computed and stored at:", best_params_path)
    return best_params_path
