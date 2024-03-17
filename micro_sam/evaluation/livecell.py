"""Inference and evaluation for the [LIVECell dataset](https://www.nature.com/articles/s41592-021-01249-6) and
the different cell lines contained in it.
"""
import os
import json
import argparse
import warnings
from glob import glob
from tqdm import tqdm
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from segment_anything import SamPredictor

from ..instance_segmentation import (
    get_custom_sam_model_with_decoder,
    AutomaticMaskGenerator, InstanceSegmentationWithDecoder,
)
from ..evaluation import precompute_all_embeddings
from . import instance_segmentation, inference, evaluation


CELL_TYPES = ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"]


#
# Inference
#


def _get_livecell_paths(input_folder, split="test", n_val_per_cell_type=None):
    assert split in ["val", "test"]
    assert os.path.exists(input_folder), f"Data not found at {input_folder}. Please download the LIVECell Dataset"

    if split == "test":

        img_dir = os.path.join(input_folder, "images", "livecell_test_images")
        assert os.path.exists(img_dir), "The LIVECell Dataset is incomplete"
        gt_dir = os.path.join(input_folder, "annotations", "livecell_test_images")
        assert os.path.exists(gt_dir), "The LIVECell Dataset is incomplete"
        image_paths, gt_paths = [], []
        for ctype in CELL_TYPES:
            for img_path in glob(os.path.join(img_dir, f"{ctype}*")):
                image_paths.append(img_path)
                img_name = os.path.basename(img_path)
                gt_path = os.path.join(gt_dir, ctype, img_name)
                assert os.path.exists(gt_path), gt_path
                gt_paths.append(gt_path)
    else:

        with open(os.path.join(input_folder, "val.json")) as f:
            data = json.load(f)
        livecell_val_ids = [i["file_name"] for i in data["images"]]

        img_dir = os.path.join(input_folder, "images", "livecell_train_val_images")
        assert os.path.exists(img_dir), "The LIVECell Dataset is incomplete"
        gt_dir = os.path.join(input_folder, "annotations", "livecell_train_val_images")
        assert os.path.exists(gt_dir), "The LIVECell Dataset is incomplete"

        image_paths, gt_paths = [], []
        count_per_cell_type = {ct: 0 for ct in CELL_TYPES}

        for img_name in livecell_val_ids:
            cell_type = img_name.split("_")[0]
            if n_val_per_cell_type is not None and count_per_cell_type[cell_type] >= n_val_per_cell_type:
                continue

            image_paths.append(os.path.join(img_dir, img_name))
            gt_paths.append(os.path.join(gt_dir, cell_type, img_name))
            count_per_cell_type[cell_type] += 1

    return image_paths, gt_paths


def livecell_inference(
    checkpoint: Union[str, os.PathLike],
    input_folder: Union[str, os.PathLike],
    model_type: str,
    experiment_folder: Union[str, os.PathLike],
    use_points: bool,
    use_boxes: bool,
    n_positives: Optional[int] = None,
    n_negatives: Optional[int] = None,
    prompt_folder: Optional[Union[str, os.PathLike]] = None,
    predictor: Optional[SamPredictor] = None,
) -> None:
    """Run inference for livecell with a fixed prompt setting.

    Args:
        checkpoint: The segment anything model checkpoint.
        input_folder: The folder with the livecell data.
        model_type: The type of the segment anything model.
        experiment_folder: The folder where to save all data associated with the experiment.
        use_points: Whether to use point prompts.
        use_boxes: Whether to use box prompts.
        n_positives: The number of positive point prompts.
        n_negatives: The number of negative point prompts.
        prompt_folder: The folder where the prompts should be saved.
        predictor: The segment anything predictor.
    """
    image_paths, gt_paths = _get_livecell_paths(input_folder)
    if predictor is None:
        predictor = inference.get_predictor(checkpoint, model_type)

    if use_boxes and use_points:
        assert (n_positives is not None) and (n_negatives is not None)
        setting_name = f"box/p{n_positives}-n{n_negatives}"
    elif use_boxes:
        setting_name = "box/p0-n0"
    elif use_points:
        assert (n_positives is not None) and (n_negatives is not None)
        setting_name = f"points/p{n_positives}-n{n_negatives}"
    else:
        raise ValueError("You need to use at least one of point or box prompts.")

    # we organize all folders with data from this experiment beneath 'experiment_folder'
    prediction_folder = os.path.join(experiment_folder, setting_name)  # where the predicted segmentations are saved
    os.makedirs(prediction_folder, exist_ok=True)
    embedding_folder = os.path.join(experiment_folder, "embeddings")  # where the precomputed embeddings are saved
    os.makedirs(embedding_folder, exist_ok=True)

    # NOTE: we can pass an external prompt folder, to make re-use prompts from another experiment
    # for reproducibility / fair comparison of results
    if prompt_folder is None:
        prompt_folder = os.path.join(experiment_folder, "prompts")
        os.makedirs(prompt_folder, exist_ok=True)

    inference.run_inference_with_prompts(
        predictor,
        image_paths,
        gt_paths,
        embedding_dir=embedding_folder,
        prediction_dir=prediction_folder,
        prompt_save_dir=prompt_folder,
        use_points=use_points,
        use_boxes=use_boxes,
        n_positives=n_positives,
        n_negatives=n_negatives,
    )


def run_livecell_precompute_embeddings(
    checkpoint: Union[str, os.PathLike],
    input_folder: Union[str, os.PathLike],
    model_type: str,
    experiment_folder: Union[str, os.PathLike],
    n_val_per_cell_type: int = 25,
) -> None:
    """Run precomputation of val and test image embeddings for livecell.

    Args:
        checkpoint: The segment anything model checkpoint.
        input_folder: The folder with the livecell data.
        model_type: The type of the segmenta anything model.
        experiment_folder: The folder where to save all data associated with the experiment.
    """
    embedding_folder = os.path.join(experiment_folder, "embeddings")  # where the embeddings will be saved
    os.makedirs(embedding_folder, exist_ok=True)

    predictor = inference.get_predictor(checkpoint, model_type)

    val_image_paths, _ = _get_livecell_paths(input_folder, "val", n_val_per_cell_type=n_val_per_cell_type)
    test_image_paths, _ = _get_livecell_paths(input_folder, "test")

    precompute_all_embeddings(predictor, val_image_paths, embedding_folder)
    precompute_all_embeddings(predictor, test_image_paths, embedding_folder)


def run_livecell_iterative_prompting(
    checkpoint: Union[str, os.PathLike],
    input_folder: Union[str, os.PathLike],
    model_type: str,
    experiment_folder: Union[str, os.PathLike],
    n_val_per_cell_type: int = 25,
    start_with_box: bool = False,
    use_masks: bool = False,
) -> str:
    """Run inference on livecell with iterative prompting setting.

    Args:
        checkpoint: The segment anything model checkpoint.
        input_folder: The folder with the livecell data.
        model_type: The type of the segment anything model.
        experiment_folder: The folder where to save all data associated with the experiment.

    """
    embedding_folder = os.path.join(experiment_folder, "embeddings")  # where the embeddings will be saved
    os.makedirs(embedding_folder, exist_ok=True)

    predictor = inference.get_predictor(checkpoint, model_type)

    # where the predictions are saved
    prediction_folder = os.path.join(
        experiment_folder, "start_with_box" if start_with_box else "start_with_point"
    )

    image_paths, gt_paths = _get_livecell_paths(input_folder, "test")

    inference.run_inference_with_iterative_prompting(
        predictor=predictor,
        image_paths=image_paths,
        gt_paths=gt_paths,
        embedding_dir=embedding_folder,
        prediction_dir=prediction_folder,
        start_with_box_prompt=start_with_box,
        use_masks=use_masks,
    )
    return prediction_folder


def run_livecell_amg(
    checkpoint: Union[str, os.PathLike],
    input_folder: Union[str, os.PathLike],
    model_type: str,
    experiment_folder: Union[str, os.PathLike],
    iou_thresh_values: Optional[List[float]] = None,
    stability_score_values: Optional[List[float]] = None,
    verbose_gs: bool = False,
    n_val_per_cell_type: int = 25,
) -> str:
    """Run automatic mask generation grid-search and inference for livecell.

    Args:
        checkpoint: The segment anything model checkpoint.
        input_folder: The folder with the livecell data.
        model_type: The type of the segmenta anything model.
        experiment_folder: The folder where to save all data associated with the experiment.
        iou_thresh_values: The values for `pred_iou_thresh` used in the gridsearch.
            By default values in the range from 0.6 to 0.9 with a stepsize of 0.025 will be used.
        stability_score_values: The values for `stability_score_thresh` used in the gridsearch.
            By default values in the range from 0.6 to 0.9 with a stepsize of 0.025 will be used.
        verbose_gs: Whether to run the gridsearch for individual images in a verbose mode.
        n_val_per_cell_type: The number of validation images per cell type.

    Returns:
        The path where the predicted images are stored.
    """
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

    val_image_paths, val_gt_paths = _get_livecell_paths(input_folder, "val", n_val_per_cell_type=n_val_per_cell_type)
    test_image_paths, _ = _get_livecell_paths(input_folder, "test")

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


def run_livecell_instance_segmentation_with_decoder(
    checkpoint: Union[str, os.PathLike],
    input_folder: Union[str, os.PathLike],
    model_type: str,
    experiment_folder: Union[str, os.PathLike],
    verbose_gs: bool = False,
    n_val_per_cell_type: int = 25,
) -> str:
    """Run automatic mask generation grid-search and inference for livecell.

    Args:
        checkpoint: The segment anything model checkpoint.
        input_folder: The folder with the livecell data.
        model_type: The type of the segmenta anything model.
        experiment_folder: The folder where to save all data associated with the experiment.
        verbose_gs: Whether to run the gridsearch for individual images in a verbose mode.
        n_val_per_cell_type: The number of validation images per cell type.

    Returns:
        The path where the predicted images are stored.
    """
    embedding_folder = os.path.join(experiment_folder, "embeddings")  # where the precomputed embeddings are saved
    os.makedirs(embedding_folder, exist_ok=True)

    predictor, decoder = get_custom_sam_model_with_decoder(checkpoint, model_type)
    segmenter = InstanceSegmentationWithDecoder(predictor, decoder)
    seg_prefix = "instance_segmentation_with_decoder"

    # where the predictions are saved
    prediction_folder = os.path.join(experiment_folder, seg_prefix, "inference")
    os.makedirs(prediction_folder, exist_ok=True)

    # where the grid-search results are saved
    gs_result_folder = os.path.join(experiment_folder, seg_prefix, "grid_search")
    os.makedirs(gs_result_folder, exist_ok=True)

    val_image_paths, val_gt_paths = _get_livecell_paths(input_folder, "val", n_val_per_cell_type=n_val_per_cell_type)
    test_image_paths, _ = _get_livecell_paths(input_folder, "test")

    grid_search_values = instance_segmentation.default_grid_search_values_instance_segmentation_with_decoder()

    instance_segmentation.run_instance_segmentation_grid_search_and_inference(
        segmenter, grid_search_values,
        val_image_paths, val_gt_paths, test_image_paths,
        embedding_dir=embedding_folder, prediction_dir=prediction_folder,
        result_dir=gs_result_folder,
    )
    return prediction_folder


def run_livecell_inference() -> None:
    """Run LIVECell inference with command line tool."""
    parser = argparse.ArgumentParser()

    # the checkpoint, input and experiment folder
    parser.add_argument(
        "-c", "--ckpt", type=str, required=True,
        help="Provide model checkpoints (vanilla / finetuned)."
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="Provide the data directory for LIVECell Dataset."
    )
    parser.add_argument(
        "-e", "--experiment_folder", type=str, required=True,
        help="Provide the path where all data for the inference run will be stored."
    )
    parser.add_argument(
        "-m", "--model", type=str, required=True,
        help="Pass the checkpoint-specific model name being used for inference."
    )

    # the experiment type:
    # 1. precompute image embeddings
    # 2. interactive instance segmentation (interactive)
    #     - iterative prompting
    #         - starting with point
    #         - starting with box
    # 3. automatic segmentation (auto)
    #     - automatic mask generation (amg)
    #     - automatic instance segmentation (ais)

    parser.add_argument("-p", "--precompute_embeddings", action="store_true")
    parser.add_argument("-ip", "--interactive_segmentation", action="store_true")
    parser.add_argument("-amg", "--auto_mask_generation", action="store_true")
    parser.add_argument("-ais", "--auto_instance_segmentation", action="store_true")

    # the prompt settings for starting iterative prompting for interactive instance segmentation
    #     - (default: start with points)
    parser.add_argument(
        "-b", "--start_with_box", action="store_true", help="Start with box for iterative prompt-based segmentation."
    )
    parser.add_argument(
        "--use_masks", action="store_true",
        help="Whether to use logits from previous interactive segmentation as inputs for iterative prompting."
    )

    args = parser.parse_args()
    if sum([args.interactive_segmentation, args.auto_mask_generation, args.auto_instance_segmentation]) > 1:
        warnings.warn(
            "It's recommended to choose either from 'interactive_segmentation', 'auto_mask_generation' or "
            "'auto_instance_segmentation' at once, else it might take a while."
        )

    if args.precompute_embeddings:
        run_livecell_precompute_embeddings(args.ckpt, args.input, args.model, args.experiment_folder)

    if args.interactive_segmentation:
        run_livecell_iterative_prompting(
            args.ckpt, args.input, args.model, args.experiment_folder,
            start_with_box=args.start_with_box, use_masks=args.use_masks
        )

    if args.auto_instance_segmentation:
        run_livecell_instance_segmentation_with_decoder(args.ckpt, args.input, args.model, args.experiment_folder)

    if args.auto_mask_generation:
        run_livecell_amg(args.ckpt, args.input, args.model, args.experiment_folder)


#
# Evaluation
#


def evaluate_livecell_predictions(
    gt_dir: Union[os.PathLike, str],
    pred_dir: Union[os.PathLike, str],
    verbose: bool,
) -> None:
    """Evaluate LiveCELL predictions.

    Args:
        gt_dir: The folder with the groundtruth segmentations.
        pred_dir: The folder with the segmentation predictions.
        verbose: Whether to run the evaluation in verbose mode.
    """
    assert os.path.exists(gt_dir), gt_dir
    assert os.path.exists(pred_dir), pred_dir

    msas, sa50s, sa75s = [], [], []
    msas_ct, sa50s_ct, sa75s_ct = [], [], []

    for ct in tqdm(CELL_TYPES, desc="Evaluate livecell predictions", disable=not verbose):

        gt_pattern = os.path.join(gt_dir, f"{ct}/*.tif")
        gt_paths = glob(gt_pattern)
        assert len(gt_paths) > 0, "gt_pattern"

        pred_paths = [
            os.path.join(pred_dir, os.path.basename(path)) for path in gt_paths
        ]

        this_msas, this_sa50s, this_sa75s = evaluation._run_evaluation(
            gt_paths, pred_paths, False
        )

        msas.extend(this_msas), sa50s.extend(this_sa50s), sa75s.extend(this_sa75s)
        msas_ct.append(np.mean(this_msas))
        sa50s_ct.append(np.mean(this_sa50s))
        sa75s_ct.append(np.mean(this_sa75s))

    result_dict = {
        "cell_type": CELL_TYPES + ["Total"],
        "msa": msas_ct + [np.mean(msas)],
        "sa50": sa50s_ct + [np.mean(sa50s_ct)],
        "sa75": sa75s_ct + [np.mean(sa75s_ct)],
    }
    df = pd.DataFrame.from_dict(result_dict)
    df = df.round(decimals=4)
    return df


def run_livecell_evaluation() -> None:
    """Run LiveCELL evaluation with command line tool."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", required=True, help="Provide the data directory for LIVECell Dataset"
    )
    parser.add_argument(
        "-e", "--experiment_folder", required=True,
        help="Provide the path where the inference data is stored."
    )
    parser.add_argument(
        "-f", "--force", action="store_true",
        help="Force recomputation of already cached eval results."
    )
    args = parser.parse_args()

    gt_dir = os.path.join(args.input, "annotations", "livecell_test_images")
    assert os.path.exists(gt_dir), "The LiveCELL Dataset is incomplete"

    experiment_folder = args.experiment_folder
    save_root = os.path.join(experiment_folder, "results")

    inference_root_names = ["points", "box", "amg/inference"]
    for inf_root in inference_root_names:

        pred_root = os.path.join(experiment_folder, inf_root)
        if inf_root.startswith("amg"):
            pred_folders = [pred_root]
        else:
            pred_folders = sorted(glob(os.path.join(pred_root, "*")))

        if inf_root == "amg/inference":
            save_folder = os.path.join(save_root, "amg")
        else:
            save_folder = os.path.join(save_root, inf_root)
        os.makedirs(save_folder, exist_ok=True)

        for pred_folder in tqdm(pred_folders, desc=f"Evaluate predictions for {inf_root} prompt settings"):
            exp_name = os.path.basename(pred_folder)
            save_path = os.path.join(save_folder, f"{exp_name}.csv")
            if os.path.exists(save_path) and not args.force:
                continue
            results = evaluate_livecell_predictions(gt_dir, pred_folder, verbose=False)
            results.to_csv(save_path, index=False)
