"""Evaluation functionality for segmentation predictions from `micro_sam.evaluation.automatic_mask_generation`
and `micro_sam.evaluation.inference`.
"""

import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import imageio.v3 as imageio
from skimage.measure import label

from elf.evaluation import mean_segmentation_accuracy

from ..util import load_image_data
from ..automatic_segmentation import _has_extension


def _run_evaluation(gt_paths, prediction_paths, verbose=True, thresholds=None):
    assert len(gt_paths) == len(prediction_paths)
    msas, sa50s, sa75s = [], [], []

    for gt_path, pred_path in tqdm(
        zip(gt_paths, prediction_paths), desc="Evaluate predictions", total=len(gt_paths), disable=not verbose
    ):

        if isinstance(gt_path, np.ndarray):
            gt = gt_path
        else:
            assert os.path.exists(gt_path), gt_path
            gt = imageio.imread(gt_path)
            gt = label(gt)

        if isinstance(pred_path, np.ndarray):
            pred = pred_path
        else:
            assert os.path.exists(pred_path), pred_path
            pred = imageio.imread(pred_path)

        msa, scores = mean_segmentation_accuracy(pred, gt, thresholds=thresholds, return_accuracies=True)
        msas.append(msa)
        if thresholds is None:
            sa50, sa75 = scores[0], scores[5]
            sa50s.append(sa50), sa75s.append(sa75)

    if thresholds is None:
        return msas, sa50s, sa75s
    else:
        return msas


def run_evaluation(
    gt_paths: List[Union[np.ndarray, os.PathLike, str]],
    prediction_paths: List[Union[np.ndarray, os.PathLike, str]],
    save_path: Optional[Union[os.PathLike, str]] = None,
    verbose: bool = True,
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Run evaluation for instance segmentation predictions.

    Args:
        gt_paths: The list of paths to ground-truth images.
        prediction_paths: The list of paths with the instance segmentations to evaluate.
        save_path: Optional path for saving the results.
        verbose: Whether to print the progress.
        thresholds: The choice of overlap thresholds.

    Returns:
        A DataFrame that contains the evaluation results.
    """
    assert len(gt_paths) == len(prediction_paths)
    # if a save_path is given and it already exists then just load it instead of running the eval
    if save_path is not None and os.path.exists(save_path):
        return pd.read_csv(save_path)

    scores = _run_evaluation(
        gt_paths=gt_paths, prediction_paths=prediction_paths, verbose=verbose, thresholds=thresholds
    )
    if thresholds is None:
        msas, sa50s, sa75s = scores
    else:
        msas = scores

    results = {"mSA": [np.mean(msas)]}
    if thresholds is None:
        results["SA50"] = [np.mean(sa50s)]
        results["SA75"] = [np.mean(sa75s)]

    results = pd.DataFrame.from_dict(results)

    if save_path is not None:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        results.to_csv(save_path, index=False)

    return results


def run_evaluation_for_iterative_prompting(
    gt_paths: List[Union[os.PathLike, str]],
    prediction_root: Union[os.PathLike, str],
    experiment_folder: Union[os.PathLike, str],
    start_with_box_prompt: bool = False,
    overwrite_results: bool = False,
    use_masks: bool = False,
) -> pd.DataFrame:
    """Run evaluation for iterative prompt-based segmentation predictions.

    Args:
        gt_paths: The list of paths to ground-truth images.
        prediction_root: The folder with the iterative prompt-based instance segmentations to evaluate.
        experiment_folder: The folder where all the experiment results are stored.
        start_with_box_prompt: Whether to evaluate on experiments with iterative prompting starting with box.
        overwrite_results: Whether to overwrite the results to update them with the new evaluation run.
        use_masks: Whether to use masks for iterative prompting.

    Returns:
        A DataFrame that contains the evaluation results.
    """
    assert os.path.exists(prediction_root), prediction_root

    # Save the results in the experiment folder
    result_folder = os.path.join(
        experiment_folder, "results", "iterative_prompting_" + ("with" if use_masks else "without") + "_mask"
    )
    os.makedirs(result_folder, exist_ok=True)

    csv_path = os.path.join(
        result_folder,
        "iterative_prompts_start_box.csv" if start_with_box_prompt else "iterative_prompts_start_point.csv"
    )

    # Overwrite the previously saved results
    if overwrite_results and os.path.exists(csv_path):
        os.remove(csv_path)

    # If the results have been computed already, it's not needed to re-run it again.
    if os.path.exists(csv_path):
        print(f"Results with iterative prompting for interactive segmentation are already stored at '{csv_path}'.")
        return

    list_of_results = []
    prediction_folders = sorted(glob(os.path.join(prediction_root, "iteration*")))
    for pred_folder in prediction_folders:
        print("Evaluating", os.path.split(pred_folder)[-1])
        pred_paths = sorted(glob(os.path.join(pred_folder, "*")))
        result = run_evaluation(gt_paths=gt_paths, prediction_paths=pred_paths, save_path=None)
        list_of_results.append(result)

    res_df = pd.concat(list_of_results, ignore_index=True)
    res_df.to_csv(csv_path)


def main():
    """@private"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluating segmentations from Segment Anything model on custom data.")

    # labels and predictions for quantitative evaluation.
    parser.add_argument(
        "--labels", required=True, type=str, nargs="+",
        help="Filepath(s) to ground-truth labels or the directory where the label data is stored."
    )
    parser.add_argument(
        "--predictions", required=True, type=str, nargs="+",
        help="Filepath to predicted labels or the directory where the predicted label data is stored."
    )
    parser.add_argument(
        "--label_key", type=str, default=None,
        help="The key for accessing predicted label data, either a pattern / wildcard or with 'elf.io.open_file'. "
    )
    parser.add_argument(
        "--prediction_key", type=str, default=None,
        help="The key for accessing ground-truth label data, either a pattern / wildcard or with 'elf.io.open_file'. "
    )
    parser.add_argument(
        "-o", "--output_path", type=str, default=None,
        help="The filepath to store the evaluation results. The current support stores results in a 'csv' file."
    )
    parser.add_argument(
        "--threshold", default=None, type=float, nargs="+",
        help="The choice of overlap threshold(s) for calculating the segmentation accuracy. By default, "
        "np.arange(0.5, 1., 0.05) is used to provide the mean segmentation accurcy score over all values.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Whether to allow verbosity of evaluation."
    )

    # TODO: We can extend this in future for other metrics, eg. dice score, etc.
    # NOTE: This argument is not exposed to the user atm.
    # parser.add_argument(
    #     "--metric", type=str, default="segmentation_accuracy", choices=("segmentation_accuracy"),
    #     help="The choice of metric for evaluation. By default, it computes segmentation accuracy "
    #     "for instance segmentation."
    # )

    args = parser.parse_args()

    # Check whether the inputs are as expected.
    def _get_inputs_from_paths(paths, key):
        fpaths = []
        for path in paths:
            if _has_extension(path):  # it is just one filepath and we check whether we can access it via 'elf'.
                fpaths.append(path if key is None else load_image_data(path=path, key=key))
            else:  # otherwise, path is a directory, fetch all inputs provided with a pattern.
                assert key is not None, \
                    f"You must provide a wildcard / pattern as the filepath '{os.path.abspath(path)}' is a directory."
                fpaths.extend(natsorted(glob(os.path.join(path, key))))

        return fpaths

    labels = _get_inputs_from_paths(args.labels, args.label_key)
    predictions = _get_inputs_from_paths(args.predictions, args.prediction_key)
    assert labels and len(labels) == len(predictions)

    # Check whether output path is a csv or not, if passed.
    output_path = args.output_path
    if output_path is not None:
        if not _has_extension(output_path):  # If it is a directory, store this in "<OUTPUT_PATH>/results.csv"
            os.makedirs(output_path, exist_ok=True)
            output_path = os.path.join(output_path, "results.csv")

        if not output_path.endswith(".csv"):  # If it is a filepath missing extension / with a different extension.
            output_path = str(Path(output_path).with_suffix(".csv"))  # Limit supports to csv files for now.

    # Run evaluation on labels and predictions.
    results = run_evaluation(
        gt_paths=labels,
        prediction_paths=predictions,
        save_path=output_path,
        verbose=args.verbose,
        thresholds=args.threshold,
    )

    print("The evaluation results for the predictions are:")
    print(results)

    if args.verbose and output_path is not None:
        print(f"The evaluation results have been stored at '{os.path.abspath(output_path)}'.")
