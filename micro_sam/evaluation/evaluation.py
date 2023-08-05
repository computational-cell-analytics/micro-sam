import os
from glob import glob
from pathlib import Path
from typing import Optional, Union

import imageio.v3 as imageio
import numpy as np
import pandas as pd

from elf.evaluation import mean_segmentation_accuracy
from tqdm import tqdm


def _run_evaluation(gt_paths, prediction_paths, verbose=True):
    assert len(gt_paths) == len(prediction_paths)
    msas, sa50s, sa75s = [], [], []

    for gt_path, pred_path in tqdm(
        zip(gt_paths, prediction_paths), desc="Evaluate predictions", total=len(gt_paths), disable=not verbose
    ):
        assert os.path.exists(gt_path), gt_path
        assert os.path.exists(pred_path), pred_path

        gt = imageio.imread(gt_path)
        pred = imageio.imread(pred_path)

        msa, scores = mean_segmentation_accuracy(pred, gt, return_accuracies=True)
        sa50, sa75 = scores[0], scores[5]
        msas.append(msa), sa50s.append(sa50), sa75s.append(sa75)

    return msas, sa50s, sa75s


def run_evaluation(
    gt_folder: Union[os.PathLike, str],
    prediction_folder: Union[os.PathLike, str],
    save_path: Optional[Union[os.PathLike, str]] = None,
    pattern: str = "*.tif",
    verbose: bool = True,
) -> pd.DataFrame:
    """Run evaluation for instance segmentation predictions.

    Args:
        gt_folder: The folder with ground-truth images.
        prediction_folder: The folder with the instance segmentations to evaluate.
        save_path: Optional path for saving the results.
        pattern: Optional pattern for selecting the images to evaluate via glob.
            By default all images with ending .tif will be evaluated.
        verbose: Whether to print the progress.

    Returns:
        A DataFrame that contains the evaluation results.
    """
    # if a save_path is given and it already exists then just load it instead of running the eval
    if save_path is not None and os.path.exists(save_path):
        return pd.from_csv(save_path)

    gt_paths = glob(os.path.join(gt_folder, pattern))
    prediction_paths = [
        os.path.join(prediction_folder, os.path.basename(path)) for path in gt_paths
    ]
    assert all(os.path.exists(path) for path in prediction_paths)
    msas, sa50s, sa75s = _run_evaluation(gt_paths, prediction_paths, verbose=verbose)

    results = pd.DataFrame.from_dict({
        "msa": [np.mean(msas)],
        "sa50": [np.mean(sa50s)],
        "sa75": [np.mean(sa75s)],
    })

    if save_path is not None:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        results.to_csv(save_path, index=False)

    return results


# TODO function to evaluate full experiment and resave in one table
