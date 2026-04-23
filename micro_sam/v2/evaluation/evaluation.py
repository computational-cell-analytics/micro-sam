import os
from tqdm import tqdm
from typing import Union, List

import numpy as np
import pandas as pd
import imageio.v3 as imageio

from torch_em.util.segmentation import size_filter

from elf.evaluation import mean_segmentation_accuracy


def run_evaluation_for_iterative_prompting(
    labels: List[np.ndarray],
    prediction_fnames: List[str],
    prediction_dir: Union[os.PathLike, str],
    experiment_folder: Union[os.PathLike, str],
    start_with_box_prompt: bool,
    n_iterations: int = 8,
    min_size: int = 0,
    use_masks: bool = False,
):
    """Run evaluation for predictions using iterative prompting for interactive segmentation.

    Args:
        labels:
        prediction_fnames:
        prediction_dir: The filepath where the predictions are stored.
        start_with_box_prompt: Whether to start with box prompt for iterative prompting.
        n_iterations: The number of iterations for iterative prompting.
        min_size: The minimum pixel count to consider object for evaluation.
            (NOTE: Similar filtering criterion is set for inference).
        use_masks: Whether to use masks for iterative prompting.
    """
    save_dir = os.path.join(
        experiment_folder, "results", "iterative_prompting_" + ("with" if use_masks else "without") + "_mask"
    )
    save_path = os.path.join(save_dir, "start_with_" + ("box.csv" if start_with_box_prompt else "point.csv"))
    if os.path.exists(save_path):
        print(f"The results are already stored at '{save_path}'.")
        return

    os.makedirs(save_dir, exist_ok=True)

    results = []
    for i in tqdm(range(n_iterations)):
        prediction_paths = [
            os.path.join(prediction_dir, f"iteration{i}", f"{fname}.tif") for fname in prediction_fnames
        ]
        res_per_iteration = []
        for prediction_path, _label in zip(prediction_paths, labels):
            msa, sa = mean_segmentation_accuracy(
                segmentation=imageio.imread(prediction_path),
                groundtruth=size_filter(seg=_label, min_size=min_size),  # NOTE: filter out small ground-truth labels,
                return_accuracies=True
            )
            res_per_iteration.append(pd.DataFrame.from_dict([{"mSA": msa, "SA50": sa[0], "SA75": sa[5]}]))

        res_per_iteration = pd.concat(res_per_iteration)
        results.append(pd.DataFrame(res_per_iteration.mean(axis=0)).transpose())

    results = pd.concat(results, ignore_index=True)
    results.to_csv(save_path)
    print(results)
