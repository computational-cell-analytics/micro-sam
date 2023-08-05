import os
from glob import glob

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


# TODO
def run_evaluation(gt_folder, prediction_folder, save_path):
    pass
