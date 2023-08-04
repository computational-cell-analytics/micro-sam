import argparse
import os
from glob import glob

import imageio.v3 as imageio
import numpy as np
import pandas as pd

from elf.evaluation import mean_segmentation_accuracy
from tqdm import tqdm

from .livecell import CELL_TYPES


def analyse_livecell_predictions(gt_dir, pred_dir):
    """ Calculate mSA, SA50, SA75
    """
    assert os.path.exists(pred_dir), pred_dir

    msas, sa50s, sa75s = [], [], []
    msas_ct, sa50s_ct, sa75s_ct = [], [], []

    for ct in tqdm(CELL_TYPES, desc="Evaluate livecell predictions"):

        gt_pattern = os.path.join(gt_dir, f"{ct}/*.tif")
        gt_paths = glob(gt_pattern)
        assert len(gt_paths) > 0, f"{gt_pattern}"

        this_msas, this_sa50s, this_sa75s = [], [], []

        for gt_path in gt_paths:
            cell_id = os.path.basename(gt_path)
            pred_path = os.path.join(pred_dir, cell_id)
            gt = imageio.imread(gt_path)
            pred = imageio.imread(pred_path)

            msa, scores = mean_segmentation_accuracy(pred, gt, return_accuracies=True)  # type: ignore
            sa50, sa75 = scores[0], scores[5]

            this_msas.append(msa), this_sa50s.append(sa50), this_sa75s.append(sa75)

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


def run_livecell_evaluation(args):
    gt_dir = os.path.join(args.input, "annotations", "livecell_test_images")
    list_of_combinations = ["points", "box"]
    for pred_mode in list_of_combinations:
        pred_dir = os.path.join(args.pred_path, args.name, pred_mode)
        save_result_filename = f"{pred_mode}"
        if pred_mode == "points":
            pred_dir = os.path.join(pred_dir, "p1-n0")
            save_result_filename = f"{save_result_filename}-p1-n0"

        csv_save_dir = os.path.join(args.save_path, args.name)
        os.makedirs(csv_save_dir, exist_ok=True)

        results = analyse_livecell_predictions(gt_dir, pred_dir)
        results.to_csv(os.path.join(csv_save_dir, f"{save_result_filename}.csv"), index=False)


def livecell_evaluation_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="./livecell/",
                        help="Provide the data directory for LIVECell Dataset")
    parser.add_argument("-p", "--pred_path", type=str, default="./predictions")
    parser.add_argument("-s", "--save_path", type=str, default="./results/")
    parser.add_argument("--name", type=str, required=True)
    return parser
