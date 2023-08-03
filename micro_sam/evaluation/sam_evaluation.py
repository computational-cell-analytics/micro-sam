import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import imageio.v2 as imageio
from elf.evaluation import mean_segmentation_accuracy

CELL_TYPES = ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"]


# FIXME this is called multiple times for the 'current_metric', but we would only need to call it once
def analyse_predictions(gt_dir, pred_dir, current_metric):
    """ Calculate mSA, SA50, SA75
    """
    assert os.path.exists(pred_dir), pred_dir

    metric_list = []
    result_dict = {"cell_type": current_metric}
    for ct in tqdm(CELL_TYPES, desc=current_metric.upper()):
        metric_list_ct = []
        gt_pattern = os.path.join(gt_dir, f"{ct}/*.tif")
        gt_paths = glob(gt_pattern)
        assert len(gt_paths) > 0, f"{gt_pattern}"
        for gt_path in gt_paths:
            cell_id = os.path.basename(gt_path)
            pred_path = os.path.join(pred_dir, cell_id)
            gt = imageio.imread(gt_path)
            pred = imageio.imread(pred_path)

            m_sas, sas = mean_segmentation_accuracy(pred, gt, return_accuracies=True)  # type: ignore

            if current_metric == "msa":
                metric_score = m_sas
            elif current_metric == "sa50":
                metric_score = sas[0]
            elif current_metric == "sa75":
                metric_score = sas[5]
            else:
                raise ValueError

            metric_list.append(metric_score)
            metric_list_ct.append(metric_score)
        result_dict[ct] = np.mean(metric_list_ct)
    result_dict["Mean"] = np.mean(metric_list)
    return pd.DataFrame([result_dict])


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

        results = [analyse_predictions(gt_dir, pred_dir, current_metric) for current_metric in ["msa", "sa50", "sa75"]]
        results = pd.concat(results)
        results.to_csv(os.path.join(csv_save_dir, f"{save_result_filename}.csv"), index=False)


def livecell_evaluation_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="./livecell/",
                        help="Provide the data directory for LIVECell Dataset")
    parser.add_argument("-p", "--pred_path", type=str, default="./predictions")
    parser.add_argument("-s", "--save_path", type=str, default="./results/")
    parser.add_argument("--name", type=str, required=True)
    return parser
