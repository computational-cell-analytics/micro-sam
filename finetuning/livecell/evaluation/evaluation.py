import argparse
import os
from glob import glob

from tqdm import tqdm

from micro_sam.evaluation.livecell import evaluate_livecell_predictions
from util import get_experiment_folder, DATA_ROOT


def run_eval(gt_dir, experiment_folder, prompt_prefix):
    result_dir = os.path.join(experiment_folder, "results", prompt_prefix)
    os.makedirs(result_dir, exist_ok=True)

    pred_dirs = sorted(glob(os.path.join(experiment_folder, prompt_prefix, "*")))

    for pred_dir in tqdm(pred_dirs, desc=f"Run livecell evaluation for all {prompt_prefix}-prompt settings"):
        setting_name = os.path.basename(pred_dir)
        save_path = os.path.join(result_dir, f"{setting_name}.csv")
        if os.path.exists(save_path):
            continue
        results = evaluate_livecell_predictions(gt_dir, pred_dir, verbose=False)
        results.to_csv(save_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True)
    args = parser.parse_args()

    name = args.name

    gt_dir = os.path.join(DATA_ROOT, "annotations", "livecell_test_images")
    assert os.path.exists(gt_dir), "The LiveCELL Dataset is incomplete"

    experiment_folder = get_experiment_folder(name)

    run_eval(gt_dir, experiment_folder, "box")
    run_eval(gt_dir, experiment_folder, "points")


if __name__ == "__main__":
    main()
