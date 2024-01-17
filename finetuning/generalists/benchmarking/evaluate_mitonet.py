import os
from glob import glob

from micro_sam.evaluation.evaluation import run_evaluation

from process_mitonet_labels import ROOT


def get_mitonet_predictions(volume) -> str:
    raise NotImplementedError("The volumes have been currently generated using `empanada-napari`")


def evaluate_mitonet_predictions():
    prediction_paths = sorted(glob(os.path.join(ROOT, "slices", "*.tif")))
    gt_paths = sorted(glob("/scratch/projects/nim00007/sam/data/lucchi/slices/labels/lucchi_test_*"))
    save_path = os.path.join(ROOT, "results", "mitonet.csv")
    res = run_evaluation(gt_paths, prediction_paths, save_path)
    print(res)


def main():
    # let's generate the predictions
    # prediction_folder = get_mitonet_predictions(volume)

    # evaluation of the instance segmentations from mitonet
    evaluate_mitonet_predictions()


if __name__ == "__main__":
    main()
