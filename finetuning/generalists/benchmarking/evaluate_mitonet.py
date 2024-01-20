import os
from glob import glob

from micro_sam.evaluation.evaluation import run_evaluation

from process_mitonet_labels import ROOT


def get_mitonet_predictions(volume) -> str:
    raise NotImplementedError("The volumes have been currently generated using `empanada-napari`")


def evaluate_mitonet_predictions(dataset_name, gt_paths):
    prediction_paths = sorted(glob(os.path.join(ROOT, dataset_name, "slices", "*.tif")))
    save_path = os.path.join(ROOT, dataset_name, "results", "mitonet.csv")
    res = run_evaluation(gt_paths, prediction_paths, save_path)
    print(res)


def main():
    # let's generate the predictions
    # prediction_folder = get_mitonet_predictions(volume)

    dataset_name = "mitoem"

    if dataset_name == "lucchi":
        gt_paths = sorted(glob("/scratch/projects/nim00007/sam/data/lucchi/slices/labels/lucchi_test_*"))
    elif dataset_name == "mitoem":
        gt_paths = sorted(glob("/scratch/projects/nim00007/sam/data/mitoem/slices/*/test/labels/mitoem_*"))
    else:
        raise ValueError

    # evaluation of the instance segmentations from mitonet
    evaluate_mitonet_predictions(dataset_name, gt_paths)


if __name__ == "__main__":
    main()
