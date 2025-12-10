import os
from glob import glob
from natsort import natsorted

from util import get_image_label_paths

from micro_sam.evaluation.inference import run_apg
from micro_sam.evaluation.evaluation import run_evaluation


def run_apg_evaluation(dataset_name, model_type, experiment_folder):
    val_image_paths, val_label_paths = get_image_label_paths(dataset_name=dataset_name, split="val")
    test_image_paths, test_label_paths = get_image_label_paths(dataset_name=dataset_name, split="test")

    # HACK: For simpilicity of grid-search, we run it on 25-images only in the beginning.
    if dataset_name != "livecell":
        val_image_paths, val_label_paths = val_image_paths[:25], val_label_paths[:25]

    if dataset_name == "plantseg_root":  # It takes super duper long for the grid-search.
        val_image_paths, val_label_paths = val_image_paths[:5], val_label_paths[:5]

    # Run predictions with grid-search.
    experiment_folder = os.path.join(experiment_folder, dataset_name)
    prediction_folder = run_apg(
        checkpoint=None,
        model_type=model_type,
        experiment_folder=experiment_folder,
        val_image_paths=val_image_paths,
        val_gt_paths=val_label_paths,
        test_image_paths=test_image_paths,
    )

    # Get the prediction paths.
    prediction_paths = natsorted(glob(os.path.join(prediction_folder, "*.tif")))
    if not prediction_paths:  # Maybe the file extension is something else? eg. `.png` for CellPose images.
        prediction_paths = natsorted(glob(os.path.join(prediction_folder, "*")))

    res = run_evaluation(test_label_paths, prediction_paths, os.path.join(experiment_folder, "results", "apg.csv"))
    print(dataset_name, model_type, res)


def main(args):
    run_apg_evaluation(args.dataset_name, args.model_type, args.experiment_folder)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_name", type=str, required=True)
    parser.add_argument("-m", "--model_type", type=str, default="vit_b_lm")
    parser.add_argument(
        "-e", "--experiment_folder", type=str,
        default="/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam/apg"
    )
    args = parser.parse_args()
    main(args)
