import argparse
import os

from micro_sam.evaluation.evaluation import run_evaluation
from micro_sam.evaluation.livecell import run_livecell_instance_segmentation_with_decoder
from util import DATA_ROOT, get_checkpoint, get_experiment_folder, get_pred_and_gt_paths


def run_instance_segmentation_with_decoder(name, model_type, checkpoint, experiment_folder):
    if checkpoint is None:
        checkpoint, model_type = get_checkpoint(name)

    if experiment_folder is None:
        experiment_folder = get_experiment_folder(name)

    input_folder = DATA_ROOT
    prediction_folder = run_livecell_instance_segmentation_with_decoder(
        checkpoint,
        input_folder,
        model_type,
        experiment_folder,
        n_val_per_cell_type=25,
    )
    return prediction_folder


def eval_instance_segmentation_with_decoder(name, prediction_folder, experiment_folder):
    print("Evaluating", prediction_folder)
    pred_paths, gt_paths = get_pred_and_gt_paths(prediction_folder)

    if experiment_folder is None:
        experiment_folder = get_experiment_folder(name)

    save_path = os.path.join(experiment_folder, "results", "instance_segmentation_with_decoder.csv")
    res = run_evaluation(gt_paths, pred_paths, save_path=save_path)
    print(res)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", default=None)
    parser.add_argument(
        "-m", "--model", type=str,
        help="Provide the model type to initialize the predictor"
    )
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("-e", "--experiment_folder", type=str, default=None)
    args = parser.parse_args()

    name = args.name

    prediction_folder = run_instance_segmentation_with_decoder(
        name, args.model, args.checkpoint, args.experiment_folder
    )
    eval_instance_segmentation_with_decoder(name, prediction_folder, args.experiment_folder)


if __name__ == "__main__":
    main()
