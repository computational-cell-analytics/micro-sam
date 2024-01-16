import os
import argparse

from micro_sam.evaluation.evaluation import run_evaluation

from util import get_pred_paths, get_paths, run_instance_segmentation_with_decoder


def run_em_instance_segmentation_with_decoder(dataset_name, model_type, checkpoint, experiment_folder):
    val_image_paths, val_gt_paths = get_paths(dataset_name, split="val")
    test_image_paths, _ = get_paths(dataset_name, split="test")
    prediction_folder = run_instance_segmentation_with_decoder(
        checkpoint,
        model_type,
        experiment_folder,
        val_image_paths,
        val_gt_paths,
        test_image_paths
    )
    return prediction_folder


def eval_instance_segmentation_with_decoder(dataset_name, prediction_folder, experiment_folder):
    print("Evaluating", prediction_folder)
    _, gt_paths = get_paths(dataset_name)
    pred_paths = get_pred_paths(prediction_folder)
    save_path = os.path.join(experiment_folder, "results", "instance_segmentation_with_decoder.csv")
    res = run_evaluation(gt_paths, pred_paths, save_path=save_path)
    print(res)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, required=True,
        help="Provide the model type to initialize the predictor"
    )
    parser.add_argument("-c", "--checkpoint", type=str, required=True,)
    parser.add_argument("-e", "--experiment_folder", type=str, required=True)
    parser.add_argument("-d", "--dataset", type=str, required=True)
    args = parser.parse_args()

    prediction_folder = run_em_instance_segmentation_with_decoder(
        args.dataset, args.model, args.checkpoint, args.experiment_folder
    )
    eval_instance_segmentation_with_decoder(args.dataset, prediction_folder, args.experiment_folder)


if __name__ == "__main__":
    main()