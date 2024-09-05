import os

from micro_sam.evaluation.evaluation import run_evaluation
from micro_sam.evaluation.inference import run_instance_segmentation_with_decoder

from util import (
    get_paths,  # comment this line out and create a custom function with the same name to run ais on your data
    get_pred_paths, get_default_arguments
)


def run_instance_segmentation_with_decoder_inference(
    dataset_name, model_type, checkpoint, experiment_folder, peft_kwargs,
):
    val_image_paths, val_gt_paths = get_paths(dataset_name, split="val")
    test_image_paths, _ = get_paths(dataset_name, split="test")
    prediction_folder = run_instance_segmentation_with_decoder(
        checkpoint=checkpoint,
        model_type=model_type,
        experiment_folder=experiment_folder,
        val_image_paths=val_image_paths,
        val_gt_paths=val_gt_paths,
        test_image_paths=test_image_paths,
        peft_kwargs=peft_kwargs,
    )
    return prediction_folder


def eval_instance_segmentation_with_decoder(dataset_name, prediction_folder, experiment_folder):
    print("Evaluating", prediction_folder)
    _, gt_paths = get_paths(dataset_name, split="test")
    pred_paths = get_pred_paths(prediction_folder)
    save_path = os.path.join(experiment_folder, "results", "instance_segmentation_with_decoder.csv")
    res = run_evaluation(gt_paths, pred_paths, save_path=save_path)
    print(res)


def main():
    args = get_default_arguments()
    peft_kwargs = {"rank": args.peft_rank, "module": args.peft_module}
    prediction_folder = run_instance_segmentation_with_decoder_inference(
        args.dataset, args.model, args.checkpoint, args.experiment_folder, peft_kwargs,
    )
    eval_instance_segmentation_with_decoder(args.dataset, prediction_folder, args.experiment_folder)


if __name__ == "__main__":
    main()
