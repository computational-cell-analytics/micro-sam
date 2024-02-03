import os
import pandas as pd
from glob import glob

from micro_sam.evaluation import inference
from micro_sam.evaluation.evaluation import run_evaluation

from util import get_model, get_paths, get_pred_paths, get_default_arguments


def run_interactive_prompting(dataset_name, exp_folder, predictor, start_with_box_prompt):
    prediction_root = os.path.join(
        exp_folder, "start_with_box" if start_with_box_prompt else "start_with_point"
    )
    embedding_folder = os.path.join(exp_folder, "embeddings")
    image_paths, gt_paths = get_paths(dataset_name, split="test")
    inference.run_inference_with_iterative_prompting(
        predictor=predictor,
        image_paths=image_paths,
        gt_paths=gt_paths,
        embedding_dir=embedding_folder,
        prediction_dir=prediction_root,
        start_with_box_prompt=start_with_box_prompt
    )
    return prediction_root


def evaluate_interactive_prompting(dataset_name, prediction_root, start_with_box_prompt, exp_folder):
    assert os.path.exists(prediction_root), prediction_root

    prediction_folders = sorted(glob(os.path.join(prediction_root, "iteration*")))
    list_of_results = []
    for pred_folder in prediction_folders:
        print("Evaluating", pred_folder)
        _, gt_paths = get_paths(dataset_name, split="test")
        pred_paths = get_pred_paths(pred_folder)
        res = run_evaluation(gt_paths, pred_paths, save_path=None)
        list_of_results.append(res)
        print(res)

    df = pd.concat(list_of_results, ignore_index=True)

    # Save the results in the experiment folder.
    result_folder = os.path.join(exp_folder, "results")
    os.makedirs(result_folder, exist_ok=True)
    csv_path = os.path.join(
        result_folder,
        "iterative_prompts_start_box.csv" if start_with_box_prompt else "iterative_prompts_start_point.csv"
    )
    df.to_csv(csv_path)


def main():
    args = get_default_arguments()

    start_with_box_prompt = args.box  # overwrite to start first iters' prompt with box instead of single point

    # get the predictor to perform inference
    predictor = get_model(model_type=args.model, ckpt=args.checkpoint)

    prediction_root = run_interactive_prompting(args.dataset, args.experiment_folder, predictor, start_with_box_prompt)
    evaluate_interactive_prompting(args.dataset, prediction_root, start_with_box_prompt, args.experiment_folder)


if __name__ == "__main__":
    main()
