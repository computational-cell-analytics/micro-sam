import argparse
import os
from glob import glob

import pandas as pd

from micro_sam.evaluation import inference
from micro_sam.evaluation.evaluation import run_evaluation
from util import get_paths, get_experiment_folder, get_model, DATA_ROOT


def run_interactive_prompting(exp_folder, predictor, start_with_box_prompt):
    prediction_root = os.path.join(
        exp_folder, "start_with_box" if start_with_box_prompt else "start_with_point"
    )
    embedding_folder = os.path.join(exp_folder, "embeddings")
    image_paths, gt_paths = get_paths()
    inference.run_inference_with_iterative_prompting(
        predictor=predictor,
        image_paths=image_paths,
        gt_paths=gt_paths,
        embedding_dir=embedding_folder,
        prediction_dir=prediction_root,
        start_with_box_prompt=start_with_box_prompt
    )
    return prediction_root


def get_pg_paths(pred_folder):
    pred_paths = sorted(glob(os.path.join(pred_folder, "*.tif")))
    names = [os.path.split(path)[1] for path in pred_paths]
    gt_root = os.path.join(DATA_ROOT, "annotations_corrected/livecell_test_images")
    gt_paths = [
        os.path.join(gt_root, name.split("_")[0], name) for name in names
    ]
    assert all(os.path.exists(pp) for pp in gt_paths)
    return pred_paths, gt_paths


def evaluate_interactive_prompting(prediction_root, start_with_box_prompt, name):
    assert os.path.exists(prediction_root), prediction_root

    csv_save_dir = f"./iterative_prompting_results/{name}"
    os.makedirs(csv_save_dir, exist_ok=True)
    csv_path = os.path.join(csv_save_dir, "start_with_box.csv" if start_with_box_prompt else "start_with_point.csv")
    if os.path.exists(csv_path):
        print("The evaluated results for the expected setting already exist here:", csv_path)
        return

    prediction_folders = sorted(glob(os.path.join(prediction_root, "iteration*")))
    list_of_results = []
    for pred_folder in prediction_folders:
        print("Evaluating", pred_folder)
        pred_paths, gt_paths = get_pg_paths(pred_folder)
        res = run_evaluation(gt_paths, pred_paths, save_path=None)
        list_of_results.append(res)
        print(res)

    df = pd.concat(list_of_results, ignore_index=True)
    df.to_csv(csv_path)

    # Also save the results in the experiment folder.
    exp_folder = get_experiment_folder(name)
    csv_path = os.path.join(
        exp_folder, "iterative_prompts_start_box.csv" if start_with_box_prompt else "iterative_prompts_start_point.csv"
    )
    df.to_csv(csv_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--name", required=True)
    parser.add_argument(
        "-m", "--model", type=str,  # options: "vit_h", "vit_h_generalist", "vit_h_specialist"
        help="Provide the model type to initialize the predictor"
    )
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("--box", action="store_true", help="If passed, starts with first prompt as box")
    args = parser.parse_args()

    name = args.name
    start_with_box_prompt = args.box  # overwrite to start first iters' prompt with box instead of single point

    # get the predictor to perform inference
    predictor = get_model(name, model_type=args.model, ckpt=args.checkpoint)

    exp_folder = get_experiment_folder(name)
    prediction_root = run_interactive_prompting(exp_folder, predictor, start_with_box_prompt)
    evaluate_interactive_prompting(prediction_root, start_with_box_prompt, name)


if __name__ == "__main__":
    main()
