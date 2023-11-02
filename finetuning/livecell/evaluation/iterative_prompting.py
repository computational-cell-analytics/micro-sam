import os
import pandas as pd
from glob import glob

from micro_sam.evaluation import inference
from micro_sam.evaluation.evaluation import run_evaluation

from util import get_paths, get_checkpoint

LIVECELL_GT_ROOT = "/scratch/projects/nim00007/data/LiveCELL/annotations_corrected/livecell_test_images"
PREDICTION_ROOT = "/scratch/projects/nim00007/sam/iterative_evaluation"


def get_prediction_root(start_with_box_prompt, model_description, root_dir=PREDICTION_ROOT):
    if start_with_box_prompt:
        prediction_root = os.path.join(root_dir, model_description, "start_with_box")
    else:
        prediction_root = os.path.join(root_dir, model_description, "start_with_point")

    return prediction_root


def run_interactive_prompting(predictor, start_with_box_prompt, model_description, prediction_root):
    # we organize all the folders with data from this experiment below
    embedding_folder = os.path.join(PREDICTION_ROOT, model_description, "embeddings")
    os.makedirs(embedding_folder, exist_ok=True)

    image_paths, gt_paths = get_paths()

    inference.run_inference_with_iterative_prompting(
        predictor=predictor,
        image_paths=image_paths,
        gt_paths=gt_paths,
        embedding_dir=embedding_folder,
        prediction_dir=prediction_root,
        start_with_box_prompt=start_with_box_prompt
    )


def get_pg_paths(pred_folder):
    pred_paths = sorted(glob(os.path.join(pred_folder, "*.tif")))
    names = [os.path.split(path)[1] for path in pred_paths]
    gt_paths = [
        os.path.join(LIVECELL_GT_ROOT, name.split("_")[0], name) for name in names
    ]
    assert all(os.path.exists(pp) for pp in gt_paths)
    return pred_paths, gt_paths


def evaluate_interactive_prompting(prediction_root, start_with_box_prompt, model_description):
    assert os.path.exists(prediction_root), prediction_root

    csv_save_dir = f"./iterative_prompting_results/{model_description}"
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


def main():
    start_with_box_prompt = False  # overwrite when you want first iters' prompt to start as box instead of single point
    model_description = "vit_h_generalist"  # overwrite to specify the choice of vanilla / finetuned models

    # add the root prediction path where you would like to save the iterative prompting results
    prediction_root = get_prediction_root(start_with_box_prompt, model_description)

    # get the model checkpoints and desired model name to initialize the predictor
    checkpoint, model_type = get_checkpoint(model_description)
    # get the predictor to perform inference
    predictor = inference.get_predictor(checkpoint, model_type)

    run_interactive_prompting(predictor, start_with_box_prompt, model_description, prediction_root)
    evaluate_interactive_prompting(prediction_root, start_with_box_prompt, model_description)


if __name__ == "__main__":
    main()
