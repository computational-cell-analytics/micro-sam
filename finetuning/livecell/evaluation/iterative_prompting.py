import os
from glob import glob

from micro_sam.evaluation import inference
from micro_sam.evaluation.evaluation import run_evaluation

from util import get_checkpoint, get_paths

LIVECELL_GT_ROOT = "/scratch/projects/nim00007/data/LiveCELL/annotations_corrected/livecell_test_images"
PREDICTION_ROOT = "/scratch/projects/nim00007/sam/iterative_evaluation"


def get_prediction_root(start_with_box_prompt, model_type, root_dir=PREDICTION_ROOT):
    if start_with_box_prompt:
        prediction_root = os.path.join(root_dir, model_type, "start_with_box")
    else:
        prediction_root = os.path.join(root_dir, model_type, "start_with_point")

    return prediction_root


def run_interactive_prompting(start_with_box_prompt, model_name, prediction_root, checkpoint=None):
    if checkpoint is None:
        checkpoint, model_type = get_checkpoint(model_name)
    else:
        model_type = model_name

    image_paths, gt_paths = get_paths()

    predictor = inference.get_predictor(checkpoint, model_type)

    # we organize all the folders with data from this experiment below
    embedding_folder = os.path.join(PREDICTION_ROOT, model_type, "embeddings")
    os.makedirs(embedding_folder, exist_ok=True)

    inference.run_inference_with_iterative_prompting(
        predictor=predictor,
        image_paths=image_paths[:10],
        gt_paths=gt_paths[:10],
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


def evaluate_interactive_prompting(prediction_root):
    assert os.path.exists(prediction_root), prediction_root

    prediction_folders = sorted(glob(os.path.join(prediction_root, "iteration*")))
    for pred_folder in prediction_folders:
        print("Evaluating", pred_folder)
        pred_paths, gt_paths = get_pg_paths(pred_folder)
        res = run_evaluation(gt_paths, pred_paths, save_path=None)
        print(res)


def main():
    start_with_box_prompt = True  # overwrite when you want first iters' prompt to start as box instead of single point
    model_name = "vit_h_generalist"  # overwrite to specify the choice of vanilla / finetuned models
    checkpoint = None  # overwrite to pass your expected model's checkpoint path

    # add the root prediction path where you would like to save the iterative prompting results
    prediction_root = get_prediction_root(start_with_box_prompt, model_name)

    run_interactive_prompting(start_with_box_prompt, model_name, prediction_root, checkpoint=checkpoint)
    evaluate_interactive_prompting(prediction_root)


if __name__ == "__main__":
    main()
