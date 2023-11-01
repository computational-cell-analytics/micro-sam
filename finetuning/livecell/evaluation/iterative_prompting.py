import os
from glob import glob

from micro_sam.evaluation.evaluation import run_evaluation
from micro_sam.evaluation.inference import run_inference_with_iterative_prompting

from util import get_checkpoint, get_paths

LIVECELL_GT_ROOT = "/scratch-grete/projects/nim00007/data/LiveCELL/annotations_corrected/livecell_test_images"
PREDICTION_ROOT = "/scratch-grete/projects/nim00007/sam/iterative_evaluation"


def get_prediction_root(use_boxes, model_type, root_dir=PREDICTION_ROOT):
    if use_boxes:
        prediction_root = os.path.join(root_dir, model_type, "start_with_box")
    else:
        prediction_root = os.path.join(root_dir, model_type, "start_with_point")

    return prediction_root


def run_interactive_prompting(use_boxes, model_name, get_default_checkpoint_paths, prediction_root):
    checkpoint, model_type = get_checkpoint(model_name)
    image_paths, gt_paths = get_paths()

    run_inference_with_iterative_prompting(
        image_paths=image_paths,
        gt_paths=gt_paths,
        prediction_root=prediction_root,
        use_boxes=use_boxes,
        batch_size=16,
        checkpoint_path=checkpoint if get_default_checkpoint_paths else None,
        model_type=model_type
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
    use_boxes = True  # overwrite when you want the first iterations' input prompt to start as a box
    model_name = "vit_b"  # overwrite to specify the choice of vanilla / finetuned models
    get_default_checkpoint_paths = False  # use the default checkpoint paths or fetch from `~/.sam_models`

    # add the root prediction path where you would like to save the iterative prompting results
    prediction_root = get_prediction_root(use_boxes, model_name)

    run_interactive_prompting(use_boxes, model_name, get_default_checkpoint_paths, prediction_root)
    # evaluate_interactive_prompting(prediction_root)


if __name__ == "__main__":
    main()
