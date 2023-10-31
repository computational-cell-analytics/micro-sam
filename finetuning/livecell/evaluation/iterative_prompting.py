import os
from glob import glob

from micro_sam.evaluation.evaluation import run_evaluation
from micro_sam.evaluation.inference import run_inference_with_iterative_prompting

from util import get_checkpoint, get_paths

LIVECELL_GT_ROOT = "/scratch-grete/projects/nim00007/data/LiveCELL/annotations_corrected/livecell_test_images"
# TODO update to make fit other models
PREDICTION_ROOT = "./pred_interactive_prompting"


def run_interactive_prompting(use_get_checkpoint=False):
    prediction_root = PREDICTION_ROOT

    checkpoint, model_type = get_checkpoint("vit_b")
    image_paths, gt_paths = get_paths()

    run_inference_with_iterative_prompting(
        image_paths=image_paths,
        gt_paths=gt_paths,
        prediction_root=prediction_root,
        use_boxes=False,
        batch_size=16,
        checkpoint_path=checkpoint if use_get_checkpoint else None,
        model_type=model_type if use_get_checkpoint else "vit_b"
    )


def get_pg_paths(pred_folder):
    pred_paths = sorted(glob(os.path.join(pred_folder, "*.tif")))
    names = [os.path.split(path)[1] for path in pred_paths]
    gt_paths = [
        os.path.join(LIVECELL_GT_ROOT, name.split("_")[0], name) for name in names
    ]
    assert all(os.path.exists(pp) for pp in gt_paths)
    return pred_paths, gt_paths


def evaluate_interactive_prompting():
    prediction_root = PREDICTION_ROOT
    prediction_folders = sorted(glob(os.path.join(prediction_root, "iteration*")))
    for pred_folder in prediction_folders:
        print("Evaluating", pred_folder)
        pred_paths, gt_paths = get_pg_paths(pred_folder)
        res = run_evaluation(gt_paths, pred_paths, save_path=None)
        print(res)


def main():
    run_interactive_prompting()
    evaluate_interactive_prompting()


if __name__ == "__main__":
    main()
