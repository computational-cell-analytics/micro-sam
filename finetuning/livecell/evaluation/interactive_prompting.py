from micro_sam.evaluation.inference import run_inference_with_iterative_prompting
from util import get_checkpoint, get_paths


def main():
    checkpoint, model_type = get_checkpoint("vit_b")
    image_paths, gt_paths = get_paths()

    prediction_root = "./pred_interactive_prompting"

    run_inference_with_iterative_prompting(
        checkpoint, model_type, image_paths, gt_paths,
        prediction_root, use_boxes=False,
    )


if __name__ == "__main__":
    main()
