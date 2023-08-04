import argparse
import os
from glob import glob

from .util import get_predictor, run_inference_with_prompts

CELL_TYPES = ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3"]


def run_livecell_inference(args):
    assert os.path.exists(args.input), "Please download the LIVECell Dataset"
    img_dir = os.path.join(args.input, "images", "livecell_test_images")
    gt_dir = os.path.join(args.input, "annotations", "livecell_test_images")

    predictor = get_predictor(args.ckpt, args.model)

    if args.box:
        print("Getting box-prompted predictions for LiveCELL")
        pred_dir = os.path.join(args.pred_path, args.name, "box")
    elif args.points:
        assert args.positive and args.negative is not None
        print("Getting point-prompted predictions for LiveCELL")
        pred_dir = os.path.join(args.pred_path, args.name, "points", f"p{args.positive}-n{args.negative}")
    else:
        raise ValueError("Choose (only) either points / box")

    print("The predictions will be saved to", pred_dir)
    prompt_save_dir = os.path.join(pred_dir, "prompts")
    os.makedirs(prompt_save_dir, exist_ok=True)

    image_paths, gt_paths = [], []
    for ctype in CELL_TYPES:
        for img_path in glob(os.path.join(img_dir, f"{ctype}*")):
            image_paths.append(img_path)
            img_name = os.path.basename(img_path)
            gt_path = os.path.join(gt_dir, ctype, img_name)
            assert os.path.exists(gt_path), gt_path
            gt_paths.append(gt_path)

    root_embedding_dir = os.path.join(args.embedding_path, "embeddings", args.name)
    run_inference_with_prompts(
        predictor,
        image_paths,
        gt_paths,
        embedding_dir=root_embedding_dir,
        prediction_dir=pred_dir,
        use_points=args.points,
        use_boxes=args.boxes,
        n_positive=args.positive,
        n_negative=args.negative,
        prompt_save_dir=prompt_save_dir,
    )


def livecell_inference_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--box", action="store_true", help="Activate box-prompted based inference")
    parser.add_argument("--points", action="store_true", help="Activate point-prompt based inference")
    parser.add_argument("--positive", type=int, default=1, help="No. of positive prompts")
    parser.add_argument("--negative", type=int, default=0, help="No. of negative prompts")
    parser.add_argument("-i", "--input", type=str, default="./livecell/",
                        help="Provide the data directory for LIVECell Dataset")
    parser.add_argument("-e", "--embedding_path", type=str, default="./sam_embeddings/",
                        help="Provide the path where embeddings will be saved")
    parser.add_argument("-p", "--pred_path", type=str, default="./predictions/",
                        help="Provide the path where the predictions will be saved")
    parser.add_argument("-c", "--ckpt", type=str, required=True,
                        help="Provide model checkpoints (vanilla / finetuned)")
    parser.add_argument("-m", "--model", type=str, default="vit_b",
                        help="Pass the checkpoint-specific model name being used for inference")
    parser.add_argument("-n", "--name", type=str, default="finetuned_livecell",
                        help="Provide a name for the saving nomenclature")
    return parser
