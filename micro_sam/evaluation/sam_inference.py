import os
import argparse

from .util import get_predictor_for_amg, get_prompted_segmentations_sam


def main(args):
    assert os.path.exists(args.input), "Please download the LIVECell Dataset"
    img_dir = os.path.join(args.input, "images", "livecell_test_images")
    gt_dir = os.path.join(args.input, "annotations", "livecell_test_images")

    predictor = get_predictor_for_amg(args.ckpt, args.model)

    if args.box:
        print("Getting box-prompted predictions for LiveCELL")
        pred_dir = os.path.join(args.pred_path, args.name, "box")
    elif args.points:
        assert args.positive and args.negative is not None
        print("Getting point-prompted predictions for LiveCELL")
        pred_dir = os.path.join(args.pred_path, args.name, "points", f"p{args.positive}-n{args.negative}")
    else:
        raise ValueError("CHoose (only) either points / box")
    os.makedirs(pred_dir, exist_ok=True)

    root_embedding_dir = os.path.join(args.embedding_path, "embeddings", args.name)
    get_prompted_segmentations_sam(predictor, img_dir, gt_dir, root_embedding_dir, pred_dir, n_positive=args.positive,
                                   n_negative=args.negative, dilation=5, get_points=args.points, get_boxes=args.box)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--box", action='store_true', help="Activate box-prompted based inference")
    parser.add_argument("--points", action='store_true', help="Activate point-prompt based inference")
    parser.add_argument("--positive", type=int, default=1, help="No. of positive prompts")
    parser.add_argument("--negative", type=int, default=0, help="No.of negative prompts")
    parser.add_argument("-i", "--input", type=str, default="./livecell/",
                        help="Provide the data directory for LIVECell Dataset")
    parser.add_argument("-e", "--embedding_path", type=str, default="./sam_embeddings/",
                        help="Provide the path where embeddings will be saved")
    parser.add_argument("-p", "--pred_path", type=str, default="./predictions/",
                        help="Provide the path where the predictions will be saved")
    parser.add_argument("-c", "--ckpt", type=str, required=True, help="Provide model checkpoints (vanilla / finetuned)")
    parser.add_argument("-m", "--model", type=str, default="vit_b",
                        help="Pass the checkpoint-specific model name being used for inference")
    parser.add_argument("-n", "--name", type=str, default="finetuned_livecell",
                        help="Provide a name for the saving nomenclature")
    args = parser.parse_args()
    main(args)
