"""Run 2d semantic segmentation on the LIVECell test split and score it.

The ground-truth instance masks are converted to the same three class layout the model was trained on.
"""

import os
import argparse

import numpy as np
import imageio.v3 as imageio
from tqdm import tqdm

from elf.evaluation import dice_score

from torch_em.data.datasets import livecell

from micro_sam.v2.transforms.labels import semantic_labels
from micro_sam.v2.semantic_segmentation import get_semantic_model, semantic_segmentation


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"
NUM_CLASSES = 3
CLASS_NAMES = {1: "boundary", 2: "foreground"}


def run_inference(args):
    """Predict the LIVECell test images and report the dice score per class."""
    image_paths, label_paths = livecell.get_livecell_paths(
        path=os.path.join(args.input_path, "livecell"), split="test", download=True,
    )
    if args.n_images is not None:
        image_paths, label_paths = image_paths[:args.n_images], label_paths[:args.n_images]

    model = get_semantic_model(
        model_type=args.model_type, num_classes=NUM_CLASSES,
        checkpoint_path=args.checkpoint_path, initial_features=args.initial_features,
    )

    if args.output_path is not None:
        os.makedirs(args.output_path, exist_ok=True)

    scores = {class_id: [] for class_id in CLASS_NAMES}
    for image_path, label_path in tqdm(list(zip(image_paths, label_paths)), desc="Segment LIVECell test images"):
        image = imageio.imread(image_path)
        labels = semantic_labels(imageio.imread(label_path))

        prediction = semantic_segmentation(model, image, verbose=False)
        for class_id in scores:
            scores[class_id].append(
                dice_score(prediction == class_id, labels == class_id, threshold_seg=None, threshold_gt=None)
            )

        if args.output_path is not None:
            imageio.imwrite(os.path.join(args.output_path, os.path.basename(image_path)), prediction)

    print(f"Semantic segmentation on {len(image_paths)} LIVECell test images:")
    for class_id, values in sorted(scores.items()):
        print(f"Dice for class {class_id} ({CLASS_NAMES[class_id]}): {np.mean(values):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run 2d semantic segmentation on the LIVECell test split.")
    parser.add_argument("-i", "--input_path", default=DATA_ROOT, help="The folder that holds the LIVECell data.")
    parser.add_argument("-c", "--checkpoint_path", required=True, help="The checkpoint of the trained model.")
    parser.add_argument("-m", "--model_type", default="hvit_t", choices=["hvit_t", "hvit_s", "hvit_b", "hvit_l"])
    parser.add_argument("-o", "--output_path", default=None, help="Where to store the predicted class maps.")
    parser.add_argument("--initial_features", type=int, default=32, help="The width of the convolutional decoder.")
    parser.add_argument("--n_images", type=int, default=None, help="Predict only this many test images.")
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
