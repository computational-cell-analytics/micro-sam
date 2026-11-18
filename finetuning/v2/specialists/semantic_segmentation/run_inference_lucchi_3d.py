"""Run 3d semantic segmentation on the Lucchi test volume and score it.

The ground-truth mitochondria mask is converted to the same three class layout the model was trained on.
"""

import os
import argparse

import h5py
import numpy as np

from elf.evaluation import dice_score

from torch_em.data.datasets import lucchi

from micro_sam.v2.transforms.labels import semantic_labels
from micro_sam.v2.semantic_segmentation import get_semantic_model, semantic_segmentation


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"
NUM_CLASSES = 3
CLASS_NAMES = {1: "boundary", 2: "foreground"}


def run_inference(args):
    """Predict the Lucchi test volume and report the dice score per class."""
    data_path = lucchi.get_lucchi_paths(path=os.path.join(args.input_path, "lucchi"), split="test", download=True)
    with h5py.File(data_path, "r") as f:
        raw = f["raw"][:]
        labels = semantic_labels(f["labels"][:])

    model = get_semantic_model(
        model_type=args.model_type, num_classes=NUM_CLASSES,
        checkpoint_path=args.checkpoint_path, initial_features=args.initial_features,
    )

    tile_shape = None if args.z_slices is None else (args.z_slices, 512, 512)
    prediction = semantic_segmentation(model, raw, tile_shape=tile_shape)

    print(f"Semantic segmentation on the Lucchi test volume of shape {raw.shape}:")
    for class_id, name in sorted(CLASS_NAMES.items()):
        score = dice_score(prediction == class_id, labels == class_id, threshold_seg=None, threshold_gt=None)
        print(f"Dice for class {class_id} ({name}): {score:.4f}")

    if args.output_path is not None:
        with h5py.File(args.output_path, "a") as f:
            f.create_dataset("semantic", data=prediction.astype(np.uint8), compression="gzip")


def main():
    parser = argparse.ArgumentParser(description="Run 3d semantic segmentation on the Lucchi test volume.")
    parser.add_argument("-i", "--input_path", default=DATA_ROOT, help="The folder that holds the Lucchi data.")
    parser.add_argument("-c", "--checkpoint_path", required=True, help="The checkpoint of the trained model.")
    parser.add_argument("-m", "--model_type", default="hvit_t", choices=["hvit_t", "hvit_s", "hvit_b", "hvit_l"])
    parser.add_argument("-o", "--output_path", default=None, help="An h5 file to store the predicted class map in.")
    parser.add_argument("-z", "--z_slices", type=int, default=None, help="The number of z slices per tile.")
    parser.add_argument("--initial_features", type=int, default=32, help="The width of the convolutional decoder.")
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
