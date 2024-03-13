import os
from glob import glob

import h5py

from micro_sam.evaluation.multi_dimensional_segmentation import run_multi_dimensional_segmentation_grid_search


def get_raw_and_label_volumes(volume_path):
    with h5py.File(volume_path, "r") as f:
        raw = f["raw"][:]
        labels = f["label"][:]

    return raw, labels


def main(args):
    test_volume_paths = glob(os.path.join(args.input_path, "*.h5"))
    test_volume_path = test_volume_paths[0]
    volume, labels = get_raw_and_label_volumes(test_volume_path)

    run_multi_dimensional_segmentation_grid_search(
        volume=volume,
        ground_truth=labels,
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        embedding_path=args.embedding_path,
        result_dir="./plantseg-ovules/results_default/",
        interactive_seg_mode="box",
        verbose=False
    )


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_path", type=str, default="/scratch/projects/nim00007/sam/data/plantseg/ovules_test",
        help="Path to PlantSeg (Ovules) volumes"
    )
    parser.add_argument("-m", "--model_type", type=str, default="vit_b", help="Name of the image encoder")
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="The custom checkpoint path.")
    parser.add_argument("-e", "--embedding_path", type=str, default=None, help="Path to save embeddings")
    args = parser.parse_args()
    main(args)
