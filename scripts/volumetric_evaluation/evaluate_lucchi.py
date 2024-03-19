import os
import h5py
from skimage.measure import label

from micro_sam.evaluation.multi_dimensional_segmentation import run_multi_dimensional_segmentation_grid_search


def get_raw_and_label_volumes(volume_path):
    with h5py.File(volume_path, "r") as f:
        raw = f["raw"][:]
        labels = f["labels"][:]

    return raw, labels


def main(args):
    test_volume_path = os.path.join(args.input_path, "lucchi_test.h5")
    volume, labels = get_raw_and_label_volumes(test_volume_path)

    # applying connected components to get instances
    labels = label(labels)

    grid_search_values = {
        "iou_threshold": [0.5],
        "projection": [{'use_box': False, 'use_mask': False, 'use_points': True}],
        "box_extension": [0.075],
        "criterion_choice": [1, 2, 3]
    }

    run_multi_dimensional_segmentation_grid_search(
        volume=volume,
        ground_truth=labels,
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        embedding_path=args.embedding_path,
        result_dir=args.resdir,
        interactive_seg_mode="box",
        verbose=False,
        grid_search_values=grid_search_values
    )


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_path", type=str, default="/scratch/projects/nim00007/sam/data/lucchi", help="Path to volume"
    )
    parser.add_argument("-m", "--model_type", type=str, default="vit_b", help="Name of the image encoder")
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="The custom checkpoint path.")
    parser.add_argument("-e", "--embedding_path", type=str, default=None, help="Path to save embeddings")
    parser.add_argument("--resdir", type=str, required=True, help="Path to save the results")
    args = parser.parse_args()
    main(args)
