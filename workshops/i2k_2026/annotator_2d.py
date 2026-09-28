import os
import argparse

import imageio.v3 as imageio

from micro_sam.sam_annotator import annotator

from download_datasets import DATASETS_2D, SEGMENTATION_MODES, get_image_and_labels


def run_annotator_2d(input_path, dataset_name, embedding_dir, segmentation_folder, model_type):
    """Start the annotator for interactive and automatic segmentation of a 2d image."""
    # We use the first image of the dataset. The '0' in the file names matches the classification scripts.
    image, _ = get_image_and_labels(input_path, dataset_name)

    os.makedirs(embedding_dir, exist_ok=True)
    embedding_path = os.path.join(embedding_dir, f"{dataset_name}-0-{model_type}.zarr")

    # We load the segmentation from 'automatic_segmentation_2d.py', so that you can correct it right away.
    mode = SEGMENTATION_MODES[dataset_name]
    segmentation_path = os.path.join(segmentation_folder, f"{dataset_name}-0-{model_type}-{mode}.tif")
    segmentation = imageio.imread(segmentation_path) if os.path.exists(segmentation_path) else None

    # 'precompute_autoseg_state' caches the decoder predictions, so that 'Automatic Segmentation' runs fast.
    annotator(
        image, ndim=2, embedding_path=embedding_path, segmentation_result=segmentation,
        model_type=model_type, precompute_autoseg_state=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Run interactive and automatic segmentation for 2d images.")
    parser.add_argument(
        "-i", "--input_path", type=str, default="./data",
        help="The filepath to the folder where the image data is downloaded. By default, it is './data'."
    )
    parser.add_argument(
        "-d", "--dataset_name", type=str, default="cells_2d", choices=DATASETS_2D,
        help="The choice of 2d dataset. By default, it uses 'cells_2d'."
    )
    parser.add_argument(
        "-e", "--embedding_dir", type=str, default="./embeddings",
        help="The filepath to the folder where the image embeddings are cached. By default, it is './embeddings'."
    )
    parser.add_argument(
        "-s", "--segmentation_folder", type=str, default="./segmentations",
        help="The filepath to the folder with the segmentations from the automatic segmentation script. "
        "By default, it is './segmentations'."
    )
    parser.add_argument("-m", "--model_type", type=str, default="hvit_t_cells", help="The segmentation model.")
    args = parser.parse_args()

    run_annotator_2d(args.input_path, args.dataset_name, args.embedding_dir, args.segmentation_folder, args.model_type)


if __name__ == "__main__":
    main()
