import os
import argparse

import imageio.v3 as imageio

from micro_sam.sam_annotator import annotator

from download_datasets import DATASETS_3D, get_image_and_labels


def run_annotator_3d(input_path, dataset_name, embedding_dir, segmentation_folder, model_type):
    """Start the annotator for interactive and automatic segmentation of a 3d volume."""
    volume, _ = get_image_and_labels(input_path, dataset_name)

    os.makedirs(embedding_dir, exist_ok=True)
    embedding_path = os.path.join(embedding_dir, f"{dataset_name}-{model_type}.zarr")

    # We load the APG segmentation from 'automatic_segmentation_3d.py', so that you can correct it right away.
    segmentation_path = os.path.join(segmentation_folder, f"{dataset_name}-{model_type}-apg.tif")
    segmentation = imageio.imread(segmentation_path) if os.path.exists(segmentation_path) else None

    # 'precompute_autoseg_state' caches the decoder predictions, so that 'Automatic Segmentation' runs fast.
    annotator(
        volume, ndim=3, embedding_path=embedding_path, segmentation_result=segmentation,
        model_type=model_type, precompute_autoseg_state=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Run interactive and automatic segmentation for 3d volumes.")
    parser.add_argument(
        "-i", "--input_path", type=str, default="./data",
        help="The filepath to the folder where the image data is downloaded. By default, it is './data'."
    )
    parser.add_argument(
        "-d", "--dataset_name", type=str, default="nuclei_3d", choices=DATASETS_3D,
        help="The choice of 3d dataset. By default, it uses 'nuclei_3d'."
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

    run_annotator_3d(args.input_path, args.dataset_name, args.embedding_dir, args.segmentation_folder, args.model_type)


if __name__ == "__main__":
    main()
