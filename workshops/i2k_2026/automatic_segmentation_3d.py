import os
import argparse

import imageio.v3 as imageio

import napari

from micro_sam.v2.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation

from download_datasets import DATASETS_3D, SEGMENTATION_MODES, get_image_and_labels


# The EM volumes have densely packed objects, so the decoder predictions are post-processed with multicut.
POSTPROCESSING_MODES = {"nuclei_3d": "sparse", "neurons_em": "dense", "cells_em": "dense"}


def run_automatic_segmentation_3d(input_path, dataset_name, embedding_dir, output_folder, model_type, mode):
    """Run automatic instance segmentation for a 3d volume and show it in napari."""
    volume, _ = get_image_and_labels(input_path, dataset_name)
    mode = SEGMENTATION_MODES[dataset_name] if mode is None else mode

    os.makedirs(embedding_dir, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    embedding_path = os.path.join(embedding_dir, f"{dataset_name}-{model_type}.zarr")
    output_path = os.path.join(output_folder, f"{dataset_name}-{model_type}-{mode}.tif")

    if os.path.exists(output_path):
        segmentation = imageio.imread(output_path)
    else:
        predictor, segmenter = get_predictor_and_segmenter(model_type=model_type, segmentation_mode=mode, ndim=3)
        segmentation = automatic_instance_segmentation(
            predictor=predictor,
            segmenter=segmenter,
            input_path=volume,
            output_path=output_path,
            embedding_path=embedding_path,
            ndim=3,
            mode=POSTPROCESSING_MODES[dataset_name],
        )

    v = napari.Viewer()
    v.add_image(volume)
    v.add_labels(segmentation, name=f"segmentation-{mode}")
    napari.run()


def main():
    parser = argparse.ArgumentParser(description="Run automatic instance segmentation for 3d volumes.")
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
        "-o", "--output_folder", type=str, default="./segmentations",
        help="The filepath to the folder where the segmentations are stored. By default, it is './segmentations'."
    )
    parser.add_argument("-m", "--model_type", type=str, default="hvit_t_cells", help="The segmentation model.")
    parser.add_argument(
        "--mode", type=str, default=None, choices=("ais", "apg"),
        help="The automatic segmentation mode: instance segmentation with the decoder ('ais') or automatic prompt "
        "generation ('apg'). By default, it uses 'ais' for 'nuclei_3d' and 'apg' for the EM volumes."
    )
    args = parser.parse_args()

    run_automatic_segmentation_3d(
        args.input_path, args.dataset_name, args.embedding_dir, args.output_folder, args.model_type, args.mode
    )


if __name__ == "__main__":
    main()
