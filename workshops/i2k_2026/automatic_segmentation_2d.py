import os
import argparse

import imageio.v3 as imageio

import napari

from micro_sam.v2.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation

from download_datasets import DATASETS_2D, SEGMENTATION_MODES, get_image_and_labels


def run_automatic_segmentation_2d(input_path, dataset_name, embedding_dir, output_folder, model_type, mode):
    """Run automatic instance segmentation for a 2d image and show it in napari."""
    # We use the first image of the dataset. The '0' in the file names matches the classification scripts.
    image, _ = get_image_and_labels(input_path, dataset_name)
    mode = SEGMENTATION_MODES[dataset_name] if mode is None else mode

    os.makedirs(embedding_dir, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    embedding_path = os.path.join(embedding_dir, f"{dataset_name}-0-{model_type}.zarr")
    output_path = os.path.join(output_folder, f"{dataset_name}-0-{model_type}-{mode}.tif")

    if os.path.exists(output_path):
        segmentation = imageio.imread(output_path)
    else:
        predictor, segmenter = get_predictor_and_segmenter(model_type=model_type, segmentation_mode=mode)
        segmentation = automatic_instance_segmentation(
            predictor=predictor,
            segmenter=segmenter,
            input_path=image,
            output_path=output_path,
            embedding_path=embedding_path,
            ndim=2,
        )

    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(segmentation, name=f"segmentation-{mode}")
    napari.run()


def main():
    parser = argparse.ArgumentParser(description="Run automatic instance segmentation for 2d images.")
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
        "-o", "--output_folder", type=str, default="./segmentations",
        help="The filepath to the folder where the segmentations are stored. By default, it is './segmentations'."
    )
    parser.add_argument("-m", "--model_type", type=str, default="hvit_t_cells", help="The segmentation model.")
    parser.add_argument(
        "--mode", type=str, default=None, choices=("apg", "ais", "amg"),
        help="The automatic segmentation mode: automatic prompt generation ('apg'), instance segmentation "
        "with the decoder ('ais') or automatic mask generation ('amg'). By default, it uses the mode of the dataset."
    )
    args = parser.parse_args()

    run_automatic_segmentation_2d(
        args.input_path, args.dataset_name, args.embedding_dir, args.output_folder, args.model_type, args.mode
    )


if __name__ == "__main__":
    main()
