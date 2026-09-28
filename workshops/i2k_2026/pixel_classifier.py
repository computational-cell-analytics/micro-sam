import os
import argparse

from micro_sam.sam_annotator.pixel_classifier import pixel_classifier, batch_pixel_classifier

from download_datasets import DATASETS_2D, get_image_and_labels


DATASETS_3D = ["nuclei_3d"]


def _run_pixel_classifier_2d(input_path, dataset_name, n_images, embedding_dir, output_folder, model_type):
    images = [get_image_and_labels(input_path, dataset_name, index)[0] for index in range(n_images)]
    embedding_paths = [
        os.path.join(embedding_dir, f"{dataset_name}-{index}-{model_type}.zarr") for index in range(n_images)
    ]

    batch_pixel_classifier(
        images, output_folder=os.path.join(output_folder, f"{dataset_name}-pixel-classification"),
        embedding_paths=embedding_paths, model_type=model_type, ndim=2,
    )


def _run_pixel_classifier_3d(input_path, dataset_name, embedding_dir, model_type):
    volume, _ = get_image_and_labels(input_path, dataset_name)
    embedding_path = os.path.join(embedding_dir, f"{dataset_name}-{model_type}.zarr")
    pixel_classifier(volume, embedding_path=embedding_path, model_type=model_type, ndim=3)


def run_pixel_classifier(input_path, dataset_name, n_images, embedding_dir, output_folder, model_type):
    """Start the pixel classifier for 2d images or for a 3d volume."""
    os.makedirs(embedding_dir, exist_ok=True)

    if dataset_name in DATASETS_3D:
        _run_pixel_classifier_3d(input_path, dataset_name, embedding_dir, model_type)
    else:
        _run_pixel_classifier_2d(input_path, dataset_name, n_images, embedding_dir, output_folder, model_type)


def main():
    parser = argparse.ArgumentParser(description="Run pixel classification for 2d images or for a 3d volume.")
    parser.add_argument(
        "-i", "--input_path", type=str, default="./data",
        help="The filepath to the folder where the image data is downloaded. By default, it is './data'."
    )
    parser.add_argument(
        "-d", "--dataset_name", type=str, default="cells_2d", choices=DATASETS_2D + DATASETS_3D,
        help="The choice of dataset. By default, it uses 'cells_2d'."
    )
    parser.add_argument(
        "-n", "--n_images", type=int, default=10,
        help="The number of 2d images to classify. The 3d datasets have one volume, so they ignore this argument."
    )
    parser.add_argument(
        "-e", "--embedding_dir", type=str, default="./embeddings",
        help="The filepath to the folder where the image embeddings are cached. By default, it is './embeddings'."
    )
    parser.add_argument(
        "-o", "--output_folder", type=str, default="./classification",
        help="The filepath to the folder where the classification results of the 2d images are stored. "
        "By default, it is './classification'."
    )
    parser.add_argument("-m", "--model_type", type=str, default="hvit_t_cells", help="The segmentation model.")
    args = parser.parse_args()

    run_pixel_classifier(
        args.input_path, args.dataset_name, args.n_images, args.embedding_dir, args.output_folder, args.model_type
    )


if __name__ == "__main__":
    main()
