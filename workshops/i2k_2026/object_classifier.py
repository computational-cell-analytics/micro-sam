import os
import argparse

import imageio.v3 as imageio

from micro_sam.sam_annotator.object_classifier import object_classifier, batch_object_classifier
from micro_sam.v2.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation

from download_datasets import DATASETS_2D, SEGMENTATION_MODES, get_image_and_labels


# We use only the nuclei in 3d: 'cells_em' has too few cells, and the segmentation of 'neurons_em' is not accurate.
DATASETS_3D = ["nuclei_3d"]


def _segment_images(images, embedding_paths, segmentation_paths, model_type, mode, ndim):
    predictor, segmenter = None, None
    segmentations = []
    for image, embedding_path, segmentation_path in zip(images, embedding_paths, segmentation_paths):
        if os.path.exists(segmentation_path):
            segmentations.append(imageio.imread(segmentation_path))
            continue

        # We load the model only when at least one image does not have a segmentation yet.
        if predictor is None:
            predictor, segmenter = get_predictor_and_segmenter(model_type=model_type, segmentation_mode=mode, ndim=ndim)

        segmentation = automatic_instance_segmentation(
            predictor=predictor,
            segmenter=segmenter,
            input_path=image,
            output_path=segmentation_path,
            embedding_path=embedding_path,
            ndim=ndim,
            verbose=False,
        )
        segmentations.append(segmentation)

    return segmentations


def _run_object_classifier_2d(
    input_path, dataset_name, n_images, embedding_dir, segmentation_folder, output_folder, model_type
):
    images = [get_image_and_labels(input_path, dataset_name, index)[0] for index in range(n_images)]
    embedding_paths = [
        os.path.join(embedding_dir, f"{dataset_name}-{index}-{model_type}.zarr") for index in range(n_images)
    ]

    # We reuse the segmentations from 'automatic_segmentation_2d.py' and segment the other images the same way.
    mode = SEGMENTATION_MODES[dataset_name]
    segmentation_paths = [
        os.path.join(segmentation_folder, f"{dataset_name}-{index}-{model_type}-{mode}.tif")
        for index in range(n_images)
    ]
    segmentations = _segment_images(images, embedding_paths, segmentation_paths, model_type, mode, ndim=2)

    batch_object_classifier(
        images, segmentations, output_folder=os.path.join(output_folder, f"{dataset_name}-object-classification"),
        embedding_paths=embedding_paths, model_type=model_type, ndim=2,
    )


def _run_object_classifier_3d(input_path, dataset_name, embedding_dir, segmentation_folder, model_type):
    volume, _ = get_image_and_labels(input_path, dataset_name)
    embedding_path = os.path.join(embedding_dir, f"{dataset_name}-{model_type}.zarr")

    # We use the APG segmentation, which 'annotator_3d.py' also loads, and compute it if it does not exist.
    segmentation_path = os.path.join(segmentation_folder, f"{dataset_name}-{model_type}-apg.tif")
    segmentation = _segment_images([volume], [embedding_path], [segmentation_path], model_type, "apg", ndim=3)[0]

    object_classifier(volume, segmentation, embedding_path=embedding_path, model_type=model_type, ndim=3)


def run_object_classifier(
    input_path, dataset_name, n_images, embedding_dir, segmentation_folder, output_folder, model_type
):
    """Segment the objects in 2d images or in a 3d volume and start the object classifier for them."""
    os.makedirs(embedding_dir, exist_ok=True)
    os.makedirs(segmentation_folder, exist_ok=True)

    if dataset_name in DATASETS_3D:
        _run_object_classifier_3d(input_path, dataset_name, embedding_dir, segmentation_folder, model_type)
    else:
        _run_object_classifier_2d(
            input_path, dataset_name, n_images, embedding_dir, segmentation_folder, output_folder, model_type
        )


def main():
    parser = argparse.ArgumentParser(description="Run object classification for 2d images or for a 3d volume.")
    parser.add_argument(
        "-i", "--input_path", type=str, default="./data",
        help="The filepath to the folder where the image data is downloaded. By default, it is './data'."
    )
    parser.add_argument(
        "-d", "--dataset_name", type=str, default="histopatho", choices=DATASETS_2D + DATASETS_3D,
        help="The choice of dataset. By default, it uses 'histopatho'."
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
        "-s", "--segmentation_folder", type=str, default="./segmentations",
        help="The filepath to the folder where the segmentations are stored. By default, it is './segmentations'."
    )
    parser.add_argument(
        "-o", "--output_folder", type=str, default="./classification",
        help="The filepath to the folder where the classification results of the 2d images are stored. "
        "By default, it is './classification'."
    )
    parser.add_argument("-m", "--model_type", type=str, default="hvit_t_cells", help="The segmentation model.")
    args = parser.parse_args()

    run_object_classifier(
        args.input_path, args.dataset_name, args.n_images, args.embedding_dir,
        args.segmentation_folder, args.output_folder, args.model_type,
    )


if __name__ == "__main__":
    main()
