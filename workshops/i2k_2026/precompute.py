import os
import argparse

from micro_sam.v2.util import resolve_default_tiling
from micro_sam.sam_annotator._state import AnnotatorState
from micro_sam.sam_annotator.util import prepare_annotation_image

from download_datasets import DATASETS_2D, DATASETS_3D, get_image_and_labels


def _precompute(image, ndim, embedding_path, model_type, models):
    image, ndim, rgb = prepare_annotation_image(image, ndim=ndim)

    state = AnnotatorState()
    state.reset_state()
    state.image_shape = image.shape[:-1] if rgb else image.shape
    tile_shape, halo = resolve_default_tiling(state.image_shape, None, None, embedding_path)

    # We reuse the model of the same dimensionality, so that it is loaded only once.
    predictor, decoder = models.get(ndim, (None, None))
    state.initialize_predictor(
        image, model_type=model_type, save_path=embedding_path, halo=halo, tile_shape=tile_shape,
        precompute_autoseg_state=True, ndim=ndim, predictor=predictor, decoder=decoder, skip_load=False, use_cli=True,
    )
    models[ndim] = (state.predictor, state.decoder)


def precompute(input_path, embedding_dir, n_images, model_type):
    """Precompute the embeddings and the automatic segmentation state for all workshop data."""
    os.makedirs(embedding_dir, exist_ok=True)
    models = {}

    for dataset_name in DATASETS_2D:
        for index in range(n_images):
            print(f"Precompute the embeddings for '{dataset_name}', image {index}.")
            image, _ = get_image_and_labels(input_path, dataset_name, index)
            embedding_path = os.path.join(embedding_dir, f"{dataset_name}-{index}-{model_type}.zarr")
            _precompute(image, 2, embedding_path, model_type, models)

    for dataset_name in DATASETS_3D:
        print(f"Precompute the embeddings for '{dataset_name}'.")
        volume, _ = get_image_and_labels(input_path, dataset_name)
        embedding_path = os.path.join(embedding_dir, f"{dataset_name}-{model_type}.zarr")
        _precompute(volume, 3, embedding_path, model_type, models)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute the image embeddings and the automatic segmentation state for the workshop data."
    )
    parser.add_argument(
        "-i", "--input_path", type=str, default="./data",
        help="The filepath to the folder where the image data is downloaded. By default, it is './data'."
    )
    parser.add_argument(
        "-e", "--embedding_dir", type=str, default="./embeddings",
        help="The filepath to the folder where the image embeddings are cached. By default, it is './embeddings'."
    )
    parser.add_argument("-n", "--n_images", type=int, default=10, help="The number of images per 2d dataset.")
    parser.add_argument("-m", "--model_type", type=str, default="hvit_t_cells", help="The segmentation model.")
    args = parser.parse_args()

    precompute(args.input_path, args.embedding_dir, args.n_images, args.model_type)


if __name__ == "__main__":
    main()
