import os

import imageio.v3 as imageio

import napari

from micro_sam import util
from micro_sam.sample_data import fetch_hela_2d_example_data, fetch_wholeslide_example_data
from micro_sam.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter


DATA_CACHE = os.path.join(util.get_cache_directory(), "sample_data")
EMBEDDING_CACHE = os.path.join(util.get_cache_directory(), "embeddings")
os.makedirs(EMBEDDING_CACHE, exist_ok=True)


def cell_segmentation(use_finetuned_model):
    """Run the instance segmentation functionality from micro_sam for segmentation of HeLA cells.
    """
    image_path = fetch_hela_2d_example_data(DATA_CACHE)

    if use_finetuned_model:
        embedding_path = os.path.join(EMBEDDING_CACHE, "embeddings-hela2d-vit_b_lm.zarr")
        model_type = "vit_b_lm"
    else:
        embedding_path = os.path.join(EMBEDDING_CACHE, "embeddings-hela2d.zarr")
        model_type = "vit_h"

    # Load the image, the SAM Model, and the pre-computed embeddings.
    image = imageio.imread(image_path)

    # There are two choices of the automatic segmentation:
    # 1. AMG (automatic mask generation): Use the instance segmentation logic of Segment Anything.
    # This works by covering the image with a grid of points, getting the masks for all the poitns
    # and only keeping the plausible ones (according to the model predictions).
    # While the functionality here does the same as the implementation from Segment Anything,
    # we enable changing the hyperparameters, e.g. 'pred_iou_thresh', without recomputing masks and embeddings,
    # to support (interactive) evaluation of different hyperparameters.
    # 2. AIS (automatic instance segmentation): Use the instance segmentation oofic of 'micro-sam'.
    # This works by predicting three output channels: distance to the object center,
    # distance to the object boundary and foreground probabilities (per image).
    # Then, we compute an instance segmentation based on them using a seeded watershed.

    # Step 1: Get the 'predictor' and 'segmenter' to perform automatic segmentation.
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type,  # choice of the Segment Anything model
        checkpoint=None,  # overwrite to pass your own finetuned model.
        device=None,  # overwrite to pass device to run the model inference. by default, chooses best supported device.
    )

    # Step 2: Get the instance segmentation for the given image.
    instances = automatic_instance_segmentation(
        predictor=predictor,  # the predictor for the Segment Anything model.
        segmenter=segmenter,  # the segmenter class responsible for generating predictions.
        input_path=image,  # the filepath to image or the input array for automatic segmentation.
        embedding_path=embedding_path,  # the filepath where the computed embeddings are / will be stored.
        ndim=2,  # the number of input dimensions.
    )

    # Show the results.
    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(instances)
    napari.run()


def cell_segmentation_with_tiling(use_finetuned_model):
    """Run the instance segmentation functionality from micro_sam for segmentation of
    cells in a large image.
    """
    image_path = fetch_wholeslide_example_data(DATA_CACHE)

    if use_finetuned_model:
        embedding_path = os.path.join(EMBEDDING_CACHE, "whole-slide-embeddings-vit_b_lm.zarr")
        model_type = "vit_b_lm"
    else:
        embedding_path = os.path.join(EMBEDDING_CACHE, "whole-slide-embeddings.zarr")
        model_type = "vit_h"

    # Load the image, the SAM Model, and the pre-computed embeddings.
    image = imageio.imread(image_path)

    # There are two choices of the automatic segmentation:
    # 1. AMG (automatic mask generation): Use the instance segmentation logic of Segment Anything.
    # This works by covering the image with a grid of points, getting the masks for all the poitns
    # and only keeping the plausible ones (according to the model predictions).
    # While the functionality here does the same as the implementation from Segment Anything,
    # we enable changing the hyperparameters, e.g. 'pred_iou_thresh', without recomputing masks and embeddings,
    # to support (interactive) evaluation of different hyperparameters.
    # 2. AIS (automatic instance segmentation): Use the instance segmentation oofic of 'micro-sam'.
    # This works by predicting three output channels: distance to the object center,
    # distance to the object boundary and foreground probabilities (per image).
    # Then, we compute an instance segmentation based on them using a seeded watershed.

    # Step 1: Get the 'predictor' and 'segmenter' to perform automatic segmentation.
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type,  # choice of the Segment Anything model
        checkpoint=None,  # overwrite to pass your own finetuned model.
        device=None,  # overwrite to pass device to run the model inference. by default, chooses best supported device.
        is_tiled=True,  # whether to run automatic segmentation with tiling.
    )

    # Step 2: Get the instance segmentation for the given image.
    instances = automatic_instance_segmentation(
        predictor=predictor,  # the predictor for the Segment Anything model.
        segmenter=segmenter,  # the segmenter class responsible for generating predictions.
        input_path=image,  # the filepath to image or the input array for automatic segmentation.
        embedding_path=embedding_path,  # the filepath where the computed embeddings are / will be stored.
        ndim=2,  # the number of input dimensions.
        tile_shape=(1024, 1024),  # the tile shape for tiling-based prediction.
        halo=(256, 256),  # the overlap shape for tiling-based prediction.
    )

    # Show the results.
    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(instances)
    napari.run()


def segmentation_in_3d(use_finetuned_model):
    """Run instance segmentation in 3d.
    """
    import imageio.v3 as imageio
    from micro_sam.sample_data import fetch_nucleus_3d_example_data

    # Load the example image data: 3d nucleus segmentation.
    image_path = fetch_nucleus_3d_example_data(DATA_CACHE)
    image = imageio.imread(image_path)

    # Load the SAM model and segmentation decoder.
    model_type = "vit_b_lm"  # The model-type to use: vit_h, vit_l, vit_b etc.
    embedding_path = os.path.join(EMBEDDING_CACHE, "embeddings-3d.zarr")  # The embeddings will be cached here.

    # Step 1: Get the 'predictor' and 'segmenter' to perform automatic segmentation.
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type,  # choice of the Segment Anything model
        checkpoint=None,  # overwrite to pass your own finetuned model.
        device=None,  # overwrite to pass device to run the model inference. by default, chooses best supported device.
        is_tiled=False,  # whether to run automatic segmentation with tiling.
    )

    # Step 2: Get the instance segmentation for the given image.
    instances = automatic_instance_segmentation(
        predictor=predictor,  # the predictor for the Segment Anything model.
        segmenter=segmenter,  # the segmenter class responsible for generating predictions.
        input_path=image,  # the filepath to image or the input array for automatic segmentation.
        embedding_path=embedding_path,  # the filepath where the computed embeddings are / will be stored.
        ndim=3,  # the number of input dimensions.
    )

    # Show the results.
    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(instances)
    napari.run()


def main():
    # Whether to use the fine-tuned SAM model for light microscopy data.
    use_finetuned_model = True

    # cell_segmentation(use_finetuned_model)
    # cell_segmentation_with_tiling(use_finetuned_model)
    segmentation_in_3d(use_finetuned_model)


if __name__ == "__main__":
    main()
