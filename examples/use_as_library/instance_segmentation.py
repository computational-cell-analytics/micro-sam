import os

import imageio.v3 as imageio

import napari

from micro_sam import util
from micro_sam.sample_data import fetch_hela_2d_example_data, fetch_wholeslide_example_data
from micro_sam.v2.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter


DATA_CACHE = os.path.join(util.get_cache_directory(), "sample_data")
EMBEDDING_CACHE = os.path.join(util.get_cache_directory(), "embeddings")
os.makedirs(EMBEDDING_CACHE, exist_ok=True)

# There are three choices for the automatic segmentation ('segmentation_mode'):
# 1. AIS (automatic instance segmentation): The decoder of the micro-sam2 model predicts the foreground,
#    the distances to the object centers and boundaries, and the boundaries. The post-processing derives
#    the instance segmentation from them. This is the default for models with a decoder.
# 2. APG (automatic prompt generation): Derives one point prompt per object from the decoder predictions,
#    and segments each object with the interactive part of the model. It is slower and more accurate.
# 3. AMG (automatic mask generation): The instance segmentation of Segment Anything, which prompts a grid
#    of points and keeps the plausible masks. It needs no decoder.


def cell_segmentation(model_type):
    """Segment HeLa cells with the automatic segmentation of micro-sam2.
    """
    image = imageio.imread(fetch_hela_2d_example_data(DATA_CACHE))
    embedding_path = os.path.join(EMBEDDING_CACHE, f"embeddings-hela2d-{model_type}.zarr")

    # Step 1: Get the 'predictor' and 'segmenter' to run the automatic segmentation.
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type,  # The micro-sam2 model.
        checkpoint=None,  # Pass the path to your own finetuned model here.
        device=None,  # The device to run the model on. By default, the best available device is used.
        segmentation_mode="ais",  # The segmentation engine: 'ais', 'apg' or 'amg'.
    )

    # Step 2: Get the instance segmentation for the image.
    instances = automatic_instance_segmentation(
        predictor=predictor,  # The predictor of the micro-sam2 model.
        segmenter=segmenter,  # The segmenter, which derives the instance segmentation.
        input_path=image,  # The filepath to the image, or the image as an array.
        embedding_path=embedding_path,  # The filepath where the embeddings are (or will be) stored.
        ndim=2,  # The number of spatial dimensions.
    )

    # Show the results.
    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(instances)
    napari.run()


def cell_segmentation_with_tiling(model_type):
    """Segment the cells in a large image with tiling.
    """
    image = imageio.imread(fetch_wholeslide_example_data(DATA_CACHE))
    embedding_path = os.path.join(EMBEDDING_CACHE, f"whole-slide-embeddings-{model_type}.zarr")

    # Step 1: Get the 'predictor' and 'segmenter' to run the automatic segmentation with tiling.
    predictor, segmenter = get_predictor_and_segmenter(model_type=model_type, segmentation_mode="ais", is_tiled=True)

    # Step 2: Get the instance segmentation for the image.
    instances = automatic_instance_segmentation(
        predictor=predictor,
        segmenter=segmenter,
        input_path=image,
        embedding_path=embedding_path,
        ndim=2,
        tile_shape=(1024, 1024),  # The tile shape for tiled prediction.
        halo=(256, 256),  # The overlap between tiles.
    )

    # Show the results.
    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(instances)
    napari.run()


def segmentation_in_3d(model_type):
    """Segment nuclei in 3d.
    """
    from micro_sam.sample_data import fetch_nucleus_3d_example_data

    image = imageio.imread(fetch_nucleus_3d_example_data(DATA_CACHE))
    embedding_path = os.path.join(EMBEDDING_CACHE, f"embeddings-3d-{model_type}.zarr")

    # Step 1: Get the 'predictor' and 'segmenter' to run the automatic segmentation.
    # APG needs to know the number of dimensions, because it propagates its prompts through the volume.
    predictor, segmenter = get_predictor_and_segmenter(model_type=model_type, segmentation_mode="ais", ndim=3)

    # Step 2: Get the instance segmentation for the volume.
    instances = automatic_instance_segmentation(
        predictor=predictor, segmenter=segmenter, input_path=image, embedding_path=embedding_path, ndim=3,
    )

    # Show the results.
    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(instances)
    napari.run()


def main():
    # The micro-sam2 model for light microscopy.
    model_type = "hvit_t_cells"

    # cell_segmentation(model_type)
    # cell_segmentation_with_tiling(model_type)
    segmentation_in_3d(model_type)


if __name__ == "__main__":
    main()
