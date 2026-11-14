import os

import imageio.v3 as imageio

from micro_sam.util import get_cache_directory
from micro_sam.v2.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation
from micro_sam.sample_data import fetch_hela_2d_example_data, fetch_livecell_example_data, fetch_wholeslide_example_data


DATA_CACHE = os.path.join(get_cache_directory(), "sample_data")


def run_automatic_segmentation(image, model_type, segmentation_mode, generate_kwargs, tile_shape=None, halo=None):
    """Segment all objects in an image and show the result in napari.

    Args:
        image: The input image.
        model_type: The micro-sam2 model.
        segmentation_mode: The segmentation engine, one of 'ais', 'apg' or 'amg'.
        generate_kwargs: Parameters that override the defaults of the engine.
        tile_shape: The tile shape for tiled prediction. By default, images larger than 768 pixels are tiled.
        halo: The overlap between tiles.
    """
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type,
        checkpoint=None,  # Replace this with the path to your own finetuned model.
        segmentation_mode=segmentation_mode,
        is_tiled=tile_shape is not None,
    )

    segmentation = automatic_instance_segmentation(
        predictor=predictor,
        segmenter=segmenter,
        input_path=image,
        ndim=2,
        tile_shape=tile_shape,
        halo=halo,
        **generate_kwargs
    )

    import napari
    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(segmentation)
    napari.run()


def livecell_automatic_segmentation(model_type, segmentation_mode, generate_kwargs):
    """Run the automatic segmentation for an example image from the LIVECell dataset.

    See https://doi.org/10.1038/s41592-021-01249-6 for details on the data.
    """
    image = imageio.imread(fetch_livecell_example_data(DATA_CACHE))
    run_automatic_segmentation(image, model_type, segmentation_mode, generate_kwargs)


def hela_automatic_segmentation(model_type, segmentation_mode, generate_kwargs):
    """Run the automatic segmentation for an example image from the Cell Tracking Challenge (HeLa 2d) dataset.
    """
    image = imageio.imread(fetch_hela_2d_example_data(DATA_CACHE))
    run_automatic_segmentation(image, model_type, segmentation_mode, generate_kwargs)


def wholeslide_automatic_segmentation(model_type, segmentation_mode, generate_kwargs):
    """Run the automatic segmentation with tiling for an example whole-slide image from the
    NeurIPS Cell Segmentation challenge.
    """
    image = imageio.imread(fetch_wholeslide_example_data(DATA_CACHE))
    run_automatic_segmentation(
        image, model_type, segmentation_mode, generate_kwargs, tile_shape=(1024, 1024), halo=(256, 256),
    )


def main():
    # The micro-sam2 model. 'hvit_t_cells' is the fastest model for light microscopy,
    # 'hvit_b_cells' and 'hvit_l_cells' are larger and more accurate.
    model_type = "hvit_t_cells"

    # The segmentation engine:
    # 'ais': the automatic instance segmentation, which post-processes the decoder predictions. It is the fastest.
    # 'apg': the automatic prompt generation, which derives point prompts from the decoder predictions and
    #        segments each object with the interactive part of the model. It is the most accurate.
    # 'amg': the automatic mask generation of Segment Anything, which prompts a grid of points. It needs no decoder.
    segmentation_mode = "apg"

    # The defaults of 'ais' and 'apg' are tuned per model. Pass a parameter here to change it.
    # 'micro_sam.v2.postprocessing.default_postprocessing' (AIS) and
    # 'micro_sam.v2.automatic_prompt_generation.default_prompt_generation' (APG) list the defaults.
    # For example: {"min_size": 100} removes smaller objects in both engines.
    generate_kwargs = {}

    # Automatic segmentation for livecell data.
    livecell_automatic_segmentation(model_type, segmentation_mode, generate_kwargs)

    # Automatic segmentation for cell tracking challenge hela data.
    # hela_automatic_segmentation(model_type, segmentation_mode, generate_kwargs)

    # Automatic segmentation for a whole slide image.
    # wholeslide_automatic_segmentation(model_type, segmentation_mode, generate_kwargs)


# The corresponding CLI call for hela_automatic_segmentation:
# (replace <cache> with the cache directory on your machine)
# $ micro_sam inference segmentation -i <cache>/sample_data/hela-2d-image.png -o hela-2d-image_segmentation.tif \
#   -m hvit_t_cells --engine apg
if __name__ == "__main__":
    main()
