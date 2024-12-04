import os

import imageio.v3 as imageio

from micro_sam.util import get_cache_directory
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation
from micro_sam.sample_data import fetch_hela_2d_example_data, fetch_livecell_example_data, fetch_wholeslide_example_data


DATA_CACHE = os.path.join(get_cache_directory(), "sample_data")


def livecell_automatic_segmentation(model_type, use_amg, generate_kwargs):
    """Run the automatic segmentation for an example image from the LIVECell dataset.

    See https://doi.org/10.1038/s41592-021-01249-6 for details on the data.
    """
    example_data = fetch_livecell_example_data(DATA_CACHE)
    image = imageio.imread(example_data)

    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type,
        checkpoint=None,  # Replace this with your custom checkpoint.
        amg=use_amg,
        is_tiled=False,  # Switch to 'True' in case you would like to perform tiling-window based prediction.
    )

    segmentation = automatic_instance_segmentation(
        predictor=predictor,
        segmenter=segmenter,
        input_path=image,
        ndim=2,
        tile_shape=None,  # If you set 'is_tiled' in 'get_predictor_and_segmeter' to True, set a tile shape
        halo=None,  # If you set 'is_tiled' in 'get_predictor_and_segmeter' to True, set a halo shape.
        **generate_kwargs
    )

    import napari
    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(segmentation)
    napari.run()


def hela_automatic_segmentation(model_type, use_amg, generate_kwargs):
    """Run the automatic segmentation for an example image from the Cell Tracking Challenge (HeLa 2d) dataset.
    """
    example_data = fetch_hela_2d_example_data(DATA_CACHE)
    image = imageio.imread(example_data)

    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type,
        checkpoint=None,  # Replace this with your custom checkpoint.
        amg=use_amg,
        is_tiled=False,  # Switch to 'True' in case you would like to perform tiling-window based prediction.
    )

    segmentation = automatic_instance_segmentation(
        predictor=predictor,
        segmenter=segmenter,
        input_path=image,
        ndim=2,
        tile_shape=None,  # If you set 'is_tiled' in 'get_predictor_and_segmeter' to True, set a tile shape
        halo=None,  # If you set 'is_tiled' in 'get_predictor_and_segmeter' to True, set a halo shape.
        **generate_kwargs
    )

    import napari
    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(segmentation)
    napari.run()


def wholeslide_automatic_segmentation(model_type, use_amg, generate_kwargs):
    """Run the automatic segmentation with tiling for an example whole-slide image from the
    NeurIPS Cell Segmentation challenge.
    """
    example_data = fetch_wholeslide_example_data(DATA_CACHE)
    image = imageio.imread(example_data)

    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type,
        checkpoint=None,  # Replace this with your custom checkpoint.
        amg=use_amg,
        is_tiled=True,
    )

    segmentation = automatic_instance_segmentation(
        predictor=predictor,
        segmenter=segmenter,
        input_path=image,
        ndim=2,
        tile_shape=(1024, 1024),
        halo=(256, 256),
        **generate_kwargs
    )

    import napari
    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(segmentation)
    napari.run()


def main():
    # The choice of Segment Anything model.
    model_type = "vit_b_lm"

    # Whether to use:
    # the automatic mask generation (AMG): supported by all our models.
    # the automatic instance segmentation (AIS): supported by 'micro-sam' models.
    use_amg = False  # 'False' chooses AIS as the automatic segmentation mode.

    # Post-processing parameters for automatic segmentation.
    if use_amg:  # AMG parameters
        generate_kwargs = {
            "pred_iou_thresh": 0.88,
            "stability_score_thresh": 0.95,
            "box_nms_thresh": 0.7,
            "crop_nms_thresh": 0.7,
            "min_mask_region_area": 0,
            "output_mode": "binary_mask",
        }
    else:  # AIS parameters
        generate_kwargs = {
            "center_distance_threshold": 0.5,
            "boundary_distance_threshold": 0.5,
            "foreground_threshold": 0.5,
            "foreground_smoothing": 1.0,
            "distance_smoothing": 1.6,
            "min_size": 0,
            "output_mode": "binary_mask",
        }

    # Automatic segmentation for livecell data.
    livecell_automatic_segmentation(model_type, use_amg, generate_kwargs)

    # Automatic segmentation for cell tracking challenge hela data.
    # hela_automatic_segmentation(model_type, use_amg, generate_kwargs)

    # Automatic segmentation for a whole slide image.
    # wholeslide_automatic_segmentation(model_type, use_amg, generate_kwargs)


# The corresponding CLI call for hela_automatic_segmentation:
# (replace with cache directory on your machine)
# $ micro_sam.automatic_segmentation -i /home/pape/.cache/micro_sam/sample_data/hela-2d-image.png -o hela-2d-image_segmentation.tif  # noqa
if __name__ == "__main__":
    main()
