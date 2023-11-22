import imageio.v3 as imageio
import napari

from micro_sam import instance_segmentation, util
from micro_sam.multi_dimensional_segmentation import segment_3d_from_slice


def cell_segmentation():
    """Run the instance segmentation functionality from micro_sam for segmentation of
    HeLA cells. You need to run examples/annotator_2d.py:hela_2d_annotator once before
    running this script so that all required data is downloaded and pre-computed.
    """
    image_path = "../data/hela-2d-image.png"
    embedding_path = "../embeddings/embeddings-hela2d.zarr"

    # Load the image, the SAM Model, and the pre-computed embeddings.
    image = imageio.imread(image_path)
    predictor = util.get_sam_model()
    embeddings = util.precompute_image_embeddings(predictor, image, save_path=embedding_path)

    # Use the instance segmentation logic of SegmentAnything.
    # This works by covering the image with a grid of points, getting the masks for all the poitns
    # and only keeping the plausible ones (according to the model predictions).
    # While the functionality here does the same as the implementation from SegmentAnything,
    # we enable changing the hyperparameters, e.g. 'pred_iou_thresh', without recomputing masks and embeddings,
    # to support (interactive) evaluation of different hyperparameters.

    # Create the automatic mask generator class.
    amg = instance_segmentation.AutomaticMaskGenerator(predictor)

    # Initialize the mask generator with the image and the pre-computed embeddings.
    amg.initialize(image, embeddings, verbose=True)

    # Generate the instance segmentation. You can call this again for different values of 'pred_iou_thresh'
    # without having to call initialize again.
    instances = amg.generate(pred_iou_thresh=0.88)
    instances = instance_segmentation.mask_data_to_segmentation(
        instances, shape=image.shape, with_background=True
    )

    # Show the results.
    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(instances)
    napari.run()


def cell_segmentation_with_tiling():
    """Run the instance segmentation functionality from micro_sam for segmentation of
    cells in a large image. You need to run examples/annotator_2d.py:wholeslide_annotator once before
    running this script so that all required data is downloaded and pre-computed.
    """
    image_path = "../data/whole-slide-example-image.tif"
    embedding_path = "../embeddings/whole-slide-embeddings.zarr"

    # Load the image, the SAM Model, and the pre-computed embeddings.
    image = imageio.imread(image_path)
    predictor = util.get_sam_model()
    embeddings = util.precompute_image_embeddings(
        predictor, image, save_path=embedding_path, tile_shape=(1024, 1024), halo=(256, 256)
    )

    # Use the instance segmentation logic of SegmentAnything.
    # This works by covering the image with a grid of points, getting the masks for all the poitns
    # and only keeping the plausible ones (according to the model predictions).
    # The functionality here is similar to the instance segmentation in Segment Anything,
    # but uses the pre-computed tiled embeddings.

    # Create the automatic mask generator class.
    amg = instance_segmentation.TiledAutomaticMaskGenerator(predictor)

    # Initialize the mask generator with the image and the pre-computed embeddings.
    amg.initialize(image, embeddings, verbose=True)

    # Generate the instance segmentation. You can call this again for different values of 'pred_iou_thresh'
    # without having to call initialize again.
    instances = amg.generate(pred_iou_thresh=0.88)
    instances = instance_segmentation.mask_data_to_segmentation(
        instances, shape=image.shape, with_background=True
    )

    # Show the results.
    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(instances)
    v.add_labels(instances)
    napari.run()


def segmentation_in_3d():
    """Run instance segmentation in 3d, for segmenting all objects that intersect
    with a given slice. If you use a fine-tuned model for this then you should
    first find good parameters for 2d segmentation.
    """
    import imageio.v3 as imageio
    from micro_sam.sample_data import fetch_nucleus_3d_example_data

    # Load the example image data: 3d nucleus segmentation.
    path = fetch_nucleus_3d_example_data("./data")
    data = imageio.imread(path)

    # Load the SAM model for prediction.
    model_type = "vit_b"  # The model-type to use: vit_h, vit_l, vit_b etc.
    checkpoint_path = None  # You can specifiy the path to a custom (fine-tuned) model here.
    predictor = util.get_sam_model(model_type=model_type, checkpoint_path=checkpoint_path)

    # Run 3d segmentation for a given slice. Will segment all objects found in that slice
    # throughout the volume.

    # The slice that is used for segmentation in 2d. If you don't specify a slice
    # then the middle slice is used.
    z_slice = data.shape[0] // 2

    # The threshold for filtering objects in the 2d segmentation based on the model's
    # predicted iou score. If you use a custom model you should first find a good setting
    # for this value, e.g. with the 2d annotation tool.
    pred_iou_thresh = 0.88

    # The threshold for filtering objects in the 2d segmentation based on the model's
    # stability score for a given object. If you use a custom model you should first find a good setting
    # for this value, e.g. with the 2d annotation tool.
    stability_score_thresh = 0.95

    instances = segment_3d_from_slice(
        predictor, data, z=z_slice,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        verbose=True
    )

    # Show the results.
    v = napari.Viewer()
    v.add_image(data)
    v.add_labels(instances)
    napari.run()


def main():
    # cell_segmentation()
    # cell_segmentation_with_tiling()
    segmentation_in_3d()


if __name__ == "__main__":
    main()
