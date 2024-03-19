import imageio.v3 as imageio
import napari

from micro_sam import instance_segmentation, util
from micro_sam.multi_dimensional_segmentation import automatic_3d_segmentation


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
    """Run instance segmentation in 3d.
    """
    import imageio.v3 as imageio
    from micro_sam.sample_data import fetch_nucleus_3d_example_data

    # Load the example image data: 3d nucleus segmentation.
    path = fetch_nucleus_3d_example_data("./data")
    data = imageio.imread(path)

    # Load the SAM model and segmentation decoder.
    # TODO update this to just use vit_b_lm once it's properly released.
    model_type = "vit_b"  # The model-type to use: vit_h, vit_l, vit_b etc.
    checkpoint_path = "./vit_b_lm.pt"  # You can specifiy the path to a custom (fine-tuned) model here.
    embedding_path = "./embeddings-3d.zarr"  # The embeddings will be cached here. (Optional)

    # Load the model and create segmentation functionality.
    predictor, decoder = instance_segmentation.get_predictor_and_decoder(model_type, checkpoint_path)
    segmentor = instance_segmentation.InstanceSegmentationWithDecoder(predictor, decoder)

    # Run the automatic instance segmentation.
    instances = automatic_3d_segmentation(
        data, predictor, segmentor, embedding_path=embedding_path,
        gap_closing=2,  # This option closes small gaps (here of size 2) in the initial segmentation.
        min_object_size=100,  # The minimal object size per slice.
        center_distance_threshold=0.5,  # We can pass additional arguments for the generate function of the segmenter.
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
