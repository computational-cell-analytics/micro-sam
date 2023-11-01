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
    instances_amg = amg.generate(pred_iou_thresh=0.88)
    instances_amg = instance_segmentation.mask_data_to_segmentation(
        instances_amg, shape=image.shape, with_background=True
    )

    # Use the mutex waterhsed based instance segmentation logic.
    # Here, we generate initial segmentation masks from the image embeddings, using the mutex watershed algorithm.
    # These initial masks are used as prompts for the actual instance segmentation.
    # This class uses the same overall design as 'AutomaticMaskGenerator'.

    # Create the automatic mask generator class.
    amg_mws = instance_segmentation.EmbeddingMaskGenerator(predictor, min_initial_size=10)

    # Initialize the mask generator with the image and the pre-computed embeddings.
    amg_mws.initialize(image, embeddings, verbose=True)

    # Generate the instance segmentation. You can call this again for different values of 'pred_iou_thresh'
    # without having to call initialize again.
    # NOTE: the main advantage of this method is that it's faster than the original implementation,
    # however the quality is not as high as the original instance segmentation quality yet.
    instances_mws = amg_mws.generate(pred_iou_thresh=0.88)
    instances_mws = instance_segmentation.mask_data_to_segmentation(
        instances_mws, shape=image.shape, with_background=True
    )

    # Show the results.
    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(instances_amg)
    v.add_labels(instances_mws)
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
    instances_amg = amg.generate(pred_iou_thresh=0.88)
    instances_amg = instance_segmentation.mask_data_to_segmentation(
        instances_amg, shape=image.shape, with_background=True
    )

    # Use the mutex waterhsed based instance segmentation logic.
    # Here, we generate initial segmentation masks from the image embeddings, using the mutex watershed algorithm.
    # These initial masks are used as prompts for the actual instance segmentation.
    # This class uses the same overall design as 'AutomaticMaskGenerator'.

    # Create the automatic mask generator class.
    amg_mws = instance_segmentation.TiledEmbeddingMaskGenerator(predictor, min_initial_size=10)

    # Initialize the mask generator with the image and the pre-computed embeddings.
    amg_mws.initialize(image, embeddings, verbose=True)

    # Generate the instance segmentation. You can call this again for different values of 'pred_iou_thresh'
    # without having to call initialize again.
    # NOTE: the main advantage of this method is that it's faster than the original implementation.
    # however the quality is not as high as the original instance segmentation quality yet.
    instances_mws = amg_mws.generate(pred_iou_thresh=0.88)

    # Show the results.
    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(instances_amg)
    v.add_labels(instances_mws)
    napari.run()


def segmentation_in_3d():
    """
    """
    from micro_sam.sample_data import synthetic_data

    shape = (5, 512, 512)
    data, _ = synthetic_data(shape)
    predictor = util.get_sam_model(model_type="vit_t")
    seg = segment_3d_from_slice(predictor, data, embedding_path="./tmp_embeddings.zarr", verbose=True)

    v = napari.Viewer()
    v.add_image(data)
    v.add_labels(seg)
    napari.run()


def main():
    # cell_segmentation()
    # cell_segmentation_with_tiling()
    segmentation_in_3d()


if __name__ == "__main__":
    main()
