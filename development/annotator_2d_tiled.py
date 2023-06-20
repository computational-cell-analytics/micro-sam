import imageio.v3 as imageio
from micro_sam.sam_annotator import annotator_2d


def annotator_with_tiling():
    # whole slide image data from the neurips cell seg challenge
    im = imageio.imread(
        "/home/pape/Work/data/neurips-cell-seg/TrainUnlabeled_WholeSlide/whole_slide_00002.tiff"
    )

    im = im[:4096, :4096, :]

    # import napari
    # v = napari.Viewer()
    # v.add_image(im)
    # napari.run()

    embedding_path = "./embeddings/embeddings-tiled.zarr"
    annotator_2d(im, embedding_path, tile_shape=(1024, 1024), halo=(256, 256))


def debug():
    import numpy as np
    import micro_sam.util as util
    from micro_sam.segment_from_prompts import segment_from_points, segment_from_box

    im = imageio.imread(
        "/home/pape/Work/data/neurips-cell-seg/TrainUnlabeled_WholeSlide/whole_slide_00002.tiff"
    )
    im = im[:4096, :4096, -1]

    embedding_path = "./embeddings/embeddings-tiled.zarr"
    predictor = util.get_sam_model(model_type="vit_h", return_sam=False)
    image_embeddings = util.precompute_image_embeddings(
        predictor, im, save_path=embedding_path, ndim=2, tile_shape=(512, 512), halo=(64, 64)
    )

    if False:
        points = np.array([
            [1718.7391402, 708.53708524],
            [1759.25543047, 698.75729103],
            [1715.94491328, 663.82945459],
            [1665.64882881, 720.41254963]
        ])
        labels = np.array([1, 0, 0, 0])
        segment_from_points(predictor, points, labels, image_embeddings)

    if True:
        box = np.array([2056.1861078, 1560.13288731, 2114.97895827, 1646.06089954])
        segment_from_box(predictor, box, image_embeddings)


def main():
    # debug()
    annotator_with_tiling()


main()
