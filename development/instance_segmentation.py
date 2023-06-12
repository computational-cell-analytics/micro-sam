import napari
from elf.io import open_file

import micro_sam.util as util
from micro_sam.segment_instances import (
    segment_instances_from_embeddings,
    segment_instances_sam,
    segment_instances_from_embeddings_with_tiling
)
from micro_sam.visualization import compute_pca

INPUT_PATH = "../examples/data/Lucchi++/Test_In"
EMBEDDINGS_PATH = "../examples/embeddings/embeddings-mito2d.zarr"
TIMESERIES_PATH = "../examples/data/DIC-C2DH-HeLa/train/01"
EMBEDDINGS_TRACKING_PATH = "../examples/embeddings/embeddings-ctc.zarr"


def mito_segmentation() -> None:
    """Performs mito segmentation on the input image."""
    with open_file(INPUT_PATH) as f:
        raw = f["*.png"][-1, :768, :768]

    predictor, sam = util.get_sam_model(return_sam=True)

    print("Run SAM prediction ...")
    seg_sam = segment_instances_sam(sam, raw)

    image_embeddings = util.precompute_image_embeddings(predictor, raw, EMBEDDINGS_PATH)
    embedding_pca = compute_pca(image_embeddings["features"])

    print("Run prediction from embeddings ...")
    seg, initial_seg = segment_instances_from_embeddings(
        predictor, image_embeddings=image_embeddings, return_initial_seg=True
    )

    v = napari.Viewer()
    v.add_image(raw)
    v.add_image(embedding_pca, scale=(12, 12))
    v.add_labels(seg_sam)
    v.add_labels(seg)
    v.add_labels(initial_seg)
    napari.run()


def cell_segmentation(use_sam=False, use_mws=False, use_tiling=False) -> None:
    """Performs cell segmentation on the input timeseries."""
    with open_file(TIMESERIES_PATH, mode="r") as f:
        timeseries = f["*.tif"][:50]
    print(timeseries.shape[1:])

    frame = 11

    predictor, sam = util.get_sam_model(return_sam=True)

    image_embeddings = util.precompute_image_embeddings(
        predictor, timeseries, EMBEDDINGS_TRACKING_PATH)

    embedding_pca = compute_pca(image_embeddings["features"][frame])

    if use_mws:
        print("Run embedding segmentation ...")
        seg_mws, initial_seg = segment_instances_from_embeddings(
            predictor, image_embeddings=image_embeddings, i=frame, return_initial_seg=True
        )
    else:
        seg_mws = None

    if use_sam:
        print("Run SAM prediction ...")
        seg_sam = segment_instances_sam(sam, timeseries[frame])
    else:
        seg_sam = None

    if use_tiling:
        print("Run embedding segmentation with tiling ...")
        seg_tiled = segment_instances_from_embeddings_with_tiling(
            predictor, timeseries[frame], image_embeddings
        )
    else:
        seg_tiled = None

    v = napari.Viewer()
    v.add_image(timeseries[frame])
    v.add_image(embedding_pca, scale=(8, 8))

    if seg_mws is not None:
        v.add_labels(seg_mws)

    if seg_sam is not None:
        v.add_labels(seg_sam)

    if seg_tiled is not None:
        v.add_labels(seg_tiled)

    napari.run()


def main():
    # automatic segmentation for the data from Lucchi et al. (see 'sam_annotator_3d.py')
    # mito_segmentation()

    # automatic segmentation for data from the cell tracking challenge (see 'sam_annotator_tracking.py')
    cell_segmentation(use_mws=True)


if __name__ == "__main__":
    main()
