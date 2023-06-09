import napari
from elf.io import open_file

import micro_sam.util as util
from micro_sam.segment_instances import (
    segment_instances_from_embeddings,
    segment_instances_sam,
    segment_instances_from_embeddings_3d,
)
from micro_sam.visualization import compute_pca

INPUT_PATH = "/home/pape/Work/data/mouse-embryo/Nuclei/for_sam.h5"
EMBEDDINGS_PATH = "./embeddings/embedding-nuclei-3d.zarr"
TIMESERIES_PATH = "../examples/data/DIC-C2DH-HeLa/train/01"
EMBEDDINGS_TRACKING_PATH = "../examples/embeddings/embeddings-ctc.zarr"


def nucleus_segmentation(use_sam=False, use_mws=False) -> None:
    """Segment nuclei in 3d lightsheet data (one slice)."""
    with open_file(INPUT_PATH) as f:
        raw = f["raw"][:]

    z = 32

    predictor, sam = util.get_sam_model(return_sam=True, model_type="vit_b")
    if use_sam:
        print("Run SAM prediction ...")
        seg_sam = segment_instances_sam(sam, raw[z])
    else:
        seg_sam = None

    image_embeddings = util.precompute_image_embeddings(predictor, raw, EMBEDDINGS_PATH)
    embedding_pca = compute_pca(image_embeddings["features"])[z]

    if use_mws:
        print("Run prediction from embeddings ...")
        seg, initial_seg = segment_instances_from_embeddings(
            predictor, image_embeddings=image_embeddings, return_initial_segmentation=True,
            pred_iou_thresh=0.8, verbose=1, stability_score_thresh=0.9,
            i=z,
        )
    else:
        seg, initial_seg = None, None

    v = napari.Viewer()
    v.add_image(raw[z])
    v.add_image(embedding_pca, scale=(12, 12), visible=False)
    if seg_sam is not None:
        v.add_labels(seg_sam)
    if seg is not None:
        v.add_labels(seg)
    if initial_seg is not None:
        v.add_labels(initial_seg, visible=False)
    napari.run()


def nucleus_segmentation_3d() -> None:
    """Segment nuclei in 3d lightsheet data (3d segmentation)."""
    with open_file(INPUT_PATH) as f:
        raw = f["raw"][:]

    predictor = util.get_sam_model(model_type="vit_b")
    image_embeddings = util.precompute_image_embeddings(predictor, raw, EMBEDDINGS_PATH)
    seg = segment_instances_from_embeddings_3d(predictor, image_embeddings)

    v = napari.Viewer()
    v.add_image(raw)
    v.add_labels(seg)
    napari.run()


def cell_segmentation(use_sam=False, use_mws=False) -> None:
    """Performs cell segmentation on the input timeseries."""
    with open_file(TIMESERIES_PATH, mode="r") as f:
        timeseries = f["*.tif"][:50]

    frame = 11
    predictor, sam = util.get_sam_model(return_sam=True)

    image_embeddings = util.precompute_image_embeddings(predictor, timeseries, EMBEDDINGS_TRACKING_PATH)

    embedding_pca = compute_pca(image_embeddings["features"][frame])

    if use_mws:
        print("Run embedding segmentation ...")
        seg_mws, initial_seg = segment_instances_from_embeddings(
            predictor, image_embeddings=image_embeddings, i=frame, return_initial_segmentation=True,
            bias=0.0, distance_type="l2", verbose=2,
        )
    else:
        seg_mws = None
        initial_seg = None

    if use_sam:
        print("Run SAM prediction ...")
        seg_sam = segment_instances_sam(sam, timeseries[frame])
    else:
        seg_sam = None

    v = napari.Viewer()
    v.add_image(timeseries[frame])
    v.add_image(embedding_pca, scale=(8, 8), visible=False)

    if seg_mws is not None:
        v.add_labels(seg_mws)

    if initial_seg is not None:
        v.add_labels(initial_seg)

    if seg_sam is not None:
        v.add_labels(seg_sam)

    napari.run()


def cell_segmentation_3d() -> None:
    with open_file(TIMESERIES_PATH, mode="r") as f:
        timeseries = f["*.tif"][:50]

    predictor = util.get_sam_model()
    image_embeddings = util.precompute_image_embeddings(predictor, timeseries, EMBEDDINGS_TRACKING_PATH)

    seg = segment_instances_from_embeddings_3d(predictor, image_embeddings)

    v = napari.Viewer()
    v.add_image(timeseries)
    v.add_labels(seg)
    napari.run()


def main():
    # automatic segmentation for the data from Lucchi et al. (see 'sam_annotator_3d.py')
    # nucleus_segmentation(use_mws=True)
    nucleus_segmentation_3d()

    # automatic segmentation for data from the cell tracking challenge (see 'sam_annotator_tracking.py')
    # cell_segmentation(use_mws=True)
    # cell_segmentation_3d()


if __name__ == "__main__":
    main()
