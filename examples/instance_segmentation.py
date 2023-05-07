import micro_sam.util as util
import napari

from elf.io import open_file
from micro_sam.segment_instances import segment_from_embeddings
from micro_sam.visualization import compute_pca


def mito_segmentation():
    input_path = "./data/Lucchi++/Test_In"
    with open_file(input_path) as f:
        raw = f["*.png"][-1, :768, :768]

    predictor = util.get_sam_model()
    image_embeddings = util.precompute_image_embeddings(predictor, raw, "./embeddings/embeddings-mito2d.zarr")
    embedding_pca = compute_pca(image_embeddings["features"])

    seg, initial_seg = segment_from_embeddings(predictor, image_embeddings=image_embeddings, return_initial_seg=True)

    v = napari.Viewer()
    v.add_image(raw)
    v.add_image(embedding_pca, scale=(12, 12))
    v.add_labels(seg)
    v.add_labels(initial_seg)
    napari.run()


def cell_segmentation():
    path = "./DIC-C2DH-HeLa/train/01"
    with open_file(path, mode="r") as f:
        timeseries = f["*.tif"][:50]

    frame = 11

    predictor = util.get_sam_model()
    image_embeddings = util.precompute_image_embeddings(predictor, timeseries, "./embeddings/embeddings-ctc.zarr")
    embedding_pca = compute_pca(image_embeddings["features"][frame])

    seg, initial_seg = segment_from_embeddings(
        predictor, image_embeddings=image_embeddings, i=frame, return_initial_seg=True
    )

    v = napari.Viewer()
    v.add_image(timeseries[frame])
    v.add_image(embedding_pca, scale=(8, 8))
    v.add_labels(seg)
    v.add_labels(initial_seg)
    napari.run()


def main():
    # automatic segmentation for the data from Lucchi et al. (see 'sam_annotator_3d.py')
    # mito_segmentation()

    # automatic segmentation for data from the cell tracking challenge (see 'sam_annotator_tracking.py')
    cell_segmentation()


if __name__ == "__main__":
    main()
