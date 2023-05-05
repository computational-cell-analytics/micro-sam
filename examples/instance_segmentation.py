import micro_sam.util as util
import napari

from elf.io import open_file
from micro_sam.embedding_instance_segmentation import automatic_instance_segmentation
from micro_sam.visualization import compute_pca


def mito_segmentation():
    input_path = "./data/Lucchi++/Test_In"
    with open_file(input_path) as f:
        raw = f["*.png"][-1, :768, :768]

    predictor = util.get_sam_model()
    image_embeddings = util.precompute_image_embeddings(predictor, raw, "./embeddings/embeddings-mito2d.zarr")
    embedding_pca = compute_pca(image_embeddings["features"])

    seg, initial_seg = automatic_instance_segmentation(
        predictor, image_embeddings, return_initial_seg=True
    )

    v = napari.Viewer()
    v.add_image(raw)
    v.add_image(embedding_pca, scale=(12, 12))
    v.add_labels(seg)
    v.add_labels(initial_seg)
    napari.run()


def main():
    mito_segmentation()
    # TODO set up cell segmentation, e.g. on CTC Hela data or some IncuCyte data
    # cell_segmentation()


if __name__ == "__main__":
    main()
