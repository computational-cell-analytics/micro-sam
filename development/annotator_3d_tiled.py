import z5py
from micro_sam.sam_annotator import annotator_3d


def annotator_with_tiling():
    with z5py.File("/home/pape/Work/data/cremi/sampleA.n5", "r") as f:
        raw = f["volumes/raw/s0"][:25]
    embedding_path = "./embeddings/embeddings-tiled_3d.zarr"
    annotator_3d(raw, embedding_path, tile_shape=(512, 512), halo=(64, 64))


def segment_tiled():
    import micro_sam.util as util
    from micro_sam.segment_instances import segment_instances_from_embeddings_3d

    with z5py.File("/home/pape/Work/data/cremi/sampleA.n5", "r") as f:
        raw = f["volumes/raw/s0"][:25]
    embedding_path = "./embeddings/embeddings-tiled_3d.zarr"

    predictor = util.get_sam_model()
    image_embeddings = util.precompute_image_embeddings(
        predictor, raw, embedding_path, tile_shape=(512, 512), halo=(64, 64)
    )
    segment_instances_from_embeddings_3d(predictor, image_embeddings)


def main():
    # annotator_with_tiling()
    segment_tiled()


main()
