import z5py
from micro_sam.sam_annotator import annotator_3d


def annotator_with_tiling():
    with z5py.File("/home/pape/Work/data/cremi/sampleA.n5", "r") as f:
        raw = f["volumes/raw/s0"][:25]

    embedding_path = "./embeddings/embeddings-tiled_3d.zarr"
    annotator_3d(raw, embedding_path, tile_shape=(512, 512), halo=(64, 64))


def main():
    annotator_with_tiling()


main()
