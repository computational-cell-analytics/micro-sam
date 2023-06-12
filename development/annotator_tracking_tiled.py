from micro_sam.sam_annotator import annotator_tracking


# TODO find a suitable larger 2d dataset from the ctc for this
def annotator_with_tiling():
    timeseries = ""
    annotator_tracking(timeseries, embedding_path="./embeddings/embeddings-tiled_tracking.zarr",
                       tile_shape=(512, 512), halo=(64, 64))


def main():
    annotator_with_tiling()


main()
