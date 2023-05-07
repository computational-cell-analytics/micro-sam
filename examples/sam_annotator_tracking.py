from glob import glob

import numpy as np
from elf.io import open_file
from micro_sam.sam_annotator import annotator_tracking


def track_incucyte_data():
    pattern = "/home/pape/Work/data/incu_cyte/carmello/videos/MiaPaCa_flat_B3-3_registered/image-*"
    paths = glob(pattern)
    paths.sort()

    timeseries = []
    for p in paths[:45]:
        with open_file(p, mode="r") as f:
            timeseries.append(f["phase-contrast"][:])
    timeseries = np.stack(timeseries)

    annotator_tracking(timeseries, embedding_path="./embeddings/embeddings-tracking.zarr", show_embeddings=False)


# TODO describe how to get the data from CTC
def track_ctc_data():
    path = "./data/DIC-C2DH-HeLa/train/01"
    with open_file(path, mode="r") as f:
        timeseries = f["*.tif"][:50]

    annotator_tracking(timeseries, embedding_path="./embeddings/embeddings-ctc.zarr")


def main():
    # private data used for initial tests
    # track_incucyte_data()

    # data from the cell tracking challenges
    track_ctc_data()


if __name__ == "__main__":
    main()
