from glob import glob

import h5py
import numpy as np
from micro_sam.sam_annotator import annotator_tracking


def main():
    pattern = "/home/pape/Work/data/incu_cyte/carmello/videos/MiaPaCa_flat_B3-3_registered/image-*"
    paths = glob(pattern)
    paths.sort()

    timeseries = []
    for p in paths[:45]:
        with h5py.File(p) as f:
            timeseries.append(f["phase-contrast"][:])
    timeseries = np.stack(timeseries)

    annotator_tracking(timeseries, embedding_path="./embeddings/embeddings-tracking.zarr", show_embeddings=False)


if __name__ == "__main__":
    main()
