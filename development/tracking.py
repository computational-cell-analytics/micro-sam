from glob import glob

import numpy as np
from elf.io import open_file
from micro_sam.sam_annotator import annotator_tracking


def debug_tracking(timeseries, embedding_path):
    import micro_sam.util as util
    from micro_sam.sam_annotator.annotator_tracking import _track_from_prompts

    predictor = util.get_sam_model()
    image_embeddings = util.precompute_image_embeddings(predictor, timeseries, embedding_path)

    # seg = np.zeros(timeseries.shape, dtype="uint32")
    seg = np.load("./seg.npy")
    assert seg.shape == timeseries.shape
    slices = np.array([0])
    stop_upper = False

    _track_from_prompts(seg, predictor, slices, image_embeddings, stop_upper, threshold=0.5, projection="bounding_box")


def load_data():
    pattern = "/home/pape/Work/data/incu_cyte/carmello/videos/MiaPaCa_flat_B3-3_registered/image-*"
    paths = glob(pattern)
    paths.sort()

    timeseries = []
    for p in paths[:45]:
        with open_file(p, mode="r") as f:
            timeseries.append(f["phase-contrast"][:])
    timeseries = np.stack(timeseries)
    return timeseries


def main():
    timeseries = load_data()
    embedding_path = "./embeddings/embeddings-tracking.zarr"

    # _check_tracking(timeseries, embedding_path)
    annotator_tracking(timeseries, embedding_path=embedding_path)


if __name__ == "__main__":
    main()
