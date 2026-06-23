import os

from elf.io import open_file
from micro_sam.sam_annotator import annotator_tracking
from micro_sam.sample_data import fetch_tracking_example_data
from micro_sam.util import get_cache_directory
from micro_sam.v2.util import DEFAULT_MODEL

DATA_CACHE = os.path.join(get_cache_directory(), "sample_data")
EMBEDDING_CACHE = os.path.join(get_cache_directory(), "embeddings")
os.makedirs(EMBEDDING_CACHE, exist_ok=True)


def track_ctc_data():
    """Run interactive tracking for data from the cell tracking challenge.
    """
    # download the example data
    example_data = fetch_tracking_example_data(DATA_CACHE)
    # load the example data (load the sequence of tif files as timeseries)
    with open_file(example_data, mode="r") as f:
        timeseries = f["*.tif"]

    embedding_path = os.path.join(EMBEDDING_CACHE, f"embeddings-ctc-{DEFAULT_MODEL}.zarr")

    # start the annotator with cached embeddings
    annotator_tracking(
        timeseries, embedding_path=embedding_path, model_type=DEFAULT_MODEL,
        precompute_amg_state=True,
    )


def main():
    track_ctc_data()


# The corresponding CLI call for track_ctc_data:
# (replace with cache directory on your machine)
# $ micro_sam.annotator_tracking -i <cache>/sample_data/DIC-C2DH-HeLa.zip.unzip/DIC-C2DH-HeLa/01 -k "*.tif" \
#   -e <cache>/embeddings/embeddings-ctc-hvit_t_cells.zarr
if __name__ == "__main__":
    main()
