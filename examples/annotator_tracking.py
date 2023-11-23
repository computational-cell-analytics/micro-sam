import os

from elf.io import open_file
from micro_sam.sam_annotator import annotator_tracking
from micro_sam.sample_data import fetch_tracking_example_data
from micro_sam.util import get_cache_directory

DATA_CACHE = os.path.join(get_cache_directory(), "sample_data")
EMBEDDING_CACHE = os.path.join(get_cache_directory(), "embeddings")
os.makedirs(EMBEDDING_CACHE, exist_ok=True)


def track_ctc_data(use_finetuned_model):
    """Run interactive tracking for data from the cell tracking challenge.
    """
    # download the example data
    example_data = fetch_tracking_example_data(DATA_CACHE)
    # load the example data (load the sequence of tif files as timeseries)
    with open_file(example_data, mode="r") as f:
        timeseries = f["*.tif"]

    if use_finetuned_model:
        embedding_path = os.path.join(EMBEDDING_CACHE, "embeddings-ctc-vit_h_lm.zarr")
        model_type = "vit_h_lm"
    else:
        embedding_path = os.path.join(EMBEDDING_CACHE, "embeddings-ctc.zarr")
        model_type = "vit_h"

    # start the annotator with cached embeddings
    annotator_tracking(timeseries, embedding_path=embedding_path, show_embeddings=False, model_type=model_type)


def main():
    # whether to use the fine-tuned SAM model
    # this feature is still experimental!
    use_finetuned_model = False
    track_ctc_data(use_finetuned_model)


if __name__ == "__main__":
    main()
