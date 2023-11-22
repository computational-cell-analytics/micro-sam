import os

from elf.io import open_file
from micro_sam.sam_annotator import annotator_3d
from micro_sam.sample_data import fetch_3d_example_data
from micro_sam.util import get_cache_directory

DATA_CACHE = os.path.join(get_cache_directory(), "sample_data")
EMBEDDING_CACHE = os.path.join(get_cache_directory(), "embeddings")
os.makedirs(EMBEDDING_CACHE, exist_ok=True)


def em_3d_annotator(use_finetuned_model):
    """Run the 3d annotator for an example EM volume."""
    # download the example data
    example_data = fetch_3d_example_data(DATA_CACHE)
    # load the example data (load the sequence of tif files as 3d volume)
    with open_file(example_data) as f:
        raw = f["*.png"][:]

    if use_finetuned_model:
        embedding_path = os.path.join(EMBEDDING_CACHE, "embeddings-lucchi-vit_h_em.zarr")
        model_type = "vit_h_em"
    else:
        embedding_path = os.path.join(EMBEDDING_CACHE, "embeddings-lucchi.zarr")
        model_type = "vit_h"

    # start the annotator, cache the embeddings
    annotator_3d(raw, embedding_path, model_type=model_type, show_embeddings=False)


def main():
    # whether to use the fine-tuned SAM model
    # this feature is still experimental!
    use_finetuned_model = False

    em_3d_annotator(use_finetuned_model)


if __name__ == "__main__":
    main()
