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
        embedding_path = os.path.join(EMBEDDING_CACHE, "embeddings-lucchi-vit_b_em_organelles.zarr")
        model_type = "vit_b_em_organelles"
        precompute_amg_state = True
    else:
        embedding_path = os.path.join(EMBEDDING_CACHE, "embeddings-lucchi.zarr")
        model_type = "vit_h"
        precompute_amg_state = False

    # start the annotator, cache the embeddings
    annotator_3d(raw, embedding_path, model_type=model_type, precompute_amg_state=precompute_amg_state)


def main():
    # Whether to use the fine-tuned SAM model for mitochondria (organelles).
    use_finetuned_model = True

    em_3d_annotator(use_finetuned_model)


# The corresponding CLI call for em_3d_annotator:
# (replace with cache directory on your machine)
# $ micro_sam.annotator_3d -i /home/pape/.cache/micro_sam/sample_data/lucchi_pp.zip.unzip/Lucchi++/Test_In -k *.png -e /home/pape/.cache/micro_sam/embeddings/embeddings-lucchi.zarr
if __name__ == "__main__":
    main()
