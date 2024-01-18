import os

from elf.io import open_file
from micro_sam.sam_annotator import annotator_3d
from micro_sam.sample_data import fetch_3d_example_data
from micro_sam.util import get_cache_directory

DATA_CACHE = os.path.join(get_cache_directory(), "sample_data")
EMBEDDING_CACHE = os.path.join(get_cache_directory(), "embeddings")
os.makedirs(EMBEDDING_CACHE, exist_ok=True)


def em_3d_annotator(finetuned_model):
    """Run the 3d annotator for an example EM volume."""
    # download the example data
    example_data = fetch_3d_example_data(DATA_CACHE)
    # load the example data (load the sequence of tif files as 3d volume)
    with open_file(example_data) as f:
        raw = f["*.png"][:]

    if not finetuned_model:
        embedding_path = os.path.join(EMBEDDING_CACHE, "embeddings-lucchi.zarr")
        model_type = "vit_h"
    else:
        assert finetuned_model in ("organelles", "boundaries")
        embedding_path = os.path.join(EMBEDDING_CACHE, f"embeddings-lucchi-vit_b_em_{finetuned_model}.zarr")
        model_type = f"vit_b_em_{finetuned_model}"
        print(embedding_path)

    # start the annotator, cache the embeddings
    annotator_3d(raw, embedding_path, model_type=model_type)


def main():
    # Whether to use the fine-tuned SAM model for mitochondria (organelles) or boundaries.
    # valid choices are:
    # - None / False (will use the vanilla model)
    # - "organelles": will use the model for mitochondria and other organelles
    # - "boundaries": will use the model for boundary based structures
    finetuned_model = "boundaries"

    em_3d_annotator(finetuned_model)


# The corresponding CLI call for em_3d_annotator:
# (replace with cache directory on your machine)
# $ micro_sam.annotator_3d -i /home/pape/.cache/micro_sam/sample_data/lucchi_pp.zip.unzip/Lucchi++/Test_In -k *.png -e /home/pape/.cache/micro_sam/embeddings/embeddings-lucchi.zarr
if __name__ == "__main__":
    main()
