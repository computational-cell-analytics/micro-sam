import os

import imageio.v3 as imageio
from micro_sam.sam_annotator import annotator
from micro_sam.util import get_cache_directory
from micro_sam.sample_data import fetch_3d_example_data

DATA_CACHE = os.path.join(get_cache_directory(), "sample_data")
EMBEDDING_CACHE = os.path.join(get_cache_directory(), "embeddings")
os.makedirs(EMBEDDING_CACHE, exist_ok=True)


def em_3d_annotator(model_type):
    """Run the 3d annotator for an example EM volume of mitochondria."""
    # Download the example volume and load it.
    raw = imageio.imread(fetch_3d_example_data(DATA_CACHE))

    # Start the annotator and cache the embeddings and the automatic segmentation state.
    embedding_path = os.path.join(EMBEDDING_CACHE, f"embeddings-lucchi-{model_type}.zarr")
    annotator(raw, embedding_path=embedding_path, model_type=model_type, precompute_autoseg_state=True)


def main():
    # The micro-sam2 model for mitochondria in EM. Use 'hvit_t_cells' for light microscopy data.
    model_type = "hvit_s_em_mito"
    em_3d_annotator(model_type)


# The corresponding CLI call for em_3d_annotator:
# (replace <cache> with the cache directory on your machine)
# $ micro_sam annotator segmentation -i <cache>/sample_data/lucchi-test-volume.tif \
#   -m hvit_s_em_mito -e <cache>/embeddings/embeddings-lucchi-hvit_s_em_mito.zarr
if __name__ == "__main__":
    main()
