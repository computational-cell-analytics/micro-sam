import os

import imageio.v3 as imageio
from micro_sam.sam_annotator import annotator_2d
from micro_sam.sample_data import fetch_hela_2d_example_data, fetch_livecell_example_data, fetch_wholeslide_example_data
from micro_sam.util import get_cache_directory

DATA_CACHE = os.path.join(get_cache_directory(), "sample_data")
EMBEDDING_CACHE = os.path.join(get_cache_directory(), "embeddings")
os.makedirs(EMBEDDING_CACHE, exist_ok=True)


def livecell_annotator(use_finetuned_model):
    """Run the 2d annotator for an example image from the LiveCELL dataset.

    See https://doi.org/10.1038/s41592-021-01249-6 for details on the data.
    """
    example_data = fetch_livecell_example_data(DATA_CACHE)
    image = imageio.imread(example_data)

    if use_finetuned_model:
        embedding_path = os.path.join(EMBEDDING_CACHE, "embeddings-livecell-vit_b_lm.zarr")
        model_type = "vit_b_lm"
    else:
        embedding_path = os.path.join(EMBEDDING_CACHE, "embeddings-livecell.zarr")
        model_type = "vit_h"

    annotator_2d(image, embedding_path, model_type=model_type, precompute_amg_state=True)


def hela_2d_annotator(use_finetuned_model):
    """Run the 2d annotator for an example image form the cell tracking challenge HeLa 2d dataset.
    """
    example_data = fetch_hela_2d_example_data(DATA_CACHE)
    image = imageio.imread(example_data)

    if use_finetuned_model:
        embedding_path = os.path.join(EMBEDDING_CACHE, "embeddings-hela2d-vit_b_lm.zarr")
        model_type = "vit_b_lm"
    else:
        embedding_path = os.path.join(EMBEDDING_CACHE, "embeddings-hela2d.zarr")
        model_type = "vit_h"

    annotator_2d(image, embedding_path, model_type=model_type)


def wholeslide_annotator(use_finetuned_model):
    """Run the 2d annotator with tiling for an example whole-slide image from the
    NeuRIPS cell segmentation challenge.

    See https://neurips22-cellseg.grand-challenge.org/ for details on the data.
    """
    example_data = fetch_wholeslide_example_data(DATA_CACHE)
    image = imageio.imread(example_data)

    if use_finetuned_model:
        embedding_path = os.path.join(EMBEDDING_CACHE, "whole-slide-embeddings-vit_b_lm.zarr")
        model_type = "vit_b_lm"
    else:
        embedding_path = os.path.join(EMBEDDING_CACHE, "whole-slide-embeddings.zarr")
        model_type = "vit_h"

    annotator_2d(image, embedding_path, tile_shape=(1024, 1024), halo=(256, 256), model_type=model_type)


def main():
    # Whether to use the fine-tuned SAM model for light microscopy data.
    use_finetuned_model = False

    # 2d annotator for livecell data
    livecell_annotator(use_finetuned_model)

    # 2d annotator for cell tracking challenge hela data
    # hela_2d_annotator(use_finetuned_model)

    # 2d annotator for a whole slide image
    # wholeslide_annotator(use_finetuned_model)


# The corresponding CLI call for hela_2d_annotator:
# (replace with cache directory on your machine)
# $ micro_sam.annotator_2d -i /home/pape/.cache/micro_sam/sample_data/hela-2d-image.png -e /home/pape/.cache/micro_sam/embeddings/embeddings-hela2d.zarr
if __name__ == "__main__":
    main()
