import os

import imageio.v3 as imageio
from micro_sam.util import get_cache_directory
from micro_sam.sam_annotator import annotator
from micro_sam.sample_data import fetch_hela_2d_example_data, fetch_livecell_example_data, fetch_wholeslide_example_data


DATA_CACHE = os.path.join(get_cache_directory(), "sample_data")
EMBEDDING_CACHE = os.path.join(get_cache_directory(), "embeddings")
os.makedirs(EMBEDDING_CACHE, exist_ok=True)


def livecell_annotator():
    """Run the 2d annotator for an example image from the LIVEcell dataset.

    See https://doi.org/10.1038/s41592-021-01249-6 for details on the data.
    """
    example_data = fetch_livecell_example_data(DATA_CACHE)
    image = imageio.imread(example_data)

    embedding_path = os.path.join(EMBEDDING_CACHE, "embeddings-livecell-uSAM2.zarr")
    annotator(image, embedding_path=embedding_path, precompute_autoseg_state=True)


def hela_2d_annotator():
    """Run the 2d annotator for an example image from the Cell Tracking Challenge (HeLa 2d) dataset.
    """
    example_data = fetch_hela_2d_example_data(DATA_CACHE)
    image = imageio.imread(example_data)

    embedding_path = os.path.join(EMBEDDING_CACHE, "embeddings-hela2d-uSAM2.zarr")
    annotator(image, embedding_path=embedding_path)


def wholeslide_annotator():
    """Run the 2d annotator with tiling for an example whole-slide image from the
    NeurIPS Cell Segmentation challenge.

    See https://neurips22-cellseg.grand-challenge.org/ for details on the data.
    """
    example_data = fetch_wholeslide_example_data(DATA_CACHE)
    image = imageio.imread(example_data)

    embedding_path = os.path.join(EMBEDDING_CACHE, "whole-slide-embeddings-uSAM2.zarr")
    annotator(image, embedding_path=embedding_path, tile_shape=(1024, 1024), halo=(256, 256))


def main():
    # 2d annotator for livecell data
    livecell_annotator()

    # 2d annotator for cell tracking challenge hela data
    # hela_2d_annotator()

    # 2d annotator for a whole slide image
    # wholeslide_annotator()


# The corresponding CLI call for hela_2d_annotator:
# $ micro_sam annotator -i /home/pape/.cache/micro_sam/sample_data/hela-2d-image.png  # noqa
if __name__ == "__main__":
    main()
