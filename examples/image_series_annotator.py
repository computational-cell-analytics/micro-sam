import os

from micro_sam.v2.util import DEFAULT_MODEL
from micro_sam.util import get_cache_directory
from micro_sam.sam_annotator import image_folder_annotator
from micro_sam.sample_data import fetch_image_series_example_data

DATA_CACHE = os.path.join(get_cache_directory(), "sample_data")
EMBEDDING_CACHE = os.path.join(get_cache_directory(), "embeddings")
os.makedirs(EMBEDDING_CACHE, exist_ok=True)


def series_annotation(model_type):
    """Annotate a series of images, one after the other. The example data has three images."""
    example_data = fetch_image_series_example_data(DATA_CACHE)
    image_folder_annotator(
        example_data,
        output_folder="./series-segmentation-result",
        pattern="*.tif",
        embedding_path=os.path.join(EMBEDDING_CACHE, f"series-embeddings-{model_type}"),
        model_type=model_type,
        precompute_autoseg_state=True,
    )


def main():
    series_annotation(DEFAULT_MODEL)


# The corresponding CLI call for series_annotation:
# (replace <cache> with the cache directory on your machine)
# $ micro_sam annotator batch -i <cache>/sample_data/image-series.zip.unzip/series/ -p "*.tif" \
#   -e <cache>/embeddings/series-embeddings-hvit_t_cells/ -o series-segmentation-result
if __name__ == "__main__":
    main()
