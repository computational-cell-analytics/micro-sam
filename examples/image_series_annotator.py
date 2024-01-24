import os

from micro_sam.sam_annotator import image_folder_annotator
from micro_sam.sample_data import fetch_image_series_example_data
from micro_sam.util import get_cache_directory

DATA_CACHE = os.path.join(get_cache_directory(), "sample_data")
EMBEDDING_CACHE = os.path.join(get_cache_directory(), "embeddings")
os.makedirs(EMBEDDING_CACHE, exist_ok=True)


def series_annotation(use_finetuned_model):
    """Annotate a series of images. Example runs for three different images.
    """

    if use_finetuned_model:
        embedding_path = os.path.join(EMBEDDING_CACHE, "series-embeddings-vit_b_lm")
        model_type = "vit_b_lm"
    else:
        embedding_path = os.path.join(EMBEDDING_CACHE, "series-embeddings")
        model_type = "vit_h"

    example_data = fetch_image_series_example_data(DATA_CACHE)
    image_folder_annotator(
        example_data, "./series-segmentation-result",
        pattern="*.tif",
        embedding_path=embedding_path,
        model_type=model_type,
        precompute_amg_state=False,
    )


def main():
    # Whether to use the fine-tuned SAM model.
    use_finetuned_model = False
    series_annotation(use_finetuned_model)


# The corresponding CLI call for track_ctc_data:
# (replace with cache directory on your machine)
# $ micro_sam.image_series_annotator -i /home/pape/.cache/micro_sam/sample_data/image-series.zip.unzip/series/ -e /home/pape/.cache/micro_sam/embeddings/series-embeddings/ -o segmentation_results
if __name__ == "__main__":
    main()
