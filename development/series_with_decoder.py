import os

from micro_sam.sam_annotator import image_folder_annotator
from micro_sam.instance_segmentation import get_custom_sam_model_with_decoder
from micro_sam.sample_data import fetch_image_series_example_data
from micro_sam.util import get_cache_directory

DATA_CACHE = os.path.join(get_cache_directory(), "sample_data")
CHECKPOINT = "./for_decoder/lm/vit_b/best.pt"


def main():
    predictor, decoder = get_custom_sam_model_with_decoder(CHECKPOINT, model_type="vit_b")
    example_data = fetch_image_series_example_data(DATA_CACHE)

    image_folder_annotator(
        example_data, "./for_decoder/series-segmentation-result",
        pattern="*.tif",
        embedding_path="./for_decoder/embeddings.zarr",
        model_type="vit_b",
        decoder=decoder,
        precompute_amg_state=True,
    )


if __name__ == "__main__":
    main()
