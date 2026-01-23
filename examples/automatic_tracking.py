import os

from elf.io import open_file

from micro_sam.util import get_cache_directory
from micro_sam.sample_data import fetch_tracking_example_data
from micro_sam.automatic_segmentation import automatic_tracking, get_predictor_and_segmenter


DATA_CACHE = os.path.join(get_cache_directory(), "sample_data")
EMBEDDING_CACHE = os.path.join(get_cache_directory(), "embeddings")
os.makedirs(EMBEDDING_CACHE, exist_ok=True)


def example_automatic_tracking(use_finetuned_model):
    """Run automatic tracking for data from the cell tracking challenge.
    """
    # Download the example tracking data.
    example_data = fetch_tracking_example_data(DATA_CACHE)

    # Load the example data (load the sequence of tif files as timeseries)
    with open_file(example_data, mode="r") as f:
        timeseries = f["*.tif"]

    if use_finetuned_model:
        embedding_path = os.path.join(EMBEDDING_CACHE, "embeddings-ctc-vit_b_lm.zarr")
        model_type = "vit_b_lm"
    else:
        embedding_path = os.path.join(EMBEDDING_CACHE, "embeddings-ctc.zarr")
        model_type = "vit_h"

    predictor, segmenter = get_predictor_and_segmenter(model_type=model_type, segmentation_mode="ais")

    masks_tracked, _ = automatic_tracking(
        predictor=predictor,
        segmenter=segmenter,
        input_path=timeseries[:],
        output_path="./hela_ctc",
        embedding_path=embedding_path,
    )

    import napari
    v = napari.Viewer()
    v.add_image(timeseries)
    v.add_labels(masks_tracked)
    napari.run()


def main():
    # Whether to use the fine-tuned SAM model.
    use_finetuned_model = True
    example_automatic_tracking(use_finetuned_model)


if __name__ == "__main__":
    main()
