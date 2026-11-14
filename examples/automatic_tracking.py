import os

from elf.io import open_file

from micro_sam.v2.util import DEFAULT_MODEL
from micro_sam.util import get_cache_directory
from micro_sam.sample_data import fetch_tracking_example_data
from micro_sam.v2.automatic_segmentation import automatic_tracking, get_predictor_and_segmenter


DATA_CACHE = os.path.join(get_cache_directory(), "sample_data")


def example_automatic_tracking(model_type):
    """Run automatic tracking for data from the cell tracking challenge.

    Each frame is segmented with the decoder of the micro-sam2 model (AIS), and the objects are linked
    across frames with Trackastra.
    """
    # Download the example tracking data and load the sequence of tif files as a timeseries.
    example_data = fetch_tracking_example_data(DATA_CACHE)
    with open_file(example_data, mode="r") as f:
        timeseries = f["*.tif"][:]

    predictor, segmenter = get_predictor_and_segmenter(model_type=model_type, segmentation_mode="ais")

    # The tracking result is saved in the Cell Tracking Challenge format in 'output_path'.
    masks_tracked, _ = automatic_tracking(
        predictor=predictor, segmenter=segmenter, input_path=timeseries, output_path="./hela_ctc",
    )

    import napari
    v = napari.Viewer()
    v.add_image(timeseries)
    v.add_labels(masks_tracked)
    napari.run()


def main():
    example_automatic_tracking(DEFAULT_MODEL)


# The corresponding CLI call for example_automatic_tracking:
# (replace <cache> with the cache directory on your machine)
# $ micro_sam inference tracking -i <cache>/sample_data/DIC-C2DH-HeLa.zip.unzip/DIC-C2DH-HeLa/01 -k "*.tif" \
#   -o hela_ctc -m hvit_t_cells
if __name__ == "__main__":
    main()
