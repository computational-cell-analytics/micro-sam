from elf.io import open_file
from micro_sam.sam_annotator import annotator_tracking
from micro_sam.sample_data import fetch_tracking_example_data


def track_ctc_data():
    """Run interactive tracking for data from the cell tracking challenge.
    """
    # download the example data
    example_data = fetch_tracking_example_data("./data")
    # load the example data (load the sequence of tif files as timeseries)
    with open_file(example_data, mode="r") as f:
        timeseries = f["*.tif"]
    # start the annotator with cached embeddings
    annotator_tracking(timeseries, embedding_path="./embeddings/embeddings-ctc.zarr", show_embeddings=False)


def main():
    track_ctc_data()


if __name__ == "__main__":
    main()
