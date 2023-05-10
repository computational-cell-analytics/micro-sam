from elf.io import open_file
from micro_sam.sam_annotator import annotator_tracking


# This runs the interactive tracking annotator for data from the cell tracking challenge:
# It uses the training data for the HeLA dataset. You can download the data from
# http://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip
def track_ctc_data():
    path = "./data/DIC-C2DH-HeLa/train/01"
    with open_file(path, mode="r") as f:
        timeseries = f["*.tif"][:50]
    annotator_tracking(timeseries, embedding_path="./embeddings/embeddings-ctc.zarr")


def main():
    # run interactive tracking for data from the cell tracking challenge
    track_ctc_data()


if __name__ == "__main__":
    main()
