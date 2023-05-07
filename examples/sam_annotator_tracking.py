from elf.io import open_file
from micro_sam.sam_annotator import annotator_tracking


# TODO describe how to get the data from CTC
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
