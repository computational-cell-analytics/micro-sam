import os
from pathlib import Path

from elf.io import open_file
import pooch
from micro_sam.sam_annotator import annotator_tracking


def track_ctc_data():
    example_data_directory = "./data"
    with open_file(str(fetch_example_data(example_data_directory)), mode="r") as f:
        timeseries = f["*.tif"]
    annotator_tracking(timeseries, embedding_path="./embeddings/embeddings-ctc.zarr")


def fetch_example_data(save_directory):
    """Cell tracking challenge dataset DIC-C2DH-HeLa.

    Cell tracking challenge webpage: http://data.celltrackingchallenge.net
    HeLa cells on a flat glass
    Dr. G. van Cappellen. Erasmus Medical Center, Rotterdam, The Netherlands
    Training dataset: http://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip (37 MB)
    Challenge dataset: http://data.celltrackingchallenge.net/challenge-datasets/DIC-C2DH-HeLa.zip (41 MB)
    """
    save_directory = Path(save_directory)
    if not save_directory.exists():
        os.makedirs(save_directory)
        print("Created new folder for example data downloads.")
    print("Example data directory is:", save_directory.resolve())
    unpack_filenames = [os.path.join("DIC-C2DH-HeLa", "01", f"t{str(i).zfill(3)}.tif") for i in range(84)]
    unpack = pooch.Unzip(members=unpack_filenames)
    fnames = pooch.retrieve(
        url="http://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip",  # 37 MB
        known_hash="fac24746fa0ad5ddf6f27044c785edef36bfa39f7917da4ad79730a7748787af",
        fname="DIC-C2DH-HeLa.zip",
        path=save_directory,
        progressbar=True,
        processor=unpack,
    )
    cell_tracking_directory = save_directory.joinpath("DIC-C2DH-HeLa", "train", "01")
    return cell_tracking_directory


if __name__ == "__main__":
    # run interactive tracking for data from the cell tracking challenge
    track_ctc_data()
