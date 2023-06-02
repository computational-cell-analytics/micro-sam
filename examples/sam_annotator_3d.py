import os
from pathlib import Path

from elf.io import open_file
import pooch
from micro_sam.sam_annotator import annotator_3d


def main():
    example_data_directory = "./data"
    with open_file(fetch_example_data(example_data_directory)) as f:
        raw = f["*.png"][:]
    embedding_path = "./embeddings/embeddings-lucchi.zarr"
    annotator_3d(raw, embedding_path, show_embeddings=False)


def fetch_example_data(save_directory):
    # Lucchi++ Data from: https://casser.io/connectomics/
    save_directory = Path(save_directory)
    if not save_directory.exists():
        os.makedirs(save_directory)
        print("Created new folder for example data downloads.")
    print("Example data directory is:", save_directory.resolve())
    lucchi_filenames =[os.path.join("Lucchi++/Test_In/", f"mask{str(i).zfill(4)}.png") for i in range(165)]
    unpack = pooch.Unzip(members=lucchi_filenames)
    fnames = pooch.retrieve(
        url="http://www.casser.io/files/lucchi_pp.zip",
        known_hash="770ce9e98fc6f29c1b1a250c637e6c5125f2b5f1260e5a7687b55a79e2e8844d",
        fname="lucchi_pp.zip",
        path=save_directory,
        progressbar=True,
        processor=unpack,
    )
    lucchi_testin_dir = save_directory.joinpath("lucchi_pp.zip.unzip", "Lucchi++", "Test_In")
    return lucchi_testin_dir


if __name__ == "__main__":
    main()
