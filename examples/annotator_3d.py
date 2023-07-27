from elf.io import open_file
from micro_sam.sam_annotator import annotator_3d
from micro_sam.sample_data import fetch_3d_example_data


def em_3d_annotator():
    """Run the 3d annotator for an example EM volume."""
    # download the example data
    example_data = fetch_3d_example_data("./data")
    # load the example data (load the sequence of tif files as 3d volume)
    with open_file(example_data) as f:
        raw = f["*.png"][:]
    # start the annotator, cache the embeddings
    embedding_path = "./embeddings/embeddings-lucchi.zarr"
    annotator_3d(raw, embedding_path, show_embeddings=False)


def main():
    em_3d_annotator()


if __name__ == "__main__":
    main()
