from elf.io import open_file
from micro_sam.sam_annotator import annotator_3d


# Lucchi++ Data from: https://casser.io/connectomics/
def main():
    input_path = "./data/Lucchi++/Test_In"
    with open_file(input_path) as f:
        raw = f["*.png"][:]
    embedding_path = "./embeddings/embeddings-lucchi.zarr"
    annotator_3d(raw, embedding_path, show_embeddings=False)


if __name__ == "__main__":
    main()
