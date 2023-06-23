from micro_sam.sam_annotator import image_folder_annotator


def main():
    image_folder_annotator(
        "./data/series", "./segmented-series",
        embedding_path="./embeddings/series-embeddings",
        pattern="*.tif", model_type="vit_b"
    )


if __name__ == "__main__":
    main()
