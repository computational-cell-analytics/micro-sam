from micro_sam.sam_annotator import image_folder_annotator
from micro_sam.sample_data import fetch_image_series_example_data


def series_annotation():
    """Annotate a series of images. Example runs for three different example images.
    """
    example_data = fetch_image_series_example_data("./data")
    image_folder_annotator(
        example_data, "./data/series-segmentation-result", embedding_path="./embeddings/series-embeddings",
        pattern="*.tif", model_type="vit_b"
    )


def main():
    series_annotation()


if __name__ == "__main__":
    main()
