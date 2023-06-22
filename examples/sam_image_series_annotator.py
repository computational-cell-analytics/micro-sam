from micro_sam.sam_annotator import image_folder_annotator


def main():
    image_folder_annotator("./data/DIC-C2DH-HeLa/train/01", "./segmented-series", pattern="*.tif", model_type="vit_b")


if __name__ == "__main__":
    main()
