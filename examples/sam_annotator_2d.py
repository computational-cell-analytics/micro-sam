import imageio
from micro_sam.sam_annotator import annotator_2d


def main():
    im = imageio.imread(
        "/home/pape/Work/data/incu_cyte/livecell/images/livecell_test_images/A172_Phase_C7_1_01d04h00m_4.tif"
    )
    embedding_path = "./embeddings/embeddings-livecell_cropped.zarr"
    annotator_2d(im, embedding_path)


if __name__ == "__main__":
    main()
