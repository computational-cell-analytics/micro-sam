import imageio
from micro_sam.sam_annotator import annotator_2d


# TODO describe how to get the data and don't use hard-coded system path
def livecell_annotator():
    im = imageio.imread(
        "/home/pape/Work/data/incu_cyte/livecell/images/livecell_test_images/A172_Phase_C7_1_01d04h00m_4.tif"
    )
    embedding_path = "./embeddings/embeddings-livecell_cropped.zarr"
    annotator_2d(im, embedding_path, show_embeddings=True)


def main():
    # 2d annotator for livecell data
    livecell_annotator()

    # TODO
    # 2d annotator for cell tracking challenge hela data
    # hela_2d_annotator()


if __name__ == "__main__":
    main()
