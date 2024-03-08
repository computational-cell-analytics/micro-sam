import imageio.v3 as imageio

CHECKPOINT = "./for_decoder/lm/vit_b/best.pt"
IMAGE_PATH = "/home/pape/Work/data/incu_cyte/livecell/images/livecell_train_val_images/A172_Phase_A7_1_02d00h00m_1.tif"
EMBEDDING_PATH = "./for_decoder/A172_Phase_A7_1_02d00h00m_1.zarr"


def segment_with_decoder():
    import napari
    from micro_sam.instance_segmentation import (
        get_custom_sam_model_with_decoder,
        mask_data_to_segmentation,
        InstanceSegmentationWithDecoder,
    )
    from micro_sam.util import precompute_image_embeddings

    predictor, decoder = get_custom_sam_model_with_decoder(CHECKPOINT, model_type="vit_b")
    segmenter = InstanceSegmentationWithDecoder(predictor, decoder)

    image = imageio.imread(IMAGE_PATH)
    image_embeddings = precompute_image_embeddings(
        segmenter._predictor, image, EMBEDDING_PATH,
    )

    print("Start segmentation ...")
    segmenter.initialize(image, image_embeddings)
    masks = segmenter.generate(output_mode="binary_mask")
    segmentation = mask_data_to_segmentation(masks, with_background=True)
    print("Segmentation done")

    v = napari.Viewer()
    v.add_image(image)
    # v.add_image(segmenter._foreground)
    v.add_labels(segmentation)
    napari.run()


def run_annotator():
    from micro_sam.instance_segmentation import get_custom_sam_model_with_decoder
    from micro_sam.sam_annotator import annotator_2d

    predictor, decoder = get_custom_sam_model_with_decoder(CHECKPOINT, model_type="vit_b")
    image = imageio.imread(IMAGE_PATH)

    annotator_2d(image, EMBEDDING_PATH, predictor=predictor, decoder=decoder, precompute_amg_state=True)


def main():
    # segment_with_decoder()
    run_annotator()


if __name__ == "__main__":
    main()
