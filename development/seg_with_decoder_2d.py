import imageio.v3 as imageio

CHECKPOINT = "./for_decoder/lm/vit_b/best.pt"
IMAGE_PATH = "/home/pape/Work/data/incu_cyte/livecell/images/livecell_train_val_images/A172_Phase_A7_1_02d00h00m_1.tif"
EMBEDDING_PATH = "./for_decoder/A172_Phase_A7_1_02d00h00m_1.zarr"


def segment_with_decoder(args):
    import napari
    from micro_sam.instance_segmentation import (
        get_custom_sam_model_with_decoder,
        mask_data_to_segmentation,
        InstanceSegmentationWithDecoder,
    )
    from micro_sam.util import precompute_image_embeddings

    checkpoint = CHECKPOINT if args.checkpoint_path is None else args.checkpoint_path
    model_type = "vit_b" if args.model_type is None else args.model_type
    image_path = IMAGE_PATH if args.input_path is None else args.input_path
    embedding_path = EMBEDDING_PATH if args.embedding_path is None else args.embedding_path

    predictor, decoder = get_custom_sam_model_with_decoder(checkpoint, model_type=model_type)
    segmenter = InstanceSegmentationWithDecoder(predictor, decoder)

    image = imageio.imread(image_path)
    image_embeddings = precompute_image_embeddings(
        segmenter._predictor, image, embedding_path,
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


def run_annotator(args):
    from micro_sam.instance_segmentation import get_custom_sam_model_with_decoder
    from micro_sam.sam_annotator import annotator_2d
    
    model_type = "vit_b" if args.model_type is None else args.model_type
    checkpoint = CHECKPOINT if args.checkpoint_path is None else args.checkpoint_path
    image_path = IMAGE_PATH if args.input_path is None else args.input_path
    embedding_path = EMBEDDING_PATH if args.embedding_path is None else args.embedding_path

    predictor, decoder = get_custom_sam_model_with_decoder(checkpoint, model_type=model_type)
    image = imageio.imread(image_path)

    annotator_2d(
        image,
        embedding_path,
        predictor=predictor,
        decoder=decoder,
        precompute_amg_state=True
    )


def main(args):
    # segment_with_decoder(args)
    run_annotator(args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str)
    parser.add_argument("-m", "--model_type", type=str)
    parser.add_argument("-e", "--embedding_path", type=str)
    parser.add_argument("-c", "--checkpoint_path", type=str)

    args = parser.parse_args()
    main(args)
