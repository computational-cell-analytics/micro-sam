from elf.io import open_file

CHECKPOINT = "./for_decoder/em/vit_b/best.pt"
IMAGE_PATH = "/home/pape/.cache/micro_sam/sample_data/lucchi_pp.zip.unzip/Lucchi++/Test_In"
EMBEDDING_PATH = "./for_decoder/lucchi-embeddings-vitb.zarr"


def run_annotator():
    from micro_sam.instance_segmentation import get_custom_sam_model_with_decoder
    from micro_sam.sam_annotator import annotator_3d

    with open_file(IMAGE_PATH, "r") as f:
        image = f["*.png"][:]

    predictor, decoder = get_custom_sam_model_with_decoder(CHECKPOINT, model_type="vit_b")

    annotator_3d(image, EMBEDDING_PATH, predictor=predictor, decoder=decoder, precompute_amg_state=True)


def main():
    run_annotator()


if __name__ == "__main__":
    main()
