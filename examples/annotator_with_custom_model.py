import h5py
import micro_sam.sam_annotator as annotator
from micro_sam.util import get_sam_model

# TODO add an example for the 2d annotator with a custom model


def annotator_3d_with_custom_model():
    with h5py.File("./data/gut1_block_1.h5") as f:
        raw = f["raw"][:]

    custom_model = "/home/pape/Work/data/models/sam/user-study/vit_h_nuclei_em_finetuned.pt"
    embedding_path = "./embeddings/nuclei3d-custom-vit-h.zarr"
    predictor = get_sam_model(checkpoint_path=custom_model, model_type="vit_h")
    annotator.annotator_3d(raw, embedding_path, predictor=predictor)


def main():
    annotator_3d_with_custom_model()


if __name__ == "__main__":
    main()
