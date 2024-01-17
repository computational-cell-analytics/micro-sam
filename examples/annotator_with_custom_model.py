import os

import imageio
import h5py
import micro_sam.sam_annotator as annotator

from micro_sam.util import get_sam_model
from micro_sam.util import get_cache_directory
from micro_sam.sample_data import fetch_hela_2d_example_data


DATA_CACHE = os.path.join(get_cache_directory(), "sample_data")


def annotator_2d_with_custom_model():
    example_data = fetch_hela_2d_example_data(DATA_CACHE)
    image = imageio.imread(example_data)

    custom_model = "/home/pape/Downloads/exported_models/vit_b_lm.pth"
    predictor = get_sam_model(checkpoint_path=custom_model, model_type="vit_b")
    annotator.annotator_2d(image, predictor=predictor)


def annotator_3d_with_custom_model():
    with h5py.File("./data/gut1_block_1.h5") as f:
        raw = f["raw"][:]

    custom_model = "/home/pape/Work/data/models/sam/user-study/vit_h_nuclei_em_finetuned.pt"
    embedding_path = "./embeddings/nuclei3d-custom-vit-h.zarr"
    predictor = get_sam_model(checkpoint_path=custom_model, model_type="vit_h")
    annotator.annotator_3d(raw, embedding_path, predictor=predictor)


def main():
    annotator_2d_with_custom_model()
    # annotator_3d_with_custom_model()


if __name__ == "__main__":
    main()
