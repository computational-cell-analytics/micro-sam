import imageio.v3 as imageio

import micro_sam.util as util
from micro_sam.sam_annotator import annotator_2d


def run_annotator_with_finetuned_model():
    """Run the 2d anntator with a custom (finetuned) model.

    Here, we use the model that is produced by `finetuned_hela.py` and apply it
    for an image from the validation set.
    """
    # take the last frame, which is part of the val set, so the model was not directly trained on it
    im = imageio.imread("./data/DIC-C2DH-HeLa.zip.unzip/DIC-C2DH-HeLa/01/t083.tif")

    # set the checkpoint and the path for caching the embeddings
    checkpoint = "./finetuned_hela_model.pth"
    embedding_path = "./embeddings/embeddings-finetuned.zarr"

    model_type = "vit_b"  # We finetune a vit_b in the example script.
    # Adapt this if you finetune a different model type, e.g. vit_h.

    # Load the custom model.
    predictor = util.get_sam_model(model_type=model_type, checkpoint_path=checkpoint)

    # Run the 2d annotator with the custom model.
    annotator_2d(
        im, embedding_path=embedding_path, predictor=predictor, precompute_amg_state=True,
    )


if __name__ == "__main__":
    run_annotator_with_finetuned_model()
