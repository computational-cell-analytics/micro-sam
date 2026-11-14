"""Export a model to the bioimage.io format.

The exported micro-sam2 model does interactive segmentation with box, point and mask prompts. If the model has
a decoder, it also does automatic instance segmentation when it gets only the image.
"""
import os

import imageio.v3 as imageio

from micro_sam.sample_data import fetch_livecell_example_data, synthetic_data


def export_micro_sam2_model():
    """Export the finetuned micro-sam2 model 'hvit_t_cells' with an image of the LIVECell dataset."""
    from micro_sam.v2.bioimageio import export_sam2_model

    image = imageio.imread(fetch_livecell_example_data("./data"))
    # The model has a decoder, so the prompts of the test come from its automatic segmentation.
    # For a model without a decoder, pass the segmentation of the image instead of None.
    export_sam2_model(
        image, None, model_type="hvit_t_cells", name="micro-sam2 hvit_t_cells", output_path="./hvit_t_cells.zip",
    )


def export_own_micro_sam2_model():
    """Export your own micro-sam2 model, for example the result of 'micro_sam train'."""
    from micro_sam.v2.bioimageio import export_sam2_model

    # The checkpoint of joint or automatic training holds the model and the decoder.
    checkpoint_path = "./checkpoints/my_model/best.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Train a model first, '{checkpoint_path}' does not exist.")
        return

    image = imageio.imread(fetch_livecell_example_data("./data"))
    export_sam2_model(
        image, None, model_type="my_model", name="my micro-sam2 model", output_path="./my_model.zip",
        checkpoint_path=checkpoint_path,
    )


def export_sam1_model():
    """Export a SAM1 model of micro-sam v1 with synthetic data."""
    from micro_sam.v1.bioimageio import export_sam_model

    image, labels = synthetic_data(shape=(1024, 1022))
    # Pass the path to your own finetuned SAM1 model here, or None to export the original weights.
    export_sam_model(
        image, labels, model_type="vit_t", name="sam-test-vit-t", output_path="./test_export.zip",
        checkpoint_path=None,
    )


def main():
    export_micro_sam2_model()
    # export_own_micro_sam2_model()
    # export_sam1_model()


if __name__ == "__main__":
    main()
