import imageio.v3 as imageio
from micro_sam.sam_annotator import annotator_2d
from micro_sam.sample_data import fetch_hela_2d_example_data, fetch_livecell_example_data, fetch_wholeslide_example_data


def livecell_annotator(use_finetuned_model):
    """Run the 2d annotator for an example image from the LiveCELL dataset.

    See https://doi.org/10.1038/s41592-021-01249-6 for details on the data.
    """
    example_data = fetch_livecell_example_data("./data")
    image = imageio.imread(example_data)

    if use_finetuned_model:
        embedding_path = "./embeddings/embeddings-livecell-vit_h_lm.zarr"
        model_type = "vit_h_lm"
    else:
        embedding_path = "./embeddings/embeddings-livecell.zarr"
        model_type = "vit_h"

    annotator_2d(image, embedding_path, show_embeddings=False, model_type=model_type)


def hela_2d_annotator(use_finetuned_model):
    """Run the 2d annotator for an example image form the cell tracking challenge HeLa 2d dataset.
    """
    example_data = fetch_hela_2d_example_data("./data")
    image = imageio.imread(example_data)

    if use_finetuned_model:
        embedding_path = "./embeddings/embeddings-hela2d-vit_h_lm.zarr"
        model_type = "vit_h_lm"
    else:
        embedding_path = "./embeddings/embeddings-hela2d.zarr"
        model_type = "vit_h"

    annotator_2d(image, embedding_path, show_embeddings=False, model_type=model_type)


def wholeslide_annotator(use_finetuned_model):
    """Run the 2d annotator with tiling for an example whole-slide image from the
    NeuRIPS cell segmentation challenge.

    See https://neurips22-cellseg.grand-challenge.org/ for details on the data.
    """
    example_data = fetch_wholeslide_example_data("./data")
    image = imageio.imread(example_data)

    if use_finetuned_model:
        embedding_path = "./embeddings/whole-slide-embeddings-vit_h_lm.zarr"
        model_type = "vit_h_lm"
    else:
        embedding_path = "./embeddings/whole-slide-embeddings.zarr"
        model_type = "vit_h"

    annotator_2d(image, embedding_path, tile_shape=(1024, 1024), halo=(256, 256), model_type=model_type)


def main():
    # whether to use the fine-tuned SAM model
    # this feature is still experimental!
    use_finetuned_model = False

    # 2d annotator for livecell data
    # livecell_annotator(use_finetuned_model)

    # 2d annotator for cell tracking challenge hela data
    # hela_2d_annotator(use_finetuned_model)

    # 2d annotator for a whole slide image
    wholeslide_annotator(use_finetuned_model)


if __name__ == "__main__":
    main()
