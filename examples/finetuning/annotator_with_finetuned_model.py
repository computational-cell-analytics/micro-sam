import imageio.v3 as imageio

from micro_sam.sam_annotator import annotator


def run_annotator_with_finetuned_model():
    """Run the annotator with a custom (finetuned) micro-sam2 model.

    Here, we use the model that is produced by `finetune_hela.py` and apply it
    to an image from the validation set.
    """
    # Take the last frame, which is part of the val set, so the model was not directly trained on it.
    im = imageio.imread("./data/DIC-C2DH-HeLa.zip.unzip/DIC-C2DH-HeLa/01/t083.tif")

    # Set the checkpoint and the path for caching the embeddings.
    # The checkpoint of 'finetune_hela.py' has no decoder, so the annotator then offers only the automatic
    # segmentation without a decoder (AMG). Use './checkpoints/sam2_hela_instance_segmentation/best.pt',
    # the model of 'finetune_hela_instance_segmentation.py', to use its decoder for automatic segmentation.
    checkpoint_path = "./checkpoints/sam2_hela/best.pt"
    embedding_path = "./embeddings/embeddings-finetuned.zarr"

    # The model size is read from the checkpoint. The 'model_type' selects the model that provides
    # the weights that are not in the checkpoint, e.g. the prompt encoder for the decoder-only checkpoint.
    model_type = "hvit_t_cells"

    # Run the annotator with the custom model.
    annotator(im, model_type=model_type, embedding_path=embedding_path, checkpoint_path=checkpoint_path)


if __name__ == "__main__":
    run_annotator_with_finetuned_model()
