import os

from micro_sam.evaluation import precompute_all_embeddings

from util import get_paths  # comment this and create a custom function with the same name to execute on your data
from util import get_model, get_default_arguments
from micro_sam.util import get_sam_model


def main():
    args = get_default_arguments()

    predictor = get_sam_model(model_type=args.model, checkpoint_path=args.checkpoint, lora_rank=args.lora_rank)
    embedding_dir = os.path.join(args.experiment_folder, "embeddings")
    os.makedirs(embedding_dir, exist_ok=True)

    # getting the embeddings for the val set
    image_paths, _ = get_paths(args.dataset, "val")
    precompute_all_embeddings(predictor, image_paths, embedding_dir)

    # getting the embeddings for the test set
    image_paths, _ = get_paths(args.dataset, "test")
    precompute_all_embeddings(predictor, image_paths, embedding_dir)


if __name__ == "__main__":
    main()
