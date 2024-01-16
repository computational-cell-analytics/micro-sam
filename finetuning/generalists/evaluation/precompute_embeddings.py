import argparse
import os

from micro_sam.evaluation import precompute_all_embeddings
from util import get_paths, get_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-c", "--checkpoint", type=str, required=True)
    parser.add_argument("-e", "--experiment_folder", type=str, required=True)
    parser.add_argument("-d", "--dataset", type=str, required=True)
    args = parser.parse_args()

    predictor = get_model(model_type=args.model, ckpt=args.checkpoint)
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
