import argparse
import os

from micro_sam.evaluation import precompute_all_embeddings
from util import get_paths, get_model, get_experiment_folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True)
    args = parser.parse_args()

    name = args.name

    image_paths, _ = get_paths()
    predictor = get_model(name)
    exp_folder = get_experiment_folder(name)
    embedding_dir = os.path.join(exp_folder, "embeddings")
    os.makedirs(embedding_dir, exist_ok=True)
    precompute_all_embeddings(predictor, image_paths, embedding_dir)


if __name__ == "__main__":
    main()
