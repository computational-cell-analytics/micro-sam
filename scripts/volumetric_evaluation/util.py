import argparse
import warnings


def _get_default_args(input_path):
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_path", type=str, default=input_path, help="Path to volume."
    )
    parser.add_argument(
        "-m", "--model_type", type=str, default="vit_b", help="Name of the image encoder."
    )
    parser.add_argument(
        "-c", "--checkpoint", type=str, default=None, help="The custom checkpoint path."
    )
    parser.add_argument(
        "-e", "--embedding_path", type=str, default=None, help="Path to save embeddings."
    )
    parser.add_argument(
        "--resdir", type=str, required=True, help="Path to save the results."
    )
    parser.add_argument(
        "--ais", action="store_true", help="Whether to perforn 3d ais."
    )
    parser.add_argument(
        "--int", action="store_true", help="Whether to perform 3d interactive instance segmentation."
    )
    parser.add_argument(
        "--species", type=str, default=None, help="Relevant for MitoEM."
    )
    args = parser.parse_args()
    return args
