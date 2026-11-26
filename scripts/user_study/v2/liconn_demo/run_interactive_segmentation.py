"""Interactive segmentation of a LICONN block with micro-sam2 in napari.

Run it with '--precompute' one time on a GPU. This caches the embeddings and the decoder predictions.
Then run it without arguments to open the annotator. The annotator does not need a GPU.
Set '--data_root' and '--demo_root' as for 'run_automatic_segmentation.py'.

Usage:
    python run_interactive_segmentation.py --precompute
    python run_interactive_segmentation.py
    python run_interactive_segmentation.py --data_root /path/to/liconn --demo_root /path/to/liconn_demo --precompute
"""

import os
import argparse

from micro_sam.sam_annotator import annotator
from micro_sam.sam_annotator._state import AnnotatorState

from run_automatic_segmentation import MODEL_TYPE, add_path_arguments, load_block


def precompute(raw, embedding_path):
    state = AnnotatorState()
    state.image_shape = raw.shape
    state.initialize_predictor(
        raw, model_type=MODEL_TYPE, ndim=3, save_path=embedding_path, precompute_autoseg_state=True, use_cli=True
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precompute", action="store_true", help="Compute the embeddings. This needs a GPU.")
    add_path_arguments(parser)
    args = parser.parse_args()

    raw, _ = load_block(args.data_root)
    embedding_path = os.path.join(args.demo_root, "embeddings.zarr")
    if args.precompute:
        precompute(raw, embedding_path)
    else:
        annotator(raw, ndim=3, embedding_path=embedding_path, model_type=MODEL_TYPE)


if __name__ == "__main__":
    main()
