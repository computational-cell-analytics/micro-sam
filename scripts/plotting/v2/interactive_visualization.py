"""Interactive 3d segmentation with 'hvit_t_cells' in the micro-sam annotator on LICONN.

The beke volumes moved to 'scripts/user_study/v2/explore_huisken_data.py' ('--embryo').

Usage:
    python interactive_visualization.py --view  # only the raw volume in napari
    python interactive_visualization.py --precompute  # the embeddings, over all visible GPUs
    python interactive_visualization.py  # the annotator
    python interactive_visualization.py --slice_norm  # each slice normalized by its own percentiles
"""

import os
import argparse

import zarr
import numpy as np

import napari

from micro_sam.sam_annotator import Annotator
from micro_sam.v2.normalization import normalize_raw
from micro_sam.sam_annotator._state import AnnotatorState
from micro_sam.sam_annotator._titles import get_dock_title
from micro_sam.sam_annotator.util import _sync_embedding_widget
from micro_sam.v2.util import get_sam2_model, precompute_image_embeddings


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"
EMBEDDING_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/interactive_visualization"

# The torch-em store at 18x18x24 nm, the resolution the models were trained on.
LICONN_ZARR_PATH = os.path.join(DATA_ROOT, "liconn", "liconn.zarr")
# ZYX, centred in the proofread region ('LICONN_ROI' of the generalist loader).
LICONN_ROI = (slice(288, 416), slice(2048, 2560), slice(1656, 2168))


def get_roi_name(roi):
    return "_".join(f"{axis}{r.start}-{r.stop}" for axis, r in zip("zyx", roi))


def load_volume():
    volume = zarr.open_array(os.path.join(LICONN_ZARR_PATH, "raw"), mode="r")[LICONN_ROI]
    return volume, f"liconn_18x18x24nm_{get_roi_name(LICONN_ROI)}"


def normalize_per_slice(volume):
    # Every slice keeps >= 2% of its pixels at 0 and 1, so the volume bounds of the embeddings become (0, 1).
    return np.stack([normalize_raw(z_slice) for z_slice in volume])


def view_raw(volume, title):
    viewer = napari.Viewer(title=title)
    viewer.add_image(volume, name="raw")
    napari.run()


def precompute_embeddings(volume, embedding_path, model_type):
    predictor = get_sam2_model(model_type=model_type, input_type="videos")
    # 'devices=None' spreads the slices over every visible GPU.
    embeddings = precompute_image_embeddings(predictor, volume, save_path=embedding_path, ndim=3, lazy_loading=True)
    embeddings.close()


def run_annotator(volume, embedding_path, model_type):
    # 'annotator' tiles slices above 768 pixels, so build the tool here to segment over the full XY.
    state = AnnotatorState()
    state.image_shape = volume.shape
    state.initialize_predictor(
        volume, model_type=model_type, ndim=3, save_path=embedding_path, batch_size=None, use_cli=True
    )

    viewer = napari.Viewer()
    viewer.add_image(volume, name="image")
    annotator = Annotator(viewer, ndim=3, reset_state=False)
    annotator._update_image()
    viewer.window.add_dock_widget(annotator, name=get_dock_title("segmentation"))
    _sync_embedding_widget(
        widget=state.widgets["embeddings"], model_type=model_type, save_path=embedding_path,
        checkpoint_path=None, device=None, tile_shape=None, halo=None,
    )
    napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", default="hvit_t_cells")
    parser.add_argument("--view", action="store_true", help="Only show the raw volume in napari.")
    parser.add_argument("--precompute", action="store_true", help="Only precompute the embeddings.")
    parser.add_argument("--slice_norm", action="store_true", help="Normalize each slice instead of the volume.")
    args = parser.parse_args()

    volume, name = load_volume()
    print(f"Loaded {name} with shape {volume.shape} and dtype {volume.dtype}")
    if args.slice_norm:
        volume = normalize_per_slice(volume)
        name = f"{name}_slice_norm"

    if args.view:
        view_raw(volume, title=name)
        return

    embedding_path = os.path.join(EMBEDDING_ROOT, f"{name}_{args.model_type}.zarr")
    if args.precompute:
        precompute_embeddings(volume, embedding_path, args.model_type)
    else:
        run_annotator(volume, embedding_path, args.model_type)


if __name__ == "__main__":
    main()
