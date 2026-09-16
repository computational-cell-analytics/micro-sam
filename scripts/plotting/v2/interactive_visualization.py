"""Interactive 3d segmentation with 'hvit_t_cells' in the micro-sam annotator on beke_data and LICONN.

Usage:
    python interactive_visualization.py -d beke_big --view  # only the raw volume in napari
    python interactive_visualization.py -d liconn --precompute  # the embeddings, over all visible GPUs
    python interactive_visualization.py -d liconn  # the annotator
    python interactive_visualization.py -d beke_big --slice_norm  # each slice normalized by its own percentiles
    python interactive_visualization.py -d beke_small_crop  # the embryo of beke_small, upsampled 2x by SAM2
    python interactive_visualization.py -d beke_big_crop  # the embryo of beke_big, upsampled 2x by SAM2
"""

import os
import argparse

import zarr
import tifffile
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
DATASETS = ["beke_big", "beke_big_crop", "beke_small", "beke_small_crop", "liconn"]

# The two beke volumes: big is fused t0 (208, 1024, 1024), small is 3.5hpf (79, 1024, 1024).
BEKE_BIG_PATH = os.path.join(DATA_ROOT, "beke_data", "fused_t000000_R0000_C02_Dfinal.tif")
# ZYX, the 512x512 region around the embryo, identical to the raw of 'embryo_fused_t0_crop.h5'.
BEKE_BIG_ROI = (slice(0, 208), slice(215, 727), slice(253, 765))
BEKE_SMALL_PATH = os.path.join(DATA_ROOT, "beke_data", "C2-Pdu_Composite_3.5hpf_HRas-mScarlet_H2B-GFP.tif")
# ZYX, the 512x512 region around the embryo, identical to the raw of 'embryo_3.5hpf_crop.h5'.
BEKE_SMALL_ROI = (slice(0, 79), slice(211, 723), slice(255, 767))

# The torch-em store at 18x18x24 nm, the resolution the models were trained on.
LICONN_ZARR_PATH = os.path.join(DATA_ROOT, "liconn", "liconn.zarr")
# ZYX, centred in the proofread region ('LICONN_ROI' of the generalist loader).
LICONN_ROI = (slice(288, 416), slice(2048, 2560), slice(1656, 2168))


def get_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def get_roi_name(roi):
    return "_".join(f"{axis}{r.start}-{r.stop}" for axis, r in zip("zyx", roi))


def load_volume(dataset):
    if dataset == "beke_big":
        return tifffile.imread(BEKE_BIG_PATH), get_name(BEKE_BIG_PATH)
    if dataset == "beke_big_crop":
        volume = tifffile.imread(BEKE_BIG_PATH)[BEKE_BIG_ROI]
        return volume, f"{get_name(BEKE_BIG_PATH)}_{get_roi_name(BEKE_BIG_ROI)}"
    if dataset == "beke_small":
        return tifffile.imread(BEKE_SMALL_PATH), get_name(BEKE_SMALL_PATH)
    if dataset == "beke_small_crop":
        volume = tifffile.imread(BEKE_SMALL_PATH)[BEKE_SMALL_ROI]
        return volume, f"{get_name(BEKE_SMALL_PATH)}_{get_roi_name(BEKE_SMALL_ROI)}"

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
        volume, model_type=model_type, ndim=3, save_path=embedding_path, batch_size=None, skip_load=False, use_cli=True
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
    parser.add_argument("-d", "--dataset", required=True, choices=DATASETS)
    parser.add_argument("-m", "--model_type", default="hvit_t_cells")
    parser.add_argument("--view", action="store_true", help="Only show the raw volume in napari.")
    parser.add_argument("--precompute", action="store_true", help="Only precompute the embeddings.")
    parser.add_argument("--slice_norm", action="store_true", help="Normalize each slice instead of the volume.")
    args = parser.parse_args()

    volume, name = load_volume(args.dataset)
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
