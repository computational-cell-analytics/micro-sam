"""Automatic segmentation of a LICONN block with micro-sam2, with AIS and APG.

Run it with '--precompute ais' and '--precompute apg' one time on a GPU. Each run adds its segmentation to the file
with the block and the proofread labels. Then run it without arguments to show the results in napari.
Run 'run_interactive_segmentation.py --precompute' first, so that both modes use its embeddings.

Usage:
    python run_automatic_segmentation.py --precompute ais
    python run_automatic_segmentation.py --precompute apg
    python run_automatic_segmentation.py
"""

import os
import argparse

import h5py
import zarr

import napari

from elf.evaluation import cremi_score

from torch_em.data.datasets.light_microscopy.liconn import get_liconn_paths

from micro_sam.v2.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data/liconn"
DEMO_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/liconn_demo"
MODEL_TYPE = "hvit_l_cells"
MODES = ("ais", "apg")
# The propagation rounds drop the candidates that earlier rounds already cover, so APG finishes on the dense neurites.
GENERATE_KWARGS = {"ais": {}, "apg": {"propagation_waves": 4}}
ROI = (slice(384, 512), slice(3072, 3584), slice(2560, 3072))


def load_block():
    f = zarr.open_group(get_liconn_paths(DATA_ROOT), mode="r")
    return f["raw"][ROI], f["seg_proofread"][ROI]


def precompute(mode):
    raw, labels = load_block()
    predictor, segmenter = get_predictor_and_segmenter(model_type=MODEL_TYPE, segmentation_mode=mode, ndim=3)
    segmentation = automatic_instance_segmentation(
        predictor, segmenter, input_path=raw, embedding_path=os.path.join(DEMO_ROOT, "embeddings.zarr"), ndim=3,
        **GENERATE_KWARGS[mode],
    )
    # The unlabeled voxels are not proofread, so the score ignores them.
    vi_split, vi_merge, _, cremi = cremi_score(segmentation, labels, ignore_gt=[0])
    print(f"{mode.upper()}: CREMI {cremi:.3f}, VI split {vi_split:.3f}, VI merge {vi_merge:.3f}")

    os.makedirs(DEMO_ROOT, exist_ok=True)
    with h5py.File(os.path.join(DEMO_ROOT, "automatic_segmentation.h5"), "a") as f:
        if "raw" not in f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")
        if mode in f:
            del f[mode]
        f.create_dataset(mode, data=segmentation, compression="gzip")


def show():
    viewer = napari.Viewer()
    with h5py.File(os.path.join(DEMO_ROOT, "automatic_segmentation.h5"), "r") as f:
        viewer.add_image(f["raw"][:], name="raw")
        viewer.add_labels(f["labels"][:], name="proofread labels")
        for mode in MODES:
            if mode in f:
                viewer.add_labels(f[mode][:], name=mode.upper())
    napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precompute", choices=MODES, help="Segment the block with this mode. This needs a GPU.")
    args = parser.parse_args()
    if args.precompute is None:
        show()
    else:
        precompute(args.precompute)


if __name__ == "__main__":
    main()
