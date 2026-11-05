"""Write the data of two candidate in-domain test sets of the EM nucleus panel (user, 2026-11-05).

- dauer_larva_nucleus: boxes of the OpenOrganelle jrc_dauer-larva volume (C. elegans, ssTEM). The volume is not in
  OPENORGANELLE_NUCLEUS_VOLUMES of the generalist loader, so the nucleus model never saw it. Its labels are the
  automatic COSEM nucleus segmentation ('labels/inference/nucleus_seg', 70 x 40 x 40 nm). The raw is read at 32 nm and
  resampled onto the label grid. The boxes with the most nuclei are kept: DAUER_TEST_BOXES are scored, one more tunes.
- human_organoids_nucleus: the validation part of the four human organoid volumes of the nucleus generalist (the
  last HUMAN_ORGANOIDS_VAL_FRACTION of the slices, manual labels). It selected the training checkpoint, so it is not
  unseen data. The 32 slices before it (training data) tune.

Usage: python prepare_em_nucleus_candidates.py [--datasets dauer_larva_nucleus human_organoids_nucleus]

The files go to DATA_ROOT/<dataset>/{test,val}/*.h5 with 'raw' and 'labels'. Then place the evaluation crops with
'preprocess_nucleus_dataset.py --dataset eval_crops --eval_datasets <dataset>'.
"""

import os
import argparse

import h5py
import numpy as np
from scipy.ndimage import zoom

from common import DATA_ROOT

# The box shape on the label grid (z, y, x) and the number of boxes that are scored; one more box tunes.
DAUER_BOX_SHAPE = (64, 1024, 1024)
DAUER_TEST_BOXES = 3
DAUER_NAME = "jrc_dauer-larva"
DAUER_LABELS = "labels/inference/nucleus_seg"
DAUER_RAW = "em/tem-uint8"
# The raw level read for the boxes (32 nm), the closest one to the 40 nm label grid.
DAUER_RAW_LEVEL = "s3"
# The coarse label level that ranks the boxes (640 nm in plane, 16 times coarser than s0).
DAUER_COARSE_LEVEL = "s4"
# The slices of each organoid volume before its validation part that tune.
ORGANOID_TUNING_SLICES = 32


def write_h5(path, raw, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(f"{path}.tmp", "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip", chunks=True)
        f.create_dataset("labels", data=labels, compression="gzip", chunks=True)
    os.replace(f"{path}.tmp", path)
    print(f"Wrote {path}: {raw.shape}, {len(np.unique(labels)) - 1} objects.")


def rank_dauer_boxes():
    """Return the starts of the non-overlapping boxes on the label grid, the ones with the most nuclei first."""
    from torch_em.data.datasets.electron_microscopy import openorganelle_nucleus as oo
    url = f"s3://janelia-cosem-datasets/{DAUER_NAME}/{DAUER_NAME}.zarr/recon-1/{DAUER_LABELS}"
    coarse = np.asarray(oo._open_remote_zarr(f"{url}/{DAUER_COARSE_LEVEL}")[:])
    full_shape = oo._open_remote_zarr(f"{url}/s0").shape
    factor = np.array(full_shape) / np.array(coarse.shape)
    window = [max(1, int(round(s / f))) for s, f in zip(DAUER_BOX_SHAPE, factor)]
    candidates = []
    for z in range(0, coarse.shape[0] - window[0] + 1, window[0]):
        for y in range(0, coarse.shape[1] - window[1] + 1, window[1] // 2):
            for x in range(0, coarse.shape[2] - window[2] + 1, window[2] // 2):
                ids = np.unique(coarse[z:z + window[0], y:y + window[1], x:x + window[2]])
                candidates.append((len(ids) - (ids[0] == 0), (z, y, x)))
    chosen = []
    for count, start in sorted(candidates, reverse=True):
        box = [(s, s + w) for s, w in zip(start, window)]
        if all(any(b[1] <= c[0] or c[1] <= b[0] for b, c in zip(box, other)) for other in chosen):
            chosen.append(box)
            print(f"Box at {start} (coarse grid) with {count} nuclei.")
        if len(chosen) == DAUER_TEST_BOXES + 1:
            break
    return [tuple(int(round(lo * f)) for (lo, _), f in zip(box, factor)) for box in chosen]


def prepare_dauer_larva(root):
    from torch_em.data.datasets.electron_microscopy import openorganelle_nucleus as oo
    base = f"s3://janelia-cosem-datasets/{DAUER_NAME}/{DAUER_NAME}.zarr/recon-1"
    labels = oo._open_remote_zarr(f"{base}/{DAUER_LABELS}/s0")
    raw = oo._open_remote_zarr(f"{base}/{DAUER_RAW}/{DAUER_RAW_LEVEL}")
    label_scale = oo._multiscale_levels(f"{oo.BUCKET_URL}{DAUER_NAME}/{DAUER_NAME}.zarr/recon-1/{DAUER_LABELS}")["s0"]
    raw_scale = oo._multiscale_levels(f"{oo.BUCKET_URL}{DAUER_NAME}/{DAUER_NAME}.zarr/recon-1/{DAUER_RAW}")[
        DAUER_RAW_LEVEL
    ]
    ratio = np.array(label_scale) / np.array(raw_scale)
    for k, start in enumerate(rank_dauer_boxes()):
        box = tuple(slice(s, min(s + w, n)) for s, w, n in zip(start, DAUER_BOX_SHAPE, labels.shape))
        label_data = np.asarray(labels[box]).astype("uint32")
        raw_box = tuple(slice(int(sl.start * r), int(np.ceil(sl.stop * r))) for sl, r in zip(box, ratio))
        raw_native = np.asarray(raw[raw_box])
        raw_data = zoom(raw_native, [t / s for t, s in zip(label_data.shape, raw_native.shape)], order=1)
        split = "test" if k < DAUER_TEST_BOXES else "val"
        name = "_".join(str(sl.start) for sl in box)
        write_h5(os.path.join(root, "dauer_larva_nucleus", split, f"box_{name}.h5"), raw_data.astype("uint8"),
                 label_data)


def prepare_human_organoids(root):
    from micro_sam.v2.datasets.generalist_loader import (
        HUMAN_ORGANOIDS_VOLUMES, HUMAN_ORGANOIDS_VAL_FRACTION, HUMAN_ORGANOIDS_NUCLEUS_IGNORE_LABEL,
    )
    import zarr
    for volume in HUMAN_ORGANOIDS_VOLUMES:
        name = volume.replace("_bin2.h5", "")
        with h5py.File(os.path.join(root, "human_organoids", "11380", "data", volume), "r") as f:
            raw = f["raw"]
            n_train = int(raw.shape[0] * (1 - HUMAN_ORGANOIDS_VAL_FRACTION))
            parts = {"test": np.s_[n_train:], "val": np.s_[n_train - ORGANOID_TUNING_SLICES:n_train]}
            labels = zarr.open(os.path.join(root, "human_organoids_nucleus", f"{name}_nucleus_labels.zarr"), mode="r")
            labels = labels["labels"] if not hasattr(labels, "shape") else labels
            for split, part in parts.items():
                label_data = np.asarray(labels[part]).astype("uint32")
                # The ignored fragments and the entotic cell are not scored.
                label_data[label_data == HUMAN_ORGANOIDS_NUCLEUS_IGNORE_LABEL] = 0
                write_h5(os.path.join(root, "human_organoids_nucleus_val", split, f"{name}.h5"), raw[part],
                         label_data)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    choices = ["dauer_larva_nucleus", "human_organoids_nucleus"]
    parser.add_argument("--datasets", nargs="+", default=choices, choices=choices)
    parser.add_argument("--root", default=DATA_ROOT)
    args = parser.parse_args()
    if "human_organoids_nucleus" in args.datasets:
        prepare_human_organoids(args.root)
    if "dauer_larva_nucleus" in args.datasets:
        prepare_dauer_larva(args.root)


if __name__ == "__main__":
    main()
