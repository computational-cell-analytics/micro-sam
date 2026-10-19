"""Overview figures for the MICrONS Zenodo pinky volumes with (sparse) mitochondria labels."""
import os
from glob import glob

import h5py
import numpy as np

from figure_utils import DATA_ROOT, get_output_dir, save_volume_figure, write_info


DATASET = "microns"
DATA_DIR = os.path.join(DATA_ROOT, DATASET, "pinky")
MITO_KEY = "volumes/mitochondria"
NOTES = "ssTEM mouse visual cortex (pinky), 4x4x40 nm, mito instance labels (uint8), sparse"


def get_stats(f, n_slices=8):
    """Estimate mito and mask fractions and count mito ids on evenly spaced slices."""
    labels, mask = f[MITO_KEY], f["volumes/mask"]
    ids, mito_fraction, mask_fraction = set(), [], []
    for z in np.linspace(0, labels.shape[0] - 1, n_slices).astype(int):
        lab = labels[z]
        ids.update(np.unique(lab).tolist())
        mito_fraction.append((lab > 0).mean())
        mask_fraction.append((mask[z] > 0).mean())
    ids.discard(0)
    return {
        "n_mito_ids_sampled": len(ids), "max_mito_id_sampled": max(ids) if ids else 0,
        "mito_fraction": round(float(np.mean(mito_fraction)), 4),
        "mask_fraction": round(float(np.mean(mask_fraction)), 4),
    }


def render_volume(path, out_dir):
    name = os.path.splitext(os.path.basename(path))[0]
    with h5py.File(path, "r") as f:
        raw, labels = f["volumes/image"], f[MITO_KEY]
        save_volume_figure(raw, {"mito": labels}, os.path.join(out_dir, f"{name}.png"), DATASET, name, notes=NOTES)
        return {"name": name, "shape": raw.shape, "voxel_size_nm": (40, 4, 4), "path": path, **get_stats(f)}


def main():
    out_dir = get_output_dir(DATASET)
    paths = sorted(glob(os.path.join(DATA_DIR, "*.h5")))
    with_mito, without_mito = [], []
    for path in paths:
        with h5py.File(path, "r") as f:
            (with_mito if MITO_KEY in f else without_mito).append(os.path.basename(path))
    volumes = [render_volume(os.path.join(DATA_DIR, name), out_dir) for name in with_mito]
    write_info(DATASET, {
        "classes": ["mito"], "modality": "ssTEM", "dimensionality": "3d",
        "label_quality": "sparse instance labels (pinky only), uint8 ids",
        "n_volumes": len(volumes), "volumes": volumes, "data_path": DATA_DIR, "download": "none, already local",
        "notes": "basil and minnie have no mito labels; volumes/mask marks the region of the cutout",
        "issues": "pinky volumes without mito labels: " + ", ".join(without_mito) + "; "
        "pinky_stitched_vol19-vol34.h5 and pinky_stitched_vol19-vol34_realigned.h5 cover the same region",
        "observations": (
            "stitched vol19-vol34 (and its realigned copy) are near exhaustive; vol101-104, 401, "
            "501-503 and vol40-41 only label mitos inside one or a few cells (unlabeled mitos next to "
            "them), so they need the volumes/mask region or an ignore label; uint8 ids reach 255 in the "
            "stitched volumes, so different mitos share ids"
        ),
    })


if __name__ == "__main__":
    main()
