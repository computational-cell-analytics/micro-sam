"""Overview figures for the UroCell FIB-SEM volumes that have mitochondria labels."""
import os
from glob import glob

import h5py
import numpy as np

from figure_utils import DATA_ROOT, get_output_dir, save_volume_figure, write_info


DATASET = "uro_cell"
DATA_DIR = os.path.join(DATA_ROOT, DATASET)
NOTES = "FIB-SEM mouse urothelium, 16 nm isotropic, manual instance mito labels"


def render_volume(path, out_dir):
    name = os.path.splitext(os.path.basename(path))[0]
    with h5py.File(path, "r") as f:
        raw, labels = f["raw"], f["labels/mito"]
        save_volume_figure(raw, {"mito": labels}, os.path.join(out_dir, f"{name}.png"), DATASET, name, notes=NOTES)
        data = labels[:]
        return {
            "name": name, "shape": raw.shape, "voxel_size_nm": (16, 16, 16), "path": path,
            "n_mito_ids": int(len(np.unique(data)) - 1), "mito_fraction": round(float((data > 0).mean()), 4),
            "other_labels": [key for key in f["labels"].keys() if key != "mito"],
        }


def main():
    out_dir = get_output_dir(DATASET)
    paths = sorted(glob(os.path.join(DATA_DIR, "*.h5")))
    with_mito, without_mito = [], []
    for path in paths:
        with h5py.File(path, "r") as f:
            (with_mito if "labels/mito" in f else without_mito).append(path)
    volumes = [render_volume(path, out_dir) for path in with_mito]
    write_info(DATASET, {
        "classes": ["mito"], "modality": "FIB-SEM", "dimensionality": "3d", "label_quality": "manual, instance",
        "n_volumes": len(volumes), "volumes": volumes, "data_path": DATA_DIR, "download": "none, already local",
        "notes": NOTES + "; raw stored as float64",
        "issues": "volumes without mito labels (raw only): " + ", ".join(os.path.basename(p) for p in without_mito),
        "observations": (
            "exhaustive and aligned in all 5 labeled volumes; dark lysosome-like bodies are correctly "
            "left unlabeled; gray stripe artifact in the top rows of some volumes"
        ),
    })


if __name__ == "__main__":
    main()
