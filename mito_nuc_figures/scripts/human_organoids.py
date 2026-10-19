import os
from glob import glob

import h5py
import numpy as np

from figure_utils import DATA_ROOT, get_output_dir, save_volume_figure, write_info


DATASET = "human_organoids"
DATA_DIR = os.path.join(DATA_ROOT, DATASET, "11380", "data")
NOTES = "FIB-SEM organoids/spheroids (bin2), voxel size not stored, manual semantic masks (int8: -128 bg, 127 fg)"
DESCRIPTIONS = {
    "F059": "BT474 breast cancer cell line spheroid",
    "F107_A1": "patient-derived colorectal cancer organoid, acquisition 1",
    "F107_A2": "patient-derived colorectal cancer organoid, acquisition 2",
    "F107_A3": "patient-derived colorectal cancer organoid, acquisition 3",
}
ISSUES = (
    "Raw and labels are int8 (MRC mode 0): labels are -128 background / 127 foreground and the loader "
    "passes them unchanged, so anything treating non-zero as foreground marks the whole volume; use "
    "labels > 0. Voxel size is lost (MRC headers not copied, original MRCs deleted by _prepare_dataset). "
    "Large black alignment padding (-128) in all volumes, especially F059."
)
OBSERVATIONS = (
    "Exhaustive and aligned in all 4 volumes: every visible nucleus and mitochondrion is labeled, in all "
    "cells. Mitos are small round profiles. F107_A2 (and partly F059) mito masks show horizontal streaks "
    "in the ZX view, i.e. slice-wise annotation that is not fully consistent across z; F059 nucleus ZX "
    "shows one seam. F107_A1 also has actin/entotic_cell/junctions, F107_A2/A3 actin/junctions."
)


class Foreground:
    """Lazy view of a label dataset that returns the binary foreground (value > 0) on slicing."""

    def __init__(self, data):
        self.data = data
        self.shape = data.shape
        self.ndim = data.ndim

    def __getitem__(self, key):
        return (self.data[key] > 0).astype("uint8")


def render_volume(path, out_dir):
    name = os.path.basename(path).replace("_bin2.h5", "")
    with h5py.File(path, "r") as f:
        raw = f["raw"]
        labels = {"mito": Foreground(f["mitos"]), "nucleus": Foreground(f["nuclei"])}
        notes = f"{DESCRIPTIONS[name]}; {NOTES}"
        save_volume_figure(raw, labels, os.path.join(out_dir, f"{name}.png"), DATASET, name, notes=notes)
        info = {
            "name": name, "description": DESCRIPTIONS[name], "shape": raw.shape, "voxel_size_nm": "unknown (bin2)",
            "path": path, "label_keys": sorted(k for k in f.keys() if k != "raw"),
            "raw_dtype": str(raw.dtype), "label_values": np.unique(f["mitos"][::64, ::16, ::16]).tolist(),
        }
        for class_name, label_volume in labels.items():
            info[f"{class_name}_fraction"] = round(float(label_volume[::16, ::16, ::16].mean()), 4)
    print(info)
    return info


def main():
    out_dir = get_output_dir(DATASET)
    volumes = [render_volume(path, out_dir) for path in sorted(glob(os.path.join(DATA_DIR, "*.h5")))]
    write_info(DATASET, {
        "classes": ["mito", "nucleus"], "modality": "FIB-SEM", "dimensionality": "3d",
        "label_quality": "manual, semantic binary masks", "n_volumes": len(volumes), "volumes": volumes,
        "data_path": DATA_DIR, "download": "already local (8.0 GB, EMPIAR-11380, 4 h5 files)", "notes": NOTES,
        "issues": ISSUES, "observations": OBSERVATIONS,
    })


if __name__ == "__main__":
    main()
