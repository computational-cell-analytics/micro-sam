"""Render the 3D volume EM datasets added to torch-em on 2026-10-18 into mito/TBD and nuc/TBD.

Each figure shows a few xy slices with the most labeled pixels and one zx slice per volume, raw next to the label
overlay (instances in tab20 colors, semantic masks as one filled color), with the layout of render_mito_overview.py.
The figures of the datasets the user has decided on were moved into the category folders, so they are not listed any
more; CATEGORIES sends the decided figures that are still listed to their folder.
The data is read slice by slice, so no volume is loaded whole.
"""
import argparse
import os
from glob import glob

import h5py
import numpy as np
import zarr

from render_mito_overview import REPO, save_grid, text_figure

DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"
N_XY = 3
N_CANDIDATES = 24


def h5_volume(path, raw_key, label_key, value=None, title=None, zx=True):
    return {"path": path, "raw_key": raw_key, "label_key": label_key, "value": value,
            "title": title or os.path.basename(path), "zx": zx}


def zarr_volumes(pattern, value=None):
    return [h5_volume(p, "raw", "labels", value) for p in sorted(glob(pattern))]


def h5_volumes(pattern, raw_key, label_key, value=None):
    return [h5_volume(p, raw_key, label_key, value) for p in sorted(glob(pattern))]


# The category folder of the figures the user has decided on (all others go to TBD).
CATEGORIES = {("mito", "parlakgul_fasting_liver"): "training"}

# Per figure: the model folder, the dataset name, the volumes and the title lines.
FIGURES = [
    ("mito", "hela_metaphase", h5_volumes(os.path.join(DATA_ROOT, "hela_metaphase", "*.h5"), "raw",
                                          "labels/organelles", value=1),
     "HeLa Metaphase (EMPIAR-10994): SBF-SEM of two metaphase HeLa cells (control, REEP3/4 kd), 30 nm sections. "
     "Mito = class 1 of the semantic MIB map: small (10-20 px wide), many mitos unlabeled; no nucleus (metaphase)."),
    ("mito", "parlakgul_fasting_liver",
     zarr_volumes(os.path.join(DATA_ROOT, "parlakgul_fasting_liver", "*_mito_*.zarr")),
     "Parlakgul fasting liver (EMPIAR-12017): FIB-SEM mouse liver, 8 nm, mito instance segmentation (32-bit) of the "
     "authors; one 64 x 1024 x 1024 box per sample (new samples, not our Parlakgul evaluation boxes)."),
    ("nuc", "parlakgul_fasting_liver",
     zarr_volumes(os.path.join(DATA_ROOT, "parlakgul_fasting_liver", "*_nuclear_membrane_*.zarr")),
     "Parlakgul fasting liver (EMPIAR-12017): FIB-SEM mouse liver, 8 nm, NUCLEAR MEMBRANE binary masks "
     "(not filled nuclei)."),
]


def open_arrays(volume):
    path = volume["path"]
    if path.endswith(".zarr"):
        group = zarr.open_group(path, mode="r")
    else:
        group = h5py.File(path, "r")
    return group[volume["raw_key"]], group[volume["label_key"]]


def target(labels, value):
    labels = np.asarray(labels)
    return (labels == value).astype("uint32") if value is not None else labels.astype("uint32")


def crop_aspect(raw, lab, max_ratio=1.5):
    """Crop a long 2D slice to at most max_ratio times its short side, centered on the labels."""
    raw, lab = np.asarray(raw), np.asarray(lab)
    h, w = lab.shape
    if w > max_ratio * h or h > max_ratio * w:
        axis = 1 if w > h else 0
        size = int(max_ratio * min(h, w))
        profile = (lab > 0).sum(axis=1 - axis)
        center = int(np.average(np.arange(len(profile)), weights=profile)) if profile.sum() else len(profile) // 2
        start = int(np.clip(center - size // 2, 0, len(profile) - size))
        window = np.s_[:, start:start + size] if axis == 1 else np.s_[start:start + size, :]
        raw, lab = raw[window], lab[window]
    return raw, lab


def volume_pairs(volume):
    raw, labels = open_arrays(volume)
    shape = labels.shape
    candidates = np.unique(np.linspace(0, shape[0] - 1, N_CANDIDATES).round().astype(int))
    stride = max(1, max(shape[1:]) // 256)
    counts = [int((target(labels[z, ::stride, ::stride], volume["value"]) > 0).sum()) for z in candidates]
    # The slice with the most labeled pixels in each third of the volume, so that the slices spread over z.
    chosen = []
    for part in np.array_split(np.arange(len(candidates)), N_XY):
        best = max(part, key=lambda i: counts[i])
        if counts[best] > 0:
            chosen.append(int(candidates[best]))
    chosen = chosen or [int(candidates[len(candidates) // 2])]
    pairs = []
    for z in chosen:
        lab = target(labels[z], volume["value"])
        n_ids = len(np.unique(lab[lab > 0]))
        title = f"{volume['title']} xy z={z}/{shape[0]}, shape {shape}, {n_ids} label id(s)"
        pairs.append((*crop_aspect(raw[z], lab), None, title))
    if not volume["zx"]:
        return pairs
    y_candidates = np.linspace(0.1 * shape[1], 0.9 * shape[1], 12).round().astype(int)
    z_stride = max(1, shape[0] // 64)

    def fg(yy):
        return int((target(labels[::z_stride, yy], volume["value"]) > 0).sum())

    y = int(max(y_candidates, key=fg))
    lab = target(labels[:, y], volume["value"])
    pairs.append((*crop_aspect(raw[:, y], lab), None, f"{volume['title']} zx y={y} ({shape[0]} slices)"))
    return pairs


def render(model, name, volumes, title):
    category = CATEGORIES.get((model, name), "TBD")
    out_dir = os.path.join(REPO, "mito_nuc_figures", model, category)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.png")
    volumes = [v for v in volumes if os.path.exists(v["path"])]
    if not volumes:
        message = f"{title} Data not available locally (download failed)."
        text_figure(out_path, f"{model} / {category} / {name}", message)
        return
    pairs = []
    for volume in volumes:
        pairs += volume_pairs(volume)
    save_grid(pairs, out_path, f"{model} / {category} / {name}\n{title}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+", default=None, help="Only render these datasets.")
    args = parser.parse_args()
    for model, name, volumes, title in FIGURES:
        if args.names and name not in args.names:
            continue
        render(model, name, volumes, title)


if __name__ == "__main__":
    main()
