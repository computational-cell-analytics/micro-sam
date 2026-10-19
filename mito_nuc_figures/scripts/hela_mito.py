import os

import h5py
import numpy as np

from figure_utils import DATA_ROOT, get_output_dir, save_image_grid, save_volume_figure, write_info
from torch_em.data.datasets.electron_microscopy import hela_mito


DATASET = "hela_mito"
NOTES = "SBF-SEM HeLa, 10 nm px, manual mito instances, sparse"
# The GT2 masks are numbered sequentially, not by raw slice. Edge-alignment and IoU against GT show the true raw slice.
ISSUES = (
    "Loader bug, hela_mito.py:26-36 (SLICES): the GT2 files z0002, z0004, z0026, z0028 are numbered sequentially, "
    "not by raw slice. They are second annotations of raw z0030, z0060, z0116, z0150 (best edge alignment over all "
    "24 raw slices; IoU vs GT of the same slice 0.81, 0.76, 0.96, 0.93). The loader pairs them with raw "
    "z0002/z0004/z0026/z0028, so 4 of 9 training images have misaligned labels. Only 5 distinct slices are labeled "
    "(z0001, z0030, z0060, z0116, z0150). Fix: drop the 4 GT2 entries, or pair them with the correct raw slices."
)
GT2_TRUE_SLICE = {"z0002": "z0030", "z0004": "z0060", "z0026": "z0116", "z0028": "z0150"}

OBSERVATIONS = (
    "z0001, z0030, z0060, z0116, z0150 (GT): aligned, tight instance outlines. z0002, z0004, z0026, z0028 (GT2): "
    "misaligned, labels fall on the nucleus or on empty cytoplasm (z0028 clearly inside the nucleus). Only the "
    "central cell is annotated, the neighbouring cells at the image border are unlabeled, and even in the central "
    "cell some visible mitos are missed (roughly 70-85% coverage by eye). Instances are separate ids (15-40 per slice)."
)


def render_slice(path, out_dir):
    name = os.path.splitext(os.path.basename(path))[0]
    with h5py.File(path, "r") as f:
        raw, labels = f["raw"][:], f["labels"][:]
    notes = NOTES
    if name in GT2_TRUE_SLICE:
        notes = f"MISALIGNED: this GT2 mask belongs to raw {GT2_TRUE_SLICE[name]}"
    save_volume_figure(raw, {"mito": labels}, os.path.join(out_dir, f"{name}.png"), DATASET, name, notes=notes)
    volume = {
        "name": name, "shape": raw.shape, "voxel_size_nm": (10, 10), "path": path,
        "n_objects": int((np.unique(labels) != 0).sum()), "foreground_fraction": round(float((labels > 0).mean()), 4),
        "aligned": name not in GT2_TRUE_SLICE, "true_raw_slice": GT2_TRUE_SLICE.get(name, name),
    }
    return volume, (name if name not in GT2_TRUE_SLICE else f"{name} (MISALIGNED)", raw, {"mito": labels})


def main():
    data_path = os.path.join(DATA_ROOT, DATASET)
    paths = hela_mito.get_hela_mito_paths(data_path, download=True)
    out_dir = get_output_dir(DATASET)
    results = [render_slice(path, out_dir) for path in paths]
    volumes = [result[0] for result in results]
    samples = [result[1] for result in results]
    grid_path = os.path.join(out_dir, "all_slices.png")
    save_image_grid(samples, grid_path, DATASET, "all 9 loader slices", notes=NOTES, ncols=5)
    info = {
        "classes": ["mito"], "modality": "SBF-SEM (EMPIAR-10094 HeLa, 10x10 nm px, 50 nm sections)",
        "dimensionality": "2d", "label_quality": "manual, instance, sparse: only the central cell is labeled",
        "n_volumes": len(volumes), "volumes": volumes, "data_path": data_path,
        "download": "9 per-slice h5 files (2000x2000 uint8 raw + labels), 26 MB, get_hela_mito_paths(download=True)",
        "notes": NOTES, "issues": ISSUES, "observations": OBSERVATIONS,
    }
    write_info(DATASET, info)


if __name__ == "__main__":
    main()
