"""Overview figures for the Drosophila VNC ssTEM mitochondria data."""
import os

import h5py

from figure_utils import DATA_ROOT, get_output_dir, save_volume_figure, write_info


DATASET = "vnc"
DATA_DIR = os.path.join(DATA_ROOT, DATASET)
NOTES = "ssTEM Drosophila VNC, 4.6x4.6x45 nm, {labels}"


def render_split(split, out_dir):
    path = os.path.join(DATA_DIR, f"vnc_{split}.h5")
    with h5py.File(path, "r") as f:
        raw = f["raw"]
        has_labels = "labels/mitochondria" in f
        labels = {"mito": f["labels/mitochondria"]} if has_labels else {}
        label_note = "manual mito labels, instances from 3D connected components" if has_labels else "no labels"
        out_path = os.path.join(out_dir, f"{split}.png")
        save_volume_figure(raw, labels, out_path, DATASET, split, notes=NOTES.format(labels=label_note))
        volume = {"name": split, "shape": raw.shape, "voxel_size_nm": (45, 4.6, 4.6), "path": path}
        volume["has_labels"] = has_labels
        if has_labels:
            volume["n_mito_ids"] = int(f["labels/mitochondria"][:].max())
        return volume


def main():
    out_dir = get_output_dir(DATASET)
    volumes = [render_split(split, out_dir) for split in ("train", "test")]
    write_info(DATASET, {
        "classes": ["mito"], "modality": "ssTEM", "dimensionality": "3d",
        "label_quality": "manual, semantic masks turned into instances by connected components",
        "n_volumes": len(volumes), "volumes": volumes, "data_path": DATA_DIR, "download": "none, already local",
        "notes": "stack1 (train) has mito + synapse labels, stack2 (test) has raw only",
        "issues": "vnc_test.h5 has no labels, so only the 20-slice train volume is usable", "observations": (
            "train: exhaustive and aligned, 20 thin slices only (45 nm), so 3D context is minimal; "
            "test: raw only"
        ),
    })


if __name__ == "__main__":
    main()
