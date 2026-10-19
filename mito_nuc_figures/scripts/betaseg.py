import os
from glob import glob

import h5py
import numpy as np

from figure_utils import DATA_ROOT, get_output_dir, save_volume_figure, write_info


DATASET = "betaseg"
NOTES = "FIB-SEM mouse pancreatic beta cells, ~16 nm isotropic, manual semantic masks (mito, nucleus)"
ISSUES = (
    "Voxel size is not stored in the h5 files; 16 nm isotropic is inferred (whole beta cell in ~1000 px). "
    "Masks are 0/255. Loader bug: betaseg.py:128 iterates over the undefined name 'label_choices' instead "
    "of 'label_choice', so passing a list of organelles raises UnboundLocalError."
)
OBSERVATIONS = (
    "Only the central target beta cell is annotated (its nucleus and its mitochondria); nuclei and "
    "mitochondria of the neighbouring cells in the field of view are unlabeled, so whole-volume training "
    "sees many false negatives (restrict patches to the target cell). Inside the target cell the mito "
    "labels look exhaustive and aligned, nucleus masks are clean and aligned in XY and ZX. Exactly one "
    "nucleus per volume."
)


def foreground_fraction(labels, stride=8):
    return float((labels[::stride, ::stride, ::stride] > 0).mean())


def render_volume(path, out_dir):
    name = os.path.basename(path).replace("_source.h5", "")
    with h5py.File(path, "r") as f:
        raw = f["raw"]
        labels = {"mito": f["labels/mitochondria"], "nucleus": f["labels/nucleus"]}
        save_volume_figure(raw, labels, os.path.join(out_dir, f"{name}.png"), DATASET, name, notes=NOTES)
        info = {"name": name, "shape": raw.shape, "voxel_size_nm": (16, 16, 16), "path": path}
        for class_name, label_volume in labels.items():
            info[f"{class_name}_values"] = np.unique(label_volume[::32, ::8, ::8]).tolist()
            info[f"{class_name}_fraction"] = round(foreground_fraction(label_volume), 4)
    print(info)
    return info


def main():
    out_dir = get_output_dir(DATASET)
    paths = sorted(glob(os.path.join(DATA_ROOT, DATASET, "data", "*.h5")))
    volumes = [render_volume(path, out_dir) for path in paths]
    write_info(DATASET, {
        "classes": ["mito", "nucleus"], "modality": "FIB-SEM", "dimensionality": "3d",
        "label_quality": "manual, semantic (binary masks per organelle)", "n_volumes": len(volumes),
        "volumes": volumes, "data_path": os.path.join(DATA_ROOT, DATASET, "data"),
        "download": "already local (2.1 GB, 7 h5 files)", "notes": NOTES,
        "issues": ISSUES,
        "observations": OBSERVATIONS,
    })


if __name__ == "__main__":
    main()
