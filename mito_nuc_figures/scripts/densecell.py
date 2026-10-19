import os

import h5py
import numpy as np
from skimage.measure import label as connected_components

from figure_utils import DATA_ROOT, get_output_dir, save_volume_figure, write_info


DATASET = "densecell"
NOTES = "SBF-SEM human platelets, 10x10x50 nm, manual dense semantic labels (class 2 = mitochondrion shown)"
OBSERVATIONS = (
    "Labels aligned with the raw data and look exhaustive. Platelet mitos are small and few: 0.3-1.4% foreground, "
    "11-34 3D connected components per volume. 50 nm sections, so the ZX view is coarse. "
    "Tiny dataset for mito (3 small volumes, 195 slices total)."
)


def render_split(split, out_dir):
    path = os.path.join(DATA_ROOT, DATASET, f"densecell_{split}.h5")
    with h5py.File(path, "r") as f:
        raw, mito = f["raw"], f["labels/mitochondrion"]
        original = f["labels/original"][:]
        mito_data = mito[:]
        n_components = int(connected_components(mito_data > 0).max())
        out_path = os.path.join(out_dir, f"{split}.png")
        save_volume_figure(raw, {"mito": mito}, out_path, DATASET, split, notes=NOTES)
        return {
            "name": split, "shape": raw.shape, "raw_dtype": str(raw.dtype), "voxel_size_nm": (50, 10, 10),
            "path": path, "label_key": "labels/mitochondrion",
            "mito_foreground_fraction": float((mito_data > 0).mean()),
            "n_mito_3d_connected_components": n_components,
            "original_class_ids": np.unique(original).tolist(),
        }


def main():
    out_dir = get_output_dir(DATASET)
    volumes = [render_split(split, out_dir) for split in ("train", "val", "test")]
    write_info(DATASET, {
        "classes": ["mito"], "modality": "SBF-SEM", "dimensionality": "3d",
        "label_quality": "manual, dense, semantic (multi-class map, mitochondrion = class 2)",
        "n_volumes": len(volumes), "volumes": volumes, "data_path": os.path.join(DATA_ROOT, DATASET),
        "download": "already present (densecell_{train,val,test}.h5, 150 MB total)", "notes": NOTES,
        "issues": "Semantic labels only, no mito instances (touching mitos would merge). Raw dtype differs per split.",
        "observations": OBSERVATIONS,
    })


if __name__ == "__main__":
    main()
