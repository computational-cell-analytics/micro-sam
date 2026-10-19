import os

import h5py

from figure_utils import DATA_ROOT, get_output_dir, save_volume_figure, write_info


DATASET = "lucchi"
NOTES = "FIB-SEM mouse hippocampus, 5 nm isotropic, manual semantic mito labels"


def render_split(split, out_dir):
    path = os.path.join(DATA_ROOT, DATASET, f"lucchi_{split}.h5")
    with h5py.File(path, "r") as f:
        raw, labels = f["raw"], f["labels"]
        out_path = os.path.join(out_dir, f"{split}.png")
        save_volume_figure(raw, {"mito": labels}, out_path, DATASET, split, notes=NOTES)
        return {"name": split, "shape": raw.shape, "voxel_size_nm": (5, 5, 5), "path": path}


def main():
    out_dir = get_output_dir(DATASET)
    volumes = [render_split(split, out_dir) for split in ("train", "test")]
    write_info(DATASET, {
        "classes": ["mito"], "modality": "FIB-SEM", "dimensionality": "3d", "label_quality": "manual, semantic",
        "n_volumes": len(volumes), "volumes": volumes, "data_path": os.path.join(DATA_ROOT, DATASET),
        "notes": NOTES, "issues": "",
    })


if __name__ == "__main__":
    main()
