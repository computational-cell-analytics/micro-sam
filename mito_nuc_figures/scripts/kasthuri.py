"""Overview figures for the Kasthuri mitochondria data (Lucchi++ style re-annotation by Casser et al.)."""
import os

import h5py
import numpy as np

from figure_utils import DATA_ROOT, get_output_dir, save_volume_figure, write_info


DATASET = "kasthuri"
DATA_DIR = os.path.join(DATA_ROOT, DATASET)
NOTES = "ATUM-SEM mouse cortex, 3x3x30 nm, manual semantic mito labels (-1 = ignore region, shown as background)"


class ClippedLabels:
    """Lazy label view that maps the ignore label -1 to background."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.shape = dataset.shape
        self.ndim = dataset.ndim

    def __getitem__(self, index):
        return np.clip(self.dataset[index], 0, None)


def render_split(split, out_dir):
    path = os.path.join(DATA_DIR, f"kasthuri_{split}.h5")
    with h5py.File(path, "r") as f:
        raw, labels = f["raw"], f["labels"]
        out_path = os.path.join(out_dir, f"{split}.png")
        save_volume_figure(raw, {"mito": ClippedLabels(labels)}, out_path, DATASET, split, notes=NOTES)
        data = labels[:]
        ignore_fraction = float((data == -1).mean())
        mito_fraction = float((data == 1).mean())
        return {
            "name": split, "shape": raw.shape, "voxel_size_nm": (30, 3, 3), "path": path,
            "label_values": np.unique(data).tolist(), "ignore_fraction": round(ignore_fraction, 3),
            "mito_fraction": round(mito_fraction, 3),
        }


def main():
    out_dir = get_output_dir(DATASET)
    volumes = [render_split(split, out_dir) for split in ("train", "test")]
    write_info(DATASET, {
        "classes": ["mito"], "modality": "ATUM-SEM (serial section SEM)", "dimensionality": "3d",
        "label_quality": "manual, semantic (0 / 1, -1 = unlabeled border region)",
        "n_volumes": len(volumes), "volumes": volumes, "data_path": DATA_DIR, "download": "none, already local",
        "notes": NOTES, "issues": "labels contain -1 for the region outside the annotated area; the loader has a TODO "
        "for masking it, so -1 currently reaches the label transform", "observations": (
            "exhaustive and aligned in both volumes; -1 is the black padding around the rotated ATUM "
            "section (43% of train voxels), not tissue"
        ),
    })


if __name__ == "__main__":
    main()
