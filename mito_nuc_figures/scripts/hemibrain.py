"""Overview figures for the cached Hemibrain FIB-SEM boxes with mitochondria labels (class id 1 only)."""
import os
from glob import glob

import h5py
import numpy as np

from figure_utils import DATA_ROOT, get_output_dir, save_volume_figure, write_info


DATASET = "hemibrain"
DATA_DIR = os.path.join(DATA_ROOT, DATASET)
MITO_ID = 1
NOTES = "FIB-SEM Drosophila brain, 8 nm isotropic, mito detection class id 1 (16 nm, upsampled), semantic"


class ClassMask:
    """Lazy binary view of one class id in a multi-class label volume."""

    def __init__(self, dataset, class_id):
        self.dataset = dataset
        self.class_id = class_id
        self.shape = dataset.shape
        self.ndim = dataset.ndim

    def __getitem__(self, index):
        return (self.dataset[index] == self.class_id).astype("uint8")


def get_class_fractions(labels, n_slices=8):
    """Estimate the voxel fraction of each label id on evenly spaced slices."""
    counts = {}
    for z in np.linspace(0, labels.shape[0] - 1, n_slices).astype(int):
        ids, n = np.unique(labels[z], return_counts=True)
        for i, c in zip(ids.tolist(), n.tolist()):
            counts[i] = counts.get(i, 0) + c
    total = sum(counts.values())
    return {int(i): round(c / total, 4) for i, c in sorted(counts.items())}


def render_box(path, out_dir):
    with h5py.File(path, "r") as f:
        raw, labels = f["raw"], f["labels"]
        bbox = f.attrs["bounding_box"].tolist()
        name = "mito_x{}_y{}_z{}".format(bbox[0], bbox[2], bbox[4])
        mito = ClassMask(labels, MITO_ID)
        save_volume_figure(raw, {"mito": mito}, os.path.join(out_dir, f"{name}.png"), DATASET, name, notes=NOTES)
        return {
            "name": name, "shape": raw.shape, "voxel_size_nm": (8, 8, 8), "path": path,
            "bounding_box_xyz_8nm": bbox, "class_fractions": get_class_fractions(labels),
        }


def main():
    out_dir = get_output_dir(DATASET)
    volumes = [render_box(path, out_dir) for path in sorted(glob(os.path.join(DATA_DIR, "mito_*.h5")))]
    write_info(DATASET, {
        "classes": ["mito"], "modality": "FIB-SEM", "dimensionality": "3d",
        "label_quality": "automated mito detection (Janelia flyem 'mito' volume), semantic, 16 nm upsampled to 8 nm",
        "n_volumes": len(volumes), "volumes": volumes, "data_path": DATA_DIR, "download": "none, already local",
        "notes": "only one 1024^3 box is cached; the full volume is ~34k x 40k x 41k voxels and is streamed by box",
        "issues": "", "observations": (
            "class 1 is aligned and looks exhaustive, but boundaries are blocky (16 nm upsampled) and a "
            "few large dark mitos are only partly covered; class 2/3/4 are "
            "interior/background/unlabeled and are not mito"
        ),
    })


if __name__ == "__main__":
    main()
