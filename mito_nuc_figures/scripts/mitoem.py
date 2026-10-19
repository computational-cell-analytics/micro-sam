"""Overview figures for MitoEM (rat and human cortex, ATUM-SEM, instance mito labels for train and val)."""
import os

import numpy as np
import z5py

from figure_utils import DATA_ROOT, get_output_dir, save_volume_figure, write_info


DATASET = "mitoem"
DATA_DIR = os.path.join(DATA_ROOT, DATASET)
NOTES = "ATUM-SEM {species} cortex, 8x8x30 nm (z,y,x = 30,8,8), {labels}"


CROP_SIZE = 1024


class StridedArray:
    """Lazy wrapper for z5py datasets, which do not support strided indexing, with an optional in-plane crop."""

    def __init__(self, dataset, offset=(0, 0, 0), shape=None):
        self.dataset = dataset
        self.offset = offset
        self.shape = tuple(dataset.shape) if shape is None else tuple(shape)
        self.ndim = len(self.shape)

    def __getitem__(self, index):
        index = index if isinstance(index, tuple) else (index,)
        dense, strides = [], []
        for item, offset, size in zip(index, self.offset, self.shape):
            if isinstance(item, slice):
                start, stop = item.start or 0, size if item.stop is None else item.stop
                dense.append(slice(start + offset, stop + offset))
                strides.append(slice(None, None, item.step))
            else:
                dense.append(item + offset)
        return np.asarray(self.dataset[tuple(dense)])[tuple(strides)]


def render_center_crop(f, name, notes, out_dir):
    """Render a full-depth center crop at full in-plane resolution, where single mitochondria are visible."""
    nz, ny, nx = f["raw"].shape
    offset = (0, ny // 2 - CROP_SIZE // 2, nx // 2 - CROP_SIZE // 2)
    shape = (nz, CROP_SIZE, CROP_SIZE)
    raw = StridedArray(f["raw"], offset, shape)
    labels = {"mito": StridedArray(f["labels"], offset, shape)}
    crop_name = f"{name}_center_crop_y{offset[1]}_x{offset[2]}"
    out_path = os.path.join(out_dir, f"{crop_name}.png")
    save_volume_figure(raw, labels, out_path, DATASET, crop_name, notes=notes + ", full-res center crop")


def render_volume(species, split, out_dir):
    path = os.path.join(DATA_DIR, f"{species}_{split}.n5")
    f = z5py.File(path, "r")
    raw = StridedArray(f["raw"])
    has_labels = "labels" in f
    labels = {"mito": StridedArray(f["labels"])} if has_labels else {}
    label_note = "manual (proofread) instance mito labels" if has_labels else "no labels (challenge test volume)"
    notes = NOTES.format(species=species, labels=label_note)
    name = f"{species}_{split}"
    save_volume_figure(raw, labels, os.path.join(out_dir, f"{name}.png"), DATASET, name, notes=notes)
    if has_labels:
        render_center_crop(f, name, notes, out_dir)
    volume = {"name": name, "shape": raw.shape, "voxel_size_nm": (30, 8, 8), "path": path, "has_labels": has_labels}
    if has_labels:
        volume["max_id"] = f["labels"].attrs.get("maxId")
    return volume


def main():
    out_dir = get_output_dir(DATASET)
    volumes = [
        render_volume(species, split, out_dir) for species in ("human", "rat") for split in ("train", "val", "test")
    ]
    write_info(DATASET, {
        "classes": ["mito"], "modality": "ATUM-SEM (serial section SEM)", "dimensionality": "3d",
        "label_quality": "manual, proofread instance labels (train / val only)",
        "n_volumes": len(volumes), "volumes": volumes, "data_path": DATA_DIR, "download": "none, already local",
        "notes": "train = slices 0-399, val = 400-499, test = 499-999 of each 1000 x 4096 x 4096 volume",
        "issues": (
            "test volumes have no labels; the test volume has 501 slices because torch-em mitoem.py "
            "_create_volume starts it at the last val slice (z_start = slices[-1]), so slice 499 is in val and test"
        ),
        "observations": (
            "train / val labels are exhaustive and aligned in the full-res center crops of both species (dark "
            "lysosome-like bodies left unlabeled); full-volume panels are too downsampled to judge single mitos; "
            "occasional section artifacts (white lines in ZX)"
        ),
    })


if __name__ == "__main__":
    main()
