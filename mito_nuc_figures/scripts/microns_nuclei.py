import os
from glob import glob

import h5py
import numpy as np

from figure_utils import DATA_ROOT, get_output_dir, save_image_grid, save_volume_figure, write_info


DATASET = "microns_nuclei"
VOXEL_SIZE = (40, 64, 64)
NOTES = "ssTEM mouse visual cortex (MICrONS minnie65), ~64x64x40 nm (mip3), automatic nucleus segmentation (instance)"
SPLITS = ("train", "val", "test")


def count_ids(labels):
    """Count the nucleus ids on a strided view; nuclei are large, so the stride does not miss any."""
    return int(len(np.unique(labels[::2, ::4, ::4])) - 1)


def render_crop(path, split, out_dir):
    name = os.path.splitext(os.path.basename(path))[0]
    with h5py.File(path, "r") as f:
        raw, labels = f["raw"], f["labels/nuclei"]
        out_path = os.path.join(out_dir, split, f"{name}.png")
        save_volume_figure(raw, {"nucleus": labels}, out_path, DATASET, f"{split}/{name}", notes=NOTES)
        z = raw.shape[0] // 2
        sample = (name, raw[z], {"nucleus": labels[z]})
        volume = {
            "name": f"{split}/{name}", "shape": raw.shape, "voxel_size_nm": VOXEL_SIZE, "path": path,
            "n_nuclei": count_ids(labels),
        }
    print(f"Rendered {split}/{name}")
    return volume, sample


def render_split(split, out_dir):
    paths = sorted(glob(os.path.join(DATA_ROOT, DATASET, split, "*.h5")))
    results = [render_crop(path, split, out_dir) for path in paths]
    samples = [sample for _, sample in results]
    out_path = os.path.join(out_dir, f"overview_{split}.png")
    save_image_grid(samples, out_path, DATASET, f"{split} (middle slices)", notes=NOTES)
    return [volume for volume, _ in results]


def main():
    out_dir = get_output_dir(DATASET)
    volumes = []
    for split in SPLITS:
        volumes.extend(render_split(split, out_dir))
    write_info(DATASET, {
        "classes": ["nucleus"], "modality": "ssTEM (MICrONS minnie65, bossdb)", "dimensionality": "3d",
        "label_quality": "automatic nucleus segmentation (MICrONS minnie65 nuclei), instance",
        "n_volumes": len(volumes), "volumes": volumes, "data_path": os.path.join(DATA_ROOT, DATASET),
        "download": "already local (owncloud zip, 1.1 GB extracted)",
        "notes": NOTES,
        "issues": (
            "Voxel size not stored in the h5 files; raw is minnie65 EM at mip3 (64x64x40 nm), labels are the minnie65 "
            "nuclei volume at mip0 (torch-em scripts/datasets/electron_microscopy/check_microns_nuclei.py). "
            "Labels are an automatic segmentation, not manual GT."
        ),
        "observations": (
            "Aligned in xy and ZX. Close to exhaustive, 3-8 large nuclei per slice, including thin endothelial / glial "
            "nuclei next to blood vessels. Outlines are jagged at this resolution and some nuclei have small notches. "
            "Raw has strong vertical stripe artifacts, folds / cracks (train_block_0), many partly blank sections "
            "(horizontal lines in the ZX views) and large blank vessel lumens."
        ),
    })


if __name__ == "__main__":
    main()
