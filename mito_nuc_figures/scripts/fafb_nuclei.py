import os
from glob import glob

import h5py
import numpy as np

from figure_utils import DATA_ROOT, get_output_dir, save_image_grid, save_volume_figure, write_info


DATASET = "fafb_nuclei"
VOXEL_SIZE = (40, 32, 32)
NOTES = "ssTEM adult fly brain (FAFB), ~32x32x40 nm (mip3), automatic FlyWire nucleus segmentation (instance)"
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
        "classes": ["nucleus"], "modality": "ssTEM (FAFB / FlyWire v1 aligned)", "dimensionality": "3d",
        "label_quality": "automatic nucleus segmentation (FlyWire nucleus seg v5), instance",
        "n_volumes": len(volumes), "volumes": volumes, "data_path": os.path.join(DATA_ROOT, DATASET),
        "download": "already local (owncloud zip, 778 MB extracted)",
        "notes": NOTES,
        "issues": (
            "Voxel size not stored in the h5 files; raw is FlyWire v1 aligned EM at mip3 (32x32x40 nm), labels are the "
            "nucleus seg v5_z_intp_intp at its mip0 (torch-em scripts/datasets/electron_microscopy/"
            "check_fafb_nuclei.py). "
            "Labels are an automatic segmentation, not manual GT."
        ),
        "observations": (
            "Aligned in xy and ZX. Close to exhaustive: nearly every nucleus is labeled, 7-35 nuclei per middle slice, "
            "smooth outlines. Raw has occasional missing / black sections (seen in the ZX view of train_block_0) and "
            "big blank tracheae; labels are interpolated through missing sections. Very dense neuronal cell-body "
            "rinds, so crops are nucleus-rich."
        ),
    })


if __name__ == "__main__":
    main()
