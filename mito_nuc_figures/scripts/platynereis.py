import os
from glob import glob

import h5py
import numpy as np

from figure_utils import DATA_ROOT, get_output_dir, save_volume_figure, write_info


DATASET = "platynereis"
LABEL_KEY = "volumes/labels/nucleus_instance_labels"
IGNORE_LABEL = -1
NOTES = "SBEM Platynereis larva (6 dpf), ~80x80x100 nm (est.), instance nucleus labels, outside the box = -1 (ignore)"


def get_annotated_bbox(labels):
    """Get the bounding box of all voxels that are not the ignore label."""
    coords = np.where(labels != IGNORE_LABEL)
    return tuple(slice(int(c.min()), int(c.max()) + 1) for c in coords)


def summarize_labels(data):
    ids = np.unique(data)
    return {
        "n_nuclei": int((ids > 0).sum()),
        "fraction_nucleus": round(float((data > 0).mean()), 3),
        "fraction_background": round(float((data == 0).mean()), 3),
        "fraction_unlabeled_ignore": round(float((data == IGNORE_LABEL).mean()), 3),
    }


def render_volume(path, out_dir):
    name = os.path.splitext(os.path.basename(path))[0]
    with h5py.File(path, "r") as f:
        shape = f["volumes/raw"].shape
        labels = f[LABEL_KEY][:]
        bbox = get_annotated_bbox(labels)
        raw = f["volumes/raw"][bbox]
    bbox_str = ", ".join(f"{b.start}:{b.stop}" for b in bbox)
    crop_labels = labels[bbox]
    overlays = {"nucleus": np.where(crop_labels == IGNORE_LABEL, 0, crop_labels)}
    out_path = os.path.join(out_dir, f"{name}.png")
    volume_name = f"nuclei/{name} {tuple(shape)} cropped to annotated box [{bbox_str}]"
    save_volume_figure(raw, overlays, out_path, DATASET, volume_name, notes=NOTES)
    volume = {
        "name": name, "shape": shape, "voxel_size_nm": "~(100, 80, 80), not stored in file", "path": path,
        "annotated_bbox_zyx": bbox_str, "unlabeled_inside_bbox": round(float((crop_labels == IGNORE_LABEL).mean()), 3),
    }
    volume.update(summarize_labels(labels))
    print(f"Rendered {name}")
    return volume


def main():
    out_dir = get_output_dir(DATASET)
    paths = sorted(glob(os.path.join(DATA_ROOT, DATASET, "nuclei", "train_data_nuclei_*.h5")))
    volumes = [render_volume(path, out_dir) for path in paths]
    write_info(DATASET, {
        "classes": ["nucleus"], "modality": "SBEM", "dimensionality": "3d",
        "label_quality": "instance nucleus labels with an ignore label (-1) for unlabeled regions",
        "n_volumes": len(volumes), "volumes": volumes, "data_path": os.path.join(DATA_ROOT, DATASET, "nuclei"),
        "download": "already local (zenodo 3675220 nuclei.zip, extracted)",
        "notes": NOTES,
        "issues": (
            "nucleus_instance_labels uses -1 for unlabeled voxels: a halo of 16 z-slices and 32 xy-pixels around every "
            "volume (25 px / 2 slices for volume 05), 29-45% of all voxels. "
            "get_platynereis_nuclei_dataset passes these labels unchanged, so with binary=True the -1 halo becomes "
            "background and instance-label transforms can treat it "
            "as an extra object; mask it (or use the rois kwarg to crop to the annotated box) for training. "
            "Voxel size is not stored; ~80x80x100 nm is an estimate. "
            "DATA_ROOT/platynereis_nuclei is not a separate dataset: vframes/images/0000-0131.tif are the 132 z-slices "
            "of platynereis/nuclei/train_data_nuclei_01.h5 volumes/raw as single tifs (checked equal on slices 0, 50, "
            "131), no labels, probably a SAM2 video-frame export."
        ),
        "observations": (
            "Inside the annotated box the labels are exhaustive and aligned (0% ignore inside the box). Volumes 01, "
            "03, 05 and 08 are dense nucleus-rich brain regions (100-141 nuclei); 06, 07, 10-12 are sparse muscle / "
            "gut regions (23-47 nuclei) where large dark cells without a visible nucleus stay unlabeled. Nuclei show "
            "heterochromatin texture; labels follow the nuclear envelope well."
        ),
    })


if __name__ == "__main__":
    main()
