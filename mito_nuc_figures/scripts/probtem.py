"""Figures for ProbTEM (2D TEM skeletal muscle, semantic mito).

Only part of the test split is present locally: 6 test inputs but 3 test targets, and no train split.
A fresh gdown download of the Google Drive folder fails with "Access denied" for all image files.
So only the 3 complete test images (processed by torch-em into processed/test/x_{0,1,2}.h5) are rendered.
"""
import os
from glob import glob

import h5py
from skimage.measure import label as connected_components

from figure_utils import DATA_ROOT, get_output_dir, save_volume_figure, write_info


DATASET = "probtem"
DATA_PATH = os.path.join(DATA_ROOT, DATASET)
SOURCE_ROOT = os.path.join(DATA_PATH, "Deeppi-EM", "mitoseg_deploy", "datasets", "Skeletal_muscle")
NOTES = "TEM skeletal muscle, px size n/a, manual semantic"


def render_image(path, out_dir):
    name = os.path.splitext(os.path.basename(path))[0]
    with h5py.File(path, "r") as f:
        raw, labels = f["raw"], f["labels"]
        label_data = labels[:]
        out_path = os.path.join(out_dir, f"test_{name}.png")
        save_volume_figure(raw, {"mito": labels}, out_path, DATASET, f"test {name}", notes=NOTES)
        print(f"Rendered {out_path}")
        return {
            "name": f"test/{name}", "shape": raw.shape, "voxel_size_nm": None, "path": path,
            "mito_foreground_fraction": float((label_data > 0).mean()),
            "n_mito_connected_components": int(connected_components(label_data > 0).max()),
        }


def list_source_files():
    files = {}
    for split in ("train", "test"):
        for folder in ("input", "target"):
            pattern = os.path.join(SOURCE_ROOT, split, folder, "*")
            files[f"{split}/{folder}"] = sorted(os.path.basename(path) for path in glob(pattern))
    return files


def main():
    out_dir = get_output_dir(DATASET)
    paths = sorted(glob(os.path.join(DATA_PATH, "processed", "test", "*.h5")))
    volumes = [render_image(path, out_dir) for path in paths]
    source_files = list_source_files()
    write_info(DATASET, {
        "classes": ["mito"], "modality": "TEM", "dimensionality": "2d", "label_quality": "manual, semantic",
        "n_volumes": len(volumes), "volumes": volumes, "data_path": DATA_PATH, "local_source_files": source_files,
        "download": (
            "nothing new: gdown of the Google Drive folder (id 1n2ZqbJEHPyMB_6a6OTBBACt5Jct2PZJc) into "
            "DATA_ROOT/probtem_redownload got only the __MACOSX resource forks, every image file failed with "
            "'Access denied' (Drive quota / permission)"
        ),
        "notes": NOTES,
        "issues": (
            "Local copy is incomplete: no train split (21 images expected), test has 6 inputs but only 3 targets "
            "(y_0..y_2). Only 3 of 27 labeled images are available. get_probtem_data returns early once "
            "processed/test has any h5 file, so it never notices the 3 missing test targets."
        ),
        "observations": (
            "Labels aligned with the raw data. Very large, densely packed mitos (27-42% foreground per image); "
            "x_0 and x_2 look exhaustive, x_1 leaves a few darker / ragged mitos in the center unlabeled. "
            "Touching mitos are separated by thin background gaps in the semantic mask, so connected components "
            "give usable instances in most places."
        ),
    })


if __name__ == "__main__":
    main()
