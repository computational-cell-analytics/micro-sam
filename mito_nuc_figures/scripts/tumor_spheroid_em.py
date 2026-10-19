import os
from glob import glob

import h5py
import numpy as np

from figure_utils import DATA_ROOT, get_output_dir, save_image_grid, save_volume_figure, write_info


DATASET = "tumor_spheroid_em"
RESOLUTIONS = ("50-50-50", "100-100-100")
NOTES = "SBF-SEM tumor spheroid with gold NPs, {res} nm, manual 2D instance nucleus labels ({axis} slice)"


def render_slice(path, resolution, out_dir):
    name = os.path.splitext(os.path.basename(path))[0]
    axis = name.split("-")[-1].split("_")[0]
    with h5py.File(path, "r") as f:
        raw, nuclei, cells = f["raw"][:], f["labels/nuclei"][:], f["labels/cells"][:]
    notes = NOTES.format(res=resolution, axis=axis)
    out_path = os.path.join(out_dir, resolution, f"{name}.png")
    save_volume_figure(raw, {"nucleus": nuclei}, out_path, DATASET, f"2d_manual/{resolution}/{name}", notes=notes)
    volume = {
        "name": f"{resolution}/{name}", "shape": raw.shape, "voxel_size_nm": resolution, "path": path,
        "slice_axis": axis, "n_nuclei": int(len(np.unique(nuclei)) - 1), "n_cells": int(len(np.unique(cells)) - 1),
        "nucleus_fraction": round(float((nuclei > 0).mean()), 3),
    }
    print(f"Rendered {resolution}/{name}")
    return volume, (name.replace("Au_01-vol_01-", ""), raw, {"nucleus": nuclei})


def render_resolution(resolution, out_dir):
    paths = sorted(glob(os.path.join(DATA_ROOT, DATASET, "2d_manual", resolution, "*.h5")))
    results = [render_slice(path, resolution, out_dir) for path in paths]
    notes = f"SBF-SEM tumor spheroid, {resolution} nm, manual 2D instance nucleus labels, x/y/z slices"
    out_path = os.path.join(out_dir, f"overview_{resolution}.png")
    save_image_grid([sample for _, sample in results], out_path, DATASET, f"2d_manual/{resolution}", notes=notes)
    return [volume for volume, _ in results]


def main():
    out_dir = get_output_dir(DATASET)
    volumes = []
    for resolution in RESOLUTIONS:
        volumes.extend(render_resolution(resolution, out_dir))
    write_info(DATASET, {
        "classes": ["nucleus"], "modality": "SBF-SEM", "dimensionality": "2d",
        "label_quality": "manual 2D instance labels (nuclei and cells) on single slices in xy, xz and yz",
        "n_volumes": len(volumes), "volumes": volumes, "data_path": os.path.join(DATA_ROOT, DATASET, "2d_manual"),
        "download": "already local (2d_manual, 48 MB); 3d_automatic (67 GB zarr, automatic seg) not downloaded",
        "notes": "SBF-SEM tumor spheroid (Au_01-vol_01) with gold nanoparticles, single annotated slices",
        "issues": (
            "The 100-100-100 slices are the same 20 slices as 50-50-50, 2x downsampled (identical nucleus and cell "
            "counts per slice), so there are only 20 independent annotated images. Several z-slices are only 2-7 "
            "sections apart (z 0073/0075, 0169/0173/0180), so they are near-duplicates. The 3d_automatic source "
            "(67 GB zarr, automatic segmentation of cells, nuclei and NPs) is not local and was not downloaded."
        ),
        "observations": (
            "Labels align with the raw data. Close to exhaustive for nuclei in each slice (6-25 per slice, 9-21% of "
            "pixels), cut nucleus profiles at cell borders are included. A few labels have irregular outlines. Raw "
            "contrast is inverted (dark background, bright gold NP clusters), x/y slices are 701 px high (35 um z)."
        ),
    })


if __name__ == "__main__":
    main()
