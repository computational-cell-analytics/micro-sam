"""Export the tracked nuclei and the cell segmentation of the Beke time-lapse (classic_lsm/timelapse_3-6hpf) as tiffs.

Each time point is one ZYX tiff stack, with the voxel size in the ImageJ metadata:

    <output_folder>/<version>/tracked_nuclei/t000.tif ...  # the nuclei, the id of each nucleus is its track id
    <output_folder>/<version>/cells/t000.tif ...  # the cells of the membrane segmentation
    <output_folder>/<version>/tracked_nuclei/res_track.txt  # the lineage in the CTC format

The lineage file has one row per track: track_id t_start t_end parent_id, where parent_id 0 means no parent.

Usage:
    python export_huisken_tiffs.py /path/to/export  # all time points of the old defaults
    python export_huisken_tiffs.py /path/to/export --version new_defaults -t 0 44 88  # some time points
"""

import os
import argparse

import h5py
import tifffile
import numpy as np

from analyze_huisken_volume_ratio import DATA_DIR, VERSIONS

# The voxel size in micrometer (z, y, x).
VOXEL_SIZE = (2.5, 0.406, 0.406)
SEGMENTATIONS = {"tracked_nuclei": "C02_nuclei_tracks_fixed.h5", "cells": "C01_membrane_seg.h5"}


def write_stack(path, stack):
    tifffile.imwrite(
        path, stack, imagej=True, compression="zlib",
        resolution=(1 / VOXEL_SIZE[2], 1 / VOXEL_SIZE[1]),
        metadata={"axes": "ZYX", "spacing": VOXEL_SIZE[0], "unit": "um"},
    )


def export_segmentation(input_path, output_folder, timepoints):
    os.makedirs(output_folder, exist_ok=True)
    with h5py.File(input_path, "r") as f:
        ds = f["segmentation"]
        timepoints = range(ds.shape[0]) if timepoints is None else timepoints
        for t in timepoints:
            if not 0 <= t < ds.shape[0]:
                raise ValueError(f"Time point {t} is not in the {ds.shape[0]} time points of {input_path}.")
            write_stack(os.path.join(output_folder, f"t{t:03}.tif"), ds[t])
            print(f"Exported t{t} of {os.path.basename(input_path)}", flush=True)
        lineage = f["lineage"][:] if "lineage" in f else None
    if lineage is not None:
        np.savetxt(os.path.join(output_folder, "res_track.txt"), lineage, fmt="%d")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_folder", help="The folder for the tiffs, they go to a subfolder per version.")
    parser.add_argument("--version", default="old_defaults", choices=VERSIONS, help="The results to export.")
    parser.add_argument("-t", "--timepoints", type=int, nargs="+", help="The time points to export, default all.")
    args = parser.parse_args()
    for name, file_name in SEGMENTATIONS.items():
        input_path = os.path.join(DATA_DIR, args.version, file_name)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"The segmentation {input_path} does not exist.")
        export_segmentation(input_path, os.path.join(args.output_folder, args.version, name), args.timepoints)


if __name__ == "__main__":
    main()
