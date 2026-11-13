"""Sweep the APG settings of the nuclei segmentation in the first division round of the time-lapse of Beke.

In t9 to t14 most nuclei are in mitosis. H2B-GFP then marks the condensed chromosomes, small bright bars and spots,
which APG with the defaults of 'hvit_l_cells' skips (2 to 6 nuclei instead of 7 to 14). The settings lower the APG
thresholds for small and weak objects and resample z, on the same top-hat input as 'segment_huisken_timelapse.py'.
Each result is filtered by brightness like the merge of that script and scored against the membrane cells.

The 'spots' settings detect the chromosomes by a threshold instead ('add_spots' of 'segment_huisken_timelapse.py'),
added to the 'default' APG result. 'spots_t0.6_r6' is the setting of the merge there.

Usage:
    python sweep_huisken_timelapse_nuclei.py -s min25  # one setting on t9 to t14 (see 'submit_..._nuclei_sweep.sh')
    python sweep_huisken_timelapse_nuclei.py --spots  # the threshold detection on t9 to t14, on CPU
    python sweep_huisken_timelapse_nuclei.py --score  # the scores of all results, as a table
    python sweep_huisken_timelapse_nuclei.py --collect  # one h5 per time point with all results, for napari
"""

import os
import argparse

import h5py
import tifffile
import numpy as np

from segment_huisken_timelapse import (
    OUTPUT_DIR, NORM_PERCENTILES, load_timepoint, tophat, get_norm_bounds, remove_dim_objects, add_spots,
)
from explore_huisken_data import SMALL_DEFAULT_SETTINGS, segment_small


SWEEP_DIR = os.path.join(OUTPUT_DIR, "nuclei_division_sweep")
COLLECT_DIR = "/mnt/vast-nhr/home/archit/u12090/micro-sam/timelapse_sweep_nuclei_h5"
CELL_PATH = os.path.join(OUTPUT_DIR, "C01_membrane_seg.h5")
TIMEPOINTS = (9, 10, 11, 12, 13, 14)
# The defaults of 'hvit_l' for volumes are foreground 0.7, candidates (1.5, 10.0), score 0.6 and min_size 100.
LOOSE = {"min_size": 25, "score_threshold": 0.4, "foreground_threshold": 0.5, "candidate_threshold": (0.5, 10.0)}
SETTINGS = {
    "default": {},
    "min25": {"apg_params": {"min_size": 25}},
    "min25_score04": {"apg_params": {"min_size": 25, "score_threshold": 0.4}},
    "min25_fg05": {"apg_params": {"min_size": 25, "foreground_threshold": 0.5}},
    "min25_cand05": {"apg_params": {"min_size": 25, "candidate_threshold": (0.5, 10.0)}},
    "loose": {"apg_params": LOOSE},
    "z2": {"scale": (2, 1, 1)},
    "z2_loose": {"scale": (2, 1, 1), "apg_params": {**LOOSE, "min_size": 50}},
}
SPOT_SETTINGS = {f"spots_t{threshold}_r6": {"threshold": threshold, "radius": 6} for threshold in (0.5, 0.6, 0.7)}
# A nucleus belongs to the cell that holds at least this fraction of it, as in 'analyze_huisken_volume_ratio.py'.
MIN_NUCLEUS_OVERLAP = 0.5


def get_path(t, name):
    return os.path.join(SWEEP_DIR, name, f"t{t:03d}.tif")


def run(name, timepoints):
    os.makedirs(os.path.join(SWEEP_DIR, name), exist_ok=True)
    settings = {**SMALL_DEFAULT_SETTINGS, "norm_percentiles": NORM_PERCENTILES, **SETTINGS[name]}
    for t in timepoints:
        if os.path.exists(get_path(t, name)):
            continue
        volume = tophat(load_timepoint(t))
        segmentation, n_objects, seconds = segment_small(volume, "apg", settings, get_norm_bounds(volume))
        tifffile.imwrite(get_path(t, name), segmentation, compression="zlib")
        print(f"t{t:03d} {name}: {n_objects} objects in {seconds:.1f} s", flush=True)


def run_spots(name, timepoints):
    """Add the spots to the 'default' APG result, as the merge of 'segment_huisken_timelapse.py' does."""
    os.makedirs(os.path.join(SWEEP_DIR, name), exist_ok=True)
    settings = SPOT_SETTINGS[name]
    for t in timepoints:
        raw = load_timepoint(t)
        nuclei, _ = remove_dim_objects(tifffile.imread(get_path(t, "default")), raw)
        segmentation, n_spots = add_spots(nuclei, raw, settings["threshold"], settings["radius"])
        tifffile.imwrite(get_path(t, name), segmentation, compression="zlib")
        print(f"t{t:03d} {name}: {n_spots} spots added", flush=True)


def load_result(t, name):
    segmentation = tifffile.imread(get_path(t, name))
    # 'add_spots' filters the spot results already.
    if name in SETTINGS:
        segmentation, _ = remove_dim_objects(segmentation, load_timepoint(t))
    return segmentation


def score(nuclei, cells):
    """Count the nuclei and the cells with no, one or more nuclei, and the nuclei outside of all cells."""
    nucleus_ids = np.unique(nuclei)[1:]
    cell_ids = np.unique(cells)[1:]
    nucleus_voxels = np.bincount(nuclei.ravel())
    owners = []
    for nucleus_id in nucleus_ids:
        ids, counts = np.unique(cells[nuclei == nucleus_id], return_counts=True)
        counts[ids == 0] = 0
        best = np.argmax(counts)
        owners.append(ids[best] if counts[best] >= MIN_NUCLEUS_OVERLAP * nucleus_voxels[nucleus_id] else 0)
    owners = np.array(owners, dtype="int64")
    per_cell = np.array([np.sum(owners == cell_id) for cell_id in cell_ids])
    return {
        "nuclei": len(nucleus_ids), "cells": len(cell_ids), "cells_0": int(np.sum(per_cell == 0)),
        "cells_1": int(np.sum(per_cell == 1)), "cells_2+": int(np.sum(per_cell >= 2)),
        "outside": int(np.sum(owners == 0)),
    }


def score_all(names):
    rows = []
    with h5py.File(CELL_PATH, "r") as f:
        cells = {t: f["segmentation"][t] for t in TIMEPOINTS}
    for name in names:
        for t in TIMEPOINTS:
            if os.path.exists(get_path(t, name)):
                rows.append({"setting": name, "t": t, **score(load_result(t, name), cells[t])})
    for row in rows:
        print("\t".join(str(value) for value in row.values()))
    totals = {}
    for row in rows:
        total = totals.setdefault(row["setting"], {"nuclei": 0, "cells_1": 0, "cells_2+": 0, "outside": 0})
        for key in total:
            total[key] += row[key]
    print("\nsetting\tnuclei\tcells_1\tcells_2+\toutside (summed over t9 to t14)")
    for name, total in totals.items():
        print(name, *total.values(), sep="\t")


def collect(names):
    """Write one h5 per time point with the raw data, the membrane cells and each filtered result."""
    os.makedirs(COLLECT_DIR, exist_ok=True)
    for t in TIMEPOINTS:
        path = os.path.join(COLLECT_DIR, f"sweep_nuclei_division_t{t:03d}.h5")
        with h5py.File(path, "w") as f, h5py.File(CELL_PATH, "r") as f_cells:
            f.create_dataset("raw", data=load_timepoint(t), compression="gzip")
            f.create_dataset("cells", data=f_cells["segmentation"][t], compression="gzip")
            for name in names:
                if os.path.exists(get_path(t, name)):
                    segmentation = load_result(t, name)
                    ds = f.create_dataset(name, data=segmentation, compression="gzip")
                    ds.attrs["n_objects"] = len(np.unique(segmentation)) - 1
        print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-s", "--settings", nargs="+", choices=(*SETTINGS, *SPOT_SETTINGS), help="By default all.")
    parser.add_argument("-t", "--timepoints", nargs="+", type=int, default=list(TIMEPOINTS))
    parser.add_argument("--spots", action="store_true", help="Run the threshold detection.")
    parser.add_argument("--score", action="store_true", help="Score all results against the membrane cells.")
    parser.add_argument("--collect", action="store_true", help="Write one h5 per time point for napari.")
    parser.add_argument("--list", action="store_true", help="Print the names of the settings.")
    args = parser.parse_args()

    if args.list:
        print(" ".join(SETTINGS))
    elif args.spots:
        for name in args.settings or SPOT_SETTINGS:
            run_spots(name, args.timepoints)
    elif args.score:
        score_all(args.settings or [*SETTINGS, *SPOT_SETTINGS])
    elif args.collect:
        collect(args.settings or [*SETTINGS, *SPOT_SETTINGS])
    else:
        for name in args.settings or SETTINGS:
            run(name, args.timepoints)


if __name__ == "__main__":
    main()
