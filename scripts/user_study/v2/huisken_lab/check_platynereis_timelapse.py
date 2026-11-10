"""Look at the segmentation and the tracks of the Platynereis time-lapse (Huisken lab, 3-6 hpf) in napari.

The results of micro-sam2 with the old and the new (#1399) default parameters are in DATA_DIR/<version>:
C02_nuclei_seg.h5 (nuclei), C01_membrane_seg.h5 (cells) and C02_nuclei_tracks_fixed.h5 (tracked nuclei).

Usage:
    python check_platynereis_timelapse.py --nuclei --membrane -t 0 44 88
    python check_platynereis_timelapse.py --tracks --version new_defaults
"""

import os
import argparse

import h5py
import numpy as np

import napari


DATA_DIR = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/huisken_data/platynereis_timelapse_3-6hpf"
# Time, then the voxel size in z over the voxel size in xy (2.5 / 0.406 um).
SCALE = (1, 2.5 / 0.406, 1, 1)
VERSIONS = ["old_defaults", "new_defaults"]


def load(path, key, timepoints):
    with h5py.File(path, "r") as f:
        return f[key][:] if timepoints is None else f[key][timepoints]


def load_tracks(path, timepoints):
    with h5py.File(path, "r") as f:
        tracks, lineage = f["tracks"][:], f["lineage"][:]
    if timepoints is not None:
        tracks = tracks[np.isin(tracks[:, 1], timepoints)]
        tracks[:, 1] = np.searchsorted(timepoints, tracks[:, 1])
    tracks = tracks[np.lexsort((tracks[:, 1], tracks[:, 0]))]
    # The graph maps each daughter track to its parent track.
    track_ids = set(tracks[:, 0].astype(int))
    graph = {
        int(track_id): [int(parent)] for track_id, _, _, parent in lineage
        if parent > 0 and track_id in track_ids and parent in track_ids
    }
    return tracks, graph


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nuclei", action="store_true", help="Show the nuclei.")
    parser.add_argument("--membrane", action="store_true", help="Show the cells.")
    parser.add_argument("--tracks", action="store_true", help="Show the tracked nuclei and their tracks.")
    parser.add_argument("--version", nargs="+", default=VERSIONS, choices=VERSIONS, help="The results to show.")
    parser.add_argument("-t", "--timepoints", type=int, nargs="+", help="The time points to show (default: all 89).")
    args = parser.parse_args()
    timepoints = None if args.timepoints is None else sorted(args.timepoints)

    viewer = napari.Viewer()
    # The raw data is the same for both versions, so load it once from the first one.
    first = os.path.join(DATA_DIR, args.version[0])
    if args.nuclei or args.tracks:
        raw = load(os.path.join(first, "C02_nuclei_seg.h5"), "raw", timepoints)
        viewer.add_image(raw, name="raw nuclei", scale=SCALE)
    if args.membrane:
        raw = load(os.path.join(first, "C01_membrane_seg.h5"), "raw", timepoints)
        viewer.add_image(raw, name="raw membrane", scale=SCALE)

    for version in args.version:
        folder = os.path.join(DATA_DIR, version)
        if args.nuclei:
            nuclei = load(os.path.join(folder, "C02_nuclei_seg.h5"), "segmentation", timepoints)
            viewer.add_labels(nuclei, name=f"nuclei {version}", scale=SCALE)
        if args.membrane:
            cells = load(os.path.join(folder, "C01_membrane_seg.h5"), "segmentation", timepoints)
            viewer.add_labels(cells, name=f"cells {version}", scale=SCALE)
        if args.tracks:
            path = os.path.join(folder, "C02_nuclei_tracks_fixed.h5")
            viewer.add_labels(load(path, "segmentation", timepoints), name=f"tracked nuclei {version}", scale=SCALE)
            tracks, graph = load_tracks(path, timepoints)
            viewer.add_tracks(tracks, graph=graph, name=f"tracks {version}", scale=SCALE, tail_length=10)
    napari.run()


if __name__ == "__main__":
    main()
