"""Segmentation of the nuclei with micro-sam2 in the time-lapse of Beke (huisken_data/classic_lsm/timelapse_3-6hpf).

The time-lapse runs from 3 hpf (4 to 8 cells) to 6 hpf every 2 min, 89 time points of (92, 609, 604) uint16 (ZYX),
voxel size 0.406 x 0.406 x 2.5 um, one tif per time point and channel: C01 the membranes, C02 the nuclei (H2B-GFP).

The nuclei are segmented with APG and 'hvit_l_cells' on every time point on its own, the setting chosen by eye on
t0, t44 and t88 (AIS splits the nuclei, z-scaling and other normalizations change little):
- From t~16 on, some cells are filled with diffuse H2B-GFP, which APG segments as one object of the whole cell. A
  top-hat of every z-slice (TOPHAT_SIZE) flattens these cells (~200-300 px) and keeps the nuclei (~35-60 px). Larger
  sizes add false objects at t0.
- The intensity drops ~4x over the time-lapse, so every time point is normalized with the percentiles of its own
  volume after the top-hat (NORM_PERCENTILES).
- The cells that remain and dim streaks next to the nuclei are removed afterwards by their brightness
  (MIN_RELATIVE_BRIGHTNESS).
- In the first division round (SPOT_TIMEPOINTS) most nuclei are in mitosis, and APG skips the condensed chromosomes.
  There the merge adds them by a threshold detection (see 'add_spots' and 'sweep_huisken_timelapse_nuclei.py').
'registered_t000069_C02.tif' is empty (all zeros) in the upload, so t69 has no nuclei.

Usage:
    python segment_huisken_timelapse.py -t 0-4  # these time points, one tif each (see 'submit_huisken_timelapse.sh')
    python segment_huisken_timelapse.py --merge  # all time points with the raw data into one h5 of (T, Z, Y, X)
"""

import os
import argparse
from glob import glob

import h5py
import tifffile
import numpy as np
from scipy import ndimage

from explore_huisken_data import DATA_ROOT, OUTPUT_ROOT, MODEL_TYPE, SMALL_DEFAULT_SETTINGS, segment_small


INPUT_DIR = os.path.join(DATA_ROOT, "classic_lsm", "timelapse_3-6hpf")
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "timelapse_3-6hpf")
# The segmentation of every time point, kept so that a resubmitted job continues with the missing ones.
RUN_DIR = os.path.join(OUTPUT_DIR, f"nuclei_{MODEL_TYPE}_raw_tophat41_apg")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "C02_nuclei_seg.h5")
N_TIMEPOINTS = 89
VOXEL_SIZE = (2.5, 0.406, 0.406)
ENGINE = "apg"
# The background of every z-slice is a grey opening of this size (px) of the slice smoothed with TOPHAT_SIGMA, which
# keeps the noise minima out of the opening. The opening is blocky, so it is smoothed at a quarter of its size.
TOPHAT_SIZE = 41
TOPHAT_SIGMA = 3
# The nuclei are < 1% of the voxels after the top-hat, so the upper bound is the 99.9th percentile (the nuclei are at
# ~0.6 of it, the 98th is in the background noise).
NORM_PERCENTILES = (2, 99.9)
# The objects darker than this fraction of the median object brightness of their time point are removed: the cells
# filled with diffuse signal and the dim streaks (<= 0.37), the real nuclei are >= 0.67 at t0, t44 and t88.
MIN_RELATIVE_BRIGHTNESS = 0.5
CHUNKS = (1, 32, 256, 256)
# The threshold detection of the chromosomes: the smoothed top-hat above SPOT_THRESHOLD of its 99.9th percentile, the
# pieces within SPOT_RADIUS px in xy as one object. The objects that are small, sparse in their bounding box (joined
# noise) or long in xy (light-sheet streaks) are removed. A spot that lies mostly in an APG nucleus is that nucleus.
SPOT_TIMEPOINTS = (9, 10, 11, 12, 13, 14)
SPOT_THRESHOLD, SPOT_RADIUS, SPOT_SIGMA = 0.6, 6, 1
SPOT_MIN_SIZE, SPOT_MIN_FILL, SPOT_MAX_ELONGATION = 100, 0.08, 4
SPOT_MAX_NUCLEUS_OVERLAP = 0.25


def load_timepoint(t):
    return tifffile.imread(os.path.join(INPUT_DIR, f"registered_t{t:06d}_C02.tif"))


def tophat(raw):
    volume = raw.astype("float32")
    smoothed = ndimage.gaussian_filter(volume, (0, TOPHAT_SIGMA, TOPHAT_SIGMA))
    background = ndimage.grey_opening(smoothed, size=(1, TOPHAT_SIZE, TOPHAT_SIZE))
    background = ndimage.gaussian_filter(background, (0, TOPHAT_SIZE / 4, TOPHAT_SIZE / 4))
    return np.clip(volume - background, 0, None)


def get_norm_bounds(volume):
    # The (lower, upper) bounds in the per-channel shape of 'automatic_instance_segmentation'.
    lower, upper = np.percentile(volume, NORM_PERCENTILES)
    return np.array([lower], dtype="float32"), np.array([upper], dtype="float32")


def parse_timepoints(values):
    timepoints = []
    for value in values:
        start, _, stop = value.partition("-")
        timepoints.extend(range(int(start), int(stop or start) + 1))
    return timepoints


def run_timepoints(timepoints):
    os.makedirs(RUN_DIR, exist_ok=True)
    settings = {**SMALL_DEFAULT_SETTINGS, "norm_percentiles": NORM_PERCENTILES}
    for t in timepoints:
        out = os.path.join(RUN_DIR, f"t{t:03d}.tif")
        if os.path.exists(out):
            continue
        volume = tophat(load_timepoint(t))
        segmentation, n_objects, seconds = segment_small(volume, ENGINE, settings, get_norm_bounds(volume))
        tifffile.imwrite(out, segmentation, compression="zlib")
        print(f"t{t:03d}: {n_objects} objects in {seconds:.1f} s", flush=True)


def remove_dim_objects(segmentation, raw):
    """Remove the objects whose mean raw intensity is below MIN_RELATIVE_BRIGHTNESS of the median over all objects."""
    ids = np.unique(segmentation)[1:]
    if len(ids) == 0:
        return segmentation, 0
    brightness = ndimage.mean(raw, segmentation, ids)
    dim = ids[brightness < MIN_RELATIVE_BRIGHTNESS * np.median(brightness)]
    return np.where(np.isin(segmentation, dim), 0, segmentation), len(dim)


def detect_spots(volume, threshold, radius):
    """Label the bright spots of the top-hat volume, with the pieces within 'radius' px in xy as one object."""
    smoothed = ndimage.gaussian_filter(volume, (0, SPOT_SIGMA, SPOT_SIGMA))
    mask = smoothed > threshold * np.percentile(volume, 99.9)
    disk = np.hypot(*np.mgrid[-radius:radius + 1, -radius:radius + 1]) <= radius
    labels, _ = ndimage.label(ndimage.binary_dilation(mask, structure=disk[None]))
    labels[~mask] = 0
    for spot_id, box in enumerate(ndimage.find_objects(labels), 1):
        if box is None:
            continue
        size = np.sum(labels[box] == spot_id)
        extent = [s.stop - s.start for s in box]
        fill = size / np.prod(extent)
        elongation = max(extent[1:]) / min(extent[1:])
        if size < SPOT_MIN_SIZE or fill < SPOT_MIN_FILL or elongation > SPOT_MAX_ELONGATION:
            labels[box][labels[box] == spot_id] = 0
    return labels


def add_spots(nuclei, raw, threshold=SPOT_THRESHOLD, radius=SPOT_RADIUS):
    """Add the spots that lie mostly outside the nuclei to the nuclei, then remove the dim objects again.

    The nuclei must be filtered by brightness already, so that a spot is not dropped for an object that the filter
    removes anyway. Returns the nuclei and the number of added spots.
    """
    spots = detect_spots(tophat(raw), threshold, radius)
    overlap = np.bincount(spots[nuclei > 0], minlength=spots.max() + 1) / np.maximum(np.bincount(spots.ravel()), 1)
    spots[overlap[spots] >= SPOT_MAX_NUCLEUS_OVERLAP] = 0
    spots[nuclei > 0] = 0
    ids = np.unique(spots)[1:]
    relabel = np.zeros(spots.max() + 1, dtype="uint32")
    relabel[ids] = np.arange(1, len(ids) + 1) + nuclei.max()
    segmentation, _ = remove_dim_objects(np.where(spots > 0, relabel[spots], nuclei).astype("uint32"), raw)
    return segmentation, len(ids)


def merge_timepoints(output_path):
    """Write the raw data and the filtered segmentation of all time points to one h5 of (T, Z, Y, X)."""
    paths = sorted(glob(os.path.join(RUN_DIR, "t*.tif")))
    assert len(paths) == N_TIMEPOINTS, f"{len(paths)} of {N_TIMEPOINTS} time points in {RUN_DIR}"
    shape = (N_TIMEPOINTS,) + tifffile.imread(paths[0]).shape
    with h5py.File(output_path, "w") as f:
        raw_ds = f.create_dataset("raw", shape=shape, dtype="uint16", chunks=CHUNKS, compression="gzip")
        raw_ds.attrs["axes"], raw_ds.attrs["voxel_size_zyx_um"] = "tzyx", VOXEL_SIZE
        raw_ds.attrs["time_interval_min"], raw_ds.attrs["start_hpf"] = 2.0, 3.0
        seg_ds = f.create_dataset("segmentation", shape=shape, dtype="uint16", chunks=CHUNKS, compression="gzip")
        seg_ds.attrs["axes"], seg_ds.attrs["model_type"], seg_ds.attrs["engine"] = "tzyx", MODEL_TYPE, ENGINE
        seg_ds.attrs["tophat_size"], seg_ds.attrs["norm_percentiles"] = TOPHAT_SIZE, NORM_PERCENTILES
        seg_ds.attrs["min_relative_brightness"] = MIN_RELATIVE_BRIGHTNESS
        seg_ds.attrs["spot_timepoints"], seg_ds.attrs["spot_threshold"] = SPOT_TIMEPOINTS, SPOT_THRESHOLD
        seg_ds.attrs["spot_radius"] = SPOT_RADIUS
        for t, path in enumerate(paths):
            raw = load_timepoint(t)
            segmentation, n_removed = remove_dim_objects(tifffile.imread(path), raw)
            n_spots = 0
            if t in SPOT_TIMEPOINTS:
                segmentation, n_spots = add_spots(segmentation, raw)
            assert segmentation.max() <= np.iinfo("uint16").max
            raw_ds[t], seg_ds[t] = raw, segmentation
            n_objects = len(np.unique(segmentation)) - 1
            print(f"t{t:03d}: {n_objects} objects, {n_removed} dim objects removed, {n_spots} spots added", flush=True)
    print(f"Saved {shape} to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-t", "--timepoints", nargs="+", default=[f"0-{N_TIMEPOINTS - 1}"],
                        help="Time points or ranges, e.g. '0 44 88' or '0-29'.")
    parser.add_argument("--merge", action="store_true", help="Merge all time points into one h5.")
    parser.add_argument("-o", "--output_path", default=OUTPUT_PATH, help="The h5 of the merge.")
    args = parser.parse_args()

    if args.merge:
        merge_timepoints(args.output_path)
    else:
        run_timepoints(parse_timepoints(args.timepoints))


if __name__ == "__main__":
    main()
