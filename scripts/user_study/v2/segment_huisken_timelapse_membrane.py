"""Segment the membranes with micro-sam2 in the time-lapse of Beke (C01 of huisken_data/classic_lsm/timelapse_3-6hpf).

APG with 'hvit_l_cells' segments each time point on its own, after a 2x resampling in z.
The user chose this setting by eye on t44 and t88 of 'sweep_huisken_timelapse_membrane.py'.
The membrane signal is close to the noise before t8. These time points are denoised first: an average over
the time points t-2 to t+2, then non-local means per slice. Without it, APG finds no cells there.
APG also finds small false objects in these time points, mostly at the top and bottom of the volume.
The merge keeps only the cells with a nucleus (from 'segment_huisken_timelapse.py') there.
It joins a cell without a nucleus to a neighbour cell with a nucleus, and removes it if there is none.
From t8 on, the merge removes the artifacts with the tracks of the cells (see 'remove_artifacts').

Usage:
    python segment_huisken_timelapse_membrane.py -t 0-4  # these time points, one tif each
    python segment_huisken_timelapse_membrane.py --merge  # all time points with the raw data into one h5
"""

import os
import argparse
from glob import glob

import h5py
import tifffile
import numpy as np
from scipy import ndimage

from explore_huisken_data import MODEL_TYPE, SMALL_DEFAULT_SETTINGS, enhance, segment_small
from segment_huisken_timelapse import (
    INPUT_DIR, OUTPUT_DIR, OUTPUT_PATH as NUCLEI_PATH, N_TIMEPOINTS, VOXEL_SIZE, CHUNKS, parse_timepoints
)


# The tif of each time point, so that a new job continues with the missing time points.
RUN_DIR = os.path.join(OUTPUT_DIR, f"membrane_{MODEL_TYPE}_z2_none_apg")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "C01_membrane_seg.h5")
ENGINE = "apg"
Z_SCALE = 2
SETTINGS = {**SMALL_DEFAULT_SETTINGS, "scale": (Z_SCALE, 1, 1), "anisotropy": VOXEL_SIZE[0] / VOXEL_SIZE[1] / Z_SCALE}
# The time points before this one are denoised with DENOISER (see 'denoise').
DENOISE_UNTIL = 8
DENOISER = "tavg2_nlm"
# A track links a cell to the cell that covers at least TRACK_MIN_COVERAGE of it at the next or previous time point.
# A cell is an artifact if its track finds no cell with a nucleus within TRACK_WINDOW time points and spans fewer
# than TRACK_MIN_LENGTH time points. Some real cells have no nucleus for a long time, because their H2B fills the cell.
TRACK_MIN_COVERAGE = 0.5
TRACK_WINDOW = 2
TRACK_MIN_LENGTH = 5
# The nuclei are missing during the divisions of t9 to t15, so the filter would remove real cells there.
SKIP_ARTIFACT_FILTER = tuple(range(9, 16))
NUCLEUS_MIN_OVERLAP = 0.5


def load_timepoint(t):
    return tifffile.imread(os.path.join(INPUT_DIR, f"registered_t{t:06d}_C01.tif"))


def estimate_noise(image):
    # The median absolute deviation of the differences between neighbour pixels, without PyWavelets.
    diff = np.diff(image, axis=1).ravel()
    return 1.4826 * np.median(np.abs(diff - np.median(diff))) / np.sqrt(2)


def denoise(t, denoiser):
    """Denoise the membranes of one time point.

    'tavgN' averages the time points t-N to t+N, 'gauss' is a 3D gaussian, 'nlm' is non-local means per slice.
    """
    from skimage.restoration import denoise_nl_means

    if denoiser.startswith("tavg"):
        n = int(denoiser[len("tavg")])
        timepoints = range(max(t - n, 0), min(t + n, N_TIMEPOINTS - 1) + 1)
        volume = np.mean([load_timepoint(tt).astype("float32") for tt in timepoints], axis=0)
    else:
        volume = load_timepoint(t).astype("float32")
    if "gauss" in denoiser:
        volume = ndimage.gaussian_filter(volume, (0.5, 1.5, 1.5))
    if "nlm" in denoiser:
        slices = []
        for image in volume:
            sigma = estimate_noise(image)
            slices.append(denoise_nl_means(image, h=0.8 * sigma, sigma=sigma, patch_size=5, patch_distance=6))
        volume = np.stack(slices).astype("float32")
    return volume


def run_timepoints(timepoints):
    os.makedirs(RUN_DIR, exist_ok=True)
    for t in timepoints:
        out = os.path.join(RUN_DIR, f"t{t:03d}.tif")
        if os.path.exists(out):
            continue
        raw = denoise(t, DENOISER) if t < DENOISE_UNTIL else load_timepoint(t)
        volume, norm_bounds = enhance(raw, SETTINGS)
        segmentation, n_objects, seconds = segment_small(volume, ENGINE, SETTINGS, norm_bounds)
        tifffile.imwrite(out, segmentation, compression="zlib")
        print(f"t{t:03d}: {n_objects} objects in {seconds:.1f} s", flush=True)


def get_cells_with_nucleus(segmentation, nuclei):
    """Get the ids of the cells with a nucleus.

    A nucleus belongs to a cell if the cell contains at least NUCLEUS_MIN_OVERLAP of its voxels.
    Without this limit, a large background object that touches the edges of many nuclei gets a nucleus.
    """
    mask = (nuclei > 0) & (segmentation > 0)
    offset = int(segmentation.max()) + 1
    pairs, counts = np.unique(nuclei[mask].astype("int64") * offset + segmentation[mask], return_counts=True)
    nucleus_sizes = np.bincount(nuclei.ravel())
    return {
        int(cell_id) for nucleus_id, cell_id, count in zip(pairs // offset, pairs % offset, counts)
        if count >= NUCLEUS_MIN_OVERLAP * nucleus_sizes[nucleus_id]
    }


def get_best_matches(segmentation, other):
    """Map each cell to the cell of 'other' that covers at least TRACK_MIN_COVERAGE of it.

    TRACK_MIN_COVERAGE is at least 0.5, so at most one cell of 'other' matches.
    """
    mask = segmentation > 0
    # Each pair of ids is one integer, because np.unique is much faster in 1D than with axis=1.
    offset = int(other.max()) + 1
    pairs, counts = np.unique(segmentation[mask].astype("int64") * offset + other[mask], return_counts=True)
    sizes = np.bincount(segmentation[mask])
    return {
        int(cell_id): int(other_id) for cell_id, other_id, count in zip(pairs // offset, pairs % offset, counts)
        if other_id != 0 and count >= TRACK_MIN_COVERAGE * sizes[cell_id]
    }


def follow_track(t, cell_id, links, step, anchored):
    """Follow the track of a cell in the direction 'step' for at most TRACK_MIN_LENGTH - 1 time points.

    Returns the number of linked time points and if the track finds a nucleus within TRACK_WINDOW time points.
    """
    found = False
    for n_linked in range(TRACK_MIN_LENGTH - 1):
        if not 0 <= t + step < len(anchored):
            return n_linked, found
        cell_id = links[t].get(cell_id)
        t += step
        if cell_id is None:
            return n_linked, found
        found = found or (n_linked < TRACK_WINDOW and cell_id in anchored[t])
    return TRACK_MIN_LENGTH - 1, found


def is_artifact(t, cell_id, forward, backward, anchored):
    if cell_id in anchored[t]:
        return False
    n_forward, found_forward = follow_track(t, cell_id, forward, 1, anchored)
    n_backward, found_backward = follow_track(t, cell_id, backward, -1, anchored)
    return not (found_forward or found_backward or n_forward + n_backward + 1 >= TRACK_MIN_LENGTH)


def remove_artifacts(segmentations, nuclei, start):
    """Remove the cells from 'start' on that are short-lived and have no nucleus.

    The track follows the best match (see 'get_best_matches') forward and backward in time.
    A cell is kept if its track finds a cell with a nucleus within TRACK_WINDOW time points,
    or if the track spans at least TRACK_MIN_LENGTH time points.
    The result does not depend on the order of the cells, so it is the same for each run.
    Returns the number of removed cells for each time point.
    """
    n_timepoints = len(segmentations)
    anchored = [get_cells_with_nucleus(seg, nuc) for seg, nuc in zip(segmentations, nuclei)]
    forward = [get_best_matches(segmentations[t], segmentations[t + 1]) for t in range(n_timepoints - 1)]
    backward = [get_best_matches(segmentations[t], segmentations[t - 1]) if t > 0 else {} for t in range(n_timepoints)]

    artifacts = []
    for t in range(start, n_timepoints):
        removed = [] if t in SKIP_ARTIFACT_FILTER else [
            cell_id for cell_id in np.unique(segmentations[t])[1:]
            if is_artifact(t, cell_id, forward, backward, anchored)
        ]
        artifacts.append((t, removed))
    # The cells are removed after all tracks are known, so that a removed cell does not change another track.
    for t, removed in artifacts:
        segmentations[t][np.isin(segmentations[t], removed)] = 0
    return {t: len(removed) for t, removed in artifacts}


def keep_cells_with_nucleus(segmentation, nuclei):
    """Merge each cell without a nucleus into the neighbour cell with a nucleus that it touches most.

    The function removes the cells without a nucleus that touch no cell with a nucleus.
    Returns the segmentation, the number of merged cells and the number of removed cells.
    """
    cell_ids = np.unique(segmentation)
    cell_ids = cell_ids[cell_ids > 0]
    anchored = get_cells_with_nucleus(segmentation, nuclei)

    lut = np.arange(segmentation.max() + 1, dtype=segmentation.dtype)
    anchored_ids = np.array(sorted(anchored), dtype=segmentation.dtype)
    unanchored = [cell_id for cell_id in cell_ids if cell_id not in anchored]
    bboxes = ndimage.find_objects(segmentation)
    # A piece can touch a cell with a nucleus only through other pieces, so repeat until nothing changes.
    changed = True
    while changed:
        changed = False
        for cell_id in unanchored:
            if lut[cell_id] in anchored:
                continue
            bbox = tuple(slice(max(axis.start - 2, 0), axis.stop + 2) for axis in bboxes[cell_id - 1])
            mask = segmentation[bbox] == cell_id
            border = lut[segmentation[bbox]][ndimage.binary_dilation(mask, iterations=2) & ~mask]
            border = border[np.isin(border, anchored_ids)]
            if border.size == 0:
                continue
            ids, counts = np.unique(border, return_counts=True)
            lut[cell_id] = ids[np.argmax(counts)]
            changed = True
    removed = [cell_id for cell_id in unanchored if lut[cell_id] not in anchored]
    lut[removed] = 0
    return lut[segmentation], len(unanchored) - len(removed), len(removed)


def merge_timepoints(output_path):
    """Write the raw data and the segmentation of all time points to one h5 with the axes (T, Z, Y, X)."""
    paths = sorted(glob(os.path.join(RUN_DIR, "t*.tif")))
    assert len(paths) == N_TIMEPOINTS, f"{len(paths)} of {N_TIMEPOINTS} time points are in {RUN_DIR}"
    with h5py.File(NUCLEI_PATH, "r") as f:
        nuclei = f["segmentation"][:]
    segmentations, messages = [], []
    for t, path in enumerate(paths):
        segmentation = tifffile.imread(path).astype("uint32")
        if t < DENOISE_UNTIL:
            segmentation, n_merged, n_removed = keep_cells_with_nucleus(segmentation, nuclei[t])
            messages.append(f"{n_merged} merged, {n_removed} removed without nucleus")
        segmentations.append(segmentation)
    n_artifacts = remove_artifacts(segmentations, nuclei, start=DENOISE_UNTIL)
    messages += [f"{n_artifacts[t]} artifacts removed" for t in range(DENOISE_UNTIL, N_TIMEPOINTS)]

    shape = (N_TIMEPOINTS,) + segmentations[0].shape
    with h5py.File(output_path, "w") as f:
        raw_ds = f.create_dataset("raw", shape=shape, dtype="uint16", chunks=CHUNKS, compression="gzip")
        raw_ds.attrs["axes"], raw_ds.attrs["voxel_size_zyx_um"] = "tzyx", VOXEL_SIZE
        raw_ds.attrs["time_interval_min"], raw_ds.attrs["start_hpf"] = 2.0, 3.0
        seg_ds = f.create_dataset("segmentation", shape=shape, dtype="uint16", chunks=CHUNKS, compression="gzip")
        seg_ds.attrs["axes"], seg_ds.attrs["model_type"], seg_ds.attrs["engine"] = "tzyx", MODEL_TYPE, ENGINE
        seg_ds.attrs["z_scale"] = Z_SCALE
        seg_ds.attrs["denoiser"], seg_ds.attrs["denoise_until"] = DENOISER, DENOISE_UNTIL
        seg_ds.attrs["keep_cells_with_nucleus_until"] = DENOISE_UNTIL
        seg_ds.attrs["track_min_coverage"], seg_ds.attrs["track_window"] = TRACK_MIN_COVERAGE, TRACK_WINDOW
        seg_ds.attrs["nucleus_min_overlap"], seg_ds.attrs["track_min_length"] = NUCLEUS_MIN_OVERLAP, TRACK_MIN_LENGTH
        seg_ds.attrs["skip_artifact_filter"] = SKIP_ARTIFACT_FILTER
        for t, (segmentation, message) in enumerate(zip(segmentations, messages)):
            assert segmentation.max() <= np.iinfo("uint16").max
            raw_ds[t], seg_ds[t] = load_timepoint(t), segmentation.astype("uint16")
            print(f"t{t:03d}: {len(np.unique(segmentation)) - 1} cells, {message}", flush=True)
    print(f"Saved {shape} to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-t", "--timepoints", nargs="+", default=[f"0-{N_TIMEPOINTS - 1}"], help="The time points or ranges."
    )
    parser.add_argument("--merge", action="store_true", help="Merge all time points into one h5.")
    parser.add_argument("-o", "--output_path", default=OUTPUT_PATH, help="The path to the h5 of the merge.")
    args = parser.parse_args()

    if args.merge:
        merge_timepoints(args.output_path)
    else:
        run_timepoints(parse_timepoints(args.timepoints))


if __name__ == "__main__":
    main()
