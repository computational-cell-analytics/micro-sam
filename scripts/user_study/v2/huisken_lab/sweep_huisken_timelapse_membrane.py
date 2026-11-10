"""Sweep the settings of the membrane segmentation with micro-sam2 in the time-lapse of Beke.

The script segments the membranes (C01 of timelapse_3-6hpf) with 'hvit_l_cells' on single time points.
It scores each result against the nuclei of the same time point (see 'segment_huisken_timelapse.py').
Each cell must contain one nucleus.

Usage:
    python sweep_huisken_timelapse_membrane.py -t 44 -s z2_ridge_clahe  # one setting on one time point
    python sweep_huisken_timelapse_membrane.py --watershed  # the watersheds from the nuclei, on CPU
    python sweep_huisken_timelapse_membrane.py --score  # the scores of all results, as a table
    python sweep_huisken_timelapse_membrane.py --collect -t 0 44 88  # one h5 per time point with all results
"""

import os
import argparse
from glob import glob

import h5py
import numpy as np
from scipy import ndimage

from segment_huisken_timelapse_membrane import load_timepoint, denoise
from segment_huisken_timelapse import OUTPUT_DIR, OUTPUT_PATH as NUCLEI_PATH
from explore_huisken_data import MODEL_TYPE, ENGINES, SMALL_DEFAULT_SETTINGS, enhance, segment_small


SWEEP_DIR = os.path.join(OUTPUT_DIR, "membrane_sweep")
COLLECT_DIR = "/mnt/vast-nhr/home/archit/u12090/micro-sam/timelapse_sweep_membrane_h5"
ANISOTROPY = 2.5 / 0.406
SCALINGS = {
    "raw": {"anisotropy": ANISOTROPY},
    "z2": {"scale": (2, 1, 1), "anisotropy": ANISOTROPY / 2},
    "z4": {"scale": (4, 1, 1), "anisotropy": ANISOTROPY / 4},
}
# The contrast steps of 'enhance'.
ENHANCEMENTS = {
    "none": {},
    "smooth": {"smooth": 1.0},
    "ridge": {"ridge": 0.5},
    "clahe": {"clahe": 0.02},
    "gamma": {"gamma": 0.5},
    "ridge_clahe": {"ridge": 0.5, "clahe": 0.02},
    "smooth_ridge_clahe": {"ridge": 0.5, "clahe": 0.02, "smooth": 1.0},
    "invert": {"invert": True},
}
SETTINGS = {
    f"{scaling}_{name}": {**scale, **enhancement}
    for scaling, scale in SCALINGS.items() for name, enhancement in ENHANCEMENTS.items()
}
for name in ("none", "ridge_clahe"):
    base = SETTINGS[f"z2_{name}"]
    SETTINGS[f"z2_{name}_p1-99"] = {**base, "norm_percentiles": (1, 99)}
    SETTINGS[f"z2_{name}_p1-99.9"] = {**base, "norm_percentiles": (1, 99.9)}
    SETTINGS[f"z2_{name}_t256"] = {**base, "tile_shape": (256, 256), "halo": (64, 64)}
# 'tavgN' averages the time points t-N to t+N, 'gauss' is a 3D gaussian, 'nlm' is non-local means per slice.
DENOISERS = ("tavg1", "tavg2", "gauss", "nlm", "tavg2_nlm")
for denoiser in DENOISERS:
    for name in ("raw_none", "raw_ridge_clahe", "z2_none"):
        SETTINGS[f"{name}_{denoiser}"] = {**SETTINGS[name], "denoise": denoiser}
# The watershed grows each nucleus over the membrane image, inside the embryo mask.
WATERSHED_SETTINGS = {
    f"ws_{height}{'_z2' if z_scale == 2 else ''}": {"height": height, "z_scale": z_scale}
    for height in ("smooth", "ridge", "smooth_ridge") for z_scale in (1, 2)
}
EMBRYO_MASK_SIGMA = (1, 8, 8)


def get_path(t, name):
    return os.path.join(SWEEP_DIR, f"t{t:03d}_{MODEL_TYPE}_{name}.h5")


def run(t, name, engines):
    os.makedirs(SWEEP_DIR, exist_ok=True)
    settings = {**SMALL_DEFAULT_SETTINGS, **SETTINGS[name]}
    raw = denoise(t, settings["denoise"]) if "denoise" in settings else load_timepoint(t)
    volume, norm_bounds = enhance(raw, settings)
    for engine in engines:
        segmentation, n_objects, seconds = segment_small(volume, engine, settings, norm_bounds)
        print(f"t{t:03d} {name} {engine}: {n_objects} objects in {seconds:.1f} s", flush=True)
        with h5py.File(get_path(t, name), "a") as f:
            if engine in f:
                del f[engine]
            ds = f.create_dataset(engine, data=segmentation, compression="gzip")
            ds.attrs["n_objects"], ds.attrs["seconds"] = n_objects, seconds


def get_embryo_mask(raw):
    from skimage.filters import threshold_otsu

    smoothed = ndimage.gaussian_filter(raw.astype("float32"), EMBRYO_MASK_SIGMA)
    mask = smoothed > threshold_otsu(smoothed)
    mask = np.stack([ndimage.binary_fill_holes(z) for z in mask])
    return ndimage.binary_fill_holes(mask)


def run_watershed(t, name):
    from skimage.filters import sato
    from skimage.segmentation import watershed

    os.makedirs(SWEEP_DIR, exist_ok=True)
    settings = WATERSHED_SETTINGS[name]
    raw = load_timepoint(t)
    with h5py.File(NUCLEI_PATH, "r") as f:
        nuclei = f["segmentation"][t]
    volume = raw.astype("float32")
    if "smooth" in settings["height"]:
        volume = ndimage.gaussian_filter(volume, (0, 2, 2))
    if "ridge" in settings["height"]:
        volume = np.stack([sato(z, sigmas=(1, 2), black_ridges=False) for z in volume])
    mask = get_embryo_mask(raw)
    z_scale = settings["z_scale"]
    if z_scale > 1:
        volume = ndimage.zoom(volume, (z_scale, 1, 1), order=1)
        seeds, mask = np.repeat(nuclei, z_scale, axis=0), np.repeat(mask, z_scale, axis=0)
    else:
        seeds = nuclei
    segmentation = watershed(volume, markers=np.where(mask, seeds, 0), mask=mask)
    segmentation = segmentation[::z_scale].astype("uint32")
    n_objects = len(np.unique(segmentation)) - 1
    print(f"t{t:03d} {name} watershed: {n_objects} objects", flush=True)
    with h5py.File(get_path(t, name), "w") as f:
        ds = f.create_dataset("watershed", data=segmentation, compression="gzip")
        ds.attrs["n_objects"], ds.attrs["seconds"] = n_objects, 0.0


def score(cells, nuclei, min_overlap=0.5):
    """Get the precision, recall and F1 of the cells that contain exactly one nucleus.

    A nucleus is in a cell if at least 'min_overlap' of its voxels are in the cell.
    The precision divides the matched cells by all cells. The recall divides them by all nuclei.
    """
    nucleus_ids, nucleus_sizes = np.unique(nuclei, return_counts=True)
    nucleus_ids, nucleus_sizes = nucleus_ids[nucleus_ids > 0], nucleus_sizes[nucleus_ids > 0]
    cell_ids = np.unique(cells)
    cell_ids = cell_ids[cell_ids > 0]
    if len(nucleus_ids) == 0 or len(cell_ids) == 0:
        return 0.0, 0.0, 0.0
    mask = nuclei > 0
    pairs, counts = np.unique(np.stack([nuclei[mask], cells[mask]]), axis=1, return_counts=True)
    nuclei_per_cell = {}
    for (nucleus, cell), count in zip(pairs.T, counts):
        size = nucleus_sizes[np.searchsorted(nucleus_ids, nucleus)]
        if cell != 0 and count >= min_overlap * size:
            nuclei_per_cell[cell] = nuclei_per_cell.get(cell, 0) + 1
    n_matched = sum(1 for n in nuclei_per_cell.values() if n == 1)
    precision, recall = n_matched / len(cell_ids), n_matched / len(nucleus_ids)
    f1 = 2 * precision * recall / (precision + recall) if n_matched else 0.0
    return precision, recall, f1


def run_score():
    paths = sorted(glob(os.path.join(SWEEP_DIR, f"t*_{MODEL_TYPE}_*.h5")))
    nuclei, rows = {}, []
    for path in paths:
        name = os.path.basename(path)[len("t000_") + len(MODEL_TYPE) + 1:-len(".h5")]
        t = int(os.path.basename(path)[1:4])
        if t not in nuclei:
            with h5py.File(NUCLEI_PATH, "r") as f:
                nuclei[t] = f["segmentation"][t]
        with h5py.File(path, "r") as f:
            for engine in f:
                cells = f[engine][:]
                sizes = np.bincount(cells.ravel())[1:]
                n_small = int(((sizes > 0) & (sizes < 1000)).sum())
                n_objects, seconds = int(f[engine].attrs["n_objects"]), float(f[engine].attrs["seconds"])
                rows.append((t, name, engine, n_objects, n_small, *score(cells, nuclei[t]), seconds))
    print("t setting engine cells small precision recall f1 seconds")
    for t, name, engine, n, n_small, precision, recall, f1, seconds in sorted(rows, key=lambda row: (row[0], -row[7])):
        print(f"{t} {name} {engine} {n} {n_small} {precision:.2f} {recall:.2f} {f1:.2f} {seconds:.0f}")
    print("nuclei per time point:", {t: len(np.unique(seg)) - 1 for t, seg in nuclei.items()})


def collect(timepoints):
    """Write one h5 per time point with the raw data, the nuclei and each result as '<setting>_<engine>'."""
    os.makedirs(COLLECT_DIR, exist_ok=True)
    for t in timepoints:
        out = os.path.join(COLLECT_DIR, f"sweep_membrane_t{t:03d}.h5")
        with h5py.File(out, "w") as f_out:
            f_out.create_dataset("raw", data=load_timepoint(t), compression="gzip", chunks=True)
            with h5py.File(NUCLEI_PATH, "r") as f:
                f_out.create_dataset("nuclei", data=f["segmentation"][t], compression="gzip", chunks=True)
            for path in sorted(glob(os.path.join(SWEEP_DIR, f"t{t:03d}_{MODEL_TYPE}_*.h5"))):
                name = os.path.basename(path)[len("t000_") + len(MODEL_TYPE) + 1:-len(".h5")]
                with h5py.File(path, "r") as f:
                    for engine in f:
                        data = f[engine][:].astype("uint16")
                        ds = f_out.create_dataset(f"{name}_{engine}", data=data, compression="gzip", chunks=True)
                        ds.attrs.update(dict(f[engine].attrs))
            print(f"Saved {out} with {len(f_out) - 2} segmentations")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-t", "--timepoints", type=int, nargs="+", default=[0, 44, 88])
    parser.add_argument("-s", "--settings", nargs="+", choices=tuple(SETTINGS))
    parser.add_argument("-e", "--engines", nargs="+", choices=ENGINES, default=list(ENGINES))
    parser.add_argument("--score", action="store_true", help="Score all results against the nuclei.")
    parser.add_argument("--collect", action="store_true", help="Collect the results of each time point into one h5.")
    parser.add_argument("--list", action="store_true", help="Print the names of the settings.")
    parser.add_argument("--watershed", action="store_true", help="Run the watersheds from the nuclei, on CPU.")
    args = parser.parse_args()

    if args.list:
        print(" ".join(SETTINGS))
    elif args.score:
        run_score()
    elif args.collect:
        collect(args.timepoints)
    elif args.watershed:
        for t in args.timepoints:
            for name in WATERSHED_SETTINGS:
                run_watershed(t, name)
    else:
        for t in args.timepoints:
            for name in args.settings or list(SETTINGS):
                run(t, name, args.engines)


if __name__ == "__main__":
    main()
