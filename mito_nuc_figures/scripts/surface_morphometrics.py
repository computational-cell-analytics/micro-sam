import os

import mrcfile
import numpy as np
import zarr
from scipy import ndimage

from denoised_view import DenoisedView
from figure_utils import DATA_ROOT, get_output_dir, save_volume_figure, write_info
from torch_em.data.datasets.electron_microscopy import surface_morphometrics as sm


DATASET = "surface_morphometrics"
DATA_PATH = os.path.join(DATA_ROOT, DATASET)
# Representative subset (2 per group: thapsigargin T / untreated U x E / F), picked as the smallest files.
TOMOGRAM_IDS = ["UF5", "TF5", "TE11", "TE7", "UE3", "UE2", "TF2", "UF6"]
# Label values, determined visually: 1 outer mito membrane, 2 inner mito membrane incl. cristae, 3 ER membrane.
MITO_VALUES = (1, 2)
ISSUES = (
    "Loader bug 1, surface_morphometrics.py get_surface_morphometrics_paths / _read_label_mrc: the label volume is "
    "y-flipped relative to the raw tomogram for the float32 (MRC mode 2) tomograms (UF5, TF5, TE11, UE2, TF2, UF6 "
    "here, ridge score as stored ~0 vs 1.2-2.1 flipped) but aligned as stored for the int16 (mode 1) ones (TE7, UE3). "
    "The loader never flips, so 6/8 checked training volumes have misaligned labels; by header, the mode 1 ids are "
    "TE2, TE7, TE9, TE12, UE3, UE4, UE6 and all others are mode 2 (likely flipped). Fix: flip labels[:, ::-1] when "
    "the raw MRC is mode 2 (or check alignment per tomogram). "
    "Loader bug 2: header shapes differ for TF3 (raw 960x928x300 vs labels 928x960x350, nx/ny/nz) and UE8 (raw "
    "952x952x500 vs labels 960x928x187, identical header to UE2/UE5 labels, likely a wrong file in the deposit). "
    "_read_label_mrc reshapes the label bytes to the raw shape, so TF3 gets silently scrambled labels and UE8 raises "
    "a reshape error. Exclude both. Also the loader keeps both the MRC files and the zarr copy (2x disk)."
)
OBSERVATIONS = (
    "Label values verified visually on TF5, UF5, UE2, TE7: 1 = outer mito membrane, 2 = inner mito membrane incl. "
    "cristae, 3 = ER membrane (hidden in the figures). Membrane labels, 2-4 voxels thick, not filled mitochondria. "
    "After the per-tomogram y-flip all 8 are well aligned. Coverage of mitochondria in the field is near exhaustive "
    "and cristae are traced in detail; gaps only where membranes run parallel to the beam (missing wedge). "
    "1-4 mitochondria per tomogram."
)
DOWNLOAD = (
    "8 of 34 tomograms via get_surface_morphometrics_paths(download=True), 2 per group (T/U x E/F), the smallest "
    "files: UF5, TF5, TE11, TE7, UE3, UE2, TF2, UF6; 8.4 GB incl. zarr cache (full deposit ~42 GB of MRCs)."
)
NOTES = "cryo-ET MEFs, {vs} nm voxels, mito MEMBRANES 1=outer 2=inner+cristae (not filled, ER label hidden), semantic"


def ridge_score(raw, labels, z_values):
    """Mean darkness of the smoothed raw under the label voxels, relative to the slice mean (membranes are dark)."""
    scores = []
    for z in z_values:
        image = ndimage.gaussian_filter(np.asarray(raw[z]).astype("float32"), 2)
        image = (image.mean() - image) / image.std()
        mask = labels[z] > 0
        if mask.any():
            scores.append(image[mask].mean())
    return float(np.mean(scores))


def find_orientation(raw, labels):
    """Check whether the labels align with raw as stored, or only after flipping the y axis."""
    z_values = np.linspace(raw.shape[0] * 0.25, raw.shape[0] * 0.75, 5).astype(int)
    as_stored = ridge_score(raw, labels, z_values)
    flipped = ridge_score(raw, labels[:, ::-1], z_values)
    return flipped > as_stored, round(as_stored, 3), round(flipped, 3)


def render_tomogram(tomogram_id, out_dir):
    zarr_path = sm.get_surface_morphometrics_paths(DATA_PATH, [tomogram_id], download=True)[0]
    root = zarr.open_group(zarr_path, mode="r")
    raw, labels = root["raw"], root["labels"][:]
    flip_y, score_stored, score_flipped = find_orientation(raw, labels)
    if flip_y:
        labels = labels[:, ::-1]
    values, counts = np.unique(labels, return_counts=True)
    mito = np.where(np.isin(labels, MITO_VALUES), labels, 0)
    with mrcfile.open(os.path.join(DATA_PATH, f"{tomogram_id}_tomo.rec.mrc"), header_only=True, permissive=True) as f:
        voxel_size_nm = round(float(f.voxel_size.x) / 10, 3)
    notes = NOTES.format(vs=voxel_size_nm)
    if flip_y:
        notes += "; labels y-flipped to align"
    out_path = os.path.join(out_dir, f"{tomogram_id}.png")
    save_volume_figure(DenoisedView(raw, 2, 1.0), {"mito": mito}, out_path, DATASET, tomogram_id, notes=notes)
    return {
        "name": tomogram_id, "shape": raw.shape, "voxel_size_nm": voxel_size_nm, "path": zarr_path,
        "label_values": {int(v): int(c) for v, c in zip(values, counts)}, "labels_flipped_y_to_align": bool(flip_y),
        "ridge_score_as_stored": score_stored, "ridge_score_flipped_y": score_flipped,
    }


def main():
    out_dir = get_output_dir(DATASET)
    volumes = []
    for tomogram_id in TOMOGRAM_IDS:
        volume = render_tomogram(tomogram_id, out_dir)
        print(volume["name"], volume["shape"], volume["labels_flipped_y_to_align"], volume["label_values"], flush=True)
        volumes.append(volume)
    info = {
        "classes": ["mito (outer + inner membrane, multi-class 1/2)"], "modality": "cryo-ET (EMPIAR-11370)",
        "dimensionality": "3d", "label_quality": "manual/semi-automated membrane voxel segmentation (Amira), semantic",
        "n_volumes": len(volumes), "volumes": volumes, "data_path": DATA_PATH, "download": DOWNLOAD, "notes": NOTES,
        "issues": ISSUES, "observations": OBSERVATIONS,
    }
    write_info(DATASET, info)


if __name__ == "__main__":
    main()
