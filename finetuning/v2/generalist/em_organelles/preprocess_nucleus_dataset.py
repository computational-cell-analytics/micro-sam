"""Build the derived nucleus data that the nucleus EM generalist and its evaluation read.

- betaseg: nucleus boxes with the ignore label outside the target cell -> <output>/betaseg_nucleus_boxes/.
- human_organoids: cleaned nucleus instance labels -> <output>/human_organoids_nucleus/<volume>_nucleus_labels.zarr.
- tumor_spheroid: nucleus slices with the cells without a nucleus ignored -> <output>/tumor_spheroid_nucleus/.
- waenet: the WAE-Net nucleus volumes and 2D stacks -> <output>/waenet_nucleus/.
- cellmap: the large CellMap nucleus crops -> <output>/cellmap_nucleus_crops/<volume>_crop<id>.zarr and index.json.
- cardiomyocyte, perez, vem, u2os, etchacell, aive, cerebellum, macrophage: the nucleus data of these torch-em
  datasets -> <output>/<folder>/ (see BUILDERS) with build_info.json.
- human_liver_eval: the human liver EM nucleus evaluation boxes -> <output>/human_liver_em_nucleus_boxes/.
- eval_crops: the fixed evaluation crops of the EM nucleus panel -> finetuning/v2/evaluation/eval_crops_3d.json.
"""
import io
import os
import re
import sys
import glob
import json
import time
import shutil
import socket
import zipfile
import argparse
from concurrent.futures import ThreadPoolExecutor

import cc3d
import h5py
import zarr
import requests
import tifffile
import numpy as np
from PIL import Image
from scipy import ndimage
import imageio.v3 as imageio
from imagecodecs import delta_decode, lzw_decode

from skimage.measure import label
from skimage.morphology import convex_hull_image, disk
from skimage.transform import downscale_local_mean, resize

from elf.io import open_file

from torch_em.data.datasets.electron_microscopy.human_liver_em import _HttpFile

from preprocess_mito_dataset import CELLMAP_ZIP, DATA_ROOT, get_cellmap_em_levels, load_cellmap_crop


IGNORE_LABEL = np.iinfo("uint32").max

BETASEG_MISSING_DATA_MIN_AREA = 1000

# The human organoids volumes with the raw value of their alignment padding.
ORGANOIDS_VOLUMES = {"F059": -128, "F107_A1": -128, "F107_A2": -108, "F107_A3": -108}
ORGANOIDS_PADDING_MIN_AREA = 2000
ORGANOIDS_MIN_NUCLEUS_SIZE = 100000
ORGANOIDS_CLOSING_RADIUS = 4
ORGANOIDS_CHUNKS = (32, 256, 256)
LABEL_SLAB = 128

WAENET_VAL_FRACTION = 0.1
WAENET_CELL_BAND = 10
WAENET_MIN_NUCLEUS_AREA = 100
WAENET_PADDING_MIN_AREA = 1000
WAENET_SIZE = 512
WAENET_DS3_VAL_SECTIONS = range(38, 46)
WAENET_TEM_DATASETS = (4, 5, 6, 7)
# Images whose masks do not follow the cell (found by screening every image).
WAENET_DROPPED_IMAGES = {
    "dataset_5/49.h5": "nucleus and cytoplasm masks are offset from the cell and overlap a neighbour",
    "dataset_6/75.h5": "nucleus outline does not match the nucleus in the raw",
    "dataset_7/108.h5": "nucleus boundary is not visible in the low-contrast raw, the mask cannot be verified",
}

CELLMAP_NUCLEUS_TRAIN_CROPS = {
    "jrc_mus-liver-zon-1": [289, 336, 337, 349, 351],
    "jrc_mus-liver-zon-2": [358, 367],
    "jrc_mus-kidney": [179],
    "jrc_mus-kidney-3": [472],
    "jrc_zf-cardiac-1": [378, 379],
    "jrc_mus-heart-1": [423, 452],
}
CELLMAP_NUCLEUS_TEST_CROPS = {"jrc_mus-liver-zon-2": [357]}
CELLMAP_NUCLEUS_CHUNKS = (64, 64, 64)

MIN_OBJECT_SIZE = 200
CARDIOMYOCYTE_SCALE = 4
VEM_TARGET_DIAMETER = 250
VEM_3D_SCALE = 2
VEM_EDGE_NUCLEI_INCLUDED = (8, 10)
VEM_MISSING = (9,)
VEM_BORDER_PERCENTILE = 90
U2OS_MIN_ENVELOPE_PIXELS = 500
ETCHACELL_SCALE = 4
ETCHACELL_CLOSING_RADIUS = 5
AIVE_SCALE = (3, 4, 4)
AIVE_NUCLEUS_CLASSES = (6, 11)
AIVE_VOLUMES = (1, 2, 3)
CEREBELLUM_SCALE = (2, 2, 2)
CEREBELLUM_MIN_NUCLEUS_SIZE = 100000
MACROPHAGE_SCALE = (1, 4, 4)
MACROPHAGE_VOLUMES = ("sbf_a431", "sbf_oa_chol_7t", "sbf_oa_chol_av1", "sbf_oa_chol_av3", "sbf_no_load")
U2OS_CLOSING_RADIUS = 10
U2OS_MIN_PIECE_FRACTION = 0.1
U2OS_MAX_PIECE_GAP = 50
U2OS_MIN_FILLED_FRACTION = 0.05

EMPIAR_BASE = "https://ftp.ebi.ac.uk/empiar/world_availability/13356/data"
HUMAN_LIVER_ROOT = "human_liver_em_nucleus_boxes"
HUMAN_LIVER_BOX_SIZE = 4096
HUMAN_LIVER_SCALE = 4
HUMAN_LIVER_N_SLICES = 597
HUMAN_LIVER_BLOCK = 128
# The lower part of the tissue (fibrous / portal region) has unlabeled nuclei, and the lower right holds an artefact.
HUMAN_LIVER_MAX_Y = 11000
# The evaluation boxes as (x_min, x_max, y_min, y_max, z_min, z_max) at full resolution, with their role.
HUMAN_LIVER_BOXES = {
    "test_a": {"bbox_xxyyzz": [14336, 18432, 2048, 6144, 80, 112], "role": "test"},
    "test_b": {"bbox_xxyyzz": [14336, 18432, 3584, 7680, 290, 322], "role": "test"},
    "test_c": {"bbox_xxyyzz": [8192, 12288, 4608, 8704, 550, 582], "role": "test"},
    "tune": {"bbox_xxyyzz": [7168, 11264, 1024, 5120, 160, 176], "role": "tuning"},
}

EVALUATION_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "evaluation"))
EVAL_CROPS_PATH = os.path.join(EVALUATION_ROOT, "eval_crops_3d.json")
EVAL_OBJECT_NAMES = {
    "parlakgul_liver": "mitochondria", "cerebellum_fibsem_mito": "mitochondria",
    "macrophage_a431_mito": "mitochondria", "mito_anf": "mitochondria", "snemi_mito": "mitochondria",
    "donglai_wei_mito": "mitochondria",
}
EVAL_CROP_SHAPE = (32, 512, 512)
EVAL_MIN_TILE_SHAPE = (16, 256, 256)
# These volumes are smaller than the default tile in-plane: (crop shape, minimum tile shape).
EVAL_TILE_SHAPES = {
    "nucmm_zebrafish": ((64, 64, 64), (16, 64, 64)),
    "nucmm_mouse": ((32, 192, 192), (16, 192, 192)),
    "aive_nucleus": ((32, 512, 512), (16, 128, 128)),
}
EVAL_MAX_TILES_PER_VOLUME = 16
EVAL_MIN_RELATIVE_COUNT = 0.2
EVAL_NOTES = {
    "cellmap_nucleus": "Nucleus panel: the held-out jrc_mus-liver-zon-2 crop357 (32 nm, manual instances).",
    "fafb_nuclei": "Nucleus panel: the four official FAFB test blocks (automatic FlyWire nucleus segmentation).",
    "microns_nuclei": "Nucleus panel: the four official MICrONS test blocks (automatic minnie65 nucleus segmentation).",
    "platynereis_nuclei": "Nucleus panel: Platynereis volume 03 inside its annotated box (manual instances).",
    "nucmm_zebrafish": (
        "Nucleus panel (out-of-domain): the scored NucMM-Z volumes (ssEM at about 0.5 um, 64^3), each crop a whole "
        "volume; manual instances."
    ),
    "nucmm_mouse": (
        "Nucleus panel (out-of-domain): the four NucMM-M val volumes (X-ray micro-CT at 0.72 um, 192^3, not EM); "
        "manual instances."
    ),
    "aive_nucleus": (
        "Nucleus panel (out-of-domain): AIVE datasets 2 and 3 below the tuning slab (the first 40 slices), one nucleus "
        "per volume with the nucleoli folded in."
    ),
    "parlakgul_liver": (
        "Mitochondria panel (out-of-domain): the eight Parlakgul liver boxes, scored below the tuning slab (the first "
        "13 slices); semantic labels made instances by connected components."
    ),
    "cerebellum_fibsem_mito": (
        "Mitochondria panel (out-of-domain): the Cerebellum FIB-SEM volume below the tuning slab (the first 106 "
        "slices); student-labeled mitochondria instances, not necessarily proofread."
    ),
    "donglai_wei_mito": (
        "Mitochondria panel (out-of-domain): internal data from Donglai Wei (not public), the MICrONS phase 1 volume "
        "below the tuning slab (the first 150 slices). The labels are dense in the crops (checked by eye)."
    ),
    "snemi_mito": (
        "Mitochondria panel (in-domain): the test split of the SNEMI Mito stack (z 70-99); the train and valid splits "
        "are training data, the valid split tunes. Manual semantic labels, made instances by connected components."
    ),
    "macrophage_a431_mito": (
        "Mitochondria panel (out-of-domain): the Macrophage / A431 fib_control block (z 848-975) below the tuning slab "
        "(the first 26 slices); quick semantic MIB mitochondria labels, made instances by connected components."
    ),
    "mito_anf": (
        "Mitochondria panel (out-of-domain): the two MitoANF boxes below their tuning slabs (the first 26 slices); "
        "only the mitochondria of traced auditory nerve fibers are labeled (about half of the visible ones)."
    ),
    "human_liver_em_nucleus": (
        "Nucleus panel (out-of-domain): three 4x downsampled human liver boxes (automated segmentation, every visible "
        "nucleus labeled)."
    ),
    "etchacell_nucleus": (
        "Nucleus panel (out-of-domain): the scored Etch a Cell ROIs (HeLa SBF-SEM, 4x downsampled in-plane, one "
        "nucleus per ROI, filled from the consensus nuclear envelope labels)."
    ),
}


def betaseg_nucleus_box(nucleus, margin):
    idx = [np.where(nucleus.any(axis=tuple(a for a in range(3) if a != ax)))[0] for ax in range(3)]
    return tuple(
        slice(max(0, int(i.min()) - margin), min(size, int(i.max()) + 1 + margin))
        for i, size in zip(idx, nucleus.shape)
    )


def betaseg_missing_data_mask(raw):
    """Connected raw == 0 or raw == 255 regions of a slice above BETASEG_MISSING_DATA_MIN_AREA."""
    mask = np.zeros(raw.shape, dtype=bool)
    for z in range(raw.shape[0]):
        for value in (0, 255):
            components, _ = ndimage.label(raw[z] == value)
            areas = np.bincount(components.ravel())
            mask[z] |= np.isin(components, np.where(areas > BETASEG_MISSING_DATA_MIN_AREA)[0]) & (components > 0)
    return mask


def write_betaseg_box(path, margin, out_dir):
    name = os.path.basename(path).replace("_source.h5", "")
    with h5py.File(path, "r") as f:
        nucleus = f["labels/nucleus"][:] > 0
        box = betaseg_nucleus_box(nucleus, margin)
        raw = f["raw"][box]
        cell = f["labels/membrane"][box] > 0
    nucleus = nucleus[box]
    missing = betaseg_missing_data_mask(raw)
    ignore = (~cell & ~nucleus) | missing

    labels = nucleus.astype("uint32")
    labels[ignore] = IGNORE_LABEL
    out_path = os.path.join(out_dir, f"{name}.h5")
    tmp_path = f"{out_path}.incomplete"
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip", chunks=(32, 256, 256))
        f.create_dataset("labels/nucleus", data=labels, compression="gzip", chunks=(32, 256, 256))
        f.attrs["box"] = [[b.start, b.stop] for b in box]
        f.attrs["ignore_label"] = int(IGNORE_LABEL)
    os.replace(tmp_path, out_path)

    info = {
        "box": [[b.start, b.stop] for b in box], "shape": list(raw.shape),
        "nucleus_fraction": float(nucleus.mean()), "outside_cell_fraction": float((~cell).mean()),
        "missing_data_fraction": float(missing.mean()), "ignore_fraction": float(ignore.mean()),
        "nucleus_outside_cell_fraction": float((nucleus & ~cell).sum() / nucleus.sum()),
    }
    print(f"{name}: {info}")
    return name, info


def preprocess_betaseg(input_root, output_root, margin):
    """Write the BetaSeg nucleus training boxes.

    Only the nucleus of the central target cell is labeled in each volume. Each box is the bounding box of the labeled
    nucleus plus a margin. Neighbouring cells can bring unlabeled nuclei into the box, for example in high_c4 and
    low_c2. The labels carry IGNORE_LABEL outside the target cell mask and in the regions without image data, so the
    loss excludes these nuclei. The target cell mask is the 'membrane' label, a filled cell mask.
    """
    out_dir = os.path.join(output_root, "betaseg_nucleus_boxes")
    os.makedirs(out_dir, exist_ok=True)
    boxes = dict(
        write_betaseg_box(path, margin, out_dir)
        for path in sorted(glob.glob(os.path.join(input_root, "betaseg", "data", "*.h5")))
    )
    with open(os.path.join(out_dir, "boxes.json"), "w") as f:
        json.dump({"margin": margin, "ignore_label": int(IGNORE_LABEL), "boxes": boxes}, f, indent=2)


def organoids_padding_mask(raw_slice, value):
    components, _ = ndimage.label(raw_slice == value)
    areas = np.bincount(components.ravel())
    return np.isin(components, np.where(areas > ORGANOIDS_PADDING_MIN_AREA)[0]) & (components > 0)


def close_and_fill(mask):
    """Close the mask with a disk of ORGANOIDS_CLOSING_RADIUS and fill its holes. The padding keeps border nuclei."""
    radius = ORGANOIDS_CLOSING_RADIUS
    padded = np.pad(mask, radius)
    closed = ndimage.binary_closing(padded, structure=disk(radius))
    closed = closed[radius:-radius, radius:-radius] | mask
    return ndimage.binary_fill_holes(closed)


def label_slabwise(mask):
    """3D connected components (6-connectivity) into uint32, labeled in z-slabs that are merged across their borders.

    For the largest volume, the job memory cannot hold either alternative:
    - scipy's label holds an extra 8-byte array of the volume.
    - The cc3d output for the whole volume is uint64.

    Per slab, cc3d labels into uint32.
    """
    instances = np.zeros(mask.shape, dtype="uint32")
    offset = 0
    for z0 in range(0, mask.shape[0], LABEL_SLAB):
        labels, n = cc3d.connected_components(mask[z0:z0 + LABEL_SLAB], connectivity=6, return_N=True)
        labels = labels.astype("uint32", copy=False)
        labels[labels > 0] += offset
        instances[z0:z0 + LABEL_SLAB] = labels
        offset += n
    parent = np.arange(offset + 1)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for z in range(LABEL_SLAB, mask.shape[0], LABEL_SLAB):
        below, above = instances[z - 1], instances[z]
        touching = (below > 0) & (above > 0)
        for a, b in np.unique(np.stack([below[touching], above[touching]], axis=1), axis=0):
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                parent[max(root_a, root_b)] = min(root_a, root_b)
    roots = np.array([find(i) for i in range(offset + 1)])
    unique_roots, lut = np.unique(roots, return_inverse=True)
    lut = lut.astype("uint32")
    for z0 in range(0, mask.shape[0], LABEL_SLAB):
        instances[z0:z0 + LABEL_SLAB] = lut[instances[z0:z0 + LABEL_SLAB]]
    return instances, len(unique_roots) - 1


def clean_organoids_volume(path, padding_value):
    """Clean the nucleus labels of one volume, holding only the boolean mask and the instances in memory."""
    with h5py.File(path, "r") as f:
        raw, labels = f["raw"], f["nuclei"]
        nucleus = np.zeros(labels.shape, dtype=bool)
        n_filled = n_chipped = 0
        for z in range(labels.shape[0]):
            mask = labels[z] > 0
            filled = close_and_fill(mask)
            pad = organoids_padding_mask(raw[z], padding_value)
            n_filled += int((filled & ~mask).sum())
            n_chipped += int((filled & pad).sum())
            nucleus[z] = filled & ~pad
    instances, n_ids = label_slabwise(nucleus)
    del nucleus
    # Counted per slice: np.bincount casts its input to int64, a copy too large for the memory for the largest volume.
    sizes = np.zeros(n_ids + 1, dtype="int64")
    for plane in instances:
        sizes += np.bincount(plane.ravel(), minlength=n_ids + 1)
    is_fragment = sizes < ORGANOIDS_MIN_NUCLEUS_SIZE
    is_fragment[0] = False
    n_fragment_voxels = n_entotic = n_nucleus = 0
    with h5py.File(path, "r") as f:
        entotic = f["entotic_cell"] if "entotic_cell" in f else None
        for z in range(instances.shape[0]):
            plane = instances[z]
            fragment = is_fragment[plane]
            n_fragment_voxels += int(fragment.sum())
            plane[fragment] = IGNORE_LABEL
            if entotic is not None:
                cell = entotic[z] > 0
                n_entotic += int(cell.sum())
                plane[cell] = IGNORE_LABEL
            n_nucleus += int(((plane > 0) & (plane != IGNORE_LABEL)).sum())
    info = {
        "n_nuclei": int(n_ids - is_fragment.sum()), "n_fragments_ignored": int(is_fragment.sum()),
        "fragment_voxels": n_fragment_voxels, "voxels_added_by_closing_and_filling": n_filled,
        "padding_voxels_removed": n_chipped, "entotic_voxels_ignored": n_entotic,
        "nucleus_fraction": n_nucleus / instances.size,
    }
    return instances, info


def write_organoids_labels(instances, out_path):
    tmp_path = f"{out_path}.incomplete"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    group = zarr.open_group(tmp_path, mode="w")
    array = group.create_array("labels", shape=instances.shape, chunks=ORGANOIDS_CHUNKS, dtype="uint32")
    for z in range(0, instances.shape[0], ORGANOIDS_CHUNKS[0]):
        array[z:z + ORGANOIDS_CHUNKS[0]] = instances[z:z + ORGANOIDS_CHUNKS[0]]
    group.attrs["ignore_label"] = int(IGNORE_LABEL)
    # The labels of an earlier run are replaced.
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.replace(tmp_path, out_path)


def preprocess_human_organoids(input_root, output_root, volumes):
    """Write the cleaned nucleus instance labels of the human organoids volumes.

    The nucleus labels are manual semantic masks (int8, -128 background, 127 nucleus). The cleaning has these steps:
    1. The speckled masks are closed and the holes inside the nuclei are filled, per z-slice.
    2. The labels on the black alignment padding are removed, so that nuclei are only labeled on valid raw. The padding
       is a connected region of the padding value of the volume above ORGANOIDS_PADDING_MIN_AREA pixels of a slice.
    3. 3D connected components separate the nuclei into instances.
    4. Components below ORGANOIDS_MIN_NUCLEUS_SIZE voxels are label fragments (speckle clouds and scribbles), not
       nuclei. They get IGNORE_LABEL. The entotic cell of F107_A1 also gets IGNORE_LABEL, because its nucleus is not
       labeled.

    F107_A3 needs a large-memory node.
    """
    out_dir = os.path.join(output_root, "human_organoids_nucleus")
    os.makedirs(out_dir, exist_ok=True)
    for volume in volumes:
        path = os.path.join(input_root, "human_organoids", "11380", "data", f"{volume}_bin2.h5")
        instances, info = clean_organoids_volume(path, ORGANOIDS_VOLUMES[volume])
        write_organoids_labels(instances, os.path.join(out_dir, f"{volume}_nucleus_labels.zarr"))
        print(f"{volume}: {info}")
        with open(os.path.join(out_dir, f"{volume}_info.json"), "w") as f:
            json.dump(info, f, indent=2)


def write_spheroid_slice(path, out_dir):
    with h5py.File(path, "r") as f:
        raw = f["raw"][:]
        nuclei = f["labels/nuclei"][:].astype("uint32")
        cells = f["labels/cells"][:]
    cell_ids = np.unique(cells[cells > 0])
    with_nucleus = np.unique(cells[(cells > 0) & (nuclei > 0)])
    without_nucleus = np.setdiff1d(cell_ids, with_nucleus)
    ignore = np.isin(cells, without_nucleus) & (nuclei == 0)
    labels = nuclei.copy()
    labels[ignore] = IGNORE_LABEL
    out_path = os.path.join(out_dir, os.path.basename(path))
    tmp_path = f"{out_path}.incomplete"
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("labels", data=labels, compression="gzip")
        f.attrs["ignore_label"] = int(IGNORE_LABEL)
    os.replace(tmp_path, out_path)
    info = {
        "n_nuclei": int(len(np.unique(nuclei)) - 1), "n_cells": int(len(cell_ids)),
        "n_cells_without_nucleus": int(len(without_nucleus)), "ignore_fraction": float(ignore.mean()),
    }
    print(os.path.basename(path), info)
    return os.path.basename(path), info


def preprocess_tumor_spheroid(input_root, output_root):
    """Write the tumor spheroid nucleus labels, with the cells without a nucleus ignored.

    The 20 manually annotated slices (50 nm) label cells and nuclei. A few large cells per slice carry a cell label but
    no nucleus label, although their dark, mottled content looks like nucleoplasm. It is unclear whether the annotation
    misses their nucleus or the section does not contain it. Thus, these cells get IGNORE_LABEL. The nuclei keep their
    ids. The labels are uint32, so that the ignore label survives torch-em's cast of the labels to their stored dtype.
    """
    source = os.path.join(input_root, "tumor_spheroid_em", "2d_manual", "50-50-50")
    out_dir = os.path.join(output_root, "tumor_spheroid_nucleus")
    os.makedirs(out_dir, exist_ok=True)
    infos = dict(write_spheroid_slice(path, out_dir) for path in sorted(glob.glob(os.path.join(source, "*.h5"))))
    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump({"ignore_label": int(IGNORE_LABEL), "slices": infos}, f, indent=2)


def file_number(name):
    return int(re.sub(r"\D", "", os.path.splitext(name)[0]))


def sorted_names(folder):
    return sorted(os.listdir(folder), key=file_number)


def write_waenet_h5(out_path, raw, labels, attrs):
    tmp_path = f"{out_path}.incomplete"
    with h5py.File(tmp_path, "w") as f:
        is_stack = raw.ndim == 3 and raw.shape[1:] == (WAENET_SIZE, WAENET_SIZE)
        chunks = (1, WAENET_SIZE, WAENET_SIZE) if is_stack else True
        f.create_dataset("raw", data=raw, compression="gzip", chunks=chunks)
        f.create_dataset("labels", data=labels, compression="gzip", chunks=chunks)
        for key, value in attrs.items():
            f.attrs[key] = value
    os.replace(tmp_path, out_path)


def waenet_padding_mask(raw):
    components, _ = ndimage.label(raw == 0)
    areas = np.bincount(components.ravel())
    return np.isin(components, np.where(areas > WAENET_PADDING_MIN_AREA)[0]) & (components > 0)


def waenet_instances_2d(nucleus, ignore=None):
    labels, _ = ndimage.label(nucleus)
    labels = labels.astype("uint32")
    sizes = np.bincount(labels.ravel())
    small = np.isin(labels, np.where(sizes < WAENET_MIN_NUCLEUS_AREA)[0]) & (labels > 0)
    labels[small] = IGNORE_LABEL
    if ignore is not None:
        labels[ignore & (labels == 0)] = IGNORE_LABEL
    return labels


def build_waenet_dataset_1(root, out_dir):
    names = sorted_names(os.path.join(root, "dataset_1"))
    raw, nucleus = [], []
    for name in names:
        with h5py.File(os.path.join(root, "dataset_1", name), "r") as f:
            raw.append(f["raw"][:])
            nucleus.append(f["labels/nucleus"][:] > 0)
    raw, nucleus = np.stack(raw), np.stack(nucleus)
    labels, n_ids = ndimage.label(nucleus)
    write_waenet_h5(os.path.join(out_dir, "dataset_1.h5"), raw, labels.astype("uint32"), {"slices": names})
    n_val = int(round(raw.shape[0] * WAENET_VAL_FRACTION))
    return {"shape": list(raw.shape), "n_nuclei": int(n_ids), "n_val_slices": n_val}


def build_waenet_dataset_3(root, out_dir):
    images = {"train": [], "val": []}
    n_removed = n_empty = 0
    for name in sorted_names(os.path.join(root, "dataset_3")):
        with h5py.File(os.path.join(root, "dataset_3", name), "r") as f:
            raw, nucleus = f["raw"][:], f["labels/nucleus"][:] > 0
        pad = waenet_padding_mask(raw)
        n_removed += int((nucleus & pad).sum())
        nucleus &= ~pad
        if not nucleus.any():
            n_empty += 1
            continue
        split = "val" if file_number(name) in WAENET_DS3_VAL_SECTIONS else "train"
        images[split].append((name, raw, waenet_instances_2d(nucleus)))
    info = {"n_without_nucleus": n_empty, "padding_pixels_removed": n_removed}
    for split, items in images.items():
        write_waenet_h5(
            os.path.join(out_dir, f"dataset_3_{split}.h5"), np.stack([it[1] for it in items]),
            np.stack([it[2] for it in items]), {"images": [it[0] for it in items]},
        )
        info[f"n_{split}"] = len(items)
    return info


def fit_to_size(raw, labels):
    """Crop around the labeled cell or pad with IGNORE_LABEL to WAENET_SIZE x WAENET_SIZE."""
    cell = labels != IGNORE_LABEL
    out_raw = np.zeros((WAENET_SIZE, WAENET_SIZE), dtype=raw.dtype)
    out_labels = np.full((WAENET_SIZE, WAENET_SIZE), IGNORE_LABEL, dtype=labels.dtype)
    src, dst = [], []
    for ax in range(2):
        n = raw.shape[ax]
        if n >= WAENET_SIZE:
            center = int(np.mean(np.where(cell.any(axis=1 - ax))[0]))
            start = min(max(center - WAENET_SIZE // 2, 0), n - WAENET_SIZE)
            src.append(slice(start, start + WAENET_SIZE))
            dst.append(slice(0, WAENET_SIZE))
        else:
            src.append(slice(0, n))
            dst.append(slice(0, n))
    out_raw[tuple(dst)] = raw[tuple(src)]
    out_labels[tuple(dst)] = labels[tuple(src)]
    return out_raw, out_labels


def build_waenet_tem(root, out_dir):
    images = {"train": [], "val": []}
    info = {}
    for dataset in WAENET_TEM_DATASETS:
        folder = f"dataset_{dataset}"
        kept, n_empty, n_dropped, n_resized = [], 0, 0, 0
        for name in sorted_names(os.path.join(root, folder)):
            with h5py.File(os.path.join(root, folder, name), "r") as f:
                raw = f["raw"][:]
                nucleus = f["labels/nucleus"][:] > 0
                cytoplasm = f["labels/cytoplasm"][:] > 0
            if not nucleus.any():
                n_empty += 1
                continue
            if f"{folder}/{name}" in WAENET_DROPPED_IMAGES:
                n_dropped += 1
                continue
            band = ndimage.binary_dilation(nucleus | cytoplasm, iterations=WAENET_CELL_BAND)
            labels = waenet_instances_2d(nucleus, ignore=~band)
            if raw.shape != (WAENET_SIZE, WAENET_SIZE):
                raw, labels = fit_to_size(raw, labels)
                n_resized += 1
            kept.append((f"{folder}/{name}", raw, labels))
        n_val = int(round(len(kept) * WAENET_VAL_FRACTION))
        images["train"].extend(kept[:len(kept) - n_val])
        images["val"].extend(kept[len(kept) - n_val:])
        info[folder] = {
            "n_without_nucleus": n_empty, "n_dropped": n_dropped, "n_fitted_to_512": n_resized,
            "n_train": len(kept) - n_val, "n_val": n_val,
        }
    for split, items in images.items():
        labels = np.stack([it[2] for it in items])
        write_waenet_h5(
            os.path.join(out_dir, f"tem_{split}.h5"), np.stack([it[1] for it in items]), labels,
            {"images": [it[0] for it in items], "ignore_label": int(IGNORE_LABEL)},
        )
        info[f"ignore_fraction_{split}"] = float((labels == IGNORE_LABEL).mean())
    return info


def preprocess_waenet(input_root, output_root):
    """Write the WAE-Net nucleus training data.

    - Dataset 1 (FIB-SEM of pancreatic carcinoid cells, all nuclei labeled): the 323 slices, stacked in numeric file
      order into one volume. The nuclei are 3D connected components. The last WAENET_VAL_FRACTION of z validates.
    - Dataset 3 (unregistered serial ssTEM sections of one interphase cell): one 2D image per section. The nuclei are
      2D connected components, and the labels on the black rotation padding are removed. A contiguous block of
      sections validates, because neighbouring sections are near-duplicates.
    - Datasets 4-7 (single TEM images of Jurkat, primary T, J558L B and expanded T cells): only the central cell is
      labeled. Its mask is the union of the 'cytoplasm' and 'nucleus' labels. The loss is restricted to the central
      cell mask dilated by WAENET_CELL_BAND pixels. All other pixels get IGNORE_LABEL. Images without a nucleus label
      and the WAENET_DROPPED_IMAGES are not used. Images that are not 512 x 512 are cropped around the central cell or
      padded with IGNORE_LABEL. The last WAENET_VAL_FRACTION of the images of each dataset (by file number) validates,
      because consecutive images can be serial sections of one cell.

    The 2D datasets are stacks of 512 x 512 images, one h5 file per split.
    """
    root = os.path.join(input_root, "waenet")
    out_dir = os.path.join(output_root, "waenet_nucleus")
    os.makedirs(out_dir, exist_ok=True)
    info = {
        "dataset_1": build_waenet_dataset_1(root, out_dir),
        "dataset_3": build_waenet_dataset_3(root, out_dir),
        "tem": build_waenet_tem(root, out_dir),
        "dropped_images": WAENET_DROPPED_IMAGES, "cell_band": WAENET_CELL_BAND, "ignore_label": int(IGNORE_LABEL),
    }
    print(json.dumps(info, indent=2))
    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)


def write_cellmap_crop(store, zip_file, volume, crop_id, em_levels, out_dir):
    raw, labels, info = load_cellmap_crop(store, zip_file, volume, crop_id, em_levels, "nuc")
    labels = labels.astype("uint32")
    assert raw.shape == labels.shape, f"{raw.shape} != {labels.shape}"
    name = f"{volume}_crop{crop_id}"
    out_path = os.path.join(out_dir, f"{name}.zarr")
    tmp_path = f"{out_path}.incomplete"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    group = zarr.open_group(tmp_path, mode="w")
    group.create_array("raw", data=raw, chunks=CELLMAP_NUCLEUS_CHUNKS)
    group.create_array("labels", data=labels, chunks=CELLMAP_NUCLEUS_CHUNKS)
    group.attrs.update({
        "volume": volume, "crop": crop_id, "voxel_size_nm": info["voxel_size_nm"],
        "translation_nm": info["translation_nm"], "em_level": info["em_level"],
    })
    os.replace(tmp_path, out_path)
    ids = np.unique(labels)
    entry = {
        "name": name, "path": out_path, "volume": volume, "crop": crop_id, "shape": list(raw.shape),
        "voxel_size_nm": info["voxel_size_nm"], "n_ids": int((ids != 0).sum()), "fg_frac": float((labels > 0).mean()),
        "clipped_to_em": info["clipped_to_em"], "raw_zero_frac": info["raw_zero_frac"],
        "n_unknown_voxels_mapped_to_0": info["n_unknown_voxels"],
    }
    print(entry)
    return entry


def preprocess_cellmap(input_root, output_root):
    """Cache the CellMap nucleus crops.

    Only the large tissue crops (32-64 nm, many complete nuclei) are used. Most small high-resolution crops hold one
    nucleus cut by the crop borders, and 15 lie fully inside a nucleus. The in-domain evaluation uses crop357 of
    jrc_mus-liver-zon-2, because it does not overlap any other crop. The crops crop358 and crop367 overlap each other,
    so neither of them can be held out on its own. Each crop is one zarr store with 'raw' (uint8) and 'labels' (the
    nucleus instance ids, uint32) at the label scale.
    """
    zip_path = os.path.join(input_root, CELLMAP_ZIP)
    out_dir = os.path.join(output_root, "cellmap_nucleus_crops")
    os.makedirs(out_dir, exist_ok=True)
    store = zarr.storage.ZipStore(zip_path, mode="r")
    zip_file = zipfile.ZipFile(zip_path)
    index = []
    for split, crops in (("train", CELLMAP_NUCLEUS_TRAIN_CROPS), ("test", CELLMAP_NUCLEUS_TEST_CROPS)):
        for volume, crop_ids in crops.items():
            em_levels = get_cellmap_em_levels(zip_file, volume)
            for crop_id in crop_ids:
                entry = write_cellmap_crop(store, zip_file, volume, crop_id, em_levels, out_dir)
                index.append({**entry, "split": split})
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)


def to_uint8(raw):
    if raw.dtype == np.uint8:
        return raw
    lo, hi = np.percentile(raw, (0.5, 99.5))
    return (np.clip((raw.astype("float32") - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype("uint8")


def downscale(raw, labels, factor):
    """Mean-downscale the raw data and subsample the labels by an integer factor along all axes."""
    if factor == 1:
        return raw, labels
    factors = (factor,) * raw.ndim
    small = downscale_local_mean(raw.astype("float32"), factors).astype("uint8")
    sl = tuple(slice(0, s * factor, factor) for s in small.shape)
    return small, labels[sl]


def clean_instances(labels, fill=True):
    """Fill holes per object, drop objects below MIN_OBJECT_SIZE and relabel consecutively."""
    out = np.zeros(labels.shape, dtype="uint32")
    next_id = 1
    for i, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None:
            continue
        mask = labels[box] == i
        if fill:
            mask = ndimage.binary_fill_holes(mask)
        if mask.sum() < MIN_OBJECT_SIZE:
            continue
        out[box][mask] = next_id
        next_id += 1
    return out


def write_h5(path, raw, labels, attrs=None):
    tmp_path = f"{path}.incomplete"
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("labels", data=labels, compression="gzip")
        for key, value in (attrs or {}).items():
            f.attrs[key] = value
    os.replace(tmp_path, path)


def build_cardiomyocyte(input_root, output_root):
    """The 9 TEM micrographs with a nucleus mask, downscaled CARDIOMYOCYTE_SCALE times (nuclei are 900-1600 px wide).

    The holes are filled and specks below MIN_OBJECT_SIZE are dropped.
    """
    src = os.path.join(input_root, "cardiomyocyte_tem")
    out_dir = os.path.join(output_root, "cardiomyocyte_tem_nucleus")
    os.makedirs(out_dir, exist_ok=True)
    infos = {}
    for image_path in sorted(glob.glob(os.path.join(src, "images", "*.tif"))):
        name = os.path.basename(image_path)[:-4]
        masks = [
            p for p in glob.glob(os.path.join(src, "Nucleus_Masks", "*.tif"))
            if os.path.basename(p) in (f"{name}_nucleus.tif", f"KO_{name}_nucleus.tif")
        ]
        if not masks:
            continue
        raw = to_uint8(tifffile.imread(image_path))
        labels = label(tifffile.imread(masks[0]) > 0).astype("uint32")
        raw, labels = downscale(raw, labels, CARDIOMYOCYTE_SCALE)
        labels = clean_instances(labels)
        write_h5(os.path.join(out_dir, f"{name}.h5"), raw, labels, {"scale": CARDIOMYOCYTE_SCALE})
        infos[name] = {"shape": list(raw.shape), "n_nuclei": int(labels.max())}
        print("cardiomyocyte", name, infos[name], flush=True)
    return infos


def build_perez(input_root, output_root):
    """The nucleus task tiles (train 500 px, resized to 512 px; test 1000 px), holes filled."""
    src = os.path.join(input_root, "perez", "perez", "nuclei")
    infos = {}
    for split in ("train", "test"):
        out_dir = os.path.join(output_root, "perez_nucleus", split)
        os.makedirs(out_dir, exist_ok=True)
        for image_path in sorted(glob.glob(os.path.join(src, split, "images", "*.png"))):
            name = os.path.basename(image_path)[:-4]
            raw = imageio.imread(image_path)
            raw = raw[..., 0] if raw.ndim == 3 else raw
            mask = imageio.imread(os.path.join(src, split, "masks", os.path.basename(image_path)))
            mask = (mask[..., 0] if mask.ndim == 3 else mask) > 0
            if raw.shape[0] < 512:
                raw = resize(raw, (512, 512), preserve_range=True, anti_aliasing=True).astype("uint8")
                mask = resize(mask, (512, 512), order=0, anti_aliasing=False).astype(bool)
            labels = clean_instances(label(mask).astype("uint32"))
            write_h5(os.path.join(out_dir, f"{name}.h5"), raw, labels)
            infos[f"{split}/{name}"] = int(labels.max())
    print("perez", {split: sum(1 for k in infos if k.startswith(split)) for split in ("train", "test")}, flush=True)
    return infos


def border_ignore(labels, width):
    """IGNORE_LABEL for the unlabeled pixels within width px of the image border."""
    band = np.zeros(labels.shape, dtype=bool)
    band[:width] = band[-width:] = True
    band[:, :width] = band[:, -width:] = True
    out = labels.copy()
    out[band & (labels == 0)] = IGNORE_LABEL
    return out


def build_vem(input_root, output_root):
    """The 2D array tomography images and the two 3D FIB-SEM volumes of vem_nuclei.

    Each 2D image is downscaled by a power of two, so that nuclei are about VEM_TARGET_DIAMETER px wide. The 3D volumes
    are downscaled VEM_3D_SCALE times. Most 2D label images (all except VEM_EDGE_NUCLEI_INCLUDED) do not label the
    nuclei that touch the image border. Thus, the unlabeled pixels in a border band of the largest nucleus diameter get
    IGNORE_LABEL. The archive truncates image id_9, so it is left out.
    """
    src = os.path.join(input_root, "vem_nuclei")
    out_dir = os.path.join(output_root, "vem_nuclei_nucleus")
    os.makedirs(out_dir, exist_ok=True)
    infos = {}
    for folder in sorted(glob.glob(os.path.join(src, "id_*")), key=lambda p: int(p.rsplit("_", 1)[1])):
        sample_id = int(folder.rsplit("_", 1)[1])
        if sample_id in VEM_MISSING:
            continue
        raw = tifffile.imread(glob.glob(os.path.join(folder, "img_*"))[0])
        labels = tifffile.imread(glob.glob(os.path.join(folder, "lbl_*"))[0])
        if raw.ndim == 3:
            raw, labels = downscale(to_uint8(raw), label(labels > 0).astype("uint32"), VEM_3D_SCALE)
            labels = clean_instances(labels, fill=False)
            factor = VEM_3D_SCALE
        else:
            labels = labels.astype("uint32")
            sizes = np.bincount(labels.ravel())[1:]
            diameter = float(np.sqrt(4 * np.median(sizes[sizes > 0]) / np.pi))
            factor = int(max(1, 2 ** round(np.log2(diameter / VEM_TARGET_DIAMETER))))
            raw, labels = downscale(to_uint8(raw), labels, factor)
            labels = clean_instances(labels)
            if sample_id not in VEM_EDGE_NUCLEI_INCLUDED and labels.max() > 0:
                boxes = [b for b in ndimage.find_objects(labels) if b is not None]
                extents = [max(b[0].stop - b[0].start, b[1].stop - b[1].start) for b in boxes]
                width = int(np.percentile(extents, VEM_BORDER_PERCENTILE))
                labels = border_ignore(labels, width)
        write_h5(os.path.join(out_dir, f"id_{sample_id}.h5"), raw, labels, {"scale": factor})
        n = len(np.unique(labels[(labels > 0) & (labels != IGNORE_LABEL)]))
        infos[f"id_{sample_id}"] = {
            "shape": list(raw.shape), "scale": factor, "n_nuclei": n,
            "ignore_fraction": float((labels == IGNORE_LABEL).mean()),
        }
        print("vem", sample_id, infos[f"id_{sample_id}"], flush=True)
    return infos


def fill_envelope(envelope):
    """The nucleus part of a crop that cuts the nucleus, from the envelope arc, or None if it is not reliable.

    The crops cut the nucleus, so the envelope is an arc (often dashed) with the nucleus on its right. A nucleus is
    convex. Thus, its part in the crop is the convex hull of the arc and the right border stretch that the arc spans in
    y. The arc has these parts:
    - the envelope pieces (after closing small gaps) of at least U2OS_MIN_PIECE_FRACTION of the envelope.
    - the smaller pieces within U2OS_MAX_PIECE_GAP px of them, which are the ends of a dashed arc.

    Stray fragments farther away are dropped, because they would pull the hull across the cytoplasm.
    """
    if envelope.sum() < U2OS_MIN_ENVELOPE_PIXELS:
        return None
    pieces = label(ndimage.binary_closing(envelope, structure=disk(U2OS_CLOSING_RADIUS)))
    piece_sizes = np.bincount(pieces[envelope])
    piece_sizes[0] = 0
    kept = set(np.nonzero(piece_sizes >= U2OS_MIN_PIECE_FRACTION * envelope.sum())[0].tolist())
    while True:
        near = ndimage.distance_transform_edt(~np.isin(pieces, list(kept))) <= U2OS_MAX_PIECE_GAP
        added = set(np.unique(pieces[near & envelope]).tolist()) - kept - {0}
        if not added:
            break
        kept |= added
    arc = envelope & np.isin(pieces, list(kept))
    ys = np.nonzero(arc)[0]
    border = np.zeros_like(arc)
    border[ys.min():ys.max() + 1, -1] = True
    nucleus = convex_hull_image(arc | border)
    if not U2OS_MIN_FILLED_FRACTION <= nucleus.mean() <= 1 - U2OS_MIN_FILLED_FRACTION:
        return None
    return nucleus.astype("uint32")


def build_u2os(input_root, output_root):
    """The 44 nuclear envelope slices of one U2-OS cell (evaluation), filled into nucleus masks.

    The crops cut the nucleus. Thus, the nucleus part in a crop is the convex hull of the envelope arc and the right
    border that the arc spans. The right border is the last column with data, before the constant padding of some
    slices.
    """
    src = os.path.join(input_root, "u2os_fibsem", "1_NE", "1_Train_NE")
    out_dir = os.path.join(output_root, "u2os_fibsem_nucleus")
    os.makedirs(out_dir, exist_ok=True)
    infos = {}
    for label_path in sorted(glob.glob(os.path.join(src, "labels_tif", "*.tif"))):
        name = os.path.basename(label_path)[:-4]
        raw = to_uint8(tifffile.imread(os.path.join(src, "images_tif", f"{name}.tif")))
        # Some slices end in constant padding columns on the right; the nucleus reaches the last column with data.
        width = int(np.nonzero(raw.std(axis=0) >= 1)[0].max()) + 1
        filled = fill_envelope(tifffile.imread(label_path)[:, :width] > 0)
        nucleus = None if filled is None else np.pad(filled, ((0, 0), (0, raw.shape[1] - width)))
        out_path = os.path.join(out_dir, f"{name}.h5")
        if nucleus is None:
            # The output of an earlier build is removed, so a rejected slice is not evaluated.
            if os.path.exists(out_path):
                os.remove(out_path)
            infos[name] = {"used": False}
            continue
        write_h5(out_path, raw, nucleus)
        infos[name] = {"used": True, "nucleus_fraction": float((nucleus > 0).mean())}
    print("u2os", sum(i["used"] for i in infos.values()), "of", len(infos), "slices filled", flush=True)
    return infos


def build_etchacell(input_root, output_root):
    """The HeLa SBF-SEM ROIs (evaluation, 10 x 10 x 50 nm) with nuclear envelope consensus labels.

    The labels are filled into nucleus masks, per slice after closing small gaps, and in 3D. The 3D fill also fills the
    cap slices, where the cut through the envelope is tangential. The ROIs are downscaled ETCHACELL_SCALE times in xy,
    so that a nucleus fits a 512 px field of view.
    """
    src = os.path.join(input_root, "etchacell")
    out_dir = os.path.join(output_root, "etchacell_nucleus")
    os.makedirs(out_dir, exist_ok=True)
    infos = {}
    for raw_path in sorted(glob.glob(os.path.join(src, "raw", "*.tiff"))):
        name = os.path.basename(raw_path)[:-5]
        out_path = os.path.join(out_dir, f"{name}.h5")
        envelope = tifffile.imread(os.path.join(src, "labels", f"{name}.tiff")) > 0
        closing = disk(ETCHACELL_CLOSING_RADIUS)[None]
        envelope = ndimage.binary_closing(envelope, structure=closing)
        filled = np.stack([ndimage.binary_fill_holes(plane) for plane in envelope])
        filled |= ndimage.binary_fill_holes(envelope)
        del envelope
        raw = tifffile.imread(raw_path)
        small = downscale_local_mean(raw.astype("float32"), (1, ETCHACELL_SCALE, ETCHACELL_SCALE)).astype("uint8")
        del raw
        labels = filled[:, ::ETCHACELL_SCALE, ::ETCHACELL_SCALE][:, :small.shape[1], :small.shape[2]]
        labels = clean_instances(label(labels).astype("uint32"), fill=False)
        write_h5(out_path, small, labels, {"scale_xy": ETCHACELL_SCALE, "voxel_size_nm": [50, 40, 40]})
        infos[name] = {
            "shape": list(small.shape), "n_nuclei": int(labels.max()), "nucleus_fraction": float((labels > 0).mean()),
        }
        print("etchacell", name, infos[name], flush=True)
    return infos


def build_aive(input_root, output_root):
    """The nucleus of the three AIVE FIB-SEM overview volumes with a nucleus label (6.5 x 6.5 x 10 nm).

    The 'chromatin' class marks the nucleoli inside the nucleus, so it is folded into the nucleus class. The holes are
    filled. A nucleus is about 3000 px long at full resolution. Thus, the volumes are downscaled AIVE_SCALE times
    (z, y, x) to 30 x 26 x 26 nm.
    """
    out_dir = os.path.join(output_root, "aive_nucleus")
    os.makedirs(out_dir, exist_ok=True)
    infos = {}
    for volume in AIVE_VOLUMES:
        with h5py.File(os.path.join(input_root, "aive", f"aive_dataset{volume}.h5"), "r") as f:
            raw_ds, label_ds = f["raw"], f["labels"]
            fz, fy, fx = AIVE_SCALE
            nz, ny, nx = (s // factor for s, factor in zip(raw_ds.shape, AIVE_SCALE))
            raw = np.zeros((nz, ny, nx), dtype="uint8")
            mask = np.zeros((nz, ny, nx), dtype=bool)
            step = 16 * fz
            for z0 in range(0, nz * fz, step):
                z1 = min(z0 + step, nz * fz)
                block = raw_ds[z0:z1, :ny * fy, :nx * fx].astype("float32")
                raw[z0 // fz:z1 // fz] = downscale_local_mean(block, AIVE_SCALE).astype("uint8")
                labels = label_ds[z0:z1:fz, :ny * fy:fy, :nx * fx:fx]
                mask[z0 // fz:z1 // fz] = np.isin(labels, AIVE_NUCLEUS_CLASSES)
        mask = np.stack([ndimage.binary_fill_holes(plane) for plane in mask])
        mask = ndimage.binary_fill_holes(mask)
        instances = clean_instances(label(mask).astype("uint32"), fill=False)
        write_h5(os.path.join(out_dir, f"aive_dataset{volume}.h5"), raw, instances, {"voxel_size_nm": [30, 26, 26]})
        infos[f"aive_dataset{volume}"] = {
            "shape": list(raw.shape), "n_nuclei": int(instances.max()),
            "nucleus_fraction": float((instances > 0).mean()),
        }
        print("aive", volume, infos[f"aive_dataset{volume}"], flush=True)
    return infos


def downscale_volume(path, raw_key, label_key, scale, binary):
    """Read a volume slab-wise and downscale it: mean for the raw data, subsampling for the labels."""
    with h5py.File(path, "r") as f:
        raw_ds, label_ds = f[raw_key], f[label_key]
        fz, fy, fx = scale
        nz, ny, nx = (s // factor for s, factor in zip(raw_ds.shape, scale))
        raw = np.zeros((nz, ny, nx), dtype="uint8")
        labels = np.zeros((nz, ny, nx), dtype="uint32")
        step = 16 * fz
        for z0 in range(0, nz * fz, step):
            z1 = min(z0 + step, nz * fz)
            block = raw_ds[z0:z1, :ny * fy, :nx * fx].astype("float32")
            raw[z0 // fz:z1 // fz] = downscale_local_mean(block, scale).astype("uint8")
            block_labels = label_ds[z0:z1:fz, :ny * fy:fy, :nx * fx:fx]
            labels[z0 // fz:z1 // fz] = block_labels > 0 if binary else block_labels
    return raw, label(labels > 0).astype("uint32") if binary else labels


def build_volumes(output_root, folder, volumes, scale, binary, min_nucleus_size=None):
    out_dir = os.path.join(output_root, folder)
    os.makedirs(out_dir, exist_ok=True)
    infos = {}
    for name, path, label_key in volumes:
        if not os.path.exists(path):
            print("missing", path, flush=True)
            continue
        raw, labels = downscale_volume(path, "raw", label_key, scale, binary)
        labels = clean_instances(labels, fill=binary)
        if min_nucleus_size is not None:
            sizes = np.bincount(labels.ravel())
            small = np.nonzero(sizes < min_nucleus_size)[0]
            labels[np.isin(labels, small[small > 0])] = IGNORE_LABEL
        write_h5(os.path.join(out_dir, f"{name}.h5"), raw, labels, {"scale_zyx": list(scale)})
        n_nuclei = len(np.setdiff1d(np.unique(labels), [0, IGNORE_LABEL]))
        infos[name] = {"shape": list(raw.shape), "n_nuclei": n_nuclei, "ignored": bool((labels == IGNORE_LABEL).any())}
        print(folder, name, infos[name], flush=True)
    return infos


def build_cerebellum(input_root, output_root):
    """The mouse cerebellum FIB-SEM volume (10 nm isotropic) with its nucleus instances.

    The volume is downscaled CEREBELLUM_SCALE times, because nuclei are up to 600 px wide at full resolution. The
    labels below CEREBELLUM_MIN_NUCLEUS_SIZE voxels get IGNORE_LABEL. They are three pieces of one dark object inside a
    myelinated axon at the end of the volume, not a nucleus by eye.
    """
    path = os.path.join(input_root, "cerebellum_fibsem", "cerebellum_fibsem.h5")
    return build_volumes(
        output_root, "cerebellum_fibsem_nucleus", [("cerebellum_fibsem", path, "labels/nuclei")], CEREBELLUM_SCALE,
        False, min_nucleus_size=CEREBELLUM_MIN_NUCLEUS_SIZE,
    )


def build_macrophage(input_root, output_root):
    """The macrophage / A431 SBF-SEM volumes with the nucleus class (quick semantic MIB annotations).

    The volumes are downscaled MACROPHAGE_SCALE times in xy. Each volume has one nucleus of about 1300 px. The voxel
    size is not stored. The nucleus extents suggest sections about 10 times thicker than the pixel size. Thus, z / xy
    is about 2.5 after downscaling.
    """
    root = os.path.join(input_root, "macrophage_a431")
    volumes = [(name, os.path.join(root, f"{name}.h5"), "labels/nucleus") for name in MACROPHAGE_VOLUMES]
    return build_volumes(output_root, "macrophage_a431_nucleus", volumes, MACROPHAGE_SCALE, True)


# The builders of the nucleus data of the new torch-em datasets, with their output folder.
BUILDERS = {
    "cardiomyocyte": (build_cardiomyocyte, "cardiomyocyte_tem_nucleus"), "perez": (build_perez, "perez_nucleus"),
    "vem": (build_vem, "vem_nuclei_nucleus"), "u2os": (build_u2os, "u2os_fibsem_nucleus"),
    "etchacell": (build_etchacell, "etchacell_nucleus"), "aive": (build_aive, "aive_nucleus"),
    "cerebellum": (build_cerebellum, "cerebellum_fibsem_nucleus"),
    "macrophage": (build_macrophage, "macrophage_a431_nucleus"),
}


def preprocess_new_dataset(name, input_root, output_root):
    """Build the nucleus data of one new torch-em dataset. Every file is written under a temporary name."""
    build, folder = BUILDERS[name]
    infos = build(input_root, output_root)
    with open(os.path.join(output_root, folder, "build_info.json"), "w") as f:
        json.dump(infos, f, indent=2)


def with_retries(fn, n_attempts=12, max_wait=300):
    """Call *fn*, retrying with an exponential backoff on request errors.

    Under load the proxy answers 503 or drops the connection, and EMPIAR answers 403 for a while.
    """
    for attempt in range(n_attempts):
        try:
            return fn()
        except requests.RequestException as e:
            if attempt == n_attempts - 1:
                raise
            wait = min(2 ** attempt, max_wait)
            print(f"Request failed ({type(e).__name__}), retrying in {wait} s", flush=True)
            time.sleep(wait)


def read_nucleus_mask(z):
    Image.MAX_IMAGE_PIXELS = None
    url = f"{EMPIAR_BASE}/humanliver_nucleus_mask/humanliver_nucleus_mask_{z:03d}.png"

    def get():
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        return r

    r = with_retries(get)
    mask = np.array(Image.open(io.BytesIO(r.content)))
    return (mask[..., 0] if mask.ndim == 3 else mask) > 0


def block_fraction(mask):
    ny, nx = mask.shape[0] // HUMAN_LIVER_BLOCK, mask.shape[1] // HUMAN_LIVER_BLOCK
    block = HUMAN_LIVER_BLOCK
    return mask[:ny * block, :nx * block].reshape(ny, block, nx, block).mean(axis=(1, 3)).astype("float32")


def map_slice(z):
    fraction = block_fraction(read_nucleus_mask(z))
    print(f"z={z}: nucleus fraction {fraction.mean():.4f}", flush=True)
    return fraction


def map_human_liver_nuclei(output, step):
    """Map the nucleus fraction per HUMAN_LIVER_BLOCK x HUMAN_LIVER_BLOCK block of every step-th slice to place boxes.

    The nucleus mask PNG of a slice is about 0.85 MB. The maps of all sampled slices go to one npz file, together with
    the sampled z indices.
    """
    z_ids = list(range(0, HUMAN_LIVER_N_SLICES, step))
    with ThreadPoolExecutor(4) as pool:
        maps = list(pool.map(map_slice, z_ids))
    out_path = os.path.join(output, "nucleus_map.npz")
    np.savez_compressed(out_path, maps=np.stack(maps), z=np.array(z_ids), block=HUMAN_LIVER_BLOCK)
    print(f"Saved {out_path}")


def select_windows(map_path, depth, step, n_best):
    """Score 4096 x 4096 windows of depth slices by the labeled nuclei in the sampled slices of the nucleus map."""
    data = np.load(map_path)
    maps, z_ids, block = data["maps"], data["z"], int(data["block"])
    win = HUMAN_LIVER_BOX_SIZE // block
    candidates = []
    for z0 in range(0, int(z_ids.max()) - depth + 2, step):
        sel = (z_ids >= z0) & (z_ids < z0 + depth)
        if sel.sum() == 0:
            continue
        stack = maps[sel]
        for by in range(0, (HUMAN_LIVER_MAX_Y - HUMAN_LIVER_BOX_SIZE) // block + 1, 4):
            for bx in range(0, stack.shape[2] - win + 1, 4):
                window = stack[:, by:by + win, bx:bx + win]
                n_nuclei = np.mean([ndimage.label(w > 0.5)[1] for w in window])
                candidates.append({
                    "z": [z0, z0 + depth], "y": [by * block, by * block + HUMAN_LIVER_BOX_SIZE],
                    "x": [bx * block, bx * block + HUMAN_LIVER_BOX_SIZE],
                    "nucleus_fraction": float(window.mean()), "n_nuclei": float(n_nuclei),
                })
    candidates.sort(key=lambda c: (c["n_nuclei"], c["nucleus_fraction"]), reverse=True)
    return candidates[:n_best]


def tiff_strips(z):
    url = f"{EMPIAR_BASE}/humanliver_raw_images/humanliver_raw_{z:03d}.tif"

    def read_header():
        with tifffile.TiffFile(_HttpFile(url)) as tif:
            page = tif.pages[0]
            return url, page.dataoffsets, page.databytecounts, page.imagewidth, np.dtype(page.dtype)

    return with_retries(read_header)


def read_raw_rows(z, x0, x1, y0, y1):
    """Read rows y0:y1 of a raw slice with one range request over their strips (as the torch-em loader does)."""
    url, offsets, bytecounts, width, dtype = tiff_strips(z)
    start, end = offsets[y0], offsets[y1 - 1] + bytecounts[y1 - 1] - 1

    def get():
        r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=600)
        r.raise_for_status()
        if len(r.content) != end - start + 1:
            raise requests.RequestException("Incomplete range response.")
        return r

    data = with_retries(get).content
    rows = []
    for y in range(y0, y1):
        strip = data[offsets[y] - start:offsets[y] - start + bytecounts[y]]
        decoded = np.frombuffer(lzw_decode(strip), dtype=dtype).copy()
        rows.append(delta_decode(decoded, axis=-1, dist=1, out=decoded).reshape(width)[x0:x1])
    return np.stack(rows)


def download_box(name, bbox, output):
    """Download the raw and the nucleus mask of a box slice by slice, then write them to <output>/fullres.

    Every slice is saved on its own under a temporary name that is moved in place when complete. Thus, a failed or
    requeued job only downloads the slices that are missing.
    """
    fullres = os.path.join(output, "fullres")
    final_path = os.path.join(fullres, f"{name}.zarr")
    if os.path.exists(final_path):
        return final_path
    x0, x1, y0, y1, z0, z1 = bbox
    slice_dir = os.path.join(fullres, f"{name}_slices")
    os.makedirs(slice_dir, exist_ok=True)
    for z in range(z0, z1):
        slice_path = os.path.join(slice_dir, f"z{z:03d}.npz")
        if os.path.exists(slice_path):
            continue
        raw = read_raw_rows(z, x0, x1, y0, y1)
        nucleus = read_nucleus_mask(z)[y0:y1, x0:x1]
        tmp_path = f"{slice_path[:-4]}.{socket.gethostname()}.{os.getpid()}.npz"
        np.savez(tmp_path, raw=raw, nucleus=nucleus.astype("uint8"))
        os.replace(tmp_path, slice_path)
        print(f"{name}: slice {z} done", flush=True)

    tmp_path = f"{final_path[:-5]}.{socket.gethostname()}.{os.getpid()}.incomplete.zarr"
    group = zarr.open_group(tmp_path, mode="w")
    slices = [np.load(os.path.join(slice_dir, f"z{z:03d}.npz")) for z in range(z0, z1)]
    group.create_array("raw", data=np.stack([s["raw"] for s in slices]), chunks=(16, 512, 512))
    group.create_array("nucleus", data=np.stack([s["nucleus"] for s in slices]), chunks=(16, 512, 512))
    group.attrs["bounding_box"] = list(bbox)
    if os.path.exists(final_path):
        shutil.rmtree(tmp_path)
    else:
        os.replace(tmp_path, final_path)
    return final_path


def finalize_box(name, role, output):
    """Downsample a full resolution box 4x in xy and write the evaluation box with nucleus instances."""
    source = zarr.open_group(os.path.join(output, "fullres", f"{name}.zarr"), mode="r")
    bbox = list(source.attrs["bounding_box"])
    raw = source["raw"][:].astype("float32")
    nucleus = source["nucleus"][:] > 0
    nz, ny, nx = raw.shape
    scale = HUMAN_LIVER_SCALE
    raw = raw.reshape(nz, ny // scale, scale, nx // scale, scale).mean(axis=(2, 4))
    nucleus = nucleus.reshape(nz, ny // scale, scale, nx // scale, scale).mean(axis=(2, 4)) > 0.5
    lo, hi = np.percentile(raw, [0.1, 99.9])
    raw = (np.clip((raw - lo) / (hi - lo), 0, 1) * 255).astype("uint8")
    labels, n_nuclei = ndimage.label(nucleus)

    out_path = os.path.join(output, f"{name}.zarr")
    tmp_path = f"{out_path}.incomplete"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    group = zarr.open_group(tmp_path, mode="w")
    group.create_array("raw", data=raw, chunks=(16, 256, 256))
    group.create_array("labels", data=labels.astype("uint32"), chunks=(16, 256, 256))
    group.attrs.update({"bounding_box": bbox, "voxel_size_nm_zyx": [55, 32, 32], "role": role})
    # A rerun rebuilds the box: a directory cannot replace a non-empty one.
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.replace(tmp_path, out_path)
    return {
        "bounding_box_xxyyzz": bbox, "role": role, "shape": list(raw.shape), "n_nuclei": int(n_nuclei),
        "nucleus_fraction": float(nucleus.mean()), "zarr": f"{name}.zarr",
    }


def preprocess_human_liver(input_root, output_root, steps, names, map_step, depth, n_best):
    """Map, select, download and downsample the human liver EM nucleus evaluation boxes (EMPIAR-13356).

    The nucleus labels are the automated segmentation of the authors. They only cover the hepatocyte nuclei in the
    upper part of the tissue. Thus, the boxes are placed where nuclei are labeled. The steps:
    - 'map' maps the nucleus fraction of every map_step-th slice to <output>/nucleus_map.npz.
    - 'select' prints the 4096 x 4096 windows of depth slices with the most labeled nuclei in the map.
    - 'download' streams a box of HUMAN_LIVER_BOXES at full resolution slice by slice (resumable) into
      <output>/fullres.
    - 'finalize' downsamples the full resolution boxes 4x in xy and writes the evaluation boxes with nucleus instances.
    """
    output = os.path.join(output_root, HUMAN_LIVER_ROOT)
    os.makedirs(output, exist_ok=True)
    names = names or list(HUMAN_LIVER_BOXES)
    if "map" in steps:
        map_human_liver_nuclei(output, map_step)
    if "select" in steps:
        map_path = os.path.join(input_root, HUMAN_LIVER_ROOT, "nucleus_map.npz")
        for candidate in select_windows(map_path, depth, 10, n_best):
            print(candidate)
    if "download" in steps:
        for name in names:
            print(name, download_box(name, HUMAN_LIVER_BOXES[name]["bbox_xxyyzz"], output), flush=True)
    if "finalize" in steps:
        infos = {name: finalize_box(name, HUMAN_LIVER_BOXES[name]["role"], output) for name in names}
        for name, info in infos.items():
            print(name, info)
        with open(os.path.join(output, "boxes.json"), "w") as f:
            json.dump(infos, f, indent=2)


def import_evaluation_common():
    """Import common.py of the evaluation folder. Only the eval_crops choice needs it, and it loads torch."""
    sys.path.insert(0, EVALUATION_ROOT)
    import common
    return common


def dump_crops(obj, level=0):
    """JSON in the layout of eval_crops_3d.json: one space per level, number lists on one line."""
    pad, inner = " " * level, " " * (level + 1)
    if isinstance(obj, dict):
        items = [f"{inner}{json.dumps(k)}: {dump_crops(v, level + 1)}" for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(obj, list):
        if all(not isinstance(x, (list, dict)) for x in obj):
            return json.dumps(obj)
        return "[\n" + ",\n".join(inner + dump_crops(x, level + 1) for x in obj) + "\n" + pad + "]"
    return json.dumps(obj)


def tile_starts(region, crop_shape, min_tile_shape):
    axes = []
    for sl, size, min_size in zip(region, crop_shape, min_tile_shape):
        axes.append([s for s in range(sl.start, sl.stop, size) if min(size, sl.stop - s) >= min_size])
    return [(z, y, x) for z in axes[0] for y in axes[1] for x in axes[2]]


def count_tile(raw, labels, start, region, min_size, crop_shape, min_tile_shape):
    window = tuple(slice(s, min(s + c, r.stop)) for s, c, r in zip(start, crop_shape, region))
    planes = np.asarray(raw[window]).reshape(window[0].stop - window[0].start, -1)
    if (planes.min(axis=1) == planes.max(axis=1)).any():
        return None
    tile = np.asarray(labels[window]).astype("int64")
    tile[tile < 0] = 0
    labeled = np.nonzero(tile.reshape(tile.shape[0], -1).any(axis=1))[0]
    if len(labeled) == 0 or labeled[-1] - labeled[0] + 1 < min_tile_shape[0]:
        return None
    components = label(tile)
    sizes = np.bincount(components.ravel())[1:]
    return int((sizes >= min_size).sum())


def place_crops(common, input_root, dataset_name):
    raw_paths, label_paths, raw_key, label_key = common.get_data_paths(dataset_name, input_root, split="test")
    min_size = common.GT_MIN_SIZE_3D.get(dataset_name, 0)
    crop_shape, min_tile_shape = EVAL_TILE_SHAPES.get(dataset_name, (EVAL_CROP_SHAPE, EVAL_MIN_TILE_SHAPE))
    counts = {}
    for raw_path, label_path in zip(raw_paths, label_paths):
        raw, labels = open_file(raw_path, "r")[raw_key], open_file(label_path, "r")[label_key]
        roi = common.em_roi(dataset_name, label_path, "test")
        roi = tuple(slice(None) for _ in labels.shape) if roi is None else roi
        region = tuple(slice(*sl.indices(n)[:2]) for sl, n in zip(roi, labels.shape))
        counts[os.path.basename(raw_path)] = {
            start: count_tile(raw, labels, start, region, min_size, crop_shape, min_tile_shape)
            for start in tile_starts(region, crop_shape, min_tile_shape)
        }
    valid = [n for tiles in counts.values() for n in tiles.values() if n is not None]
    threshold = EVAL_MIN_RELATIVE_COUNT * float(np.median(valid))
    starts, kept, n_tiles = {}, [], 0
    for volume, tiles in counts.items():
        n_tiles += len(tiles)
        passing = sorted(
            ((n, start) for start, n in tiles.items() if n is not None and n >= threshold), key=lambda t: -t[0]
        )[:EVAL_MAX_TILES_PER_VOLUME]
        starts[volume] = [list(start) for _, start in sorted(passing, key=lambda t: t[1])]
        kept.extend(n for n, _ in passing)
    note = (
        f"{EVAL_NOTES[dataset_name]} {len(kept)} of {n_tiles} tiles are kept (at most {EVAL_MAX_TILES_PER_VOLUME} per "
        f"volume, the ones with the most objects), holding {min(kept)}-{max(kept)} "
        f"{EVAL_OBJECT_NAMES.get(dataset_name, 'nuclei')}."
    )
    print(f"{dataset_name}: {note}")
    return {"crop_shape": list(crop_shape), "note": note, "starts": starts}


def preprocess_eval_crops(input_root, crops_path, datasets, dry):
    """Place the fixed evaluation crops of the EM nucleus panel and add them to eval_crops_3d.json.

    The crops follow the rule of the file. They are 32 x 512 x 512 tiles, capped by the test region, that tile the test
    region without overlap. The annotation covers the test region. These tiles are dropped:
    - tiles thinner than 16 slices or 256 px in plane.
    - tiles with a constant raw plane.
    - tiles whose labels span fewer than 16 slices, because the evaluation trims a crop to its labeled slices.
    - tiles with fewer than 20% of the median object count of the dataset.

    Of the remaining tiles, at most 16 per volume stay, the ones with the most objects. Objects are the connected
    components of at least the ground-truth size floor, as the evaluation counts them. Tiles are read one at a time,
    so no volume is loaded whole.
    """
    common = import_evaluation_common()
    datasets = datasets or common.DATASETS_3D_EM_NUCLEUS
    placed = {name: place_crops(common, input_root, name) for name in datasets}
    # The file is read after the crops are placed, so edits made in the meantime are kept.
    with open(crops_path) as f:
        crops = json.load(f)
    crops["datasets"].update(placed)
    if not dry:
        tmp_path = f"{crops_path}.tmp{os.getpid()}"
        with open(tmp_path, "w") as f:
            f.write(dump_crops(crops) + "\n")
        os.replace(tmp_path, crops_path)


def main():
    choices = [
        "betaseg", "human_organoids", "tumor_spheroid", "waenet", "cellmap", *BUILDERS, "human_liver_eval", "eval_crops"
    ]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=choices, help="The dataset to preprocess.")
    parser.add_argument("--input_root", default=DATA_ROOT, help="The data root to read the source data from.")
    parser.add_argument("--output_root", default=DATA_ROOT, help="The data root to write the derived data to.")
    parser.add_argument("--margin", type=int, default=32, help="Margin around the BetaSeg nucleus box in voxels.")
    parser.add_argument("--volumes", nargs="+", default=list(ORGANOIDS_VOLUMES), help="The human organoids volumes.")
    steps = ["map", "select", "download", "finalize"]
    parser.add_argument("--steps", nargs="+", default=steps[2:], choices=steps, help="The human liver steps.")
    parser.add_argument("--names", nargs="+", default=None, help="The human liver boxes. Defaults to all.")
    parser.add_argument("--map_step", type=int, default=10, help="Map every n-th human liver slice.")
    parser.add_argument("--depth", type=int, default=32, help="The slices of a human liver window in 'select'.")
    parser.add_argument("--n_best", type=int, default=40, help="The human liver windows that 'select' prints.")
    parser.add_argument("--eval_datasets", nargs="+", default=None, help="The nucleus evaluation sets for eval_crops.")
    parser.add_argument("--crops_path", default=EVAL_CROPS_PATH, help="The eval_crops_3d.json file to update.")
    parser.add_argument("--dry", action="store_true", help="Print the evaluation crops without writing them.")
    args = parser.parse_args()
    if args.dataset == "betaseg":
        preprocess_betaseg(args.input_root, args.output_root, args.margin)
    elif args.dataset == "human_organoids":
        preprocess_human_organoids(args.input_root, args.output_root, args.volumes)
    elif args.dataset == "tumor_spheroid":
        preprocess_tumor_spheroid(args.input_root, args.output_root)
    elif args.dataset == "waenet":
        preprocess_waenet(args.input_root, args.output_root)
    elif args.dataset == "cellmap":
        preprocess_cellmap(args.input_root, args.output_root)
    elif args.dataset in BUILDERS:
        preprocess_new_dataset(args.dataset, args.input_root, args.output_root)
    elif args.dataset == "human_liver_eval":
        preprocess_human_liver(
            args.input_root, args.output_root, args.steps, args.names, args.map_step, args.depth, args.n_best
        )
    else:
        preprocess_eval_crops(args.input_root, args.crops_path, args.eval_datasets, args.dry)


if __name__ == "__main__":
    main()
