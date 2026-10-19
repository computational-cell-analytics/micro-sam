"""Survey figures of the new torch-em nucleus and cryo-ET loaders (2026-10-18) for the TBD folders.

Datasets: microsam_em_nuclei, vem_nuclei, etchacell, cefa_hela (multi-nuclei labels), chlamydomonas_cryoet and
deepict_pombe. The figures go to mito_nuc_figures/nuc/TBD/<dataset>.png, and for the datasets with mitochondria also
to mito_nuc_figures/mito/TBD/<dataset>.png. They show raw data with the labels as the loaders provide them.
"""
import os
import glob
import argparse

import h5py
import zarr
import tifffile
import numpy as np

import scipy.io
from scipy import ndimage

from skimage.measure import label
from skimage.segmentation import find_boundaries

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa

from render_nucleus_overview import draw_pair, normalize  # noqa


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"
FIGURE_ROOT = "/mnt/vast-kisski/home/archit/u28048/micro-sam/mito_nuc_figures"
OUTLINE_COLOR = (1.0, 0.15, 0.1, 1.0)


def out_path(model, name):
    folder = os.path.join(FIGURE_ROOT, model, "TBD")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{name}.png")


def save_grid(pairs, path, suptitle, n_cols=2):
    """pairs: (raw, labels, title, mode) with mode 'instance' (colored ids) or 'outline' (mask outline)."""
    n_rows = int(np.ceil(len(pairs) / n_cols))
    fig, axes = plt.subplots(n_rows, 2 * n_cols, figsize=(5 * 2 * n_cols, 5.2 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, (raw, labels, title, mode) in enumerate(pairs):
        row, col = divmod(i, n_cols)
        ax_pair = axes[row, 2 * col:2 * col + 2]
        if mode == "instance":
            draw_pair(ax_pair, raw, labels, title=title)
            continue
        stride = max(1, int(np.ceil(max(raw.shape) / 512)))
        r, m = normalize(raw[::stride, ::stride]), labels[::stride, ::stride] > 0
        overlay = np.zeros(m.shape + (4,), dtype="float32")
        overlay[m] = (1.0, 0.3, 0.1, 0.35)
        overlay[find_boundaries(m, mode="inner")] = OUTLINE_COLOR
        ax_pair[0].imshow(r, cmap="gray")
        ax_pair[1].imshow(r, cmap="gray")
        ax_pair[1].imshow(overlay)
        ax_pair[0].set_title(title, fontsize=7, loc="left")
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.9 / fig.get_figheight()))
    fig.savefig(path, dpi=80)
    plt.close(fig)
    print("wrote", path)


def densest_window(labels, size):
    """The (y, x) window of side size with the most labeled pixels, on a coarse grid."""
    stride = max(1, size // 4)
    best, best_yx = -1, (0, 0)
    for y in range(0, max(1, labels.shape[0] - size + 1), stride):
        for x in range(0, max(1, labels.shape[1] - size + 1), stride):
            n = int((labels[y:y + size:8, x:x + size:8] > 0).sum())
            if n > best:
                best, best_yx = n, (y, x)
    y, x = best_yx
    return np.s_[y:y + size, x:x + size]


def figure_microsam_em_nuclei():
    root = os.path.join(DATA_ROOT, "microsam_em_nuclei", "data")
    with h5py.File(os.path.join(root, "train_data.h5"), "r") as f:
        raw, labels = f["raw"][:], f["labels/nuclei"][:]
    n_nuclei = len(np.unique(labels)) - 1
    pairs = [(raw[z], labels[z], f"train crop z={z}, {n_nuclei} nuclei (manual)", "instance") for z in (0, 16, 31)]
    annotation = tifffile.imread(os.path.join(root, "annotation_volume.tif"))
    for method in ("ilastik_carving", "default_sam", "finetuned_sam"):
        lab = tifffile.imread(os.path.join(root, "annotations", f"annotations_{method}.tif"))
        title = f"user-study crop z=16, {method}: {len(np.unique(lab)) - 1} nuclei"
        pairs.append((annotation[16], lab[16], title, "instance"))
    save_grid(pairs, out_path("nuc", "microsam_em_nuclei"), (
        "microsam_em_nuclei (Zenodo 14037020): two 32 x 512 x 512 SBEM crops of a Platynereis-like larva (dense small "
        "nuclei), pixel size not stored\n'train' crop: manual nucleus instances (48); 'annotation' crop: user-study "
        "annotations from ilastik carving / default SAM / finetuned SAM (36 / 45 / 60 nuclei, no reference GT)"
    ))


def figure_vem_nuclei():
    root = os.path.join(DATA_ROOT, "vem_nuclei")
    pairs = []
    for sample_id in (1, 2, 3, 4, 5, 6, 7, 8, 10):
        folder = os.path.join(root, f"id_{sample_id}")
        img_path = sorted(glob.glob(os.path.join(folder, "img_*")))[0]
        lbl_path = sorted(glob.glob(os.path.join(folder, "lbl_*")))[0]
        lab = tifffile.imread(lbl_path)
        window = densest_window(lab, 4096)
        raw = tifffile.imread(img_path)[window]
        lab_w = lab[window]
        n_total = len(np.unique(lab)) - 1
        edge = "border nuclei kept" if sample_id in (8, 10) else "border nuclei removed"
        title = (
            f"2D AT-SEM id_{sample_id}: {lab.shape}, {n_total} nuclei in image ({edge}); "
            f"densest 4096 px window, {len(np.unique(lab_w)) - 1} nuclei"
        )
        pairs.append((raw, lab_w, title, "instance"))
        del lab
    for sample_id in (11, 12):
        folder = os.path.join(root, f"id_{sample_id}")
        raw = tifffile.imread(sorted(glob.glob(os.path.join(folder, "img_*")))[0])
        lab = tifffile.imread(sorted(glob.glob(os.path.join(folder, "lbl_*")))[0])
        z = int(np.argmax((lab > 0).reshape(lab.shape[0], -1).sum(axis=1)))
        y = int(np.argmax((lab > 0).sum(axis=(0, 2))))
        title = f"3D FIB-SEM id_{sample_id} {lab.shape}: xy z={z}, {len(np.unique(lab)) - 1} labeled nucleus"
        pairs.append((raw[z], lab[z], title, "instance"))
        pairs.append((raw[:, y], lab[:, y], f"3D FIB-SEM id_{sample_id}: zx y={y}", "instance"))
    save_grid(pairs, out_path("nuc", "vem_nuclei"), (
        "vem_nuclei (S-BIAD2822): 9 2D array-tomography SEM images (up to 7448 x 49530 px) and 2 3D FIB-SEM volumes of "
        "cancer cells / tumor tissue, pixel size not stored\nlabels: NucleoNet predictions + manual proofreading "
        "(2D: all nuclei, border nuclei removed in most images; 3D: one isolated cell per volume, background masked)"
    ))


def figure_etchacell():
    root = os.path.join(DATA_ROOT, "etchacell")
    pairs = []
    for raw_path in sorted(glob.glob(os.path.join(root, "raw", "*.tiff"))):
        roi = os.path.basename(raw_path)[:-5]
        lab = tifffile.imread(os.path.join(root, "labels", f"{roi}.tiff"))
        zs = np.where((lab > 0).reshape(lab.shape[0], -1).any(axis=1))[0]
        for z in np.linspace(zs.min(), zs.max(), 4).round().astype(int)[1:3]:
            raw = tifffile.imread(raw_path, key=int(z))
            title = f"{roi} z={z} {lab.shape}: nuclear envelope (consensus, labeled z {zs.min()}-{zs.max()})"
            pairs.append((raw, lab[z], title, "outline"))
    save_grid(pairs, out_path("nuc", "etchacell"), (
        "etchacell (S-BSST448 labels, EMPIAR-10478 raw): HeLa SBF-SEM, 10 x 10 x 50 nm, 18 ROIs of up to 300 x 2000 x "
        "2000 (3 ROIs downloaded)\nlabels: binary NUCLEAR ENVELOPE (thin band, not a filled nucleus) of the central "
        "cell, citizen-science consensus; neighbour nuclei unlabeled; EMPIAR-10478 = cropped ROIs of a HeLa pellet "
        "benchmark, likely the EMPIAR-10094 acquisition (cefa_hela), not pixel-confirmed"
    ))


def figure_cefa_hela():
    root = os.path.join(DATA_ROOT, "cefa_hela")
    origin = (5050, 5580)
    pairs = []
    for raw_path in sorted(glob.glob(os.path.join(root, "empiar_crops_y5050_x5580", "slice_*.h5"))):
        z = int(os.path.basename(raw_path)[6:10])
        with h5py.File(raw_path, "r") as f:
            raw = f["raw"][:]
        multi = scipy.io.loadmat(os.path.join(root, "gt_labels_multinuclei", f"GT_Slice_{z:03d}.mat"))["groundTruth"]
        nuclei = label(multi == 2)
        title = (
            f"GT slice {z} at the template-matched origin y={origin[0]}, x={origin[1]}: multi-nuclei "
            f"'nucleus' class, {nuclei.max()} 2D components"
        )
        pairs.append((raw, nuclei[:raw.shape[0], :raw.shape[1]], title, "instance"))
    center_path = os.path.join(root, "empiar_center_crops", "slice_0128.h5")
    with h5py.File(center_path, "r") as f:
        raw = f["raw"][:]
    multi = scipy.io.loadmat(os.path.join(root, "gt_labels_multinuclei", "GT_Slice_128.mat"))["groundTruth"]
    title = "GT slice 128 on the loader's center crop (still used by the loader): misaligned"
    pairs.append((raw, label(multi == 2)[:raw.shape[0], :raw.shape[1]], title, "instance"))
    save_grid(pairs, out_path("nuc", "cefa_hela"), (
        "cefa_hela, new 'multinuclei' labels (Zenodo 6355622): HeLa SBF-SEM (EMPIAR-10094), 10 x 10 x 50 nm, 300 GT "
        "slices of 2000 x 2000\n4 semantic classes (envelope, nucleus, cell, background), several nuclei per slice; "
        "the loader still center-crops the raw (labels sit at y~5050, x~5580) and the Zenodo files-archive URL "
        "returns 404"
    ))


def cryoet_pairs(group, classes, zs):
    pairs = []
    raw = group["raw"]
    for name in classes:
        mask = group[name]
        counts = [(int((mask[z] > 0).sum()), z) for z in range(0, mask.shape[0], 8)]
        z_best = max(counts)[1] if max(counts)[0] > 0 else mask.shape[0] // 2
        for z in sorted({z_best, *zs}):
            image = raw[z].astype("float32")
            image = ndimage.uniform_filter(image, 3)
            title = f"{name} z={z}, {int((mask[z] > 0).sum())} px"
            pairs.append((image, (mask[z] > 0).astype("uint8"), title, "outline"))
    return pairs


def figure_chlamydomonas_cryoet():
    path = glob.glob(os.path.join(DATA_ROOT, "chlamydomonas_cryoet", "*.zarr"))[0]
    group = zarr.open_group(path, mode="r")
    run = os.path.basename(path)[:-5]
    note = (
        "chlamydomonas_cryoet (CryoET Data Portal 10301), run {run}: cryo-ET, 7.84 A voxels, 512 x 1024 x 1024\n"
        "{what} masks are automatic Ais / Pom predictions (not proofread); cryo-ET is out of scope"
    )
    save_grid(
        cryoet_pairs(group, ["nucleoplasm"], [256]), out_path("nuc", "chlamydomonas_cryoet"),
        note.format(run=run, what="NUCLEOPLASM (a tiny fragment in this run)"),
    )
    save_grid(
        cryoet_pairs(group, ["mitochondrion"], [200, 300]), out_path("mito", "chlamydomonas_cryoet"),
        note.format(run=run, what="mitochondrion"),
    )


def figure_deepict_pombe():
    group = zarr.open_group(os.path.join(DATA_ROOT, "deepict_pombe", "TS_045.zarr"), mode="r")
    note = (
        "deepict_pombe (CryoET Data Portal 10000), run TS_045: cryo-ET of S. pombe, 13.48 A voxels, 500 x 928 x "
        "960\nexpert-verified {what} masks; TS_045 is byte-identical to the pombe_nucleus_mito / popsicle copy "
        "(already discarded as cryo-ET); cryo-ET is out of scope"
    )
    save_grid(
        cryoet_pairs(group, ["nucleus", "nuclear_envelope"], [250]), out_path("nuc", "deepict_pombe"),
        note.format(what="nucleus and NUCLEAR ENVELOPE"),
    )
    save_grid(
        cryoet_pairs(group, ["mitochondrion"], [150, 350]), out_path("mito", "deepict_pombe"),
        note.format(what="mitochondrion"),
    )


FIGURES = {
    "microsam_em_nuclei": figure_microsam_em_nuclei, "vem_nuclei": figure_vem_nuclei, "etchacell": figure_etchacell,
    "cefa_hela": figure_cefa_hela, "chlamydomonas_cryoet": figure_chlamydomonas_cryoet,
    "deepict_pombe": figure_deepict_pombe,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+", default=list(FIGURES), choices=list(FIGURES))
    args = parser.parse_args()
    for name in args.names:
        FIGURES[name]()


if __name__ == "__main__":
    main()
