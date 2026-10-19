"""Overview figures of the nucleus EM generalist data, as the pipeline sees it, split by decision.

- training: samples drawn from the em_nucleus training and validation datasets of the generalist loader
  (_build_joint_datasets with z_slices=[8]). Each sample shows the raw (middle z) and the instance target with the
  ignored pixels shaded.
- in_domain_evaluation / ood_evaluation: slices of the evaluation volumes with their ground truth instances.
- discarded: representative slices of the source data with the reason for the discard.

The figures are written to mito_nuc_figures/nuc/<category>/<dataset>.png.
"""
import os
import glob
import json
import argparse
from types import SimpleNamespace

import h5py
import zarr
import numpy as np
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa

from figure_utils import DATA_ROOT  # noqa


OUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nuc")
CATEGORIES = ("training", "in_domain_evaluation", "ood_evaluation", "discarded")
IGNORE_COLOR = (0.15, 0.15, 0.15, 0.75)
MAX_DISPLAY = 512
N_TRAIN_SAMPLES = 6
N_VAL_SAMPLES = 2
# tab20 without its two greys, which look like unlabeled tissue.
PALETTE = [plt.get_cmap("tab20")(i) for i in range(20) if i not in (14, 15)]

# Name of the figure, the em_nucleus dataset names it covers, the voxel size and the rationale (from DECISIONS.md).
TRAINING = {
    "betaseg": (
        ("betaseg_nucleus",), "4 nm isotropic FIB-SEM",
        "nucleus-centered boxes, loss ignored outside the target cell (unlabeled neighbour nuclei)",
    ),
    "cellmap": (
        ("cellmap_nucleus",), "32-64 nm FIB-SEM", "only the 14 large tissue crops with many complete nuclei, manual",
    ),
    "human_organoids": (
        ("human_organoids_nucleus",), "bin2 FIB-SEM (voxel size not stored)",
        "speckled masks closed, holes filled, labels on black padding removed, fragments + entotic cell ignored",
    ),
    "waenet_fibsem": (
        ("waenet_fibsem_nucleus",), "FIB-SEM (voxel size not stored)", "WAE-Net dataset 1, all nuclei labeled, 3D",
    ),
    "waenet_sstem": (
        ("waenet_sstem_nucleus",), "serial ssTEM (voxel size not stored)",
        "WAE-Net dataset 3, unregistered sections of one interphase cell, trained as 2D",
    ),
    "waenet_tem": (
        ("waenet_tem_nucleus",), "TEM (voxel size not stored)",
        "WAE-Net datasets 4-7, only the central cell labeled, loss ignored outside the central cell band",
    ),
    "sxt": (
        ("sxt_nucleus",), "36.8-45.5 nm soft X-ray tomography (not EM)",
        "one labeled cell per tomogram, loss ignored outside the cell mask",
    ),
    "fafb_nuclei": (
        ("fafb_nuclei",), "32 x 32 x 40 nm ssTEM", "automatic FlyWire segmentation, clean and near-exhaustive",
    ),
    "janelia_nucleus": (
        ("janelia_nucleus",), "32-128 nm FIB-SEM", "13 Cellpose pseudo-label volumes + 3 manual NAcc volumes",
    ),
    "microns_nuclei": (
        ("microns_nuclei",), "64 x 64 x 40 nm ssTEM", "automatic minnie65 segmentation, clean and near-exhaustive",
    ),
    "openorganelle_nucleus": (
        ("openorganelle_nucleus",), "32-128 nm FIB-SEM",
        "automatic labels; tissue volumes as full arrays + single-cell volumes",
    ),
    "platynereis": (
        ("platynereis_nuclei",), "about 80 x 80 x 100 nm SBEM", "manual, valid ROI only (removes the unlabeled halo)",
    ),
    "tumor_spheroid": (
        ("tumor_spheroid_nuclei",), "50 nm SBF-SEM (2D slices)", "20 manual 2D slices, inverted contrast",
    ),
    "cardiomyocyte_tem": (
        ("cardiomyocyte_tem_nucleus",), "TEM at 3500x (pixel size not stored), 4x downscaled",
        "9 micrographs, one manual nucleus each (nucleoli filled in), 2 seeded images validate",
    ),
    "microsam_em_nuclei": (
        ("microsam_em_nuclei",), "EM (voxel size not stored)",
        "user-study 'train' crop only, all of it trains; the 'annotation' crop has no reference labels",
    ),
    "perez": (
        ("perez_nucleus",), "SBF-SEM mouse SCN tiles (2D)", "nucleus task only, train tiles train, test tiles validate",
    ),
    "vem_nuclei": (
        ("vem_nuclei",), "array tomography SEM (2D, downscaled) + FIB-SEM (3D, 2x downscaled)",
        "proofread NucleoNet labels, border band without labels ignored; one 2D image + an id_12 z-slab validate",
    ),
    "aive": (
        ("aive_nucleus",), "6.5 x 6.5 x 10 nm FIB-SEM downscaled to 26 x 26 x 30 nm",
        "nucleus class with the nucleoli ('chromatin' class) folded in, the last 40 slices of dataset 3 validate",
    ),
    "macrophage_a431": (
        ("macrophage_a431_nucleus",), "SBF-SEM, 4x downscaled in-plane",
        "nucleus class of the complete SBF-SEM volumes (quick MIB labels), z >= 115 of sbf_a431 validates",
    ),
    "cerebellum_fibsem": (
        ("cerebellum_nucleus",), "10 nm FIB-SEM downscaled to 20 nm isotropic",
        "student labels, 3 small pieces in an axon ignored, z < 200 trains, z >= 200 validates (tail of a nucleus)",
    ),
}


def normalize(image):
    image = image.astype("float32")
    lo, hi = np.percentile(image, (1, 99))
    return np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1)


def display_stride(shape):
    return max(1, int(np.ceil(max(shape) / MAX_DISPLAY)))


def instance_colors(labels):
    """RGBA overlay with a fixed color per instance id."""
    overlay = np.zeros(labels.shape + (4,), dtype="float32")
    ids = np.unique(labels)
    for label_id in ids[ids > 0]:
        color = PALETTE[(int(label_id) * 7) % len(PALETTE)]
        overlay[labels == label_id] = (color[0], color[1], color[2], 0.45)
    return overlay


def draw_pair(axes, raw, labels, ignore=None, title=""):
    """Draw the raw slice and the raw slice with the instance overlay (and the ignored pixels shaded)."""
    stride = display_stride(raw.shape)
    raw, labels = raw[::stride, ::stride], labels[::stride, ::stride]
    raw = normalize(raw)
    axes[0].imshow(raw, cmap="gray")
    axes[1].imshow(raw, cmap="gray")
    overlay = instance_colors(labels)
    if ignore is not None:
        overlay[ignore[::stride, ::stride]] = IGNORE_COLOR
    axes[1].imshow(overlay)
    axes[0].set_title(title, fontsize=7, loc="left")
    for ax in axes:
        ax.axis("off")


def text_figure(out_path, title, text):
    fig = plt.figure(figsize=(12, 3))
    fig.text(0.02, 0.6, title, fontsize=13, weight="bold")
    fig.text(0.02, 0.3, text, fontsize=10, wrap=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def save_grid(pairs, out_path, suptitle, n_pairs_per_row=2):
    """Save a grid of (raw, overlay) panel pairs; each pair is (raw, labels, ignore, title)."""
    n_rows = int(np.ceil(len(pairs) / n_pairs_per_row))
    n_cols = 2 * n_pairs_per_row
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 5.0 * n_rows + 1.2), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, (raw, labels, ignore, title) in enumerate(pairs):
        row, col = divmod(k, n_pairs_per_row)
        draw_pair(axes[row, 2 * col:2 * col + 2], raw, labels, ignore, title)
    fig.suptitle(suptitle, fontsize=10, y=1 - 0.1 / fig.get_figheight(), va="top")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.9 / fig.get_figheight()), h_pad=2.0)
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    print("wrote", out_path)


class LeafRecorder:
    """Records the leaf dataset of the last drawn sample, to show where each sample comes from."""

    def __init__(self):
        self.last = None
        self.patched = set()

    def patch(self, leaves):
        for cls in {type(leaf) for leaf in leaves}:
            if cls in self.patched:
                continue
            original = cls.__getitem__
            recorder = self

            def getitem(self, index, original=original):
                recorder.last = self
                return original(self, index)

            cls.__getitem__ = getitem
            self.patched.add(cls)


def leaf_source(leaf):
    for attr in ("raw_path", "raw_image_paths"):
        value = getattr(leaf, attr, None)
        if value is not None:
            return os.path.basename(value if isinstance(value, str) else value[0])
    return "?"


def wrapper_side(wrapper):
    shape = getattr(wrapper.leaves[0], "patch_shape", None)
    return None if shape is None else int(shape[-1])


def draw_samples(wrappers, n, recorder, rng):
    """Draw n samples with the sampling weights of the training loader (proportional to the wrapper lengths)."""
    lengths = np.array([len(w) for w in wrappers], dtype="float64")
    samples = []
    for _ in range(n):
        k = rng.choice(len(wrappers), p=lengths / lengths.sum())
        wrapper = wrappers[k]
        raw, label = wrapper[int(rng.integers(len(wrapper)))]
        samples.append((raw.numpy(), label.numpy(), wrapper_side(wrapper), leaf_source(recorder.last)))
    return samples


def group_summary(wrappers):
    total = sum(len(w) for w in wrappers)
    parts = []
    for w in wrappers:
        side = wrapper_side(w)
        parts.append(f"side {side}: {len(w.leaves)} file(s), {100 * len(w) / max(total, 1):.0f}%")
    return "; ".join(parts)


def render_training(name, rng, organoid_volumes=None):
    from micro_sam.v2.datasets import generalist_loader

    dataset_names, voxel, rationale = TRAINING[name]
    generalist_loader.EM_NUCLEUS_DATASETS = dataset_names
    if organoid_volumes is not None:
        # Only these human organoids volumes (in this process), e.g. while the labels of another one are built.
        generalist_loader.HUMAN_ORGANOIDS_VOLUMES = [f"{volume}_bin2.h5" for volume in organoid_volumes]
        rationale = f"{rationale}\nonly the volumes {', '.join(organoid_volumes)} (labels of the others pending)"
    # A dataset without a validation leaf (microsam_em_nuclei) gives an empty validation list on its own.
    concat = generalist_loader.ConcatDataset
    generalist_loader.ConcatDataset = lambda *ds: concat(*ds) if ds else SimpleNamespace(datasets=[])
    try:
        train_ds, val_ds = generalist_loader._build_joint_datasets(
            DATA_ROOT, z_slices=[8], dataset_choice="em_nucleus"
        )
    finally:
        generalist_loader.ConcatDataset = concat
    recorder = LeafRecorder()
    pairs = []
    summaries = {}
    for split, ds, n in (("train", train_ds, N_TRAIN_SAMPLES), ("val", val_ds, N_VAL_SAMPLES)):
        wrappers = list(ds.datasets)
        if not wrappers:
            summaries[split] = "none"
            continue
        for w in wrappers:
            recorder.patch(w.leaves)
        summaries[split] = group_summary(wrappers)
        for raw, label, side, source in draw_samples(wrappers, n, recorder, rng):
            z = raw.shape[1] // 2
            instances, foreground = label[0, z], label[1, z]
            n_objects = len(np.unique(label[0][label[0] > 0]))
            ignored = foreground == -1
            resize = f"{side}->512" if side is not None and side < 512 else f"{side}"
            title = (
                f"{split} | patch side {resize} | z {z}/{raw.shape[1]} | {n_objects} obj in patch\n"
                f"{source} | ignored {100 * ignored.mean():.0f}%"
            )
            pairs.append((raw[0, z], instances.astype("int64"), ignored, title))
    suptitle = (
        f"nucleus / training / {name} ({', '.join(dataset_names)}) - samples as drawn by the em_nucleus loader, "
        f"{voxel}\n{rationale}\ntrain groups: {summaries['train']}\nval groups: {summaries['val']}"
    )
    save_grid(pairs, os.path.join(OUT_ROOT, "training", f"{name}.png"), suptitle)


def pick_slices(labels, n=2):
    """z indices with the most labeled pixels in the first and the second half of the volume."""
    counts = (labels > 0).reshape(labels.shape[0], -1).sum(axis=1)
    halves = np.array_split(np.arange(labels.shape[0]), n)
    return [int(h[np.argmax(counts[h])]) for h in halves if len(h)]


def volume_pairs(volume_name, raw, labels, n_total=None):
    pairs = []
    n_total = len(np.unique(labels[labels > 0])) if n_total is None else n_total
    for z in pick_slices(labels):
        lab = np.asarray(labels[z])
        n_slice = len(np.unique(lab[lab > 0]))
        title = f"{volume_name} z={z} | {n_slice} nuclei in slice, {n_total} in volume"
        pairs.append((np.asarray(raw[z]), lab.astype("int64"), None, title))
    return pairs


def render_cellmap_test():
    root = os.path.join(DATA_ROOT, "cellmap_nucleus_crops")
    with open(os.path.join(root, "index.json")) as f:
        crops = [c for c in json.load(f) if c["split"] == "test"]
    pairs = []
    for crop in crops:
        store = zarr.open_group(os.path.join(root, f"{crop['name']}.zarr"), mode="r")
        raw, labels = store["raw"], store["labels"]
        # The crop is 1500^3: the slices are chosen on a coarse z grid of the labels.
        z_grid = np.arange(0, labels.shape[0], 25)
        coarse = labels[::25, ::8, ::8]
        counts = (coarse > 0).reshape(len(z_grid), -1).sum(axis=1)
        for half in np.array_split(np.arange(len(z_grid)), 2):
            z = int(z_grid[half[np.argmax(counts[half])]])
            lab = labels[z]
            n_slice = len(np.unique(lab[lab > 0]))
            title = f"{crop['name']} z={z} | {n_slice} nuclei in slice, {crop['n_ids']} in crop"
            pairs.append((raw[z], lab.astype("int64"), None, title))
    suptitle = (
        "nucleus / in-domain evaluation / cellmap - held-out jrc_mus-liver-zon-2 crop357 (32 nm FIB-SEM, manual), "
        "the only large crop without overlap with another crop"
    )
    save_grid(pairs, os.path.join(OUT_ROOT, "in_domain_evaluation", "cellmap.png"), suptitle)


def render_block_test(name, folder, voxel, note):
    pairs = []
    for path in sorted(glob.glob(os.path.join(DATA_ROOT, folder, "test", "*.h5"))):
        with h5py.File(path, "r") as f:
            raw, labels = f["raw"][:], f["labels/nuclei"][:]
        pairs.extend(volume_pairs(os.path.basename(path), raw, labels))
    suptitle = f"nucleus / in-domain evaluation / {name} - official test blocks ({voxel}), {note}"
    save_grid(pairs, os.path.join(OUT_ROOT, "in_domain_evaluation", f"{name}.png"), suptitle)


def render_platynereis_test():
    path = os.path.join(DATA_ROOT, "platynereis", "nuclei", "train_data_nuclei_03.h5")
    with h5py.File(path, "r") as f:
        raw, labels = f["volumes/raw"][:], f["volumes/labels/nucleus_instance_labels"][:]
    valid = np.where(labels != -1)
    roi = tuple(slice(int(c.min()), int(c.max()) + 1) for c in valid)
    pairs = volume_pairs(f"train_data_nuclei_03 valid ROI {[(s.start, s.stop) for s in roi]}", raw[roi], labels[roi])
    zx_raw, zx_labels = raw[roi].transpose(1, 0, 2), labels[roi].transpose(1, 0, 2)
    pairs += volume_pairs("train_data_nuclei_03 (zx view of the ROI)", zx_raw, zx_labels)
    suptitle = (
        "nucleus / in-domain evaluation / platynereis - volume 03 (dense brain region, manual labels, about "
        "80 x 80 x 100 nm SBEM), cropped to its valid ROI"
    )
    save_grid(pairs, os.path.join(OUT_ROOT, "in_domain_evaluation", "platynereis.png"), suptitle)


def render_human_liver_ood():
    root = os.path.join(DATA_ROOT, "human_liver_em_nucleus_boxes")
    out_path = os.path.join(OUT_ROOT, "ood_evaluation", "human_liver_em.png")
    paths = sorted(glob.glob(os.path.join(root, "*.zarr")))
    paths = [p for p in paths if not p.endswith(".incomplete")]
    if not paths:
        text_figure(
            out_path, "nucleus / ood evaluation / human_liver_em - pending",
            "The evaluation boxes (hepatocyte region, 4x downsampled, ~32 x 32 x 55 nm, automated labels) are being "
            "built in /mnt/vast-nhr/projects/cidas/cca/data/human_liver_em_nucleus_boxes.",
        )
        print("wrote", out_path, "(pending)")
        return False
    pairs = []
    for path in paths:
        store = zarr.open_group(path, mode="r")
        role = store.attrs.get("role", "?")
        pairs.extend(volume_pairs(f"{os.path.basename(path)} ({role})", store["raw"][:], store["labels"][:]))
    suptitle = (
        "nucleus / out-of-domain evaluation / human_liver_em - SBF-SEM human liver, authors' automated nucleus "
        "segmentation, boxes inside the hepatocyte region checked for complete labels, 4x downsampled"
    )
    save_grid(pairs, out_path, suptitle)
    return True


def render_waenet_ds2():
    paths = sorted(glob.glob(os.path.join(DATA_ROOT, "waenet", "dataset_2", "*.h5")))
    pairs = []
    for path in paths[::max(1, len(paths) // 6)][:6]:
        with h5py.File(path, "r") as f:
            raw, chromosomes = f["raw"][:], f["labels/chromosomes"][:] > 0
        labels, n = ndimage.label(chromosomes)
        pairs.append((raw, labels, None, f"{os.path.basename(path)} | chromosome labels ({n} pieces)"))
    suptitle = (
        "nucleus / discarded / waenet dataset 2 - ssTEM of a mitotic BON cell: only condensed chromosomes are "
        "labeled, no nucleus (the loader docstring swaps datasets 2 and 3)"
    )
    save_grid(pairs, os.path.join(OUT_ROOT, "discarded", "waenet_dataset_2.png"), suptitle)


def render_cefa_hela():
    import scipy.io

    root = os.path.join(DATA_ROOT, "cefa_hela")
    pairs = []
    for path in sorted(glob.glob(os.path.join(root, "empiar_crops_y5050_x5580", "*.h5"))):
        z = int(os.path.basename(path)[6:10])
        with h5py.File(path, "r") as f:
            raw = f["raw"][:]
        gt = scipy.io.loadmat(os.path.join(root, "gt_labels", f"GT_Slice_{z + 1:03d}.mat"))["groundTruth"]
        nucleus = np.isin(gt, (1, 2)).astype("int64")
        title = f"z={z} at the template-matched origin | nucleus + envelope, grey: other cells"
        pairs.append((raw, nucleus, gt == 3, title))
    suptitle = (
        "nucleus / discarded / cefa_hela - one labeled nucleus (central cell); the torch-em loader center-crops "
        "the wrong region, the slice registration is unverified, the Zenodo archive link returns 404"
    )
    save_grid(pairs, os.path.join(OUT_ROOT, "discarded", "cefa_hela.png"), suptitle)


def render_nuc_mm():
    zebrafish = os.path.join(DATA_ROOT, "nuc_mm", "Zebrafish (NucMM-Z)")
    pairs = []
    for label_path in sorted(glob.glob(os.path.join(zebrafish, "Label", "train", "*.h5")))[:4]:
        image_path = os.path.join(zebrafish, "Image", "train", os.path.basename(label_path).replace("seg_", "img_"))
        with h5py.File(image_path, "r") as f, h5py.File(label_path, "r") as g:
            raw, labels = f["main"][:], g["main"][:]
        n = len(np.unique(labels[labels > 0]))
        pairs.append((raw[32], labels[32].astype("int64"), None, f"{os.path.basename(label_path)} z=32 | {n} nuclei"))
    suptitle = (
        "nucleus / discarded / nuc_mm zebrafish - ssEM at ~0.5 um, 64^3 volumes, nuclei 8-10 px wide (8x upscaling to "
        "the 512 input), only 7 train label files local (Google Drive refused the other 47)"
    )
    save_grid(pairs, os.path.join(OUT_ROOT, "discarded", "nuc_mm_zebrafish.png"), suptitle)

    pairs = []
    for path in sorted(glob.glob(os.path.join(DATA_ROOT, "nuc_mm", "mouse", "train", "*.h5")))[:4]:
        with h5py.File(path, "r") as f:
            raw, labels = f["raw"][:], f["labels"][:]
        z = raw.shape[0] // 2
        n = len(np.unique(labels[labels > 0]))
        pairs.append((raw[z], labels[z].astype("int64"), None, f"{os.path.basename(path)} z={z} | {n} nuclei"))
    suptitle = "nucleus / discarded / nuc_mm mouse - X-ray micro-CT of the mouse cortex at 0.72 um, not EM"
    save_grid(pairs, os.path.join(OUT_ROOT, "discarded", "nuc_mm_mouse.png"), suptitle)


def render_openorganelle_discarded():
    reasons = {
        "jrc_ctl-id8-2": "envelope band, not a nucleus mask",
        "jrc_mus-liver-4": "nuclear membrane traces only",
        "jrc_mus-liver-5": "nuclear membrane traces only",
        "jrc_mus-liver-6": "nuclear membrane traces only",
        "jrc_mus-epididymis-1": "binary, thin sperm nuclei partly covered",
        "jrc_mus-epididymis-2": "binary, thin sperm nuclei partly covered",
        "jrc_dauer-larva": "ragged TEM tiles, resampled raw",
        "jrc_mosquito-stylet-6": "one nucleus in a 40 GB volume",
    }
    pairs = []
    for name, reason in reasons.items():
        store = zarr.open_group(os.path.join(DATA_ROOT, "openorganelle_nucleus", f"{name}.zarr"), mode="r")
        raw, labels = store["raw"][:], store["labels"][:]
        labels = ndimage.label(labels > 0)[0] if labels.max() == 1 else labels
        z = pick_slices(labels, 1)[0]
        pairs.append((raw[z], labels[z].astype("int64"), None, f"{name} (loader crop) z={z} | {reason}"))
    suptitle = (
        "nucleus / discarded / openorganelle_nucleus - discarded volumes (automatic labels under labels/inference)"
    )
    save_grid(pairs, os.path.join(OUT_ROOT, "discarded", "openorganelle_nucleus.png"), suptitle)


def cryo_slice(raw, z, half_window=4):
    """Mean of 9 slices with a gaussian filter, for visibility of the noisy tomograms."""
    window = np.asarray(raw[max(0, z - half_window):z + half_window + 1], dtype="float32").mean(axis=0)
    return ndimage.gaussian_filter(window, 2)


def render_cryo_et():
    root = os.path.join(DATA_ROOT, "popsicle_yeast", "yeast_organelles")
    pairs = []
    for split, run in (("train", "TS_027"), ("train", "TS_0009"), ("test", "TS_037")):
        raw = zarr.open(os.path.join(root, split, run, "raw.zarr"), mode="r")
        labels = zarr.open(os.path.join(root, split, run, "labels.zarr"), mode="r")["0"]
        nucleus = np.asarray(labels[::8]) == 2
        z = int(np.argmax(nucleus.reshape(nucleus.shape[0], -1).sum(axis=1))) * 8
        raw_array = raw["0"] if isinstance(raw, zarr.Group) else raw
        nucleus_slice = (np.asarray(labels[z]) == 2).astype("int64")
        pairs.append((cryo_slice(raw_array, z), nucleus_slice, None, f"{split}/{run} z={z} | nucleus class (2)"))
    suptitle = (
        "nucleus / discarded / popsicle - cryo-ET of S. pombe at 1.348 nm (out of scope for now), classes annotated "
        "per tomogram only (raw: 9-slice mean + gaussian)"
    )
    save_grid(pairs, os.path.join(OUT_ROOT, "discarded", "popsicle.png"), suptitle)

    pairs = []
    for run in ("10001_TS_0006", "10000_TS_045"):
        store = zarr.open_group(os.path.join(DATA_ROOT, "pombe_nucleus_mito", f"{run}.zarr"), mode="r")
        nucleus = store["nucleus"]
        coarse = np.asarray(nucleus[::8]) > 0
        z = int(np.argmax(coarse.reshape(coarse.shape[0], -1).sum(axis=1))) * 8
        nucleus_slice = (np.asarray(nucleus[z]) > 0).astype("int64")
        pairs.append((cryo_slice(store["raw"], z), nucleus_slice, None, f"{run} z={z} | nucleus"))
    suptitle = (
        "nucleus / discarded / pombe_nucleus_mito - cryo-ET (out of scope for now), masks byte-identical to popsicle "
        "TS_0006 / TS_045 (raw: 9-slice mean + gaussian)"
    )
    save_grid(pairs, os.path.join(OUT_ROOT, "discarded", "pombe_nucleus_mito.png"), suptitle, n_pairs_per_row=1)


def render_evaluation():
    render_cellmap_test()
    render_block_test(
        "fafb_nuclei", "fafb_nuclei", "32 x 32 x 40 nm ssTEM", "automatic FlyWire segmentation (trains on train blocks)"
    )
    render_block_test(
        "microns_nuclei", "microns_nuclei", "64 x 64 x 40 nm ssTEM",
        "automatic minnie65 segmentation (trains on train blocks)",
    )
    render_platynereis_test()
    render_human_liver_ood()


def render_discarded():
    render_waenet_ds2()
    render_cefa_hela()
    render_nuc_mm()
    render_openorganelle_discarded()
    render_cryo_et()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", nargs="+", default=["training", "evaluation", "discarded"])
    parser.add_argument("--datasets", nargs="+", default=None, help="Training figures to render (default: all).")
    parser.add_argument("--human_liver_only", action="store_true", help="Only (re)render the human liver figure.")
    parser.add_argument("--organoid_volumes", nargs="+", default=None, help="Human organoids volumes to use.")
    args = parser.parse_args()
    for category in CATEGORIES:
        os.makedirs(os.path.join(OUT_ROOT, category), exist_ok=True)
    if args.human_liver_only:
        render_human_liver_ood()
        return
    if "training" in args.parts:
        for name in args.datasets or list(TRAINING):
            render_training(name, np.random.default_rng(0), args.organoid_volumes)
    if "evaluation" in args.parts:
        render_evaluation()
    if "discarded" in args.parts:
        render_discarded()


if __name__ == "__main__":
    main()
