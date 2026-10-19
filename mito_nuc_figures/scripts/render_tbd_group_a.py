"""Render survey figures of new torch-em datasets (group A: 2D / TEM and small sets) into the TBD folders.

The figures go to mito_nuc_figures/mito/TBD (mitochondria labels) and mito_nuc_figures/nuc/TBD (nucleus, nuclear
envelope and nucleolus labels). Large micrographs are shown as a downsampled overview plus a full-resolution crop.
"""
import argparse
import glob
import os

import h5py
import imageio.v3 as iio
import numpy as np
import tifffile
import zarr
from skimage.measure import label

from render_mito_overview import save_grid, text_figure

DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"
FIGURE_ROOT = "/mnt/vast-kisski/home/archit/u28048/micro-sam/mito_nuc_figures"
CROP = 768


def read(path):
    if path.endswith((".tif", ".tiff")):
        image = tifffile.imread(path)
    else:
        image = iio.imread(path)
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        image = image[..., :3].mean(axis=-1)
    return image


def instances(mask):
    """Instance labels: connected components of a semantic mask, kept as they are for instance masks."""
    mask = np.asarray(mask)
    if len(np.unique(mask)) <= 2:
        return label(mask > 0).astype("uint32")
    return mask.astype("uint32")


def densest_crop(labels, size=CROP):
    """The crop of `size` with the most labeled pixels on a coarse grid."""
    h, w = labels.shape
    if h <= size and w <= size:
        return np.s_[:, :]
    best, best_sl = -1, None
    step = max(size // 2, 1)
    for y in range(0, max(h - size, 0) + 1, step):
        for x in range(0, max(w - size, 0) + 1, step):
            score = (labels[y:y + size, x:x + size] > 0).sum()
            if score > best:
                best, best_sl = score, np.s_[y:y + size, x:x + size]
    return best_sl


def overview_and_crop(raw, labels, name):
    """Two panel pairs: the whole image (downsampled for display) and its densest full-resolution crop."""
    n = len(np.unique(labels)) - 1
    sl = densest_crop(labels)
    return [
        (raw, labels, None, f"{name}: whole image {raw.shape}, {n} objects"),
        (raw[sl], labels[sl], None, f"{name}: full-resolution crop"),
    ]


def out_path(model, name):
    folder = os.path.join(FIGURE_ROOT, model, "TBD")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{name}.png")


def cardiomyocyte_tem():
    from torch_em.data.datasets.electron_microscopy.cardiomyocyte_tem import get_cardiomyocyte_tem_paths

    root = os.path.join(DATA_ROOT, "cardiomyocyte_tem")
    for model, choices in (("mito", ("mito",)), ("nuc", ("nucleus", "nucleolus"))):
        pairs = []
        for choice in choices:
            raw_paths, label_paths = get_cardiomyocyte_tem_paths(root, label_choice=choice)
            for raw_path, label_path in list(zip(raw_paths, label_paths))[:3 if choice != "nucleolus" else 1]:
                name = f"{os.path.basename(raw_path)[:-4][:24]} {choice}"
                pairs += overview_and_crop(read(raw_path), instances(tifffile.imread(label_path)), name)
        what = "mitochondria" if model == "mito" else "nucleus (one per image) and nucleolus (one image)"
        save_grid(pairs, out_path(model, "cardiomyocyte_tem"), (
            f"TBD / cardiomyocyte_tem ({what})\n"
            "TEM of isolated mouse cardiomyocytes, 3500x, 9 large micrographs with raw (3.5k-13k px), pixel size not "
            "stored; binary semantic masks, instances by connected components\n"
            "Mito: 290-1150 objects per image, dense; nucleus: 1 per image; nucleolus: 1 image. The mask archives hold "
            "2 more masks without released raw"
        ))


def elmi_mito():
    root = os.path.join(DATA_ROOT, "elmi_mito")
    pairs = []
    for image_path in sorted(glob.glob(os.path.join(root, "ELMI_orig_images", "*.tif"))):
        labels = instances(read(image_path.replace("ELMI_orig_images", "ELMI_manual_annotation")))
        raw = read(image_path)
        n = len(np.unique(labels)) - 1
        pairs.append((raw, labels, None, f"{os.path.basename(image_path)}: {raw.shape}, {n} objects"))
    save_grid(pairs, out_path("mito", "elmi_mito"), (
        "TBD / elmi_mito (mitochondria)\n"
        "TEM, 5 images of ~2000 x 2000 px, pixel size not stated; manual instance masks (MitoNet fine-tuning data)\n"
        "Only 3-6 labeled objects per image (22 in total, two of them 35 px specks); no licence or publication stated"
    ), n_pairs_per_row=3)


def lewy_mito():
    root = os.path.join(DATA_ROOT, "lewy_mito", "Shafiei_Mito-Segmentation")
    crops = sorted(glob.glob(os.path.join(root, "*", "*", "crops", "*.tif")))
    rng = np.random.default_rng(0)
    pairs = []
    for crop in rng.choice(crops, 12, replace=False):
        mask_path = crop.replace(f"{os.sep}crops{os.sep}", f"{os.sep}masks{os.sep}")
        if not os.path.exists(mask_path):
            continue
        labels = instances(tifffile.imread(mask_path) > 0)
        rel = os.path.relpath(crop, root)
        pairs.append((read(crop), labels, None, f"{rel}: {labels.shape}, {len(np.unique(labels)) - 1} objects"))
    save_grid(pairs, out_path("mito", "lewy_mito"), (
        "TBD / lewy_mito (mitochondria)\n"
        "TEM of post-mortem human brain (SN, ENT, HC; PD, DLB, controls), ~16,500 small crops (20-400 px), "
        "manual + proofread masks; shown: random crops of the 2 downloaded groups (SN-PD, HC-SN)\n"
        "Crops are cut around individual mitochondria (1-3 labeled per crop): neighbouring mitochondria in a crop may "
        "be unlabeled"
    ), n_pairs_per_row=3)


def snemi_mito(overlap_note):
    root = os.path.join(DATA_ROOT, "snemi_mito", "SNEMI3D_mito")
    pairs = []
    for split, index in (("train", 0), ("train", 30), ("valid", 10), ("test", 15)):
        image_path = sorted(glob.glob(os.path.join(root, split, "images", "*.png")))[index]
        labels = instances(iio.imread(image_path.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")))
        raw = read(image_path)
        pairs += overview_and_crop(raw, labels, f"{split}/{os.path.basename(image_path)}")[:1]
    save_grid(pairs, out_path("mito", "snemi_mito"), (
        "TBD / snemi_mito (mitochondria)\n"
        "ssSEM of mouse neocortex (SNEMI3D, Kasthuri et al. 2015), 100 slices of 1024 x 1024 px at 6 nm (CLAHE), "
        "50 train / 20 val / 30 test; binary masks (Kubota lab), instances by connected components\n"
        f"{overlap_note}"
    ))


def perez(overlap_note):
    root = os.path.join(DATA_ROOT, "perez", "perez")
    for model, organelles in (("mito", ("mito",)), ("nuc", ("nuclei", "nucleoli"))):
        pairs = []
        for organelle in organelles:
            for split in ("train", "test"):
                images = sorted(glob.glob(os.path.join(root, organelle, split, "images", "*")))
                for image_path in images[::max(1, len(images) // 2)][:2]:
                    mask_path = image_path.replace(f"{os.sep}images{os.sep}", f"{os.sep}masks{os.sep}")
                    labels = instances(read(mask_path) > 0)
                    raw = read(image_path)
                    pairs.append((raw, labels, None, f"{organelle} {split}/{os.path.basename(image_path)}: "
                                                     f"{raw.shape}, {len(np.unique(labels)) - 1} objects"))
        what = "mitochondria" if model == "mito" else "nuclei and nucleoli"
        save_grid(pairs, out_path(model, "perez"), (
            f"TBD / perez ({what})\n"
            "SBF-SEM of the mouse suprachiasmatic nucleus (Perez et al. 2014, RETINA preprocessing), separate 2D tile "
            "sets per organelle: 50 train tiles (500 px) and 40 test tiles (1000 px); binary masks\n"
            f"{overlap_note}"
        ))


def plantorganelle_hunter():
    note = (
        "Not downloaded: the data is on ScienceDB (doi 10.57760/sciencedb.01335), whose file API needs a login, and "
        "the archive is ~50 GB; the torch-em loader expects a manual download and extraction. TEM of plant cells, 902 "
        "micrographs (4800 x 3230 px; 541 train / 180 val / 181 test), semantic masks of chloroplasts, mitochondria, "
        "vacuoles and nuclei."
    )
    for model, what in (("mito", "mitochondria"), ("nuc", "nuclei")):
        text_figure(out_path(model, "plantorganelle_hunter"), f"TBD / plantorganelle_hunter ({what})", note)


def u2os_fibsem():
    root = os.path.join(DATA_ROOT, "u2os_fibsem")
    for model, folder, what in (
        ("mito", "1_Mitochondria/1_Train_Mito", "mitochondria"), ("nuc", "1_NE/1_Train_NE", "nuclear envelope"),
    ):
        images = sorted(glob.glob(os.path.join(root, folder, "images_tif", "*.tif")))
        pairs = []
        for image_path in images[::max(1, len(images) // 4)][:4]:
            labels = instances(tifffile.imread(image_path.replace("images_tif", "labels_tif")))
            raw = read(image_path)
            pairs.append((raw, labels, None, f"{os.path.basename(image_path)}: {raw.shape}, "
                                             f"{len(np.unique(labels)) - 1} objects"))
        save_grid(pairs, out_path(model, "u2os_fibsem"), (
            f"TBD / u2os_fibsem ({what})\n"
            "FIB-SEM of one U2-OS cell (EMPIAR-11746), DeepMIB training data: single annotated 2D slices "
            "(mito: 45 slices of 692 x 1697 px; NE: 44 crops of 1235 x 530 px); binary masks\n"
            "All slices come from one cell; the NE label is the nuclear envelope (a thin band), not a filled nucleus"
        ))


def hela_mito():
    pairs = []
    for z in ("z0001", "z0030", "z0060"):
        for version, folder in (("gt", "hela_mito_gt"), ("gt2", "hela_mito_gt2")):
            path = glob.glob(os.path.join(DATA_ROOT, folder, f"{z}*.h5"))[0]
            with h5py.File(path, "r") as f:
                raw, labels = f["raw"][:], f["labels"][:]
            pairs.append((raw, instances(labels), None, f"{z} {version}: {len(np.unique(labels)) - 1} objects"))
    save_grid(pairs, out_path("mito", "hela_mito"), (
        "TBD / hela_mito (mitochondria, GT vs GT2)\n"
        "SBF-SEM HeLa (EMPIAR-10094), 5 labeled slices (2000 x 2000 px, 10 nm), two independent manual annotations: "
        "GT 15-35 vs GT2 17-40 objects per slice, foreground IoU 0.76-0.96\n"
        "Both label only the central cell and miss visible mitochondria; GT2 adds objects but does not change the "
        "sparse coverage (earlier discard reason)"
    ))


def parlakgul_liver_eval(overlap_note):
    root = os.path.join(DATA_ROOT, "parlakgul_liver_eval")
    pairs = []
    for crop in sorted(glob.glob(os.path.join(root, "*_mito_*.zarr"))):
        group = zarr.open_group(crop, mode="r")
        manual, prediction = group["labels"][:], group["prediction"][:]
        iou = ((manual > 0) & (prediction > 0)).sum() / max(((manual > 0) | (prediction > 0)).sum(), 1)
        name = os.path.basename(crop)[:-5]
        raw = group["raw"][12]
        pairs.append((raw, instances(manual[12]), None, f"{name} manual: {len(np.unique(manual)) - 1} objects (3D)"))
        pairs.append((raw, instances(prediction[12]), None, f"{name} model: foreground IoU with manual {iou:.2f}"))
    save_grid(pairs, out_path("mito", "parlakgul_liver_eval"), (
        "TBD / parlakgul_liver_eval (mitochondria: manual annotation and the authors' model prediction per crop)\n"
        "FIB-SEM mouse liver (EMPIAR-10791), 12 manual crops of 25 x 150 x 150 voxels at 8 nm (3 per sample: 6461, "
        "6464, 9430, 1857), instance labels (1-8 per crop)\n"
        f"{overlap_note}"
    ), n_pairs_per_row=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snemi_note", default="", help="The overlap result with the Kasthuri++ volumes.")
    parser.add_argument("--perez_note", default="", help="The overlap result with the MitoLab perez2014 group.")
    parser.add_argument("--parlakgul_note", default="", help="The overlap result with our Parlakgul boxes.")
    parser.add_argument("--names", nargs="+", default=None, help="Only render these datasets.")
    args = parser.parse_args()
    renderers = {
        "cardiomyocyte_tem": cardiomyocyte_tem, "elmi_mito": elmi_mito, "lewy_mito": lewy_mito,
        "snemi_mito": lambda: snemi_mito(args.snemi_note), "perez": lambda: perez(args.perez_note),
        "plantorganelle_hunter": plantorganelle_hunter, "u2os_fibsem": u2os_fibsem, "hela_mito": hela_mito,
        "parlakgul_liver_eval": lambda: parlakgul_liver_eval(args.parlakgul_note),
    }
    for name, render in renderers.items():
        if args.names is None or name in args.names:
            render()


if __name__ == "__main__":
    main()
