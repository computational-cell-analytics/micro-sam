"""Overview figures of the mitochondria model data, as the training and evaluation pipelines see it.

One PNG per dataset in mito_nuc_figures/mito/{training,in_domain_evaluation,ood_evaluation,discarded}:
- training: patches drawn from the actual em_mito training and validation datasets (_build_joint_datasets).
  The figure shows the middle slice of each patch, with the instance target as overlay and the ignored pixels shaded.
- in_domain_evaluation / ood_evaluation: samples as the evaluation loads them (load_data of
  finetuning/v2/evaluation/common.py). They are the test samples and the tuning sample, with the ground truth as
  overlay.
- discarded: representative slices of the source data with their labels, titled with the discard reason.

The layout and colors follow render_nucleus_overview.py: (raw, overlay) panel pairs, tab20 instance colors, ignored
pixels in dark grey.
"""
import os
import sys
import json
import argparse
import textwrap
from glob import glob

import h5py
import zarr
import numpy as np
from scipy import ndimage
from skimage.measure import label as connected_components

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa

REPO = "/mnt/vast-kisski/home/archit/u28048/micro-sam"
sys.path.insert(0, os.path.join(REPO, "finetuning", "v2", "evaluation"))

DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"
FIGURE_ROOT = os.path.join(REPO, "mito_nuc_figures", "mito")
DECISIONS_PATH = os.path.join(REPO, "mito_nuc_figures", "DECISIONS.md")
# The table of DECISIONS.md the rationales are read from.
RATIONALE_SECTION = "## Mitochondria"
IGNORE_COLOR = (0.15, 0.15, 0.15, 0.75)
MAX_DISPLAY = 512

# The training datasets of the decision list, keyed by figure name, with the voxel size where known.
TRAINING_DATASETS = {
    "cem_mitolab": ("cem (CEM-MitoLab, EMPIAR-11037)", "2D tiles, mixed sources"),
    "densecell": ("densecell", "10 x 10 x 50 nm"),
    "microns_pinky": ("microns", "4 x 4 x 40 nm"),
    "mitoem": ("mitoem", "8 x 8 x 30 nm"),
    "lucchi": ("lucchi", "5 nm isotropic"),
    "mitoemv2": ("mitoemv2", "8-16 nm, 30 nm sections for stem and mossy"),
    "mitoverse": ("mitoverse", "xiao18, haberl18, MICrONS1"),
    "probtem": ("probtem", "2D TEM"),
    "uro_cell": ("uro_cell", "16 nm isotropic"),
    "vnc": ("vnc", "2D ssTEM"),
    "cellmap": ("cellmap", "2-32 nm"),
    "human_organoids": ("human_organoids", "FIB-SEM, bin2"),
    "asem": ("asem", "FIB-SEM, 4-5 nm isotropic"),
    "brain_pfibsem": ("brain_pfibsem", "PFIB-SEM mouse brain, 5 nm isotropic"),
    "snemi_mito": ("snemi_mito", "ssSEM mouse cortex, 6 x 6 x 30 nm"),
    "u2os_fibsem": ("u2os_fibsem", "FIB-SEM U2-OS cell, 2D slices"),
    "hela_stard3": ("hela_stard3", "FIB-SEM HeLa, 5-8 nm"),
    "parlakgul_fasting_liver": ("parlakgul_fasting_liver", "FIB-SEM mouse liver, 8 nm isotropic"),
    "aive": ("aive", "FIB-SEM cells and muscle, 6.5 x 6.5 x 10 nm"),
    "mito_segem": ("mito_segem", "SBF-SEM testis and brainstem, 15 x 15 x 50 nm"),
}

# The evaluation figures: name -> (DECISIONS.md key, evaluation datasets of common.py).
IN_DOMAIN_EVALUATION = {
    "densecell": ("densecell", ["densecell_mito"]),
    "lucchi": ("lucchi", ["lucchi"]),
    "snemi_mito": ("snemi_mito", ["snemi_mito"]),
    "mitoem": ("mitoem", ["mitoem_human", "mitoem_rat"]),
    "mitoemv2": ("mitoemv2", [
        f"mitoemv2_{name}" for name in ("beta", "jurkat", "macro", "podo", "sperm", "stem", "mossy")
    ]),
    "probtem": ("probtem", ["probtem"]),
    "asem": ("asem", ["asem_mito"]),
    "mitonet_kidney": ("mitonet_predicted_kidney", ["mitonet_kidney"]),
}
OOD_EVALUATION = {
    "cem_mito_benchmark": ("cem (CEM-Mito-Benchmark, EMPIAR-10982)", [
        "mitolab_c_elegans", "mitolab_fly_brain", "mitolab_glycolytic_muscle", "mitolab_hela_cell",
        "mitolab_salivary_gland", "mitolab_tem",
    ]),
    "cmito": ("cmito", ["cmito_l1", "cmito_l2", "cmito_l3", "cmito_adult", "cmito_dauer", "cmito_dauer_manual"]),
    "mito_segem": ("mito_segem", ["mito_segem_intestine"]),
    "hemibrain": ("hemibrain", ["hemibrain_mito"]),
    "kasthuri": ("kasthuri", ["kasthuri"]),
    "tnbc_mito": ("tnbc_mito", ["tnbc_mito"]),
    "human_liver_em": ("human_liver_em", ["human_liver_em"]),
    "parlakgul_liver_eval": ("parlakgul_liver_eval", ["parlakgul_liver_eval"]),
    "elmi_mito": ("elmi_mito", ["elmi_mito"]),
    "lewy_mito": ("lewy_mito", ["lewy_mito"]),
    "cerebellum_fibsem": ("cerebellum_fibsem", ["cerebellum_fibsem_mito"]),
    "macrophage_a431": ("macrophage_a431", ["macrophage_a431_mito"]),
    "mito_anf": ("mito_anf", ["mito_anf"]),
    "donglai_wei_mito": ("donglai_wei_mito", ["donglai_wei_mito"]),
    "perez": ("perez", ["perez_mito"]),
    "cardiomyocyte_tem": ("cardiomyocyte_tem", ["cardiomyocyte_tem_mito"]),
}
# Notes that the title of an evaluation figure carries, e.g. for data that is not public.
TITLE_NOTES = {"donglai_wei_mito": "internal data from Donglai Wei (not public)"}


def normalize(image):
    image = image.astype("float32")
    lo, hi = np.percentile(image, (1, 99))
    return np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1)


def display_stride(shape):
    return max(1, int(np.ceil(max(shape) / MAX_DISPLAY)))


def instance_colors(labels):
    """RGBA overlay with a fixed color per instance id; the two greys of tab20 are left out (grey marks ignore)."""
    cmap = plt.get_cmap("tab20")
    palette = [cmap(index) for index in range(20) if index not in (14, 15)]
    overlay = np.zeros(labels.shape + (4,), dtype="float32")
    ids = np.unique(labels)
    for label_id in ids[ids > 0]:
        color = palette[(int(label_id) * 7) % len(palette)]
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
    fig.text(0.02, 0.3, "\n".join(textwrap.wrap(text, 150)), fontsize=10)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print("wrote", out_path)


def save_grid(pairs, out_path, suptitle, n_pairs_per_row=2):
    """Save a grid of (raw, overlay) panel pairs; each pair is (raw, labels, ignore, title)."""
    n_rows = int(np.ceil(len(pairs) / n_pairs_per_row))
    n_cols = 2 * n_pairs_per_row
    n_title_lines = suptitle.count("\n") + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.6 * n_rows + 0.25 * n_title_lines + 0.5))
    axes = np.asarray(axes).reshape(n_rows, n_cols)
    for ax in axes.flat:
        ax.axis("off")
    for k, (raw, labels, ignore, title) in enumerate(pairs):
        row, col = divmod(k, n_pairs_per_row)
        draw_pair(axes[row, 2 * col:2 * col + 2], raw, labels, ignore, title)
    fig.suptitle(suptitle, fontsize=10, y=1 - 0.1 / fig.get_figheight(), va="top")
    top = 1 - (0.25 * n_title_lines + 0.35) / fig.get_figheight()
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=top, wspace=0.03, hspace=0.12)
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    print("wrote", out_path)


def read_rationales():
    """The 'Details' column of the RATIONALE_SECTION table of DECISIONS.md, keyed by the dataset column."""
    rationales, in_section = {}, False
    with open(DECISIONS_PATH) as f:
        for line in f:
            if line.startswith("## "):
                in_section = line.startswith(RATIONALE_SECTION)
            if in_section and line.startswith("| ") and not line.startswith("| Dataset"):
                cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
                if len(cells) >= 3:
                    rationales[cells[0]] = (cells[1], cells[2])
    return rationales


def short_rationale(rationales, key, width=170, max_lines=2, keywords=None):
    """The decision and the first sentence of its details, or the first sentences that contain one of keywords."""
    decision, details = rationales.get(key, ("", ""))
    sentences = [sentence.strip().rstrip(".") + "." for sentence in details.split(". ") if sentence.strip()]
    chosen = [s for s in sentences if keywords and any(k in s.lower() for k in keywords)][:2] or sentences[:1]
    lines = textwrap.wrap(f"{decision}: {' '.join(chosen)}", width)
    return "\n".join(lines[:max_lines])


def training_dataset_name(path):
    """The figure name of a training leaf, from its label path."""
    rel = os.path.relpath(path, DATA_ROOT)
    if rel.startswith("mitoverse_derived/han24"):
        return "mitoemv2"
    if rel.startswith("mitoverse_derived/turner22") or rel.startswith("mitoverse/"):
        return "mitoverse"
    return {
        "mitolab": "cem_mitolab", "microns": "microns_pinky", "probtem_full": "probtem",
        "cellmap_mito_crops": "cellmap", "human_organoids_mito": "human_organoids", "snemi_mito_stack": "snemi_mito",
    }.get(rel.split(os.sep)[0], rel.split(os.sep)[0])


def leaf_path(leaf):
    return getattr(leaf, "label_path", None) or (getattr(leaf, "label_images", None) or ["unknown"])[0]


def draw_training_samples(wrappers, n, rng):
    """Draw n samples round-robin over the wrappers of a dataset; returns (raw, labels, ignore, source) tuples."""
    from micro_sam.v2.datasets.wrapper import _leaves
    samples = []
    for k in range(n):
        wrapper = wrappers[k % len(wrappers)]
        raw, label = wrapper[int(rng.integers(len(wrapper)))]
        raw, label = raw.numpy(), label.numpy()
        z = raw.shape[1] // 2
        instances = label[0, z].astype("int64")
        ignore = label[1, z] == -1
        n_objects = len(np.unique(label[0][label[0] > 0]))
        leaves = _leaves(wrapper.ds)
        if hasattr(leaves[0], "label_images"):
            source = f"random tile of {len(leaves[0].label_images)}"
        else:
            source = os.path.basename(leaf_path(leaves[0]).rstrip("/"))
            if len(leaves) > 1:
                source += f" (+{len(leaves) - 1} files)"
        title = f"{source}, patch {tuple(raw.shape[1:])}, {n_objects} objects"
        if ignore.any():
            title += f", {ignore.mean():.0%} ignored"
        samples.append((raw[0, z], instances, ignore if ignore.any() else None, title))
    return samples


def render_training(names):
    from micro_sam.v2.datasets.generalist_loader import _build_joint_datasets
    out_dir = os.path.join(FIGURE_ROOT, "training")
    os.makedirs(out_dir, exist_ok=True)
    rationales = read_rationales()
    train_ds, val_ds = _build_joint_datasets(DATA_ROOT, [8], "em_mito", distance_type="directed")
    groups = {}
    from micro_sam.v2.datasets.wrapper import _leaves
    for split, ds in [("train", train_ds), ("val", val_ds)]:
        for wrapper in ds.datasets:
            name = training_dataset_name(leaf_path(_leaves(wrapper.ds)[0]))
            groups.setdefault(name, {"train": [], "val": []})[split].append(wrapper)
    rng = np.random.default_rng(0)
    np.random.seed(0)
    for name, (key, voxel) in TRAINING_DATASETS.items():
        if names and name not in names:
            continue
        wrappers = groups.get(name)
        if wrappers is None:
            text_figure(os.path.join(out_dir, f"{name}.png"), name, "No training leaf found for this dataset.")
            continue
        n_train = min(8, max(6, len(wrappers["train"])))
        n_train += n_train % 2
        pairs = [
            (raw, lab, ign, f"train: {title}")
            for raw, lab, ign, title in draw_training_samples(wrappers["train"], n_train, rng)
        ]
        pairs += [
            (raw, lab, ign, f"val: {title}") for raw, lab, ign, title in draw_training_samples(wrappers["val"], 2, rng)
        ]
        n_train_samples = sum(len(w) for w in wrappers["train"])
        n_val_samples = sum(len(w) for w in wrappers["val"])
        suptitle = (
            f"mito / training / {name} ({voxel}): {len(wrappers['train'])} train and {len(wrappers['val'])} val "
            f"leaves, {n_train_samples} train / {n_val_samples} val samples per epoch\n"
            f"{short_rationale(rationales, key)}"
        )
        save_grid(pairs, os.path.join(out_dir, f"{name}.png"), suptitle)


def pick_slice(labels):
    """The slice with the most labeled pixels."""
    if labels.ndim == 2:
        return None
    return int(np.argmax((labels > 0).reshape(labels.shape[0], -1).sum(axis=1)))


def evaluation_pairs(dataset_name, n_test):
    import common
    ndim = 2 if dataset_name in common.DATASETS_2D else 3
    n_total = common.n_samples(dataset_name, DATA_ROOT, split="test")
    ids = sorted(set(np.linspace(0, n_total - 1, min(n_test, n_total)).round().astype(int).tolist()))
    pairs = []
    for sample_id, (raw, labels, valid) in zip(ids, common.load_data(dataset_name, DATA_ROOT, ndim, sample_ids=ids)):
        pairs.append(sample_pair(raw, labels, valid, f"{dataset_name} test {sample_id + 1}/{n_total}"))
    if common.has_val_split(dataset_name):
        raw, labels, valid = next(iter(common.load_data(dataset_name, DATA_ROOT, ndim, split="val", sample_ids=[0])))
        n_val = common.n_samples(dataset_name, DATA_ROOT, split="val")
        pairs.append(sample_pair(raw, labels, valid, f"{dataset_name} tuning 1/{n_val}"))
    return pairs, n_total


def sample_pair(raw, labels, valid, title):
    raw, labels = np.asarray(raw), np.asarray(labels)
    if raw.ndim == 3 and raw.shape[-1] in (3, 4) and labels.ndim == 2:
        raw = raw[..., :3].mean(axis=-1)
    z = pick_slice(labels)
    n_objects = len(np.unique(labels[labels > 0]))
    shape = tuple(labels.shape)
    if z is not None:
        raw, labels = raw[z], labels[z]
        valid = None if valid is None else valid[z]
    ignore = None if valid is None else ~valid.astype(bool)
    return raw, labels, ignore, f"{title}, shape {shape}, {n_objects} GT objects"


def render_evaluation(category, figures, names):
    out_dir = os.path.join(FIGURE_ROOT, category)
    os.makedirs(out_dir, exist_ok=True)
    rationales = read_rationales()
    for name, (key, dataset_names) in figures.items():
        if names and name not in names:
            continue
        n_test = 3 if len(dataset_names) == 1 else 1
        pairs, counts = [], []
        for dataset_name in dataset_names:
            try:
                set_pairs, n_total = evaluation_pairs(dataset_name, n_test)
            except Exception as e:
                print(f"{dataset_name} failed: {type(e).__name__}: {e}")
                counts.append(f"{dataset_name}: failed to load")
                continue
            pairs += set_pairs
            counts.append(f"{dataset_name}: {n_total} test samples")
        if not pairs:
            text_figure(os.path.join(out_dir, f"{name}.png"), name, "; ".join(counts))
            continue
        count_text = "\n".join(textwrap.wrap("; ".join(counts), 170))
        rationale = short_rationale(rationales, key, keywords=("eval", "test", "tun", "scor", "val volume"))
        title = f"{os.path.basename(FIGURE_ROOT)} / {category} / {name}"
        if name in TITLE_NOTES:
            title += f" ({TITLE_NOTES[name]})"
        suptitle = f"{title}\n{count_text}\n{rationale}"
        save_grid(pairs, os.path.join(out_dir, f"{name}.png"), suptitle)


def open_array(path, key=None):
    if path.endswith(".h5"):
        return h5py.File(path, "r")[key]
    group = zarr.open(path, mode="r")
    return group if key is None else group[key]


def tomogram_slice(raw, z, half_window=4):
    """Mean of 9 slices with a gaussian filter, for visibility of the noisy tomograms."""
    window = np.asarray(raw[max(0, z - half_window):z + half_window + 1], dtype="float32").mean(axis=0)
    return ndimage.gaussian_filter(window, 2)


def volume_pair(raw, labels, title, z=None, tomogram=False, flip_labels_y=False):
    """The (raw, labels) pair at slice z, or at the slice with the most labels among every 8th slice."""
    if z is None:
        candidates = range(0, labels.shape[0], 8)
        z = max(candidates, key=lambda c: int((np.asarray(labels[c]) > 0).sum()))
    lab = np.asarray(labels[z]).astype("int64")
    if flip_labels_y:
        lab = lab[::-1]
    lab[lab < 0] = 0
    image = tomogram_slice(raw, z) if tomogram else np.asarray(raw[z])
    return image, lab, None, f"{title} z={z}"


def discarded_drg_axon_mito():
    pairs = []
    for name in ("10512_Position_15", "10513_Position_03"):
        root = os.path.join(DATA_ROOT, "drg_axon_mito", name)
        if not os.path.exists(root):
            continue
        raw = open_array(os.path.join(root, "raw.zarr"), "0")
        outer = open_array(os.path.join(root, "class_outer_membrane.zarr"), "0")
        inner = open_array(os.path.join(root, "class_inner_membrane.zarr"), "0")
        z = max(range(0, outer.shape[0], 8), key=lambda c: int((np.asarray(outer[c]) > 0).sum()))
        classes = np.where(np.asarray(inner[z]) > 0, 2, np.where(np.asarray(outer[z]) > 0, 1, 0))
        pairs.append((tomogram_slice(raw, z), thicken(classes), None, f"{name} (1 outer, 2 inner membrane) z={z}"))
    return pairs


def discarded_hela_mito():
    pairs = []
    for name in ("z0001", "z0030", "z0060", "z0116"):
        path = os.path.join(DATA_ROOT, "hela_mito", f"{name}.h5")
        if os.path.exists(path):
            with h5py.File(path, "r") as f:
                pairs.append((f["raw"][:], f["labels"][:].astype("int64"), None, name))
    return pairs


def discarded_mito_anf():
    pairs = []
    for path in sorted(glob(os.path.join(DATA_ROOT, "mito_anf", "*.zarr"))):
        group = zarr.open(path, mode="r")
        pairs.append(volume_pair(group["raw"], group["labels"], os.path.basename(path)))
    return pairs


def discarded_npc1_mito():
    root = os.path.join(DATA_ROOT, "npc1_mito", "25jul29a_Position_3")
    raw, labels = open_array(os.path.join(root, "raw.zarr"), "0"), open_array(os.path.join(root, "labels.zarr"), "0")
    return [volume_pair(raw, labels, "25jul29a_Position_3", tomogram=True)]


def discarded_popsicle():
    pairs = []
    for split, run in (("test", "TS_045"), ("train", "TS_0006")):
        root = os.path.join(DATA_ROOT, "popsicle_yeast", "yeast_organelles", split, run)
        mito = sorted(glob(os.path.join(root, "class_mito*.zarr")))
        if not mito:
            continue
        raw, labels = open_array(os.path.join(root, "raw.zarr"), "0"), open_array(mito[0], "0")
        pairs.append(volume_pair(raw, labels, f"{split}/{run} mito", tomogram=True))
    return pairs


def discarded_pombe_nucleus_mito():
    pairs = []
    for name in ("10000_TS_045", "10001_TS_0006"):
        group = zarr.open(os.path.join(DATA_ROOT, "pombe_nucleus_mito", f"{name}.zarr"), mode="r")
        pairs.append(volume_pair(group["raw"], group["mitochondrion"], f"{name} mito", tomogram=True))
    return pairs


def thicken(labels, iterations=2):
    """Dilate thin membrane labels so that they stay visible in the downsampled display."""
    return ndimage.grey_dilation(labels, size=(2 * iterations + 1, 2 * iterations + 1))


def discarded_surface_morphometrics():
    # Most volumes need their labels flipped in y to align with the raw (see the survey info.json).
    info = os.path.join(REPO, "mito_nuc_figures", "mito_only", "surface_morphometrics", "info.json")
    with open(info) as f:
        flips = {v["name"]: v["labels_flipped_y_to_align"] for v in json.load(f)["volumes"]}
    pairs = []
    for name in ("TE11", "TE7"):
        group = zarr.open(os.path.join(DATA_ROOT, "surface_morphometrics", f"{name}.zarr"), mode="r")
        raw, lab, ign, title = volume_pair(
            group["raw"], group["labels"], f"{name} (membrane classes)", tomogram=True, flip_labels_y=flips[name],
        )
        pairs.append((raw, thicken(lab), ign, title))
    return pairs


def discarded_betaseg():
    pairs = []
    for name in ("high_c1", "low_c2"):
        f = h5py.File(os.path.join(DATA_ROOT, "betaseg", "data", f"{name}_source.h5"), "r")
        mito = f["labels/mitochondria"]
        z = mito.shape[0] // 2
        lab = connected_components(np.asarray(mito[z]) > 0)
        pairs.append((f["raw"][z], lab, None, f"{name} mito z={z}"))
    return pairs


def discarded_sxt_ins1e_mito():
    pairs = []
    for path in sorted(glob(os.path.join(DATA_ROOT, "sxt_ins1e_mito", "*.zarr")))[:2]:
        group = zarr.open(path, mode="r")
        labels = group["labels"]
        z = labels.shape[0] // 2
        lab = connected_components(np.asarray(labels[z]) == 5)
        pairs.append((group["raw"][z], lab, None, f"{os.path.basename(path)[:40]} mito (class 5) z={z}"))
    return pairs


def discarded_waenet():
    pairs = []
    for name in ("img_17", "img_43", "img_9", "img_60"):
        path = os.path.join(DATA_ROOT, "waenet", "dataset_3", f"{name}.h5")
        if os.path.exists(path):
            with h5py.File(path, "r") as f:
                pairs.append((f["raw"][:], connected_components(f["labels/mitochondria"][:] > 0), None, f"ds3 {name}"))
    return pairs


DISCARDED = {
    "drg_axon_mito": ("drg_axon_mito", discarded_drg_axon_mito),
    "hela_mito": ("hela_mito", discarded_hela_mito),
    "mito_anf": ("mito_anf", discarded_mito_anf),
    "npc1_mito": ("npc1_mito", discarded_npc1_mito),
    "popsicle": ("popsicle", discarded_popsicle),
    "pombe_nucleus_mito": ("pombe_nucleus_mito", discarded_pombe_nucleus_mito),
    "surface_morphometrics": ("surface_morphometrics", discarded_surface_morphometrics),
    "betaseg": ("betaseg", discarded_betaseg),
    "sxt_ins1e_mito": ("sxt_ins1e_mito", discarded_sxt_ins1e_mito),
    "waenet": ("waenet", discarded_waenet),
}


def render_discarded(names):
    out_dir = os.path.join(FIGURE_ROOT, "discarded")
    os.makedirs(out_dir, exist_ok=True)
    rationales = read_rationales()
    for name, (key, reader) in DISCARDED.items():
        if names and name not in names:
            continue
        decision, details = rationales.get(key, ("skip", ""))
        reason = "\n".join(textwrap.wrap(f"{decision}: {details}", 170)[:3])
        try:
            pairs = reader()
        except Exception as e:
            print(f"{name} failed: {type(e).__name__}: {e}")
            pairs = []
        if not pairs:
            text_figure(os.path.join(out_dir, f"{name}.png"), f"mito / discarded / {name}", f"{decision}: {details}")
            continue
        save_grid(pairs, os.path.join(out_dir, f"{name}.png"), f"mito / discarded / {name}\n{reason}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "categories", nargs="+", choices=["training", "in_domain_evaluation", "ood_evaluation", "discarded"],
        help="The figure categories to render.",
    )
    parser.add_argument("--names", nargs="+", default=None, help="Only render these figures.")
    args = parser.parse_args()
    for category in args.categories:
        if category == "training":
            render_training(args.names)
        elif category == "in_domain_evaluation":
            render_evaluation(category, IN_DOMAIN_EVALUATION, args.names)
        elif category == "ood_evaluation":
            render_evaluation(category, OOD_EVALUATION, args.names)
        else:
            render_discarded(args.names)


if __name__ == "__main__":
    main()
