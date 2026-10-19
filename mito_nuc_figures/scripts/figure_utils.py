"""Shared helpers to render raw EM data with mitochondria / nucleus label overlays.

Every dataset script in this folder uses these helpers, so that all figures share one layout:
one row for the raw data, plus one row per label class, and one column per selected slice.
For 3D volumes an extra column shows an orthogonal (ZX) view through the volume center.
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa
from skimage.segmentation import find_boundaries  # noqa


FIGURE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"
MAX_PANEL_SIZE = 768
N_CANDIDATES = 16
CLASS_COLORS = {"mito": (1.0, 0.2, 0.2), "nucleus": (0.2, 0.6, 1.0)}
DATASET_GROUPS = {
    "mito_and_nucleus": [
        "betaseg", "cellmap", "human_liver_em", "human_organoids", "pombe_nucleus_mito", "popsicle", "sxt_ins1e_mito",
        "waenet",
    ],
    "mito_only": [
        "asem", "cem", "cem_mitolab", "cmito", "deepcontact", "densecell", "drg_axon_mito", "hela_mito", "hemibrain",
        "kasthuri", "lucchi", "microns", "mito_anf", "mito_segem", "mitoem", "mitoemv2", "mitonet_predicted_kidney",
        "npc1_mito", "parlakgul_liver", "probtem", "surface_morphometrics", "tnbc_mito", "uro_cell", "vnc",
    ],
    "nucleus_only": [
        "cefa_hela", "fafb_nuclei", "janelia_nucleus", "microns_nuclei", "nuc_mm", "openorganelle_nucleus",
        "platynereis", "tumor_spheroid_em",
    ],
}


def get_group(dataset_name):
    """Get the label group folder (mito_and_nucleus, mito_only or nucleus_only) of a dataset."""
    for group, datasets in DATASET_GROUPS.items():
        if dataset_name in datasets:
            return group
    raise ValueError(f"{dataset_name} is not assigned to a group in DATASET_GROUPS.")


def get_output_dir(dataset_name):
    """Get (and create) the figure folder for one dataset, inside its label group folder."""
    out_dir = os.path.join(FIGURE_ROOT, get_group(dataset_name), dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def get_stride(shape, max_size=MAX_PANEL_SIZE):
    """Get the stride that brings the larger in-plane side down to at most max_size."""
    return max(1, int(np.ceil(max(shape) / max_size)))


def normalize(image):
    """Normalize an image to [0, 1] with percentile clipping, keeping RGB images as RGB."""
    image = np.asarray(image).astype("float32")
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        image = image[..., :3]
    lo, hi = np.percentile(image, [1, 99.5])
    if hi <= lo:
        hi = lo + 1
    return np.clip((image - lo) / (hi - lo), 0, 1)


def is_semantic(labels):
    """Check whether a label image only contains background and a single foreground id."""
    return len(np.unique(labels)) <= 2


def colorize(labels, class_name):
    """Map a label image to an RGBA overlay: one color for semantic labels, random colors for instances."""
    labels = np.asarray(labels)
    overlay = np.zeros(labels.shape + (4,), dtype="float32")
    foreground = labels > 0
    if not foreground.any():
        return overlay
    if is_semantic(labels):
        overlay[foreground, :3] = CLASS_COLORS.get(class_name, (1.0, 1.0, 0.2))
    else:
        ids, inverse = np.unique(labels[foreground], return_inverse=True)
        rng = np.random.default_rng(42)
        palette = rng.uniform(0.25, 1.0, size=(len(ids), 3))
        overlay[foreground, :3] = palette[inverse]
    overlay[foreground, 3] = 0.45
    boundaries = find_boundaries(labels, mode="inner")
    overlay[boundaries, :3] = 1.0
    overlay[boundaries, 3] = 0.9
    return overlay


def count_objects(labels):
    """Count the label ids in an image, excluding background."""
    ids = np.unique(labels)
    return int((ids != 0).sum())


def pick_slices(labels, n_slices, axis_length):
    """Pick slice indices: the middle slice and the slices with the most labeled foreground.

    The foreground is estimated on N_CANDIDATES evenly spaced, subsampled slices, so that
    large volumes do not have to be loaded.
    """
    candidates = np.unique(np.linspace(0, axis_length - 1, min(N_CANDIDATES, axis_length)).astype(int))
    if not labels:
        return sorted(set(np.linspace(0, axis_length - 1, n_slices + 2).astype(int)[1:-1].tolist()))
    scores = np.zeros(len(candidates))
    for label_volume in labels.values():
        stride = get_stride(label_volume.shape[1:], max_size=256)
        for i, z in enumerate(candidates):
            scores[i] += (np.asarray(label_volume[z, ::stride, ::stride]) > 0).mean()
    chosen = [int(candidates[len(candidates) // 2])]
    for idx in np.argsort(scores)[::-1]:
        if len(chosen) == n_slices:
            break
        if int(candidates[idx]) not in chosen:
            chosen.append(int(candidates[idx]))
    return sorted(chosen)


def draw_panel(ax, raw, labels=None, class_name=None, title=None):
    """Draw a raw image panel, optionally with a label overlay."""
    raw = normalize(raw)
    ax.imshow(raw, cmap=None if raw.ndim == 3 else "gray", interpolation="nearest")
    if labels is not None:
        ax.imshow(colorize(labels, class_name), interpolation="nearest")
    if title:
        ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def format_title(dataset_name, volume_name, shape, notes):
    """Format the figure title, with the notes on a second line."""
    title = f"{dataset_name} / {volume_name} - shape {tuple(int(s) for s in shape)}"
    if notes:
        title += f"\n{notes}"
    return title


def save_volume_figure(
    raw, labels, out_path, dataset_name, volume_name, n_slices=3, notes=None, ortho_view=True, spatial_axes=None
):
    """Save a figure for a 3D volume (z, y, x) or a single 2D image (y, x) or (y, x, c).

    Args:
        raw: The raw data, a numpy array or a lazy array (h5py / zarr / n5) supporting slicing.
        labels: Dictionary mapping the class name ("mito" or "nucleus") to a label array of the same shape as raw.
            Use an empty dictionary to only show the raw data.
        out_path: Where to save the png.
        dataset_name: The dataset name, for the title.
        volume_name: The volume name, for the title.
        n_slices: The number of z-slices to show for 3D volumes.
        notes: Optional short note shown below the title (resolution, label quality, ...).
        ortho_view: Whether to add a ZX view through the volume center for 3D volumes.
        spatial_axes: The number of spatial axes, if raw has a trailing channel axis. Inferred otherwise.
    """
    if spatial_axes is None:
        spatial_axes = 3 if (raw.ndim == 3 and raw.shape[-1] not in (3, 4)) else 2
    for class_name, label_volume in labels.items():
        assert tuple(label_volume.shape) == tuple(raw.shape[:spatial_axes]), \
            f"{class_name} labels have shape {label_volume.shape}, raw has {raw.shape}"

    rows = 1 + len(labels)
    if spatial_axes == 2:
        stride = get_stride(raw.shape[:2])
        raw_2d = np.asarray(raw[::stride, ::stride])
        label_2d = {name: np.asarray(lab[::stride, ::stride]) for name, lab in labels.items()}
        fig, axes = plt.subplots(rows, 1, figsize=(5, 5 * rows), squeeze=False)
        draw_panel(axes[0, 0], raw_2d, title="raw")
        for row, (name, lab) in enumerate(label_2d.items(), start=1):
            draw_panel(axes[row, 0], raw_2d, lab, name, title=f"{name}: {count_objects(lab)} ids")
    else:
        slices = pick_slices(labels, n_slices, raw.shape[0])
        add_ortho = ortho_view and raw.shape[0] >= 8
        cols = len(slices) + int(add_ortho)
        fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows), squeeze=False)
        stride = get_stride(raw.shape[1:])
        for col, z in enumerate(slices):
            raw_2d = np.asarray(raw[z, ::stride, ::stride])
            draw_panel(axes[0, col], raw_2d, title=f"raw z={z}")
            for row, (name, lab) in enumerate(labels.items(), start=1):
                lab_2d = np.asarray(lab[z, ::stride, ::stride])
                draw_panel(axes[row, col], raw_2d, lab_2d, name, title=f"{name} z={z}: {count_objects(lab_2d)} ids")
        if add_ortho:
            y = raw.shape[1] // 2
            stride_x = get_stride(raw.shape[2:])
            raw_zx = np.asarray(raw[:, y, ::stride_x])
            draw_panel(axes[0, -1], raw_zx, title=f"raw ZX y={y}")
            for row, (name, lab) in enumerate(labels.items(), start=1):
                lab_zx = np.asarray(lab[:, y, ::stride_x])
                draw_panel(axes[row, -1], raw_zx, lab_zx, name, title=f"{name} ZX y={y}")
            for row in range(rows):
                axes[row, -1].set_aspect("auto")

    fig.suptitle(format_title(dataset_name, volume_name, raw.shape, notes), fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def save_image_grid(samples, out_path, dataset_name, group_name, notes=None, ncols=4):
    """Save a contact sheet for 2D datasets with many images.

    Args:
        samples: List of (name, raw_2d, labels_2d) tuples, where labels_2d maps the class name to a 2D label image.
        out_path: Where to save the png.
        dataset_name: The dataset name, for the title.
        group_name: The name of the image group (split, cohort, ...), for the title.
        notes: Optional short note shown below the title.
        ncols: The number of image columns. Each image takes 1 + number of classes rows.
    """
    n_classes = max(len(sample[2]) for sample in samples)
    nrows = int(np.ceil(len(samples) / ncols)) * (1 + n_classes)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, (name, raw_2d, labels_2d) in enumerate(samples):
        row, col = (i // ncols) * (1 + n_classes), i % ncols
        stride = get_stride(raw_2d.shape[:2], max_size=512)
        raw_2d = np.asarray(raw_2d)[::stride, ::stride]
        axes[row, col].axis("on")
        draw_panel(axes[row, col], raw_2d, title=f"{name} {tuple(raw_2d.shape[:2])}x{stride}")
        for offset, (class_name, lab) in enumerate(labels_2d.items(), start=1):
            lab = np.asarray(lab)[::stride, ::stride]
            axes[row + offset, col].axis("on")
            title = f"{class_name}: {count_objects(lab)} ids"
            draw_panel(axes[row + offset, col], raw_2d, lab, class_name, title=title)
    title = f"{dataset_name} / {group_name} - {len(samples)} images"
    if notes:
        title += f"\n{notes}"
    height = fig.get_figheight()
    fig.suptitle(title, fontsize=10, y=1 - 0.1 / height, va="top")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / height))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def write_info(dataset_name, info):
    """Write the dataset summary used to build the overview table.

    Expected keys: classes (list), modality, dimensionality ("2d" or "3d"), label_quality, n_volumes,
    volumes (list of dicts with name, shape, voxel_size and any extra fields), data_path, notes, issues.
    """
    out_path = os.path.join(get_output_dir(dataset_name), "info.json")
    with open(out_path, "w") as f:
        json.dump({"dataset": dataset_name, **info}, f, indent=2, default=str)
