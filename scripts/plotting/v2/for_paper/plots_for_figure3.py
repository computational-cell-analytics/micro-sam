"""Plots for Figure 3 of the micro-sam2 paper.

The figure compares 3d automatic and interactive instance segmentation of micro-sam2 (hvit_l) with the baselines.
It uses the layout and the style of Figure 2 (see plots_for_figure2.py).
The script writes these plots to 'figures/figure3/<automatic|interactive>/<quantitative|qualitative>':
- 'automatic_3d': the score of each method per modality, for the in-domain and the out-of-domain datasets.
- 'automatic_3d_qualitative': one test volume per modality and domain, with the reference and the predictions.
- 'interactive_3d': the score over the correction iterations, with a point or a box as the first prompt.
- 'interactive_3d_qualitative': the 3d renders of two objects per modality and domain, segmented from one prompt.
- 'figure_3' (in 'figures/final'): the four plots in one figure.

Usage:

    python plots_for_figure3.py
    python plots_for_figure3.py --only figure_3

The inputs are:
- RESULTS_CSV and INTERACTIVE_CSV: the scores from the scripts in 'finetuning/v2/evaluation'.
- RENDERS: the renders of 'render_pairs.py' and 'render_methods.py', made headless on a GPU node.
- INTERACTIVE_PREDICTIONS: the interactive predictions after the first prompt.
- QUALITATIVE_PREDICTIONS: the automatic predictions of 'run_posthoc_scripts.py predict_samples'.
"""

import os
import glob
import argparse

import tifffile
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.measure import label
import matplotlib.colors as colors
from skimage.segmentation import find_boundaries

from elf.evaluation import cremi_score, mean_segmentation_accuracy

import plots_for_figure2 as fig2
from plot_style import TEXT_COLOR, set_style


ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(ROOT, "figures", "cells_automatic_segmentation_3d_results.csv")
INTERACTIVE_CSV = os.path.join(ROOT, "figures", "cells_interactive_segmentation_3d_results.csv")
OUTPUT_DIR = os.path.join(ROOT, "figures", "figure3")
EXPERIMENTS = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments"
RENDERS = os.path.join(EXPERIMENTS, "qualitative_3d", "render_tryouts")
INTERACTIVE_PREDICTIONS = os.path.join(EXPERIMENTS, "interactive_evaluation_v6_center")
QUALITATIVE_PREDICTIONS = os.path.join(EXPERIMENTS, "qualitative_3d", "out")

# The (category, structure, title, score column, score label) of each modality. The EM cells go to the supplement.
PANELS = {
    "fluorescence_cells": ("fluorescence", "cell", "Fluorescence (cell)", "msa", "Mean segmentation accuracy ↑"),
    "fluorescence_nuclei": ("fluorescence", "nucleus", "Fluorescence (nucleus)", "msa", "Mean segmentation accuracy ↑"),
    "em_neurites": ("EM", "neurite", "Electron microscopy (neurite)", "cremi", "CREMI score ↓"),
}

# The 'method', 'model' and 'mode' values of each automatic method in the 3d results.
METHODS = {
    # StarDist has no 3d model, so it segments each slice and stitches the slices.
    "StarDist": ("stardist", ("2D_versatile_fluo_stitch",), None),
    "Cellpose3": ("cellpose", ("cyto3_do3d",), None),
    "FOCUS-3D": ("focus3d", ("membrane",), None),
    "SegNeuron": ("segneuron", ("default",), None),
    "micro-sam": ("micro-sam", ("vit_l_lm",), "apg"),
    "CellposeSAM": ("cellpose", ("cpsam_do3d",), None),
    "micro-sam2": ("micro-sam2", ("hvit_l",), "apg"),
}
PANEL_MODELS = {
    "fluorescence_nuclei": {"Cellpose3": ("nuclei_do3d",), "FOCUS-3D": ("nuclei",)},
}
STARRED_METHODS = {"Cellpose3", "FOCUS-3D"}
PANEL_METHODS = {
    "fluorescence_cells": ["Cellpose3", "micro-sam", "CellposeSAM", "FOCUS-3D", "micro-sam2"],
    "fluorescence_nuclei": ["StarDist", "Cellpose3", "micro-sam", "CellposeSAM", "FOCUS-3D", "micro-sam2"],
    "em_neurites": ["SegNeuron", "micro-sam2"],
}

PANEL_DATASETS = {
    "fluorescence_cells": {
        "ID": ["cartocell", "embedseg_organoid", "phmamm", "plantseg_root", "pnas_arabidopsis", "wing_disc"],
        # Three OOD datasets only (user, 2026-11-05). vibrio_cholerae is left out: its labels split many cells along
        # z, so all methods score below 0.07 mSA.
        "OOD": ["cshaper", "morphonet_patiria", "plantseg_ovules"],
    },
    "fluorescence_nuclei": {
        "ID": ["celegans_atlas", "embedseg_mouse_skull", "embedseg_platy_nuclei", "gonuclear", "nis3d", "nucverse3d"],
        # mouse_embryo is left out: only part of its nuclei have labels. parhyale_regen is left out for nus_organoids
        # (user, 2026-11-05): all methods score 0.13 mSA or less on it, so it does not tell the methods apart.
        "OOD": ["bbbc032", "bbbc033", "bbbc050", "blastospim", "morphonet_celegans", "nus_organoids"],
    },
    # SegNeuron also trains on the in-domain sources, so they are in domain for both methods.
    "em_neurites": {
        # xpress is left out (user, 2026-11-05): its test data is a single 20-slice slab, where a few neurites
        # decide the score.
        "ID": ["cremi", "fafb", "hemibrain", "liconn", "snemi", "zebrafinch_j0126"],
        # OOD means out of distribution for micro-sam2 (user, 2026-11-05): humanneurons (H01) is not in its training
        # data, though SegNeuron trains on H01. It replaces cortex_human_stg, whose labels keep block-stitching seams.
        "OOD": ["humanneurons", "ionsem", "isbi2012", "microns_basil", "nisb", "synapseweb"],
    },
}

# Whether all panels of the interactive plot stay empty, for a figure with pending results.
INTERACTIVE_PLACEHOLDER = False
INTERACTIVE_METHODS = {
    # SAM and micro-sam correct with clicks only, as the other methods do.
    "SAM": ("sam", ("vit_l",), "points"),
    "SAM2": ("sam2", ("hvit_l",), ""),
    "SAM3": ("sam3", ("default",), ""),
    "nnInteractive": ("nninteractive", ("default",), ""),
    "micro-sam": ("micro-sam", ("vit_l_lm",), "points"),
    "micro-sam2": ("micro-sam2", ("hvit_l",), ""),
}

AUTOMATIC_ROWS = [["fluorescence_cells", "fluorescence_nuclei"], ["em_neurites"]]


def plot_automatic(region=None, compact=False):
    """Plot the automatic results of all modalities, with micro-sam and micro-sam2 in AIS and APG."""
    results = {}
    for name, (category, structure, _, metric, _) in PANELS.items():
        rows = fig2.load_modes(
            category, structure, PANEL_MODELS.get(name), path=RESULTS_CSV, methods=METHODS, metric=metric,
        )
        results[name] = fig2.one_mode(fig2.select_datasets(rows, PANEL_DATASETS[name]), fig2.FIGURE_MODES)
    panels = {
        name: (title, score_label, PANEL_METHODS[name]) for name, (_, _, title, _, score_label) in PANELS.items()
    }
    # The domain titles leave room for the values of the CREMI bars.
    return fig2.plot_automatic(
        results, AUTOMATIC_ROWS, panels, STARRED_METHODS, domain_offset=0.27, region=region,
        row_height=fig2.FIGURE_ROW_HEIGHT if compact else fig2.BAR_ROW_HEIGHT,
        grid=COMBINED_GRID if compact else None, title_line=compact, ranks=True,
        group_gap=fig2.COMBINED_GROUP_GAP if compact else fig2.GROUP_GAP,
        row_gap=fig2.COMBINED_ROW_GAP if compact else None,
    )


def plot_interactive(region=None, compact=False):
    """Plot the interactive results of all modalities over the correction iterations."""
    results = {}
    for name, (category, structure, _, metric, _) in PANELS.items():
        results[name] = fig2.load_interactive(
            category, structure, PANEL_DATASETS[name], path=INTERACTIVE_CSV, methods=INTERACTIVE_METHODS,
            metric=metric,
        ).dropna(subset=["score"])
    modalities = [name for row in AUTOMATIC_ROWS for name in row]
    titles = {name: title for name, (_, _, title, _, _) in PANELS.items()}
    score_labels = {name: score_label for name, (_, _, _, _, score_label) in PANELS.items()}
    # The EM panels are lowered, so that the legend does not cover micro-sam2 there.
    return fig2.plot_interactive_by_domain(
        results, modalities, titles, STARRED_METHODS, methods=INTERACTIVE_METHODS, score_labels=score_labels,
        placeholder=INTERACTIVE_PLACEHOLDER, legend_panel=("OOD", "fluorescence_nuclei"), legend_wide=True,
        room_below=[("ID", "em_neurites"), ("OOD", "em_neurites")], prompts_on_top=True,
        region=region, panel_height=fig2.COMBINED_PANEL_HEIGHT if compact else fig2.PANEL_HEIGHT,
        grid=COMBINED_GRID if compact else None, title_line=compact, shared_rows=True, show_titles=not compact,
        score_gap=0.32 if compact else None, row_gap=fig2.COMBINED_ROW_GAP if compact else None,
        band_color=fig2.COMBINED_BAND_COLOR if compact else fig2.BOX_BAND_COLOR,
    )


# The (dataset, sample, ids of the two objects, domain) of each volume. The pairs with the largest lead of micro-sam2
# were chosen from the candidates of 'select_pairs.py'.
INTERACTIVE_QUALITATIVE = {
    "fluorescence_cells": [
        ("pnas_arabidopsis", 33, (109, 138), "ID"),
        ("plantseg_ovules", 31, (63, 95), "OOD"),
    ],
    "fluorescence_nuclei": [
        ("embedseg_platy_nuclei", 1, (102, 119), "ID"),
        ("parhyale_regen", 2, (33, 36), "OOD"),
    ],
    "em_neurites": [
        ("snemi", 1, (39, 56), "ID"),
        ("isbi2012", 0, (62, 105), "OOD"),
    ],
}
# micro-sam has no EM model, so the EM column shows SAM.
TRACING_METHODS = {
    "fluorescence_cells": ["nnInteractive", "micro-sam2"],
    "fluorescence_nuclei": ["nnInteractive", "micro-sam2"],
    "em_neurites": ["nnInteractive", "micro-sam2"],
}
TRACING_PROMPTS = ["point", "box"]
PREDICTION_PATTERNS = {
    "micro-sam2": (
        "microsam2/predictions/*hvit_l/{dataset}_center/sample_{sample:05d}/interactive_segmentation_3d/"
        "start_with_{prompt}/without_masks/iteration0/*.tif"
    ),
    "micro-sam": (
        "microsam_v1/predictions/microsam_vol_vit_l_lm/{dataset}/{prompt}_center_points_mask_iou0.8_ext0.025/"
        "sample_{sample:05d}/iteration00.tif"
    ),
    "SAM": (
        "sam_vol/predictions/microsam_vol_vit_l/{dataset}/{prompt}_center_points_mask_iou0.8_ext0.025/"
        "sample_{sample:05d}/iteration00.tif"
    ),
    "nnInteractive": "nninteractive/predictions/nninteractive/{dataset}/{prompt}/sample_{sample:05d}/iteration00.tif",
}
DATASET_NAMES = {
    "pnas_arabidopsis": "PNAS Arabidopsis",
    "plantseg_ovules": "PlantSeg Ovules",
    "embedseg_platy_nuclei": "EmbedSeg Platynereis",
    "parhyale_regen": "Parhyale Regen",
    "snemi": "SNEMI3D",
    "isbi2012": "ISBI 2012",
    "embedseg_organoid": "EmbedSeg Organoid",
    "embedseg_mouse_skull": "EmbedSeg Mouse Skull",
    "blastospim": "BlastoSPIM",
    "synapseweb": "SynapseWeb",
}
# The layout, in inches.
TRACING_PROMPT_WIDTH = 0.07
TRACING_LABEL_WIDTH = 0.17  # left of the volume and the reference: their names, written vertically
TRACING_GAP = 0.02
TRACING_MIN_REFERENCE = 0.22  # the least height of the reference over the volume
TRACING_REFERENCE_MARGIN = 0.1  # the room around the reference, as a share of its render
# The datasets whose renders are cut around the objects of all methods.
FIT_ALL = {"parhyale_regen"}
# The datasets whose two objects stand far apart: the share of the gap between them that their renders close.
OBJECT_GAPS = {"plantseg_ovules": 0.6}
TRACING_ASPECT = 0.78  # the height to width ratio of the renders, to keep the figure within a page
# The room around the objects, as a share of the render size.
TRACING_PAD = 0.01
# The method names, in points: 'nnInteractive' is as wide as a render.
TRACING_NAME_SIZE = fig2.MAX_FONT_SIZE


def prediction_path(method, dataset, sample, prompt):
    pattern = os.path.join(INTERACTIVE_PREDICTIONS, PREDICTION_PATTERNS[method])
    paths = glob.glob(pattern.format(dataset=dataset, sample=sample, prompt=prompt))
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one prediction for '{pattern}', found {len(paths)}.")
    return paths[0]


def tracing_score(dataset, sample, objects, method, prompt):
    """Return the mean IoU of the two objects in the prediction after the first prompt."""
    labels = np.load(os.path.join(RENDERS, f"{dataset}_{sample}.npz"))["labels"]
    prediction = tifffile.imread(prediction_path(method, dataset, sample, prompt))
    ious = [
        np.logical_and(prediction == o, labels == o).sum() / np.logical_or(prediction == o, labels == o).sum()
        for o in objects
    ]
    return float(np.mean(ious))


def load_render(name):
    """Return a render of 'render_pairs.py' as RGBA in [0, 1]."""
    return np.asarray(Image.open(os.path.join(RENDERS, "pairs", f"{name}.png")).convert("RGBA")) / 255


def trim(images, pad=0.0, fit=None):
    """Cut the images to the box around what is drawn in 'fit', so that they keep one scale.

    Args:
        images: The RGBA images.
        pad: The margin around the box, as a share of its size.
        fit: The images that set the box. The default is 'images'.

    Returns:
        The cut images.
    """
    fit = images if fit is None else fit
    drawn = np.any([image[..., 3] > 0 for image in fit], axis=0)
    ys, xs = np.nonzero(drawn)
    margin = int(pad * max(np.ptp(ys), np.ptp(xs)))
    y0, x0 = max(ys.min() - margin, 0), max(xs.min() - margin, 0)
    return [image[y0:ys.max() + margin + 1, x0:xs.max() + margin + 1] for image in images]


def render_axes(fig, left, bottom, width, height, image, framed=True):
    """Draw an image centred in a box, in inches, and keep its aspect."""
    if framed:
        ax = fig2.framed_axes(fig, left, bottom, width, height)
    else:
        figure_height = fig2.height_of(fig)
        ax = fig.add_axes(
            [left / fig2.FIGURE_WIDTH, bottom / figure_height, width / fig2.FIGURE_WIDTH, height / figure_height]
        )
        ax.axis("off")
    ax.imshow(image, interpolation="antialiased")
    # Pad the shorter side of the image, so that the box keeps its size.
    rows, columns = image.shape[:2]
    scale = max(columns / width, rows / height)
    extra_x, extra_y = (width * scale - columns) / 2, (height * scale - rows) / 2
    ax.set_xlim(-0.5 - extra_x, columns - 0.5 + extra_x)
    ax.set_ylim(rows - 0.5 + extra_y, -0.5 - extra_y)
    return ax


def plot_interactive_qualitative(region=None, grid=None, show_titles=True, starred=()):
    """Plot a column per modality with a block per volume of INTERACTIVE_QUALITATIVE.

    A block has the volume over the reference render, and the renders of the methods (columns) per prompt (rows).
    The arguments are as in 'plots_for_figure2.plot_interactive_qualitative'. The method names in 'starred' get a
    star.
    """
    chosen = INTERACTIVE_QUALITATIVE
    header = fig2.ZOOM_HEADER if show_titles else fig2.ZOOM_HEADER_NO_TITLES
    n_columns, n_rows = len(chosen), max(len(blocks) for blocks in chosen.values())
    gap = TRACING_GAP
    grid = grid or (0, gap, fig2.ZOOM_BLOCK_GAP + gap)
    starts, column_width = fig2.column_grid(grid, n_columns)
    n_methods = max(len(methods) for methods in TRACING_METHODS.values())
    size = (column_width - TRACING_LABEL_WIDTH - TRACING_PROMPT_WIDTH - n_methods * gap) / (n_methods + 1)
    row_height = 2 * TRACING_ASPECT * size + gap
    domain_changes = [
        k for k in range(1, n_rows)
        if any(blocks[k][3] != blocks[k - 1][3] for blocks in chosen.values() if len(blocks) > k)
    ]
    height = (
        header + fig2.ZOOM_CAPTION + n_rows * row_height + (n_rows - 1) * fig2.ZOOM_ROW_GAP
        + len(domain_changes) * (fig2.ZOOM_DOMAIN_GAP - fig2.ZOOM_ROW_GAP) + 0.02
    )
    fig = fig2.new_figure(height, region)

    for c, (modality, blocks) in enumerate(chosen.items()):
        left = starts[c]
        methods = TRACING_METHODS[modality]
        if show_titles:
            fig2.draw_title(fig, left, left + column_width, height - 0.17, PANELS[modality][2])
        top = height - header - fig2.ZOOM_CAPTION
        methods_left = left + TRACING_LABEL_WIDTH + size + gap + TRACING_PROMPT_WIDTH
        for k, (method, name) in enumerate(zip(methods, fig2.names_of(methods, set(starred)))):
            x = methods_left + k * (size + gap) + size / 2
            fig.text(
                x / fig2.FIGURE_WIDTH, (top + 0.05) / height, name, ha="center", va="baseline",
                fontsize=TRACING_NAME_SIZE, color=TEXT_COLOR,
            )
        for r, (dataset, sample, objects, domain) in enumerate(blocks):
            top -= (fig2.ZOOM_DOMAIN_GAP if r in domain_changes else fig2.ZOOM_ROW_GAP) * (r > 0)
            plot_tracing_block(fig, left, top - row_height, size, modality, dataset, sample, objects, domain)
            top -= row_height
    return fig


def wrapped(name, width):
    """Split a name into lines of at most 'width' characters, between words."""
    lines = []
    for word in name.split():
        if lines and len(lines[-1]) + 1 + len(word) <= width:
            lines[-1] += " " + word
        else:
            lines.append(word)
    return "\n".join(lines)


def closer(images, share):
    """Move what stands right of the widest empty gap between the objects of renders of one size to the left, so
    that 'share' of the gap closes. The gap is the widest run of columns that no render draws in."""
    drawn = np.any([image[..., 3] > 0 for image in images], axis=(0, 1))
    columns = np.nonzero(drawn)[0]
    runs = np.diff(columns)
    k = int(np.argmax(runs))
    start, width = columns[k] + 1, runs[k] - 1
    shift = int(share * width)
    moved = []
    for image in images:
        out = image.copy()
        out[:, start + width - shift:-shift or None] = image[:, start + width:]
        out[:, -shift:] = 0
        moved.append(out)
    return moved


def plot_tracing_block(fig, left, bottom, size, modality, dataset, sample, objects, domain):
    """Plot the block of one volume of the interactive qualitative plot. The point renders are over the box renders."""
    name = f"{dataset}_{sample}"
    methods = TRACING_METHODS[modality]
    height = fig2.height_of(fig)
    gap = TRACING_GAP
    x = left + TRACING_LABEL_WIDTH

    # A small reference over the volume, which takes the height it needs to fill the column width. The dataset name
    # stands left of the block, written vertically.
    volume = trim([load_render(f"{name}_pyramid")])[0]
    tall = TRACING_ASPECT * size
    volume_height = min(size * volume.shape[0] / volume.shape[1], 2 * tall - TRACING_MIN_REFERENCE)
    reference_height = 2 * tall - volume_height
    render_axes(fig, x, bottom, size, volume_height, volume, framed=False)
    in_domain = domain == "ID"
    fig.text(
        (x - 0.02) / fig2.FIGURE_WIDTH, (bottom + (volume_height + gap + reference_height) / 2) / height,
        wrapped(DATASET_NAMES[dataset], 12), rotation=90, ha="right", va="center", ma="center",
        fontsize=fig2.MAX_FONT_SIZE,
        color=TEXT_COLOR, linespacing=1.0, style="italic" if in_domain else "normal",
        fontweight="normal" if in_domain else "bold",
    )

    keys = [("reference", None)] + [(method, prompt) for prompt in TRACING_PROMPTS for method in methods]
    images = {
        (method, prompt): load_render(f"{name}_both_{method}" + (f"_{prompt}" if prompt else ""))
        for method, prompt in keys
    }
    if dataset in OBJECT_GAPS:
        images = dict(zip(keys, closer([images[key] for key in keys], OBJECT_GAPS[dataset])))
    # Cut all renders around the reference and micro-sam2, so that the objects fill the renders, or around all renders
    # where another method would run out of its render.
    fitted = keys if dataset in FIT_ALL else [("reference", None)] + [("micro-sam2", p) for p in TRACING_PROMPTS]
    fit = [images[key] for key in fitted]
    images = dict(zip(keys, trim([images[key] for key in keys], TRACING_PAD, fit)))
    # The reference, unframed and a bit smaller than its room, stands right over the crop of the volume.
    reference = images["reference", None]
    margin = int(TRACING_REFERENCE_MARGIN * max(reference.shape[:2]))
    reference = np.pad(reference, ((margin, 0), (margin, margin), (0, 0)))
    shown = min(reference_height, size * reference.shape[0] / reference.shape[1])
    render_axes(fig, x, bottom + volume_height, size, shown, reference, framed=False)
    methods_left = x + size + gap + TRACING_PROMPT_WIDTH
    for row, prompt in enumerate(TRACING_PROMPTS):
        y = bottom + (1 - row) * (tall + gap)
        fig.text(
            (methods_left - 0.02) / fig2.FIGURE_WIDTH, (y + tall / 2) / height, fig2.PROMPT_NAMES[prompt],
            rotation=90, ha="right", va="center", fontsize=fig2.MAX_FONT_SIZE, color=TEXT_COLOR,
        )
        for col, method in enumerate(methods):
            ax = render_axes(fig, methods_left + col * (size + gap), y, size, tall, images[method, prompt])
            score = tracing_score(dataset, sample, objects, method, prompt)
            fig2.draw_score(ax, score)


# The (dataset, sample, modality, domain) of each volume, chosen by the lead of micro-sam2 APG over the best baseline.
# Most PlantSeg Ovules volumes have labels in only a part; the chosen one has labels over a large part.
AUTOMATIC_QUALITATIVE = [
    ("embedseg_organoid", 0, "fluorescence_cells", "ID"),
    ("plantseg_ovules", 107, "fluorescence_cells", "OOD"),
    ("embedseg_mouse_skull", 1, "fluorescence_nuclei", "ID"),
    ("blastospim", 36, "fluorescence_nuclei", "OOD"),
    ("snemi", 3, "em_neurites", "ID"),
    ("synapseweb", 1, "em_neurites", "OOD"),
]
# SegNeuron replaces CellposeSAM for the EM neurites.
QUALITATIVE_BASELINES = {"em_neurites": "SegNeuron"}
PREDICTION_NAMES = {"CellposeSAM": "cellposesam", "SegNeuron": "segneuron", "micro-sam2": "microsam2_apg"}
# The layout, in inches.
QUALITATIVE_GAP = 0.03
QUALITATIVE_HEADER = 0.15
# The height to width ratio of the panels.
QUALITATIVE_ASPECT = 0.6
# The objects of a dataset that crowd their panel are drawn smaller, as a share of the panel size.
OBJECT_SCALES = {"embedseg_organoid": 0.8}
# The volume render takes this share of the width of its panel, and its planes the rest.
PLANE_SPLIT = 0.6
PLANE_GAP = 0.02
# The width of the reference outlines in the planes, in inches (0.6 pt).
PLANE_OUTLINE_WIDTH = 0.6 / 72
# Thinner outlines for the volumes with dense objects.
PLANE_OUTLINE_WIDTHS = {"embedseg_mouse_skull": 0.45 / 72, "snemi": 0.3 / 72, "synapseweb": 0.3 / 72}
# The z/xy voxel ratios of the volumes, as in their renders (see 'select_tracing.z_scale_of').
PLANE_Z_SCALES = {
    "embedseg_organoid": 6.0, "plantseg_ovules": 0.235 / 0.15, "embedseg_mouse_skull": 4.0, "blastospim": 10.0,
    "snemi": 5.0, "synapseweb": 10.0,
}
# The volumes cut to the box of their reference objects, with CROP_MARGIN around it, as in 'render_methods.py'.
CROPPED = {"plantseg_ovules"}
CROP_MARGIN = 0.1
# The direction of the long axis of the objects of a volume, in degrees from the horizontal, so that they fill the
# wide panels: along the panel diagonal, or flat.
PANEL_DIAGONAL = float(np.degrees(np.arctan(QUALITATIVE_ASPECT)))
OBJECT_ANGLES = {
    "plantseg_ovules": PANEL_DIAGONAL, "embedseg_mouse_skull": PANEL_DIAGONAL, "snemi": 0.0, "synapseweb": 0.0,
}
# The room around the objects in their panels, as a share of the render size.
QUALITATIVE_PAD = 0.0


def automatic_score(dataset, sample, method, metric):
    """Return the score of a prediction as the evaluation computes it.

    The mSA uses the reference with its objects split into connected components.
    """
    folder = os.path.join(QUALITATIVE_PREDICTIONS, dataset)
    labels = tifffile.imread(os.path.join(folder, f"labels_{sample}.tif"))
    prediction = tifffile.imread(os.path.join(folder, f"{PREDICTION_NAMES[method]}_{sample}.tif"))
    if metric == "cremi":
        return float(cremi_score(prediction, labels)[3])
    return float(mean_segmentation_accuracy(prediction, label(labels)))


def centered_windows(images, shape):
    """Cut the images to windows of 'shape' around the center of what is drawn in any of them.

    The images get transparent padding where a window reaches beyond them.
    """
    drawn = np.any([image[..., 3] > 0 for image in images], axis=0)
    ys, xs = np.nonzero(drawn)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    height, width = shape
    pad = max(height, width)
    windows = []
    for image in images:
        padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)))
        y0, x0 = cy - height // 2 + pad, cx - width // 2 + pad
        windows.append(padded[y0:y0 + height, x0:x0 + width])
    return windows


def load_method_render(dataset, sample, name):
    """Return a render of 'render_methods.py' as RGBA in [0, 1]."""
    path = os.path.join(RENDERS, "methods", f"{dataset}_{sample}_{name}.png")
    return np.asarray(Image.open(path).convert("RGBA")) / 255


def plot_automatic_qualitative(region=None):
    """Plot a column per modality, with the in-domain volume left of the out-of-domain volume.

    The rows are the volume, the reference objects, and the objects of the baseline and of micro-sam2.
    """
    chosen = AUTOMATIC_QUALITATIVE
    modalities = list(dict.fromkeys(modality for _, _, modality, _ in chosen))
    starts, column_width = fig2.column_grid(COMBINED_IMAGE_GRID, len(modalities))
    width = (column_width - QUALITATIVE_GAP) / 2
    size = QUALITATIVE_ASPECT * width
    rows = [None, "reference", "baseline", "micro-sam2"]
    height = QUALITATIVE_HEADER + len(rows) * size + (len(rows) - 1) * QUALITATIVE_GAP
    fig = fig2.new_figure(height, region)

    def bottom_of(row):
        return height - QUALITATIVE_HEADER - (row + 1) * size - row * QUALITATIVE_GAP

    for dataset, sample, modality, domain in chosen:
        left = starts[modalities.index(modality)] + (domain == "OOD") * (width + QUALITATIVE_GAP)
        bottoms = [bottom_of(row) for row in range(len(rows))]
        plot_volume_column(fig, left, bottoms, width, size, dataset, sample, modality, domain)
    names = [None, fig2.REFERENCE_NAME] + fig2.names_of(["CellposeSAM", "micro-sam2"], set())
    for row, name in enumerate(names):
        if name is None:
            continue
        fig.text(
            (starts[0] - 0.05) / fig2.FIGURE_WIDTH, (bottom_of(row) + size / 2) / height, name, ha="right",
            va="center", rotation=90, fontsize=fig2.MAX_FONT_SIZE, color=TEXT_COLOR,
        )
    return fig


def long_axis(image):
    """Return the direction of the long axis of what is drawn in an RGBA image, in degrees from the horizontal."""
    ys, xs = np.nonzero(image[..., 3] > 0.5)
    values, vectors = np.linalg.eigh(np.cov(np.stack([xs, -ys])))
    x, y = vectors[:, np.argmax(values)]
    return float(np.degrees(np.arctan2(y, x)))


def turned(images, angle):
    """Turn RGBA images of one size alike, so that the long axis of the first one points 'angle' degrees from the
    horizontal."""
    # The long axis has no sign, so the smaller of the two turns aligns it.
    turn = (angle - long_axis(images[0]) + 90) % 180 - 90
    return [np.clip(ndimage.rotate(image, turn, axes=(1, 0), reshape=True, order=1), 0, 1) for image in images]


def planes_of(dataset, sample):
    """Return the image and the reference labels of a volume, cut as for its renders (see 'render_methods.py')."""
    folder = os.path.join(QUALITATIVE_PREDICTIONS, dataset)
    image = tifffile.imread(os.path.join(folder, f"image_{sample}.tif")).astype("float32")
    labels = tifffile.imread(os.path.join(folder, f"labels_{sample}.tif"))
    if dataset in CROPPED:
        _, ys, xs = np.nonzero(labels)
        my, mx = int(CROP_MARGIN * np.ptp(ys)), int(CROP_MARGIN * np.ptp(xs))
        window = np.s_[:, max(ys.min() - my, 0):ys.max() + my + 1, max(xs.min() - mx, 0):xs.max() + mx + 1]
        image, labels = image[window], labels[window]
    return image, labels


def plane_overlay(image, labels, thickness):
    """Return a plane as RGB, with the outlines of the reference objects in the reference color of Figure 2.

    The outlines are grown to 'thickness', the number of pixels along the rows and the columns.
    """
    low, high = np.percentile(image, (1, 99.7))
    gray = np.clip((image - low) / max(high - low, 1e-6), 0, 1)
    rgb = np.stack([gray] * 3, axis=-1)
    outlines = ndimage.binary_dilation(find_boundaries(labels, mode="inner"), np.ones(thickness, dtype=bool))
    rgb[outlines] = colors.to_rgb(fig2.REFERENCE_COLOR)
    return rgb


def plot_planes(fig, left, bottom, width, height, dataset, sample, volume=None):
    """Plot the xy, the xz and the yz plane through the most objects of a volume from top to bottom, in their true
    aspect, in a column at 'left' and 'bottom', in inches. 'volume' is the (image, labels, z/xy voxel ratio) of a
    volume that is not one of QUALITATIVE_PREDICTIONS."""
    if volume is None:
        image, labels = planes_of(dataset, sample)
        z_scale = PLANE_Z_SCALES[dataset]
    else:
        image, labels, z_scale = volume
    # Per axis, the plane in the middle half with the most labelled voxels.
    point = []
    for axis in range(3):
        counts = (labels > 0).sum(axis=tuple(a for a in range(3) if a != axis))
        lo, hi = len(counts) // 4, max(len(counts) * 3 // 4, len(counts) // 4 + 1)
        point.append(lo + int(np.argmax(counts[lo:hi])))
    z, y, x = point
    planes = [
        (image[z], labels[z], 1.0), (image[:, y], labels[:, y], z_scale), (image[:, :, x], labels[:, :, x], z_scale),
    ]
    gap = PLANE_GAP
    # The planes share one scale: the widest one fills the column, unless they are too high for it.
    stacked = sum(p.shape[0] * a for p, _, a in planes)
    scale = min(width / max(p.shape[1] for p, _, _ in planes), (height - 2 * gap) / stacked)
    total = stacked * scale + 2 * gap
    top = bottom + (height + total) / 2
    for plane, plane_labels, aspect in planes:
        w, h = plane.shape[1] * scale, plane.shape[0] * aspect * scale
        top -= h
        ax = fig2.framed_axes(fig, left + (width - w) / 2, top, w, h)
        outline_width = PLANE_OUTLINE_WIDTHS.get(dataset, PLANE_OUTLINE_WIDTH)
        thickness = [max(int(round(outline_width / size)), 1) for size in (aspect * scale, scale)]
        ax.imshow(plane_overlay(plane, plane_labels, thickness), interpolation="none", aspect="auto")
        top -= gap


def plot_volume_column(fig, left, bottoms, width, size, dataset, sample, modality, domain):
    """Plot the panels of one volume from top to bottom, at 'left' and 'bottoms', in inches.

    The objects of the methods keep the scale of the reference objects.
    """
    volume = load_method_render(dataset, sample, "volume")
    reference = load_method_render(dataset, sample, "reference")
    # The volume render, with its xy, xz and yz planes right of it.
    volume_width = PLANE_SPLIT * width
    render_axes(fig, left, bottoms[0], volume_width, size, trim([volume], TRACING_PAD)[0], framed=False)
    planes_left = left + volume_width + QUALITATIVE_GAP
    plot_planes(fig, planes_left, bottoms[0], width - volume_width - QUALITATIVE_GAP, size, dataset, sample)
    in_domain = domain == "ID"
    fig.text(
        (left + width / 2) / fig2.FIGURE_WIDTH, (bottoms[0] + size + 0.04) / fig2.height_of(fig),
        DATASET_NAMES[dataset], ha="center", va="baseline", fontsize=fig2.MAX_FONT_SIZE, color=TEXT_COLOR,
        style="italic" if in_domain else "normal", fontweight="normal" if in_domain else "bold",
    )

    methods = [QUALITATIVE_BASELINES.get(modality, "CellposeSAM"), "micro-sam2"]
    renders = [load_method_render(dataset, sample, PREDICTION_NAMES[method]) for method in methods]
    if dataset in OBJECT_ANGLES:
        reference, *renders = turned([reference] + renders, OBJECT_ANGLES[dataset])
    reference = trim([reference], QUALITATIVE_PAD)[0]
    margin = (1 / OBJECT_SCALES.get(dataset, 1.0) - 1) / 2
    pad_y, pad_x = int(margin * reference.shape[0]), int(margin * reference.shape[1])
    reference = np.pad(reference, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)))
    images = centered_windows(renders, reference.shape[:2])

    render_axes(fig, left, bottoms[1], width, size, reference)
    metric = PANELS[modality][3]
    for method, image, bottom in zip(methods, images, bottoms[2:]):
        ax = render_axes(fig, left, bottom, width, size, image)
        # The row names name CellposeSAM, so another baseline is named in its panel.
        if method not in ("CellposeSAM", "micro-sam2"):
            ax.text(
                0.04, 0.96, method, transform=ax.transAxes, ha="left", va="top",
                fontsize=fig2.SMALL_FONT_SIZE, color=TEXT_COLOR,
                bbox={"boxstyle": "square,pad=0.1", "facecolor": "white", "alpha": 0.8, "linewidth": 0},
            )
        fig2.draw_score(
            ax, automatic_score(dataset, sample, method, metric), fontsize=fig2.SMALL_FONT_SIZE
        )


# The plots in their order in the combined figure, with their kind and style.
PLOTS = {
    "automatic_3d": ("automatic", "quantitative"),
    "automatic_3d_qualitative": ("automatic", "qualitative"),
    "interactive_3d": ("interactive", "quantitative"),
    "interactive_3d_qualitative": ("interactive", "qualitative"),
}
COMBINED_NAME = "figure_3"
# The resolution of the renders in the combined figure and of its png: the renders are made large enough for it.
COMBINED_DPI = 1200
# The (room left, room right, gap) of the modality columns, in inches, with room for the CREMI values and label.
COMBINED_GRID = (0.62, 0.45, 0.4)
IMAGE_WIDENING = (COMBINED_GRID[2] - fig2.IMAGE_GAP) / 2
COMBINED_IMAGE_GRID = (COMBINED_GRID[0] - IMAGE_WIDENING, COMBINED_GRID[1] - IMAGE_WIDENING, fig2.IMAGE_GAP)
# The interactive qualitative plot has a smaller gap between its columns, with the same column centers.
TRACING_COLUMN_GAP = 0.05
TRACING_WIDENING = (COMBINED_GRID[2] - TRACING_COLUMN_GAP) / 2
COMBINED_TRACING_GRID = (COMBINED_GRID[0] - TRACING_WIDENING, COMBINED_GRID[1] - TRACING_WIDENING, TRACING_COLUMN_GAP)


def make_plot(name, region=None, compact=False):
    """Make one plot, as its own figure or in a region of the combined figure.

    With 'compact', the plots use the sizes of the combined figure and the column grid COMBINED_GRID.
    """
    if name == "automatic_3d":
        return plot_automatic(region, compact)
    if name == "interactive_3d":
        return plot_interactive(region, compact)
    if name == "automatic_3d_qualitative":
        return plot_automatic_qualitative(region=region)
    return plot_interactive_qualitative(
        region=region, grid=COMBINED_TRACING_GRID if compact else None, show_titles=not compact
    )


def main():
    names = list(PLOTS) + [COMBINED_NAME]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=names, default=names, help="Make only these plots.")
    args = parser.parse_args()
    set_style()
    fig2.check_style()
    fig2.OUTPUT_DIR = OUTPUT_DIR
    for name, (kind, style) in PLOTS.items():
        if name in args.only:
            fig2.save_figure(make_plot(name), name, kind, style, dpi=fig2.IMAGE_DPI)
    if COMBINED_NAME in args.only:
        fig2.save_combined(*fig2.plot_figure(PLOTS, make_plot), name=COMBINED_NAME, dpi=COMBINED_DPI)


if __name__ == "__main__":
    main()
