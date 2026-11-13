"""Plots for Figure 5 of the micro-sam2 paper: automatic and interactive tracking. This script makes all of the figure,
from the renders to the combined figure.

The figure compares tracking with micro-sam2 (hvit_l) to the baselines. The script writes these plots to
'figures/figure5/<automatic|interactive>/<quantitative|qualitative>':
- 'automatic_tracking' (panel a): the mean TRA, SEG and LNK of each method over the datasets, for 2D+t and 3D+t, in
  domain and out of domain of micro-sam2. Each method segments every frame and Trackastra links the objects
  ('general_2d_w_SAM2_features' for 2D+t, 'ctc' for 3D+t).
- 'automatic_tracking_qualitative' (panel b): one movie per column of panel a, as space-time renders of the tracks of
  CellposeSAM and micro-sam2 with their errors.
- 'interactive_tracking' (panel c): the mean TRA, SEG and LNK over the correction iterations of the simulated user,
  with a point or a box as the first prompt.
- 'interactive_tracking_qualitative' (panel d): one dividing lineage per domain in 3d, with time going up: the movie,
  the reference lineage and the lineage of micro-sam and micro-sam2 after the first point or box prompt.
- 'beke_tracking' (panel e): the tracked nuclei of the Beke time-lapse of Platynereis.
- 'figure_5' (in 'figures/final'): panels a to e in one figure.

Usage:

    python plots_for_figure5.py  # all plots and the combined figure
    python plots_for_figure5.py --only figure_5
    python plots_for_figure5.py --render lineage_scores lineages spacetime floor beke  # the renders, on a GPU node

The inputs are the central tracking tables, written by '--collect' of the two tracking evaluation scripts, the
tracking predictions in TRACKING_ROOT, and the Beke tracks of 'scripts/user_study/v2/huisken_lab/huisken_tracks_h5'.
The renders go to TRACKING_ROOT ('lineage_renders', 'automatic_renders', 'lineage_search') and to
'figures/figure5/beke'.
"""

import os
import sys
import glob
import json
import argparse
import colorsys

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import plots_for_figure2 as fig2
from plot_style import MAX_FONT_SIZE, METHOD_COLORS, TEXT_COLOR, set_style

ROOT = os.path.dirname(os.path.abspath(__file__))
AUTOMATIC_CSV = os.path.join(ROOT, "figures", "cells_automatic_tracking_results.csv")
INTERACTIVE_CSV = os.path.join(ROOT, "figures", "cells_interactive_tracking_results.csv")
OUTPUT_DIR = os.path.join(ROOT, "figures", "figure5")

# DET is left out: it stays within 0.02 of TRA everywhere.
SCORE_LABELS = {
    "tra": "Tracking accuracy (TRA) ↑",
    "seg": "Segmentation accuracy (SEG) ↑",
}

# The ('method', 'model') values of each automatic method in the table, and the Trackastra model per dimension.
AUTOMATIC_METHODS = {
    "micro-sam": ("micro-sam", "vit_l_lm"),
    "CellposeSAM": ("cellpose", "cpsam"),
    "micro-sam2": ("micro-sam2", "hvit_l"),
}
# Trackastra links with the SAM2 features in 2D+t; Trackastra has no such model for 3D+t.
TRACKING_MODELS = {2: "general_2d_w_SAM2_features", 3: "ctc"}
PANELS = {
    "tracking_2d": (2, "2D+t"),
    "tracking_3d": (3, "3D+t"),
}
# No 2D CTC data: both models trained on its labelled movies. The in-domain movies come from training sources
# but were held out.
AUTOMATIC_DATASETS = {
    2: {
        "ID": ["TOIAM", "YeaZ", "CellACDC", "DynamicNuclearNet"],
        "OOD": ["ALFI", "Cardioblast", "EColi", "WingDisc"],
    },
    # micro-sam2 trained on none of the 3D+t CTC sets; six of the nine are shown.
    3: {
        "OOD": [
            "Fluo-C3DH-A549-SIM", "Fluo-C3DH-H157", "Fluo-C3DL-MDA231", "Fluo-N3DH-CHO", "Fluo-N3DH-SIM+",
            "Fluo-N3DL-TRIC",
        ],
    },
}
# No method gets a mark: in domain and out of domain refer to the training data of micro-sam2.
MARKS = {}

# The ('method', 'model') values of each interactive method; stock SAM2 is back in.
INTERACTIVE_METHODS = {
    "SAM2": ("sam2", "hvit_l"),
    "micro-sam": ("micro-sam", "vit_l_lm"),
    "micro-sam2": ("micro-sam2", "hvit_l"),
}
# The interactive datasets: the 2D+t datasets of the automatic plot, in the same order.
INTERACTIVE_DATASETS = {
    "toiam": ("TOIAM", "ID"),
    "yeaz": ("YeaZ", "ID"),
    "cellacdc": ("CellACDC", "ID"),
    "dynamicnuclearnet": ("DynamicNuclearNet", "ID"),
    "alfi": ("ALFI", "OOD"),
    "cardioblast": ("Cardioblast", "OOD"),
    "ecoli": ("E. coli", "OOD"),
    "wingdisc": ("Wing disc", "OOD"),
}
N_ITERATIONS = 5


def load_automatic(metric):
    """Load a score of each automatic method per dataset, as the long tables of 'plots_for_figure2'."""
    df = pd.read_csv(AUTOMATIC_CSV)
    results = {}
    for name, (ndim, _) in PANELS.items():
        rows = df[(df["ndim"] == ndim) & (df["tracking_model"] == TRACKING_MODELS[ndim])]
        domains = {dataset: domain for domain, names in AUTOMATIC_DATASETS[ndim].items() for dataset in names}
        rows = rows[rows["dataset"].isin(domains)].assign(domain=lambda r: r["dataset"].map(domains))
        parts = []
        for method, (table_method, model) in AUTOMATIC_METHODS.items():
            part = rows[(rows["method"] == table_method) & (rows["model"] == model)]
            # Only micro-sam2 shows its two modes; micro-sam shows AIS, its default.
            mode = part["mode"].fillna("") if method == "micro-sam2" else ""
            parts.append(pd.DataFrame({
                "dataset": part["dataset"], "domain": part["domain"], "score": part[metric], "method": method,
                "mode": mode,
            }))
        results[name] = pd.concat(parts, ignore_index=True)
    return results


# The scores of the automatic plot (rows) and of the interactive plot (columns).
AUTOMATIC_METRICS = ["tra", "seg", "lnk"]
# The layout of the automatic plot, in inches: one row per score and one column per group of datasets.
AUTOMATIC_GROUPS = [("tracking_2d", "ID"), ("tracking_2d", "OOD"), ("tracking_3d", "OOD")]
# The score names stand right of the rows, so the rows can be close.
AUTOMATIC_TOP = 0.36
AUTOMATIC_ROW_GAP = 0.1
AUTOMATIC_BOTTOM = 0.2
AUTOMATIC_SCORE_WIDTH = 0.16
AUTOMATIC_SHRINK = 0.86


def plot_automatic(region=None):
    """The mean TRA, SEG and LNK of each method, one row per score, for 2D+t in domain and out of domain and for
    3D+t out of domain (there is no 3D+t data in domain of micro-sam2), so that the grid has no empty panel.
    """
    methods = list(AUTOMATIC_METHODS)
    n_bars = {"micro-sam2": 2}
    y = fig2.method_positions(methods, n_bars, fig2.GROUP_GAP)
    half = np.array([fig2.BAR_HEIGHT * n_bars.get(method, 1) / 2 for method in methods])
    panel_height = ((y + half).max() - (y - half).min() + 0.48) * fig2.BAR_ROW_HEIGHT * AUTOMATIC_SHRINK
    metrics = AUTOMATIC_METRICS
    height = AUTOMATIC_TOP + len(metrics) * panel_height + (len(metrics) - 1) * AUTOMATIC_ROW_GAP + AUTOMATIC_BOTTOM
    fig = fig2.new_figure(height, region)
    grid = (fig2.NAME_WIDTH, fig2.RIGHT_MARGIN + AUTOMATIC_SCORE_WIDTH, fig2.COLUMN_GAP)
    lefts, width = fig2.column_grid(grid, len(AUTOMATIC_GROUPS))
    axes = {}
    loaded = {metric: load_automatic(metric) for metric in metrics}
    cells_of = {
        metric: [loaded[metric][dim][loaded[metric][dim]["domain"] == domain] for dim, domain in AUTOMATIC_GROUPS]
        for metric in metrics
    }
    # All panels share one value axis, since only the bottom row labels it.
    highest = max(
        cell.groupby(["method", "mode"])["score"].mean().max() for cells in cells_of.values() for cell in cells
    )
    value_max = np.ceil(round((highest + fig2.VALUE_ROOM + fig2.RANK_ROOM) / fig2.VALUE_STEP, 6)) * fig2.VALUE_STEP
    tick_step = next(
        (step for step in fig2.TICK_STEPS if width * step / value_max >= fig2.TICK_WIDTH), fig2.TICK_STEPS[-1]
    )
    for i, metric in enumerate(metrics):
        bottom = height - AUTOMATIC_TOP - (i + 1) * panel_height - i * AUTOMATIC_ROW_GAP
        cells = cells_of[metric]
        for k, ((dim, domain), cell) in enumerate(zip(AUTOMATIC_GROUPS, cells)):
            ax = fig.add_axes([
                lefts[k] / fig2.FIGURE_WIDTH, bottom / height, width / fig2.FIGURE_WIDTH, panel_height / height,
            ])
            fig2.plot_domain(ax, cell, domain, methods, value_max, list(y), tick_step, ranks=True)
            ax.set_title("")
            ax.xaxis.set_major_formatter(fig2.tick_formatter(tick_step, zero=True))
            ax.set_ylim((y - half).min() - 0.24, (y + half).max() + 0.24)
            # Only the bottom row labels its value ticks; the rows share the value range.
            if i < len(metrics) - 1:
                ax.tick_params(axis="x", labelbottom=False)
            if k == 0:
                ax.set_yticks(y)
                ax.set_yticklabels(fig2.names_of(methods, {}), fontsize=MAX_FONT_SIZE)
            else:
                ax.set_yticks(y)
                ax.tick_params(axis="y", labelleft=False)
            axes[metric, k] = ax
        # The score of the row stands vertically right of its panels.
        right = lefts[-1] + width + 0.05
        fig.text(
            right / fig2.FIGURE_WIDTH, (bottom + panel_height / 2) / height, f"{metric.upper()} ↑", ha="left",
            va="center", rotation=270, fontsize=MAX_FONT_SIZE, fontweight="bold", color=TEXT_COLOR
        )

    # The domain over each column, in the style of the dataset names, and the dimension over its columns.
    top = height - AUTOMATIC_TOP + 0.01
    for k, (dim, domain) in enumerate(AUTOMATIC_GROUPS):
        fig.text(
            (lefts[k] + width / 2) / fig2.FIGURE_WIDTH, top / height, fig2.DOMAIN_TITLES[domain], ha="center",
            va="bottom", fontsize=MAX_FONT_SIZE, color=TEXT_COLOR, **fig2.DOMAIN_STYLES[domain]
        )
    for dim, (_, title) in PANELS.items():
        columns = [k for k, (d, _) in enumerate(AUTOMATIC_GROUPS) if d == dim]
        fig2.draw_title(fig, lefts[columns[0]], lefts[columns[-1]] + width, top + 0.15, title)
    return fig


# The 2D+t datasets with the names of the interactive runs.
INTERACTIVE_PLOT_DATASETS = {
    "ID": ["toiam", "yeaz", "cellacdc", "dynamicnuclearnet"],
    "OOD": ["alfi", "cardioblast", "ecoli", "wingdisc"],
}
# The first prompt of each half of a panel and the corrections that follow it: points after a point, boxes after a box.
INTERACTIVE_SETTINGS = {"point": "points", "box": "box"}


def load_interactive(metric):
    """Load a score of each interactive method per dataset, first prompt and correction iteration, as the long tables
    of 'plots_for_figure2.plot_interactive_by_domain'.
    """
    df = pd.read_csv(INTERACTIVE_CSV)
    domains = {dataset: domain for domain, names in INTERACTIVE_PLOT_DATASETS.items() for dataset in names}
    df = df[df["dataset"].isin(domains)]
    df = df[[INTERACTIVE_SETTINGS.get(p) == c for p, c in zip(df["prompt"], df["correction"])]]
    parts = []
    for method, (table_method, model) in INTERACTIVE_METHODS.items():
        part = df[(df["method"] == table_method) & (df["model"] == model)]
        parts.append(pd.DataFrame({
            "dataset": part["dataset"], "domain": part["dataset"].map(domains), "prompt": part["prompt"],
            "iteration": part["iteration"], "score": part[metric], "method": method,
        }))
    return pd.concat(parts, ignore_index=True)


# The layout of the interactive plot, in inches.
INTERACTIVE_PANEL_HEIGHT = 0.85
INTERACTIVE_GRID = (0.55, fig2.RIGHT_MARGIN + fig2.INTERACTIVE_DOMAIN_WIDTH, 0.35)


def plot_interactive(region=None):
    """The mean TRA, SEG and LNK over the correction iterations, as Figure 2c: one column per score and one row per
    domain, each panel with the point runs left and the box runs right, averaged over the datasets of the domain.
    """
    metrics = AUTOMATIC_METRICS
    results = {metric: load_interactive(metric) for metric in metrics}
    titles = {metric: metric.upper() for metric in metrics}
    # One score label for all columns, since the column titles name the scores; each panel keeps its own value range.
    score_labels = {metric: "Mean score ↑" for metric in metrics}
    fig = fig2.plot_interactive_by_domain(
        results, metrics, titles, starred={}, methods=list(INTERACTIVE_METHODS), score_labels=score_labels,
        legend_panel=("OOD", metrics[0]), legend_wide=True, prompts_on_top=True, grid=INTERACTIVE_GRID,
        x_label=fig2.ITERATION_LABEL, region=region, panel_height=INTERACTIVE_PANEL_HEIGHT,
    )
    # The scores end at 1: an axis that the rounding took above 1 stops just above it, without ticks above 1.
    for ax in fig.axes:
        low, high = ax.get_ylim()
        if high > 1.0:
            ax.set_yticks([tick for tick in ax.get_yticks() if low <= tick <= 1.0 + 1e-6])
            ax.set_ylim(low, 1.0 + 0.02 * (1.0 - low))
    return fig


# One dividing lineage per domain that micro-sam2 splits and micro-sam does not. Final: do not change.
INTERACTIVE_QUALITATIVE = [
    ("dynamicnuclearnet", "3T3_nuc_s2_p1_1c", 6, "ID"),
    # Lineage 58 lies inside the field of view; lineage 68 is cut by its edge.
    ("wingdisc", "Ecd_20161213_F0", 58, "OOD"),
]
TRACKING_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/tracking"
# The renders of 'render_interactive_lineage_3d.py' and the lineage scores of 'lineage_search/find_lineages_both.py'.
LINEAGE_RENDERS = os.path.join(TRACKING_ROOT, "lineage_renders")
LINEAGE_SCORES = os.path.join(TRACKING_ROOT, "lineage_search")
LINEAGE_METHODS = {"micro-sam": "microsam", "micro-sam2": "microsam2"}
LINEAGE_RUNS = {"micro-sam": "s1", "micro-sam2": "s2"}
LINEAGE_PROMPTS = ["point", "box"]
# The mother and the daughters; blue instead of green, for red-green colour blindness.
LINEAGE_COLORS = {"Mother": "#ffd91a", "Daughters": ("#0072b2", "#ff40d9")}
# The layout, in inches.
LINEAGE_GRID = (0.05, 0.05, 0.25)
LINEAGE_GAP = 0.03
LINEAGE_LABEL_WIDTH = 0.15
LINEAGE_PROMPT_WIDTH = 0.1
LINEAGE_HEADER = 0.16
# The gap between the two lineage blocks, in inches.
LINEAGE_ROW_GAP = 0.08
LINEAGE_LEGEND = 0.16
LINEAGE_ASPECT = 1.0
LINEAGE_PAD = 0.03
# The size of the reference over the movie crop, and its gap to the crop.
LINEAGE_REFERENCE_HEIGHT = 0.28
LINEAGE_REFERENCE_RISE = 0.05
# The gap between the top of the reference and its label, in inches.
LINEAGE_LABEL_GAP = 0.06
# The largest width and height of the movie render.
LINEAGE_VOLUME_SCALE = 1.12
LINEAGE_VOLUME_HEIGHT = 0.6


def load_lineage_render(dataset, sample, root, key):
    """Return a render as RGBA in [0, 1]."""
    path = os.path.join(LINEAGE_RENDERS, f"{dataset}_{sample}_{root}_{key}.png")
    return np.asarray(Image.open(path).convert("RGBA")) / 255


def trim(images, pad=LINEAGE_PAD):
    """Cut the images to one box around what any of them draws, so that they keep one scale."""
    drawn = np.any([image[..., 3] > 0 for image in images], axis=0)
    ys, xs = np.nonzero(drawn)
    margin = int(pad * max(np.ptp(ys), np.ptp(xs)))
    y0, x0 = max(ys.min() - margin, 0), max(xs.min() - margin, 0)
    return [image[y0:ys.max() + margin + 1, x0:xs.max() + margin + 1] for image in images]


def image_axes(fig, left, bottom, width, height, image, framed=True):
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
    rows, columns = image.shape[:2]
    scale = max(columns / width, rows / height)
    extra_x, extra_y = (width * scale - columns) / 2, (height * scale - rows) / 2
    ax.set_xlim(-0.5 - extra_x, columns - 0.5 + extra_x)
    ax.set_ylim(rows - 0.5 + extra_y, -0.5 - extra_y)
    return ax


def lineage_scores(dataset, sample, root):
    """The mean IoU of the lineage over its tracks and frames, per (method, prompt), after the first prompt."""
    table = pd.read_csv(os.path.join(LINEAGE_SCORES, f"{dataset}.csv"), dtype={"sample": str})
    row = table[(table["sample"] == sample) & (table["root"] == root)].iloc[0]
    return {
        (method, prompt): row[f"iou_{run}_{prompt}_0"] for method, run in LINEAGE_RUNS.items()
        for prompt in LINEAGE_PROMPTS
    }


def plot_lineage_block(fig, left, bottom, size, dataset, sample, root, domain):
    """Plot one lineage: the movie crop under the reference left, the method renders per prompt (rows) right."""
    height = fig2.height_of(fig)
    gap = LINEAGE_GAP
    tall = LINEAGE_ASPECT * size
    # The tree renders: translucent masks with their centre lines, seen facing the split.
    keys = ["tree_reference"] + [f"tree_{LINEAGE_METHODS[m]}_{p}" for p in LINEAGE_PROMPTS for m in LINEAGE_METHODS]
    images = dict(zip(keys, trim([load_lineage_render(dataset, sample, root, key) for key in keys])))
    # The movie as the volume of Figure 3 panel d.
    volume = trim([load_lineage_render(dataset, sample, root, "pyramid")])[0]

    # The dataset name stands left of the block, written vertically: in italics in domain, in bold out of domain.
    x = left + LINEAGE_LABEL_WIDTH
    in_domain = domain == "ID"
    fig.text(
        (x - 0.02 - (LINEAGE_VOLUME_SCALE - 1) * size / 2) / fig2.FIGURE_WIDTH, (bottom + tall + gap / 2) / height,
        INTERACTIVE_DATASETS[dataset][0],
        rotation=90, ha="right", va="center", fontsize=MAX_FONT_SIZE, color=TEXT_COLOR,
        style="italic" if in_domain else "normal", fontweight="normal" if in_domain else "bold",
    )
    # The movie and the reference over it form one group, centred between the rows.
    block = 2 * tall + gap
    rows, columns = volume.shape[:2]
    volume_width = min(LINEAGE_VOLUME_SCALE * size, LINEAGE_VOLUME_HEIGHT * block * columns / rows)
    volume_height = volume_width * rows / columns
    reference = images["tree_reference"]
    rows, columns = reference.shape[:2]
    shown = min(size * rows / columns, LINEAGE_REFERENCE_HEIGHT * block)
    width = shown * columns / rows
    group = volume_height + LINEAGE_REFERENCE_RISE + shown
    group_bottom = bottom + (block - group) / 2
    group_top = group_bottom + group
    shift = 0.0
    image_axes(fig, x - (volume_width - size) / 2, group_bottom, volume_width, volume_height, volume, framed=False)
    image_axes(
        fig, x + (size - width) / 2, group_bottom + volume_height + LINEAGE_REFERENCE_RISE, width, shown, reference,
        framed=False
    )
    fig.text(
        (x + size / 2) / fig2.FIGURE_WIDTH, (group_top + shift + LINEAGE_LABEL_GAP) / height, "Reference", ha="center",
        va="baseline", fontsize=MAX_FONT_SIZE, color=TEXT_COLOR,
    )

    scores = lineage_scores(dataset, sample, root)
    methods_left = x + size + gap + LINEAGE_PROMPT_WIDTH
    for k, name in enumerate(fig2.names_of(list(LINEAGE_METHODS), set())):
        fig.text(
            (methods_left + k * (size + gap) + size / 2) / fig2.FIGURE_WIDTH, (bottom + 2 * tall + gap + 0.04) / height,
            name, ha="center", va="baseline", fontsize=MAX_FONT_SIZE, color=TEXT_COLOR,
        )
    for row, prompt in enumerate(LINEAGE_PROMPTS):
        y = bottom + (1 - row) * (tall + gap)
        fig.text(
            (methods_left - 0.02) / fig2.FIGURE_WIDTH, (y + tall / 2) / height, fig2.PROMPT_NAMES[prompt],
            rotation=90, ha="right", va="center", fontsize=MAX_FONT_SIZE, color=TEXT_COLOR,
        )
        for col, (method, key) in enumerate(LINEAGE_METHODS.items()):
            ax = image_axes(fig, methods_left + col * (size + gap), y, size, tall, images[f"tree_{key}_{prompt}"])
            fig2.draw_score(ax, scores[method, prompt])


def draw_lineage_legend(fig, y, center=None):
    """The colours of the mother and the daughters, in one line centred at the height 'y' and at 'center' (the middle
    of the figure by default), in inches."""
    height = fig2.height_of(fig)
    entries = [("Mother", [LINEAGE_COLORS["Mother"]]), ("Daughters", list(LINEAGE_COLORS["Daughters"]))]
    x = (fig2.FIGURE_WIDTH / 2 if center is None else center) - 0.6
    for label, swatches in entries:
        for color in swatches:
            fig.add_artist(plt.Rectangle(
                (x / fig2.FIGURE_WIDTH, (y - 0.035) / height), 0.07 / fig2.FIGURE_WIDTH, 0.07 / height,
                facecolor=color, edgecolor="none",
            ))
            x += 0.09
        fig.text(
            x / fig2.FIGURE_WIDTH, y / height, label, ha="left", va="center", fontsize=MAX_FONT_SIZE, color=TEXT_COLOR
        )
        x += 0.55


def plot_interactive_qualitative(region=None, right=None, beke=False):
    """One block per lineage of INTERACTIVE_QUALITATIVE, one under the other (DynamicNuclearNet on
    top, the wing disc under it), with a colour legend below.

    'right' is the room right of the blocks, in inches; with 'beke', the Beke tracking panel (panel e of the combined
    figure) fills it, and the blocks fit into the height of its row.
    """
    left_margin, right_margin = LINEAGE_GRID[0], LINEAGE_GRID[1] if right is None else right
    room = fig2.FIGURE_WIDTH - left_margin - right_margin
    n_methods, n_rows = len(LINEAGE_METHODS), len(INTERACTIVE_QUALITATIVE)
    extra = LINEAGE_LABEL_WIDTH + LINEAGE_PROMPT_WIDTH + (n_methods + 1) * LINEAGE_GAP
    size = (room - extra) / (n_methods + 1)
    if beke:
        # The blocks fill the height of the row, and the Beke panel takes the rest of the width.
        rows_room = BEKE_HEIGHT - LINEAGE_LEGEND - n_rows * LINEAGE_HEADER - (n_rows - 1) * LINEAGE_ROW_GAP
        size = (rows_room / n_rows - LINEAGE_GAP) / (2 * LINEAGE_ASPECT)
        room = extra + (n_methods + 1) * size
        right_margin = fig2.FIGURE_WIDTH - left_margin - room
    block_height = 2 * LINEAGE_ASPECT * size + LINEAGE_GAP
    natural = LINEAGE_LEGEND + n_rows * (LINEAGE_HEADER + block_height) + (n_rows - 1) * LINEAGE_ROW_GAP
    # Next to the Beke panel the row is at least BEKE_HEIGHT high; panel d stands in its middle.
    height = max(natural, BEKE_HEIGHT) if beke else natural
    lift = (height - natural) / 2
    left = left_margin + (room - extra - (n_methods + 1) * size) / 2
    fig = fig2.new_figure(height, region)
    # No domain titles: the style of the dataset names shows the domain.
    bottom = lift + LINEAGE_LEGEND
    for dataset, sample, root, domain in reversed(INTERACTIVE_QUALITATIVE):
        plot_lineage_block(fig, left, bottom, size, dataset, sample, root, domain)
        bottom += block_height + LINEAGE_HEADER + LINEAGE_ROW_GAP
    draw_lineage_legend(fig, lift + LINEAGE_LEGEND / 2, center=left_margin + room / 2)
    if beke:
        box_left = fig2.FIGURE_WIDTH - right_margin + BEKE_GAP
        draw_beke(fig, (box_left, 0.0, fig2.FIGURE_WIDTH - box_left, height))
        fig.text(
            box_left / fig2.FIGURE_WIDTH, 1.0, "e", ha="left", va="top", fontsize=fig2.PANEL_LABEL_SIZE,
            fontweight="bold", color=TEXT_COLOR, transform=fig.transSubfigure if region else fig.transFigure
        )
    return fig


# One movie per column of panel a, where micro-sam2 has fewer tracking errors than CellposeSAM.
AUTOMATIC_QUALITATIVE = [
    ("DynamicNuclearNet", "3T3_nuc_s2_p1_1c", "DynamicNuclearNet_3T3_nuc_s2_p1_1c_stretch1.4", "ID",
     "DynamicNuclearNet", 2),
    ("WingDisc", "Ecd_20150608_P1", "WingDisc_Ecd_20150608_P1_window0.35", "OOD", "Wing disc", 2),
    ("Fluo-N3DH-SIM+", "1", "Fluo-N3DH-SIM+_01_floor", "OOD", "Fluo-N3DH-SIM+", 3),
]
# The space-time renders of 'render_tracking_candidates.py', and the per-movie scores.
AUTOMATIC_RENDERS = os.path.join(TRACKING_ROOT, "automatic_renders")
AUTOMATIC_SEQUENCE_CSV = os.path.join(TRACKING_ROOT, "automatic_tracking_results.csv")
QUALITATIVE_METHODS = {"CellposeSAM": ("cpsam", "cpsam", None), "micro-sam2": ("microsam2_apg", "microsam2", "apg")}
# The link colours, with short names for the vertical legend.
LINK_COLORS = {"Correct": "#dbdbdb", "False": "#ff00ff", "Missed": "#00ffff"}
# The link legend stands vertically right of the renders, in the right margin.
AUTOMATIC_QUALITATIVE_GRID = (0.05, 0.2, 0.15)
AUTOMATIC_QUALITATIVE_HEADER = 0.25
AUTOMATIC_QUALITATIVE_LEGEND = 0.02
# the width of the first frame below the views, as a share of the column width, and the gap above it, in inches
PYRAMID_OVERVIEW_SHARE = 0.4
PYRAMID_GAP = 0.02


def sequence_tra(dataset, sequence, method, mode, ndim):
    """The TRA of one movie, linked with the Trackastra model of its dimension."""
    table = pd.read_csv(AUTOMATIC_SEQUENCE_CSV, dtype={"sequence": str})
    table["method"] = table["method"].fillna("microsam2")
    rows = table[
        (table["dataset"] == dataset) & (table["sequence"] == sequence) & (table["method"] == method)
        & (table["tracking_model"] == TRACKING_MODELS[ndim])
        & (table["norm_percentiles"].isna() | (table["norm_percentiles"] == "2,98"))
    ]
    if mode is not None:
        rows = rows[rows["mode"] == mode]
    return float(rows["TRA"].iloc[-1])


def draw_link_legend(fig, x, top, length):
    """The link colours, one under the other along a vertical line at 'x' from 'top' down over 'length' inches, with
    vertical text."""
    height = fig2.height_of(fig)
    y = top
    for label, color in LINK_COLORS.items():
        fig.add_artist(plt.Rectangle(
            ((x - 0.035) / fig2.FIGURE_WIDTH, (y - 0.07) / height), 0.07 / fig2.FIGURE_WIDTH, 0.07 / height,
            facecolor=color, edgecolor="#a0a0a0", linewidth=0.3,
        ))
        text = fig.text(
            x / fig2.FIGURE_WIDTH, (y - 0.09) / height, label, ha="center", va="top", rotation=270,
            fontsize=MAX_FONT_SIZE - 1, color=TEXT_COLOR
        )
        # the next entry starts below this label, measured, or at the next third of 'length' if that is lower
        figure = fig.figure if hasattr(fig, "figure") and fig.figure is not None else fig
        extent = text.get_window_extent(renderer=figure.canvas.get_renderer())
        y -= max(length / len(LINK_COLORS), 0.09 + extent.height / figure.dpi + 0.08)


def plot_automatic_qualitative(region=None):
    """One block per movie of AUTOMATIC_QUALITATIVE, side by side as the columns of panel a: on top the space-time
    views of a small window of the movie for CellposeSAM and micro-sam2, with the TRA of the movie in the corner, and
    below them the first frame with the square of the window, joined to the views by lines. A
    legend of the link colours stands on the right. The dataset name is in italics in domain, in bold out of
    domain."""
    from matplotlib.patches import ConnectionPatch, Polygon
    starts, column_width = fig2.column_grid(AUTOMATIC_QUALITATIVE_GRID, len(AUTOMATIC_QUALITATIVE))
    gap = LINEAGE_GAP
    size = (column_width - gap) / len(QUALITATIVE_METHODS)
    blocks = []
    for dataset, sequence, _, domain, label, ndim in AUTOMATIC_QUALITATIVE:
        name = f"{dataset}_{sequence.zfill(2) if ndim == 3 else sequence}"
        layout = json.load(open(os.path.join(AUTOMATIC_RENDERS, f"pyramid_{name}.json")))

        def load(key, name=name):
            path = os.path.join(AUTOMATIC_RENDERS, f"pyramid_{name}_{key}.png")
            return np.asarray(Image.open(path).convert("RGBA")) / 255
        blocks.append(
            (name, layout, load("overview"), {method: load(key) for method, (key, _, _) in QUALITATIVE_METHODS.items()})
        )
    tall = max(size * im.shape[0] / im.shape[1] for _, _, _, views in blocks for im in views.values())
    overview_w = PYRAMID_OVERVIEW_SHARE * column_width
    low = max(overview_w * ov.shape[0] / ov.shape[1] for _, _, ov, _ in blocks)
    height = AUTOMATIC_QUALITATIVE_HEADER + tall + PYRAMID_GAP + low + AUTOMATIC_QUALITATIVE_LEGEND
    fig = fig2.new_figure(height, region)
    bottom = AUTOMATIC_QUALITATIVE_LEGEND + low + PYRAMID_GAP
    for left, (dataset, sequence, _, domain, label, ndim), (name, layout, overview, views) in zip(
        starts, AUTOMATIC_QUALITATIVE, blocks
    ):
        in_domain = domain == "ID"
        fig.text(
            (left + column_width / 2) / fig2.FIGURE_WIDTH, (bottom + tall + 0.13) / height, label, ha="center",
            va="baseline", fontsize=MAX_FONT_SIZE, color=TEXT_COLOR, style="italic" if in_domain else "normal",
            fontweight="normal" if in_domain else "bold",
        )
        methods = list(QUALITATIVE_METHODS)
        axes = []
        for k, (method, display) in enumerate(zip(methods, fig2.names_of(methods, set()))):
            x = left + k * (size + gap)
            fig.text(
                (x + size / 2) / fig2.FIGURE_WIDTH, (bottom + tall + 0.03) / height, display, ha="center",
                va="baseline", fontsize=MAX_FONT_SIZE, color=TEXT_COLOR
            )
            ax = image_axes(fig, x, bottom, size, tall, views[method])
            _, table_method, mode = QUALITATIVE_METHODS[method]
            fig2.draw_score(ax, sequence_tra(dataset, sequence, table_method, mode, ndim))
            axes.append(ax)
        ov_ax = image_axes(
            fig, left + (column_width - overview_w) / 2, AUTOMATIC_QUALITATIVE_LEGEND, overview_w, low, overview,
            framed=False
        )
        box = np.array(layout["overview"]["box"])
        # white, so that the square shows on the dark frames
        ov_ax.add_patch(Polygon(box, closed=True, fill=False, edgecolor="#ffffff", lw=1.0))
        # lines from the outer corners of the square to the outer front corners of the views
        targets = [
            (axes[0], np.array(layout[QUALITATIVE_METHODS[methods[0]][0]]["corners"]), np.argmin),
            (axes[1], np.array(layout[QUALITATIVE_METHODS[methods[1]][0]]["corners"]), np.argmax)
        ]
        for (ax, corners, pick), start in zip(targets, (box[np.argmin(box[:, 0])], box[np.argmax(box[:, 0])])):
            end = corners[pick(corners[:, 0])]
            fig.add_artist(ConnectionPatch(start, end, "data", "data", axesA=ov_ax, axesB=ax, color=TEXT_COLOR, lw=0.5))
    draw_link_legend(fig, fig2.FIGURE_WIDTH - 0.1, bottom + tall, tall)
    return fig


PLOTS = {
    "automatic_tracking": ("automatic", "quantitative", plot_automatic),
    "interactive_tracking": ("interactive", "quantitative", plot_interactive),
    "interactive_tracking_qualitative": ("interactive", "qualitative", plot_interactive_qualitative),
    "automatic_tracking_qualitative": ("automatic", "qualitative", plot_automatic_qualitative),
    "beke_tracking": ("automatic", "qualitative", lambda region=None: plot_beke(region)),
}


# Panel e: the tracked nuclei of the Beke time-lapse of Platynereis; render with '--render beke'.
BEKE_TRACKS = os.path.join(
    ROOT, "..", "..", "..", "user_study", "v2", "huisken_lab", "huisken_tracks_h5", "tracks_new_defaults.h5"
)
BEKE_DATA = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/huisken_data/platynereis_timelapse_3-6hpf"
BEKE_RENDERS = os.path.join(OUTPUT_DIR, "beke")
# z spacing over the xy spacing (2.5 / 0.406 um, xy downsampled 2x), and the xy pixel size
BEKE_Z_SCALE = 2.5 / 0.812
BEKE_XY_UM = 0.812
BEKE_START_HPF, BEKE_MINUTES_PER_TIMEPOINT = 3.0, 2.0
BEKE_TIMEPOINTS = (0, 30, 60, 88)
# the scene height of one time point in the stack
BEKE_T_HEIGHT = 3.5
BEKE_GREY = np.array([0.78, 0.78, 0.76])
# The width of the Beke panel right of panel d, the gap between them and the least height of their row, in inches.
BEKE_WIDTH = 2.9
BEKE_GAP = 0.05
BEKE_HEIGHT = 4.33
MM = 1 / 25.4


def beke_to_hpf(t):
    return BEKE_START_HPF + np.asarray(t) * BEKE_MINUTES_PER_TIMEPOINT / 60


def beke_load():
    """The lineage table and the tracks of the track file (downsampled 2x in y and x)."""
    import h5py
    with h5py.File(BEKE_TRACKS, "r") as f:
        lineage = pd.DataFrame(f["lineage"][:], columns=["track_id", "t_start", "t_end", "parent_id"])
        tracks = pd.DataFrame(f["tracks"][:], columns=["track_id", "t", "z", "y", "x"])
    tracks[["track_id", "t"]] = tracks[["track_id", "t"]].astype(int)
    return lineage.set_index("track_id"), tracks


def beke_track_colors(lineage):
    """One rgb colour per track: the hue of its lineage tree, shades of it for the daughters."""
    import colorsys
    children = {i: [] for i in lineage.index}
    for i, parent in lineage["parent_id"].items():
        if parent > 0:
            children[parent].append(i)
    roots, depth = {}, {}
    for i in lineage.index:
        j, d = i, 0
        while lineage.at[j, "parent_id"] > 0:
            j, d = lineage.at[j, "parent_id"], d + 1
        roots[i], depth[i] = j, d
    root_ids = sorted(set(roots.values()), key=lambda r: (lineage.at[r, "t_start"], r))
    hues = {root: (k * 0.6180339887) % 1.0 for k, root in enumerate(root_ids)}
    shade = {}
    for i in lineage.index:
        siblings = children.get(lineage.at[i, "parent_id"], [i])
        shade[i] = siblings.index(i) if i in siblings else 0
    out = {}
    for i in lineage.index:
        d = depth[i]
        sat = 0.75 - 0.08 * min(d, 4)
        val = 0.92 - 0.12 * shade[i] if d > 0 else 0.92
        hue = (hues[roots[i]] + 0.025 * d * (1 if shade[i] == 0 else -1)) % 1.0
        out[i] = np.array(colorsys.hsv_to_rgb(hue, sat, val))
    return out


def beke_paths(lineage, tracks, smooth=2.0):
    """The smoothed centroid path of each track, with the last point of the parent first."""
    from scipy import ndimage
    out = {}
    for i, group in tracks.sort_values("t").groupby("track_id"):
        zyx = group[["z", "y", "x"]].to_numpy(float)
        if len(zyx) > 2:
            zyx = ndimage.gaussian_filter1d(zyx, smooth, axis=0, mode="nearest")
        out[i] = (group["t"].to_numpy(), zyx)
    for i in list(out):
        parent = lineage.at[i, "parent_id"]
        if parent > 0 and parent in out:
            t, zyx = out[i]
            pt, pzyx = out[parent]
            out[i] = (np.r_[pt[-1], t], np.r_[pzyx[-1:], zyx])
    return out


def beke_tube(points, colors, radius):
    import pyvista as pv
    from scipy.spatial import cKDTree
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-3]
    points, colors = points[keep], np.asarray(colors)[keep]
    if len(points) < 2:
        return None
    mesh = pv.lines_from_points(points).tube(radius=radius, n_sides=20, capping=True)
    # the tube filter drops the colours, so each tube point takes the colour of the nearest path point
    nearest = cKDTree(points).query(mesh.points)[1]
    mesh.point_data["rgb"] = (colors[nearest] * 255).astype("uint8")
    return mesh


def beke_add_rgb(plotter, mesh, specular=0.3, specular_power=30):
    if mesh is not None:
        plotter.add_mesh(
            mesh, scalars="rgb", rgb=True, smooth_shading=True, specular=specular, specular_power=specular_power,
            ambient=0.35, diffuse=0.75, show_scalar_bar=False
        )


def beke_raw_mip(raw, t):
    mip = raw[t].max(axis=0).astype("float32")
    lo, hi = np.percentile(mip, [1, 99.8])
    return np.clip((mip - lo) / max(hi - lo, 1), 0, 1)


def beke_crop_to_embryo(raw, segmentation, tracks, margin=10):
    """Crop y and x to the nuclei of all time points plus a margin, and shift the tracks."""
    rows, cols = np.nonzero((segmentation > 0).any(axis=(0, 1)))
    y0, x0 = max(rows.min() - margin, 0), max(cols.min() - margin, 0)
    y1, x1 = rows.max() + margin + 1, cols.max() + margin + 1
    tracks = tracks.copy()
    tracks["y"] -= y0
    tracks["x"] -= x0
    return raw[:, :, y0:y1, x0:x1], segmentation[:, :, y0:y1, x0:x1], tracks


def beke_render_stack(raw, segmentation, lineage, tracks, colors):
    """The stack of raw cards with the outlines of the nuclei and the paths; returns the image positions of the card
    corners and of the scale bar."""
    from skimage.segmentation import find_boundaries
    from plots_for_figure1 import d_backdrop, d_card, d_crop_white, d_project, d_screenshot, d_studio, d_view
    gap = 30 * BEKE_T_HEIGHT
    h, w = raw.shape[2:]
    cards = []
    for k, t in enumerate(BEKE_TIMEPOINTS):
        mip = beke_raw_mip(raw, t)
        labels = segmentation[t].max(axis=0)
        image = np.stack([mip] * 3, -1) * 0.85
        edge = find_boundaries(labels, mode="inner")
        for i in np.unique(labels[edge]):
            image[edge & (labels == i)] = colors[int(i)]
        cards.append(((image * 255).astype("uint8"), k * gap))
    levels = np.arange(len(BEKE_TIMEPOINTS)) * gap
    tubes = [
        beke_tube(np.c_[zyx[:, 2], -zyx[:, 1], np.interp(t, BEKE_TIMEPOINTS, levels) + 2], [colors[i] for _ in t], 1.0)
        for i, (t, zyx) in beke_paths(lineage, tracks).items()
    ]
    bounds = (0, w, -h, 0, 0, (len(BEKE_TIMEPOINTS) - 1) * gap + 4)
    size = (1800, 2200)
    view = d_view(22, -60)[0]
    plotter = d_studio(size, bounds, view, 900, 1.15)
    for image, z in cards:
        d_card(plotter, image, (0, 0, z), 1.0, 2)
    for mesh in tubes:
        beke_add_rgb(plotter, mesh)
    d_backdrop(plotter, bounds, view, 10, 300, 40)
    image = d_screenshot(plotter, size)
    corners = [d_project(plotter, [(0, 0, z + 2), (w, 0, z + 2), (w, -h, z + 2), (0, -h, z + 2)]) for _, z in cards]
    bar = d_project(plotter, [(8, -h, 0), (8 + 50 / BEKE_XY_UM, -h, 0)])
    plotter.close()
    image, offset = d_crop_white(image, 10)
    Image.fromarray(image).save(os.path.join(BEKE_RENDERS, "stack.png"))
    corners = [[(x - offset[0], y - offset[1]) for x, y in c] for c in corners + [bar]]
    return {"card_corners": corners[:-1], "stack_bar": corners[-1]}


def beke_render_nuclei(segmentation, colors):
    """The nuclei at the time points of the cards, with the same camera; y is flipped to match the cards."""
    from plots_for_figure1 import d_colored_mesh, d_px_per_unit, d_screenshot, d_studio, d_view
    _, _, h, w = segmentation.shape
    bounds = (0, w, 0, h, 0, segmentation.shape[1] * BEKE_Z_SCALE)
    view = d_view(22, -60)[0]
    ppu = None
    for t in BEKE_TIMEPOINTS:
        labels = segmentation[t][:, ::-1].astype("uint32")
        ids = np.unique(labels)
        remap = np.zeros(int(ids.max()) + 1, "uint32")
        remap[ids] = np.arange(len(ids))
        mesh = d_colored_mesh(remap[labels], np.array([BEKE_GREY] + [colors[int(i)] for i in ids[1:]]), BEKE_Z_SCALE)
        plotter = d_studio((1400, 1400), bounds, view, 700, 1.05)
        # Without shadows: no floor and no shadow pass.
        plotter.disable_shadows()
        beke_add_rgb(plotter, mesh, specular=0.2, specular_power=20)
        Image.fromarray(d_screenshot(plotter, (1400, 1400))).save(os.path.join(BEKE_RENDERS, f"nuclei_t{t:03d}.png"))
        ppu = d_px_per_unit(plotter, np.array([w / 2, h / 2, bounds[5] / 2]))
        plotter.close()
        print("rendered the Beke nuclei at", t, flush=True)
    return {"nuclei_px_per_um": ppu / BEKE_XY_UM}


def render_beke():
    """Render the stack and the nuclei views of the Beke panel (GPU node) into BEKE_RENDERS."""
    import h5py
    os.makedirs(BEKE_RENDERS, exist_ok=True)
    lineage, tracks = beke_load()
    with h5py.File(BEKE_TRACKS, "r") as f:
        raw, segmentation = f["raw"][:], f["segmentation"][:]
    raw, segmentation, tracks = beke_crop_to_embryo(raw, segmentation, tracks)
    colors = beke_track_colors(lineage)
    layout = beke_render_stack(raw, segmentation, lineage, tracks, colors)
    layout.update(beke_render_nuclei(segmentation, colors))
    with open(os.path.join(BEKE_RENDERS, "layout.json"), "w") as f:
        json.dump(layout, f, indent=1)


def beke_crop_box(image, keep, points=(), margins=(30, 30, 30, 30)):
    """The box (c0, r0, c1, r1) around the pixels to keep and the points, with margins (left, top, right, bottom)."""
    rows, cols = np.nonzero(keep)
    xs = np.r_[cols, np.asarray(points, float).reshape(-1, 2)[:, 0]]
    ys = np.r_[rows, np.asarray(points, float).reshape(-1, 2)[:, 1]]
    c0, c1, r0, r1 = xs.min(), xs.max(), ys.min(), ys.max()
    left, top, right, bottom = margins
    h, w = image.shape[:2]
    return int(max(c0 - left, 0)), int(max(r0 - top, 0)), int(min(c1 + right, w)), int(min(r1 + bottom, h))


def beke_scale_bar(ax, p0, p1, label):
    """A dark scale bar between two image points, with its label to the right."""
    (x0, y0), (x1, y1) = p0, p1
    ax.plot([x0, x1], [y0, y1], color=TEXT_COLOR, lw=1.0, solid_capstyle="butt", clip_on=False)
    ax.text(x1 + 12, y1, label, ha="left", va="center", fontsize=MAX_FONT_SIZE - 1, color=TEXT_COLOR, clip_on=False)


def draw_beke(fig, box):
    """The Beke panel in the box (left, bottom, width, height) of 'fig', in inches: the stack of raw cards with the
    paths, and beside each card the nuclei of its time point, joined by dashed lines."""
    from matplotlib.patches import ConnectionPatch
    with open(os.path.join(BEKE_RENDERS, "layout.json")) as f:
        layout = json.load(f)
    stack = np.asarray(Image.open(os.path.join(BEKE_RENDERS, "stack.png")).convert("RGB"))
    corners = np.array(layout["card_corners"])
    bar = np.array(layout["stack_bar"])
    # the bar runs along the front edge of the lowest card; it moves down into the white below the card
    bar_shift = np.array([0, 70.0])
    c0, r0, c1, r1 = beke_crop_box(
        stack, np.zeros(stack.shape[:2], bool), np.r_[corners.reshape(-1, 2), bar + bar_shift], margins=(20, 20, 20, 40)
    )
    stack = stack[r0:r1, c0:c1]
    corners = corners - [c0, r0]
    bar = bar + bar_shift - [c0, r0]
    nuclei = [
        np.asarray(Image.open(os.path.join(BEKE_RENDERS, f"nuclei_t{t:03d}.png")).convert("RGB"))
        for t in BEKE_TIMEPOINTS
    ]
    # one crop for all time points, so that they keep the same scale
    keep = np.any([image.min(-1) < 235 for image in nuclei], axis=0)
    n0, m0, n1, m1 = beke_crop_box(nuclei[0], keep, margins=(20, 20, 20, 20))
    nuclei = [image[m0:m1, n0:n1] for image in nuclei]

    fig_w, fig_h = fig2.FIGURE_WIDTH, fig2.height_of(fig)

    def axes(left, bottom, width, height):
        return fig.add_axes([left / fig_w, bottom / fig_h, width / fig_w, height / fig_h])

    box_left, box_bottom, width, height = box
    label_h, view_gap, margin = 0.12, 0.02, 0.05
    views_h = height - 2 * margin
    view_h = (views_h - 4 * label_h - 3 * view_gap) / 4
    view_w = view_h * nuclei[0].shape[1] / nuclei[0].shape[0]
    gap = 0.2
    stack_w = width - view_w - gap
    stack_h = min(stack_w * stack.shape[0] / stack.shape[1], height - 2 * margin)
    stack_w = stack_h * stack.shape[1] / stack.shape[0]
    left = box_left
    stack_ax = axes(left, box_bottom + (height - stack_h) / 2, stack_w, stack_h)
    stack_ax.imshow(stack)
    stack_ax.axis("off")
    beke_scale_bar(stack_ax, bar[0], bar[1], "50 µm")
    view_left = left + stack_w + gap
    for k, (t, image) in enumerate(zip(BEKE_TIMEPOINTS, nuclei)):
        # the latest time point on top, as in the stack
        bottom = box_bottom + margin + k * (view_h + label_h + view_gap)
        # Each view stands on a card with a hairline frame, as the renders of panels b and d.
        ax = fig2.framed_axes(fig, view_left, bottom, view_w, view_h)
        ax.imshow(image, aspect="auto")
        ax.text(
            0, 1.02, f"{beke_to_hpf(t):.1f} hpf", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=MAX_FONT_SIZE - 1, color=TEXT_COLOR, fontweight="bold"
        )
        right = corners[k][np.argmax(corners[k][:, 0])]
        # the connector runs from the right corner of the card to the left edge of its view, touching both
        fig.add_artist(ConnectionPatch(
            right, (0, 0.5), "data", "axes fraction", axesA=stack_ax, axesB=ax, color="#52514e", lw=0.8,
            ls=(0, (3, 1.5)), capstyle="butt", zorder=5
        ))


def plot_beke(region=None):
    """The Beke panel alone, in the size that it has right of panel d."""
    height = 2.6
    fig = fig2.new_figure(height, region)
    draw_beke(fig, ((fig2.FIGURE_WIDTH - BEKE_WIDTH) / 2, 0, BEKE_WIDTH, height))
    return fig


def plot_interactive_qualitative_beke(region=None):
    """Panel d with the Beke panel (e) to its right, for the combined figure."""
    return plot_interactive_qualitative(region=region, right=BEKE_WIDTH + BEKE_GAP, beke=True)


# The renders of panels b, d and e and the lineage scores of panel d, made on a GPU node with '--render'.
RENDERS_3D = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/qualitative_3d/render_tryouts"
EVALUATION = os.path.join(ROOT, "..", "..", "..", "..", "finetuning", "v2", "evaluation")
# The 2D+t movies of panel b as render specs (dataset, sequence, Trackastra model, window share, time stretch).
SPACETIME_RENDERS = [
    ("DynamicNuclearNet", "3T3_nuc_s2_p1_1c", "general_2d_w_SAM2_features", None, 1.4),
    ("WingDisc", "Ecd_20150608_P1", "general_2d_w_SAM2_features", "0.35", 1.0),
]
# The 3D+t movie of panel b, rendered in the floor view.
FLOOR_RENDERS = [("Fluo-N3DH-SIM+", "01")]
LINEAGE_PARTS = ["pyramid", "tree_reference"] + [
    f"tree_{method}_{prompt}" for method in ("microsam2", "microsam") for prompt in ("point", "box")
]


# Set by 'load_render_modules', so that the plots need no pyvista.
tifffile = pv = gaussian_filter1d = EdgeFlag = NodeFlag = None
rp = render_pairs = best_planes = block_of = surfaces = z_scale_of = tu = None
d_backdrop = d_card = d_screenshot = d_studio = d_view = d_project = None


def load_render_modules(out_folder):
    """Import the render helpers into this module. 'render_planes3d' reads its folder from sys.argv at import."""
    import tifffile
    import pyvista as pv
    from scipy.ndimage import gaussian_filter1d
    from traccuracy import EdgeFlag, NodeFlag
    sys.path[:0] = [RENDERS_3D, EVALUATION]
    argv = sys.argv
    sys.argv = [argv[0], out_folder]
    try:
        import render_planes3d as rp
        import render_pairs
    finally:
        sys.argv = argv
    from render_three_views import best_planes, block_of
    from render_pv import surfaces
    from select_tracing import z_scale_of
    import tracking_util as tu
    from plots_for_figure1 import d_backdrop, d_card, d_project, d_screenshot, d_studio, d_view
    # The colours of the lineage renders: index 1 is the mother, 2 and 3 the daughters.
    rp.PALETTE = ["#000000", LINEAGE_COLORS["Mother"], *LINEAGE_COLORS["Daughters"]]
    globals().update(
        tifffile=tifffile, pv=pv, gaussian_filter1d=gaussian_filter1d,
        EdgeFlag=EdgeFlag, NodeFlag=NodeFlag, rp=rp, render_pairs=render_pairs, best_planes=best_planes,
        block_of=block_of, surfaces=surfaces, z_scale_of=z_scale_of, tu=tu, d_backdrop=d_backdrop, d_card=d_card,
        d_screenshot=d_screenshot, d_studio=d_studio, d_view=d_view, d_project=d_project,
    )


# The renders of the former 'render_interactive_lineage_3d.py'.
LR_PRED = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/tracking/interactive/predictions"
LR_METHODS = {"microsam2": "hvit_l_cells", "microsam": "vit_l_lm"}
LR_PROMPTS = {"point": "point_c4_iou0.9", "box": "box_boxcorr_c4_iou0.9"}
LR_BUDGET = 0
# The tree renders: mask opacity, line radius, line shade and view elevation.
LR_TREE_OPACITY = 0.72
LR_TREE_RADIUS = 0.012
LR_TREE_SHADE = 0.6
LR_TREE_ELEVATION = 15.0
# A predicted mask is drawn only where it matches the reference with this IoU.
LR_MATCH_IOU = 0.5
# Less smoothing, so that the small wing disc daughters keep their size.
LR_TREE_SMOOTHING = dict(sigma=0.5, n_iter=15)
LR_MAX_GAP = 2
# The number of frames in the movie stack and the gap between them.
LR_N_SLICES = 8
LR_SLICE_GAP = 0.09
# The radius of the node at the division, in radii of the centre lines.
LR_TREE_NODE = 3.5
# Line segments of fewer frames are left out: their masks are too small for a surface.
LR_MIN_SEGMENT = 3


def lr_load(folder):
    masks = np.stack([tifffile.imread(p) for p in sorted(glob.glob(os.path.join(folder, "*.tif")))])
    table = np.loadtxt(os.path.join(folder, "man_track.txt"), dtype=int, ndmin=2)
    return masks, table


def lr_frame_slice(plotter, image, low, high, origin, xy_scale, z, overlay=()):
    """Draw one frame as a flat image at the height 'z', with a thin grey outline. 'overlay' lists (mask, color)
    pairs that are blended into the image."""
    ny, nx = image.shape
    gray = np.clip((image - low) / max(high - low, 1e-6), 0, 1)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    for mask, color in overlay:
        rgb[mask] = (1 - render_pairs.OVERLAY_ALPHA) * rgb[mask] + render_pairs.OVERLAY_ALPHA * np.array(
            rp.pv.Color(color).float_rgb
        )
    plane = rp.pv.ImageData(dimensions=(nx, ny, 1), spacing=(xy_scale, xy_scale, 1), origin=(*origin, z))
    # The points of the plane run over x first, then y, as the pixels of the image in C order.
    plane.point_data["rgb"] = (rgb.reshape(-1, 3) * 255).astype(np.uint8)
    plotter.add_mesh(plane.extract_surface(), scalars="rgb", rgb=True, lighting=False)
    x0, y0 = origin
    x1, y1 = x0 + (nx - 1) * xy_scale, y0 + (ny - 1) * xy_scale
    outline = rp.pv.lines_from_points(np.array([(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z), (x0, y0, z)]))
    plotter.add_mesh(outline, color="#a8a8a8", line_width=2.0 * render_pairs.RENDER_SCALE)


def lr_render_pyramid(raw, mother, crop, path):
    """The movie as in Figure 3 panel d ('render_pairs.render_pair'), but as stacks of 2d frames: the whole field of
    view as LR_N_SLICES frames over the lineage, the first prompt on top, with the square of the crop marked on it, and
    the crop of the first prompt cut out and enlarged over it, with the mother on it. Time runs down the stack.

    Args:
        raw: The frames of the lineage, (time, y, x), of the whole field of view.
        mother: The mask of the mother on the first frame, (y, x).
        crop: The (y0, y1, x0, x1) square of the crop.
        path: The output path, without its extension.
    """
    frames = np.linspace(0, raw.shape[0] - 1, LR_N_SLICES).round().astype(int)
    low, high = np.percentile(raw[frames], (1, 99.7))
    y0, y1, x0, x1 = crop
    ny, nx = raw.shape[1:]
    big = max(ny, nx)
    sides, _ = block_of(np.ones((2, ny, nx), np.int64), best_planes(np.ones((2, ny, nx), np.int64)))
    plotter = render_pairs.plotter_of()
    # The full frames, the first prompt on top.
    step = LR_SLICE_GAP * big
    top = (LR_N_SLICES - 1) * step
    for k, t in enumerate(frames):
        lr_frame_slice(plotter, raw[t], low, high, (0, 0), 1.0, top - k * step)
    width = render_pairs.SQUARE_WIDTH * big
    outer = rp.pv.Box((x0 - width, x1 + width, y0 - width, y1 + width, top + 0.5, top + 1.0))
    band = outer.triangulate().clip_box((x0, x1, y0, y1, top, top + 2), invert=True)
    plotter.add_mesh(band, color=render_pairs.SQUARE_COLOR, lighting=False)

    # The crop, enlarged, as a tight stack of the same frames.
    side = y1 - y0
    scale = render_pairs.CROP_SIZE * min(ny, nx) / side
    cx, cy = (nx - 1) / 2, (ny - 1) / 2
    base = top + render_pairs.CROP_GAP * big
    origin = (cx - (side - 1) * scale / 2, cy - (side - 1) * scale / 2)
    # The crop shows only the frame of the first prompt, with the mother on it.
    crop_top = base
    lr_frame_slice(
        plotter, raw[0, y0:y1, x0:x1], low, high, origin, scale, crop_top, [(mother[y0:y1, x0:x1], rp.PALETTE[1])]
    )
    low_corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    far = origin[0] + (side - 1) * scale, origin[1] + (side - 1) * scale
    high_corners = [(origin[0], origin[1]), (far[0], origin[1]), far, (origin[0], far[1])]
    for (ax, ay), (bx, by) in zip(low_corners, high_corners):
        edge = rp.pv.Line((ax, ay, top + 1.0), (bx, by, base))
        plotter.add_mesh(edge, color=render_pairs.SQUARE_COLOR, line_width=7.0 * render_pairs.RENDER_SCALE)
    bounds = (min(0, origin[0]), max(nx - 1, far[0]), min(0, origin[1]), max(ny - 1, far[1]), 0, crop_top)
    rp.camera_of(plotter, None, sides, bounds)
    render_pairs.save(plotter, path)


def lr_centre_line(labels, k, z_scale):
    """The centroid of object k on each frame where it is, as (x, y, scaled time), smoothed along time, split into
    segments at gaps of more than LR_MAX_GAP frames. Returns the segments and the first and last frame of each."""
    frames = [t for t in range(labels.shape[0]) if (labels[t] == k).any()]
    segments, spans = [], []
    for t in frames:
        if spans and t - spans[-1][1] <= LR_MAX_GAP + 1:
            spans[-1][1] = t
        else:
            spans.append([t, t])
    for first, last in spans:
        points = np.array([
            (*np.argwhere(labels[t] == k).mean(0)[::-1], t * z_scale)
            for t in range(first, last + 1) if (labels[t] == k).any()
        ], dtype=float)
        if len(points) > 3:
            points[:, :2] = gaussian_filter1d(points[:, :2], 1.5, axis=0, mode="nearest")
        segments.append(points)
    return segments, spans


def lr_tree_view(reference, z_scale):
    """The camera direction that faces the split: horizontal and normal to the line between the daughters, raised by
    LR_TREE_ELEVATION."""
    centres = [np.vstack(lr_centre_line(reference, k, z_scale)[0])[:, :2].mean(0) for k in (2, 3)]
    line = centres[1] - centres[0]
    normal = np.array([-line[1], line[0]]) / max(np.linalg.norm(line), 1e-6)
    elevation = np.radians(LR_TREE_ELEVATION)
    return np.array([*(normal * np.cos(elevation)), np.sin(elevation)])


def lr_render_tree(labels, reference, z_scale, path):
    """The objects of 'labels' (1 the mother, 2 and 3 the daughters) as translucent surfaces with their centre lines
    as a tree: each daughter line starts at the end of the mother line. The frame and the view follow the reference."""
    direction = lr_tree_view(reference, z_scale)
    reference_meshes = [mesh for _, mesh in surfaces(reference, z_scale, **LR_TREE_SMOOTHING)]
    bounds = np.array([mesh.bounds for mesh in reference_meshes])
    lo, hi = bounds[:, 0::2].min(0), bounds[:, 1::2].max(0)
    pad = 0.05 * (hi - lo).max()
    frame = [v for a, b in zip(lo - pad, hi + pad) for v in (a, b)]
    radius = LR_TREE_RADIUS * (hi - lo).max()

    plotter = render_pairs.plotter_of()
    for k, mesh in surfaces(labels, z_scale, **LR_TREE_SMOOTHING):
        kwargs = render_pairs.mesh_kwargs(rp.PALETTE[k])
        plotter.add_mesh(mesh, opacity=LR_TREE_OPACITY, **kwargs)
    lines = {k: lr_centre_line(labels, k, z_scale) for k in (1, 2, 3)}
    shade = {k: LR_TREE_SHADE * np.array(rp.pv.Color(rp.PALETTE[k]).float_rgb) for k in (1, 2, 3)}
    mother, mother_spans = lines[1]
    split = False
    for k, (segments, spans) in lines.items():
        for n, (points, (first, _)) in enumerate(zip(segments, spans)):
            # The first segment of a daughter starts where the mother line ends, if it follows the mother directly.
            if k > 1 and n == 0 and mother and first - mother_spans[-1][1] <= LR_MAX_GAP + 1:
                points = np.vstack([mother[-1][-1:], points])
                split = True
            if len(points) < max(LR_MIN_SEGMENT, 2):
                continue
            tube = rp.pv.Spline(points, 4 * len(points)).tube(radius=radius)
            plotter.add_mesh(tube, **render_pairs.mesh_kwargs(shade[k]))
    if split:
        plotter.add_mesh(
            rp.pv.Sphere(radius=LR_TREE_NODE * radius, center=mother[-1][-1]), **render_pairs.mesh_kwargs("#1a1a1a")
        )
    center = np.array([(frame[0] + frame[1]) / 2, (frame[2] + frame[3]) / 2, (frame[4] + frame[5]) / 2])
    size = max(frame[1] - frame[0], frame[3] - frame[2], frame[5] - frame[4])
    plotter.camera_position = [center + 4 * size * direction, center, (0, 0, 1)]
    plotter.reset_camera(bounds=frame)
    plotter.camera.zoom(0.95)
    render_pairs.save(plotter, path)


def lr_render(dataset, sample, root, parts):
    raw, _, _ = tu.load_interactive_sample(dataset, sample)
    run = os.path.join(LR_PRED, f"{dataset}_{sample}_{LR_METHODS['microsam2']}_{LR_PROMPTS['box']}")
    gt, table = lr_load(os.path.join(run, "gt", "TRA"))
    tracks = {r[0]: (r[1], r[2], r[3]) for r in table}
    daughters = sorted(t for t, (_, _, p) in tracks.items() if p == root)[:2]
    track_ids = [root] + daughters
    t0, t1 = tracks[root][0], max(tracks[d][1] for d in daughters)
    # The crop: the box around the lineage over all its frames, with a margin.
    _, yy, xx = np.nonzero(np.isin(gt[t0:t1 + 1], track_ids))
    margin = max(8, int(0.25 * max(np.ptp(yy), np.ptp(xx))))
    y0, y1 = max(yy.min() - margin, 0), min(yy.max() + margin + 1, gt.shape[1])
    x0, x1 = max(xx.min() - margin, 0), min(xx.max() + margin + 1, gt.shape[2])
    crop = np.s_[t0:t1 + 1, y0:y1, x0:x1]
    image, reference_ids = np.asarray(raw[crop], dtype="float32"), gt[crop]
    reference = np.zeros(reference_ids.shape, np.int64)
    for k, tid in enumerate(track_ids, start=1):
        reference[reference_ids == tid] = k
    # The time axis gets about the length of the larger side of the crop.
    z_scale = max(y1 - y0, x1 - x0) / (t1 - t0 + 1)
    name = f"{dataset}_{sample}_{root}"
    volume = (image, reference)
    if "pyramid" in parts:
        # A square crop centred on the mother at the first prompt.
        side = min(max(y1 - y0, x1 - x0), gt.shape[1], gt.shape[2])
        cy, cx = [int(c) for c in np.argwhere(gt[t0] == root).mean(0)]
        sy = int(np.clip(cy - side // 2, 0, gt.shape[1] - side))
        sx = int(np.clip(cx - side // 2, 0, gt.shape[2] - side))
        frames = np.asarray(raw[t0:t1 + 1], dtype="float32")
        lr_render_pyramid(frames, gt[t0] == root, (sy, sy + side, sx, sx + side), f"{LR_OUT}/{name}_pyramid")
        print("rendered", name, "pyramid", flush=True)
    if "tree_reference" in parts:
        lr_render_tree(reference, reference, z_scale, f"{LR_OUT}/{name}_tree_reference")
        print("rendered", name, "tree reference", flush=True)
    if "reference" in parts:
        rp.render_combined(name, z_scale, out=f"{LR_OUT}/{name}_volume.png", volume=volume, stand=True, objects=False)
        rp.render_combined(
            name, z_scale, labels=reference, out=f"{LR_OUT}/{name}_reference.png", volume=volume, image=False
        )
    for method, model in LR_METHODS.items():
        for prompt, suffix in LR_PROMPTS.items():
            key = f"{method}_{prompt}"
            if key not in parts and f"tree_{key}" not in parts:
                continue
            folder = os.path.join(LR_PRED, f"{dataset}_{sample}_{model}_{suffix}")
            pred = lr_load(os.path.join(folder, f"corrections_{LR_BUDGET}", "TRA"))[0][crop]
            labels = np.zeros(pred.shape, np.int64)
            for t in range(pred.shape[0]):
                for k in range(1, len(track_ids) + 1):
                    g = reference[t] == k
                    ids, counts = np.unique(pred[t][g], return_counts=True)
                    ids, counts = ids[ids > 0], counts[ids > 0]
                    if not len(ids):
                        continue
                    best = pred[t] == ids[np.argmax(counts)]
                    if counts.max() / (g.sum() + best.sum() - counts.max()) >= LR_MATCH_IOU:
                        labels[t][best & (labels[t] == 0)] = k
            if f"tree_{key}" in parts:
                lr_render_tree(labels, reference, z_scale, f"{LR_OUT}/{name}_tree_{key}")
            if key in parts:
                rp.render_combined(
                    name, z_scale, labels=labels, out=f"{LR_OUT}/{name}_{key}.png", volume=volume, image=False
                )
            print("rendered", name, method, prompt, flush=True)


# The renders of the former 'render_tracking_candidates.py'.
ST_PREDICTIONS = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/tracking/automatic/predictions"
ST_METHODS = {
    "micro-sam2 (APG)": ("hvit_l_cells_apg", METHOD_COLORS["micro-sam2"]),
    "CellposeSAM": ("cpsam_cpsam", METHOD_COLORS["CellposeSAM"]),
}
# The colours of divisualisation; correct links are opaque, since the shadow pass drops translucent meshes.
ST_GOOD_COLOR = np.array([0.86, 0.86, 0.86])
ST_FN_COLOR = np.array([0.0, 1.0, 1.0])
ST_FP_COLOR = np.array([1.0, 0.0, 1.0])
ST_SMOOTHING = 3
ST_GOOD_OPACITY = 1.0
ST_RENDER_SIZE = (1100, 1100)
# Without the studio: no walls, floor or shadows, only the card and the tracks on white.
ST_STUDIO = False
ST_RESULTS = os.path.join(TRACKING_ROOT, "automatic_tracking_results.csv")


def st_seg_scores():
    """The ST_SEG of each (dataset, sequence, method key) from the result table."""
    import pandas as pd
    df = pd.read_csv(ST_RESULTS, dtype={"sequence": str})
    out = {}
    for row in df.itertuples():
        method = row.method if isinstance(row.method, str) else "microsam2"
        if method == "microsam2":
            key = f"hvit_l_cells_{row.mode}"
        elif method == "cpsam":
            key = "cpsam_cpsam"
        else:
            continue
        sequence = str(row.sequence).zfill(2) if row.dataset.startswith("Fluo") else str(row.sequence)
        if row.tracking_model in ("general_2d_w_SAM2_features", "ctc"):
            out[(row.dataset, sequence, key)] = row.SEG
    return out


ST_SEG = {}


def st_hex_rgb(color):
    return np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)]) / 255


def st_match(gt_folder, pred_folder, n_frames):
    """The ground-truth and the predicted graphs with the CTC edge flags, and the scores."""
    from traccuracy import run_metrics
    from traccuracy.loaders import load_ctc_data
    from traccuracy.matchers import CTCMatcher
    from traccuracy.metrics import CTCMetrics
    gt = load_ctc_data(gt_folder, track_path=tu.valid_track_file(gt_folder), name="gt")
    pred_folder = tu.trim_ctc_frames(pred_folder, n_frames)
    try:
        pred = load_ctc_data(pred_folder, name="pred")
    except Exception:
        pred = load_ctc_data(pred_folder, name="pred", run_checks=False)
    results, matched = run_metrics(gt, pred, matcher=CTCMatcher(), metrics=[CTCMetrics()])
    return matched.gt_graph, matched.pred_graph, results[0]["results"]


def st_smooth_positions(graph, iterations=ST_SMOOTHING):
    """The x, y st_position of each node, averaged with its predecessor and successors a few times, so that the tracks
    run smoothly; the time stays."""
    g = graph.graph
    xy = {n: np.array([g.nodes[n]["x"], g.nodes[n]["y"]], float) for n in g.nodes}
    for _ in range(iterations):
        new = {}
        for n in g.nodes:
            near = [xy[m] for m in list(g.predecessors(n)) + list(g.successors(n))]
            new[n] = (2 * xy[n] + sum(near)) / (2 + len(near)) if near else xy[n]
        xy = new
    graph.smooth_xy = xy


# The origin (y, x) of the shown window of the field of view; (0, 0) shows all of it.
ST_ORIGIN = [0, 0]
# The card thickness and the track start, per 512 pixels of image size.
ST_CARD = {"thickness": 3.0, "base": 6.0}


def st_position(graph, node, height):
    nodes = graph.graph.nodes
    x, y = graph.smooth_xy[node] if hasattr(graph, "smooth_xy") else (nodes[node]["x"], nodes[node]["y"])
    return np.array([x - ST_ORIGIN[1], -(y - ST_ORIGIN[0]), ST_CARD["base"] + graph.graph.nodes[node]["t"] * height])


def st_best_window(edge_sets, shape, share, bounds=None):
    """The square window, of 'share' of the smaller image side, where the second method has the most more errors than
    the first. 'edge_sets' holds per method a list of (graph, edges) with its errors; an edge counts at its first
    node. 'bounds' (y0, y1, x0, x1) limits the window to a part of the frame. Returns (y0, x0, side)."""
    side = int(share * min(shape))
    by0, by1, bx0, bx1 = bounds if bounds is not None else (0, shape[0], 0, shape[1])
    step = max(side // 8, 1)
    counts = []
    for sets in edge_sets:
        grid = np.zeros(shape)
        for graph, edges in sets:
            for a, _ in edges:
                node = graph.graph.nodes[a]
                grid[min(int(node["y"]), shape[0] - 1), min(int(node["x"]), shape[1] - 1)] += 1
        counts.append(grid.cumsum(0).cumsum(1))

    def box(c, y, x):
        total = c[y + side - 1, x + side - 1]
        total -= c[y - 1, x + side - 1] if y else 0
        total -= c[y + side - 1, x - 1] if x else 0
        total += c[y - 1, x - 1] if y and x else 0
        return total
    best = max(
        (box(counts[1], y, x) - box(counts[0], y, x), y, x)
        for y in range(by0, by1 - side + 1, step) for x in range(bx0, bx1 - side + 1, step)
    )
    return best[1], best[2], side


def st_inside(graph, edges, window):
    y0, x0, side = window
    keep = []
    for a, b in edges:
        nodes = [graph.graph.nodes[n] for n in (a, b)]
        if all(y0 <= n["y"] < y0 + side and x0 <= n["x"] < x0 + side for n in nodes):
            keep.append((a, b))
    return keep


def st_segments_mesh(graph, edges, colors, radius, height):
    """One tube mesh of all edges, each edge in its colour."""
    if not edges:
        return None
    points, lines, rgb = [], [], []
    for k, (a, b) in enumerate(edges):
        points += [st_position(graph, a, height), st_position(graph, b, height)]
        lines += [2, 2 * k, 2 * k + 1]
        rgb += [colors[k], colors[k]]
    poly = pv.PolyData(np.array(points), lines=np.array(lines))
    poly.point_data["rgb"] = (np.array(rgb) * 255).astype("uint8")
    return poly.tube(radius=radius, n_sides=12, capping=True)


def st_lineage_colors(graph):
    """One colour per connected lineage of the graph: golden-ratio hues."""
    import networkx as nx
    colors = {}
    components = sorted(
        nx.weakly_connected_components(graph.graph), key=lambda c: min(graph.graph.nodes[n]["t"] for n in c)
    )
    for k, component in enumerate(components):
        rgb = np.array(colorsys.hsv_to_rgb((k * 0.6180339887) % 1.0, 0.65, 0.9))
        for node in component:
            colors[node] = rgb
    return colors


def st_render(raw0, meshes, shape, n_frames, height):
    card = (np.stack([raw0] * 3, -1) * 255).astype("uint8")
    top = ST_CARD["base"] + (n_frames - 1) * height
    bounds = (0, shape[1], -shape[0], 0, 0, top)
    view = d_view(18, -60)[0]
    plotter = d_studio(ST_RENDER_SIZE, bounds, view, 900, 1.0)
    if not ST_STUDIO:
        plotter.disable_shadows()
    d_card(plotter, card, (0, 0, 0), 1.0, ST_CARD["thickness"])
    for mesh, opacity in meshes:
        if mesh is not None:
            plotter.add_mesh(
                mesh, scalars="rgb", rgb=True, smooth_shading=True, specular=0.3, specular_power=30, ambient=0.35,
                diffuse=0.75, show_scalar_bar=False, opacity=opacity
            )
    if ST_STUDIO:
        d_backdrop(plotter, bounds, view, 10, 300, 40)
    image = d_screenshot(plotter, ST_RENDER_SIZE)
    plotter.close()
    mask = (image < 245).any(-1)
    rows, cols = np.nonzero(mask)
    return image[max(rows.min() - 15, 0):rows.max() + 15, max(cols.min() - 15, 0):cols.max() + 15]


def st_candidate(out_folder, dataset, sequence, tracking_model, window_share=None, stretch=1.0):
    """Render one movie. 'window_share' ('none' for the whole frame) shows only a window of the frame; 'stretch'
    makes the time axis longer."""
    window_share = None if window_share in (None, "none") else window_share
    stretch = float(stretch)
    from traccuracy import EdgeFlag
    gt_folder = os.path.join(tu.gt_folder_of(dataset, sequence), "TRA")
    n_frames = len([f for f in os.listdir(gt_folder) if f.endswith(".tif")])
    raw0 = tu.load_frames(dataset, sequence, 0, 1)[0].astype("float32")
    if raw0.ndim == 3:
        raw0 = raw0.max(axis=0)
    # Movies padded with black (as DynamicNuclearNet) show only the box of the image without the padding.
    ST_ORIGIN[:] = [0, 0]
    filled = np.nonzero(raw0 > raw0.min())
    if len(filled[0]):
        y0, y1, x0, x1 = filled[0].min(), filled[0].max() + 1, filled[1].min(), filled[1].max() + 1
        if (y1 - y0, x1 - x0) != raw0.shape:
            raw0 = raw0[y0:y1, x0:x1]
            ST_ORIGIN[:] = [y0, x0]
            print("padding removed, image box (y0, y1, x0, x1):", (y0, y1, x0, x1), flush=True)
    lo, hi = np.percentile(raw0, [1, 99.8])
    raw0 = np.clip((raw0 - lo) / max(hi - lo, 1e-6), 0, 1)
    shape = raw0.shape
    # Time stretches to about the size of the image, so that the cube is not flat.
    height = stretch * 0.8 * max(shape) / max(n_frames - 1, 1)
    radius = max(shape) / 260

    renders, titles, found = [], [], []
    from traccuracy import NodeFlag
    for name, (key, color) in ST_METHODS.items():
        tag = f"{dataset}_{sequence}_{key}_{tracking_model}_greedy"
        pred_folder = os.path.join(ST_PREDICTIONS, tag)
        # Sparse ground truth (DRO, TRIC) is scored on the annotated cells only, as in the evaluation.
        if dataset in tu.SPARSE_GT_DATASETS and os.path.isdir(pred_folder + "_annotated"):
            pred_folder += "_annotated"
        gt_graph, pred_graph, scores = st_match(gt_folder, pred_folder, n_frames)
        st_smooth_positions(gt_graph)
        st_smooth_positions(pred_graph)
        # False links: the CTC false positive edges and the links of false detections (as CellposeSAM's jumps).
        false_nodes = set(pred_graph.get_nodes_with_flag(NodeFlag.CTC_FALSE_POS))
        false_pos = set(pred_graph.get_edges_with_flag(EdgeFlag.CTC_FALSE_POS))
        false_pos |= {e for e in pred_graph.graph.edges if e[0] in false_nodes or e[1] in false_nodes}
        good = [e for e in pred_graph.graph.edges if e not in false_pos]
        missed = list(gt_graph.get_edges_with_flag(EdgeFlag.CTC_FALSE_NEG))
        found.append((name, pred_graph, gt_graph, good, list(false_pos), missed, scores))
    # With 'window_share', only the square window where CellposeSAM has the most more errors than micro-sam2 is shown.
    window = None
    if window_share is not None:
        edge_sets = [[(f[1], f[4]), (f[2], f[5])] for f in found]
        window = st_best_window(edge_sets, shape, float(window_share))
        y0, x0, side = window
        ST_ORIGIN[:] = [y0, x0]
        raw0 = raw0[y0:y0 + side, x0:x0 + side]
        shape = raw0.shape
        height = stretch * 0.8 * max(shape) / max(n_frames - 1, 1)
        radius = max(shape) / 260
        print("window (y0, x0, side):", window, flush=True)
    ST_CARD.update(thickness=3.0 * max(shape) / 512, base=6.0 * max(shape) / 512)
    for name, pred_graph, gt_graph, good, false_pos, missed, scores in found:
        key = ST_METHODS[name][0]
        if window is not None:
            good, false_pos, missed = (
                st_inside(pred_graph, good, window), st_inside(pred_graph, false_pos, window),
                st_inside(gt_graph, missed, window)
            )
        # Thin correct links and thick errors, so that the errors stand out.
        meshes = [
            (st_segments_mesh(pred_graph, good, [ST_GOOD_COLOR] * len(good), radius * 0.6, height), ST_GOOD_OPACITY),
            (st_segments_mesh(pred_graph, list(false_pos), [ST_FP_COLOR] * len(false_pos), radius * 1.6, height), 1.0),
            (st_segments_mesh(gt_graph, missed, [ST_FN_COLOR] * len(missed), radius * 1.6, height), 1.0),
        ]
        renders.append(st_render(raw0, meshes, shape, n_frames, height))
        seg = ST_SEG.get((dataset, sequence, key))
        titles.append(
            f"{name}\nSEG {seg:.2f} · TRA {scores['TRA']:.2f} · {len(false_pos)} false / {len(missed)} " "missed links"
        )
        print(
            dataset, sequence, name, {k: round(scores[k], 3) for k in ("TRA", "DET", "LNK")}, len(false_pos),
            len(missed), flush=True
        )

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.6))
    for ax, image, title in zip(axes, renders, titles):
        ax.imshow(image)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(
        f"{dataset} {sequence}: space-time view (x, y on the floor, time up); grey = correct links, "
        "magenta = false links, cyan = missed links", fontsize=10
    )
    suffix = ("" if window is None else f"_window{window_share}") + ("" if stretch == 1.0 else f"_stretch{stretch:g}")
    fig.savefig(
        os.path.join(out_folder, f"automatic_tracking_{dataset}_{sequence}{suffix}.png"), dpi=200, bbox_inches="tight"
    )
    plt.close(fig)
    for image, title in zip(renders, ["microsam2_apg", "cpsam"]):
        Image.fromarray(image).save(os.path.join(out_folder, f"render_{dataset}_{sequence}{suffix}_{title}.png"))


# The renders of the former 'render_tracking_3d_views.py'.
FV_NUCLEUS_COLOR = (0.8, 0.8, 0.8)
# The bands of the first and last nuclei and the volume block, per larger image side.
FV_FLOOR_BAND = 0.08
FV_BLOCK_THICKNESS = 0.12


def fv_gray_rgb(image):
    lo, hi = np.percentile(image, [1, 99.8])
    gray = np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1)
    return (np.repeat(gray[..., None], 3, -1) * 255).astype("uint8")


def fv_textured_face(plotter, rgb, dims, origin, spacing):
    """A flat image on one face of the block; the points of the face run over its first axis first."""
    grid = pv.ImageData(dimensions=dims, spacing=spacing, origin=origin)
    grid.point_data["rgb"] = rgb.reshape(-1, 3)
    plotter.add_mesh(grid, scalars="rgb", rgb=True, ambient=1.0, diffuse=0.0, specular=0.0, show_scalar_bar=False)


def fv_volume_block(plotter, raw, thickness, view):
    """The volume (z, y, x) as a block from z = 0 to 'thickness' (x right, y turned, as the tracks): the maximum
    projection along z on its top face, and the projections along x and y on the two side faces the camera sees."""
    nz, ny, nx = raw.shape
    dz = thickness / max(nz - 1, 1)
    # The top face: row j of the face is image row ny - 1 - j, since y is turned.
    fv_textured_face(plotter, fv_gray_rgb(raw.max(axis=0))[::-1], (nx, ny, 1), (0, -(ny - 1), thickness), (1, 1, 1))
    # The side faces towards the camera: x at its far end or start, y (turned) at its far end or start.
    side_x = nx - 1 if view[0] > 0 else 0
    side_y = -(ny - 1) if view[1] < 0 else 0
    along_x = fv_gray_rgb(raw.max(axis=2))[:, ::-1]  # (z, y), the columns run over the turned y
    fv_textured_face(plotter, along_x, (1, ny, nz), (side_x, -(ny - 1), 0), (1, 1, dz))
    along_y = fv_gray_rgb(raw.max(axis=1))  # (z, x)
    fv_textured_face(plotter, along_y, (nx, 1, nz), (0, side_y, 0), (1, 1, dz))
    box = pv.Box((0, nx - 1, -(ny - 1), 0, 0, thickness)).extract_feature_edges()
    plotter.add_mesh(box, color="#a8a8a8", line_width=2.0)


def fv_load_frame(folder, t):
    return tifffile.imread(sorted(glob.glob(os.path.join(folder, "*.tif")))[t])


def fv_xyz(graph, node, z_scale):
    n = graph.graph.nodes[node]
    return np.array([n["x"], -n["y"], n.get("z", 0.0) * z_scale])


def fv_tubes(points_pairs, color, radius):
    if not points_pairs:
        return None
    points, lines = [], []
    for k, (a, b) in enumerate(points_pairs):
        points += [a, b]
        lines += [2, 2 * k, 2 * k + 1]
    return pv.PolyData(np.array(points), lines=np.array(lines)).tube(radius=radius, n_sides=12, capping=True), color


def fv_add(plotter, item):
    if item is not None:
        mesh, color = item
        plotter.add_mesh(
            mesh, color=color, smooth_shading=True, specular=0.3, specular_power=30, ambient=0.35, diffuse=0.75
        )


def fv_nuclei_meshes(labels, z_scale, squash=None, lift=0.0):
    """The surfaces of the objects, with y turned as in the tracks; 'squash' scales z to that height."""
    meshes = []
    for _, mesh in surfaces(labels.astype(np.int64), z_scale, sigma=1.0, n_iter=20):
        points = mesh.points.copy()
        points[:, 1] *= -1
        mesh.points = points
        meshes.append(mesh)
    if squash is not None and meshes:
        top = max(m.bounds[5] for m in meshes)
        for mesh in meshes:
            points = mesh.points.copy()
            points[:, 2] = lift + points[:, 2] / max(top, 1e-6) * squash
            mesh.points = points
    return meshes


def fv_render_scene(add_content, bounds, card, card_z):
    """Render the scene; 'card' is the raw card at 'card_z', or a function that draws the floor."""
    view = d_view(18, -60)[0]
    plotter = d_studio(ST_RENDER_SIZE, bounds, view, 900, 1.0)
    if not ST_STUDIO:
        plotter.disable_shadows()
    if callable(card):
        card(plotter, view)
    else:
        d_card(plotter, card, (0, 0, card_z), 1.0, ST_CARD["thickness"])
    add_content(plotter)
    if ST_STUDIO:
        d_backdrop(plotter, bounds, view, 10, 300, 40)
    image = d_screenshot(plotter, ST_RENDER_SIZE)
    plotter.close()
    mask = (image < 245).any(-1)
    rows, cols = np.nonzero(mask)
    return image[max(rows.min() - 15, 0):rows.max() + 15, max(cols.min() - 15, 0):cols.max() + 15]


# The renders of the former 'lineage_search/find_lineages_both.py'.
LS_PRED = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/tracking/interactive/predictions"
LS_DATASETS = ["dynamicnuclearnet", "toiam", "yeaz", "cellacdc", "alfi", "ecoli", "cardioblast", "wingdisc"]
LS_RUNS = {
    "s2_point": "hvit_l_cells_point_c4_iou0.9", "s2_box": "hvit_l_cells_box_boxcorr_c4_iou0.9",
    "s1_point": "vit_l_lm_point_c4_iou0.9", "s1_box": "vit_l_lm_box_boxcorr_c4_iou0.9",
}
LS_BUDGETS = (0,)


def ls_load(folder):
    masks = np.stack([tifffile.imread(p) for p in sorted(glob.glob(os.path.join(folder, "*.tif")))])
    table = np.loadtxt(os.path.join(folder, "man_track.txt"), dtype=int, ndmin=2)
    return masks, table


def ls_best_matches(gt, pred):
    """For each frame, the best predicted id and its IoU for each ground-truth id."""
    out = {}
    for t in range(gt.shape[0]):
        g, p = gt[t].ravel(), pred[t].ravel()
        fg = g > 0
        if not fg.any():
            continue
        g_ids, g_area = np.unique(g[fg], return_counts=True)
        p_ids, p_area = np.unique(p[p > 0], return_counts=True)
        p_area = dict(zip(p_ids, p_area))
        g_area = dict(zip(g_ids, g_area))
        both = fg & (p > 0)
        pairs, counts = np.unique(np.stack([g[both], p[both]]), axis=1, return_counts=True)
        best = {}
        for (gi, pi), c in zip(pairs.T, counts):
            iou = c / (g_area[gi] + p_area[pi] - c)
            if iou > best.get(gi, (0, 0.0))[1]:
                best[gi] = (pi, iou)
        for gi in g_ids:
            out[t, gi] = best.get(gi, (0, 0.0))
    return out


def ls_sample_lineages(args):
    dataset, sample = args
    root_folder = os.path.join(LS_PRED, f"{dataset}_{sample}_{LS_RUNS['s2_box']}")
    gt, table = ls_load(os.path.join(root_folder, "gt", "TRA"))
    tracks = {r[0]: (r[1], r[2], r[3]) for r in table}
    children = {}
    for tid, (_, _, parent) in tracks.items():
        if parent in tracks:
            children.setdefault(parent, []).append(tid)
    roots = [tid for tid, (_, _, parent) in tracks.items() if parent not in tracks]

    def descendants(tid):
        out = [tid]
        for c in children.get(tid, []):
            out += descendants(c)
        return out
    lineages = {r: descendants(r) for r in roots if children.get(r) or any(children.get(d) for d in descendants(r))}
    if not lineages:
        return []
    rows = {}
    for r, ts in lineages.items():
        rows[r] = {
            "dataset": dataset, "sample": sample, "root": r, "n_tracks": len(ts),
            "n_divisions": sum(1 for t in ts if len(children.get(t, [])) >= 2),
            "start": min(tracks[t][0] for t in ts), "end": max(tracks[t][1] for t in ts),
            "n_track_frames": sum(tracks[t][1] - tracks[t][0] + 1 for t in ts),
        }
    for run, suffix in LS_RUNS.items():
        folder = os.path.join(LS_PRED, f"{dataset}_{sample}_{suffix}")
        if not os.path.isdir(folder):
            continue
        for budget in LS_BUDGETS:
            pred, ptable = ls_load(os.path.join(folder, f"corrections_{budget}", "TRA"))
            pparent = {r[0]: r[3] for r in ptable}
            matches = ls_best_matches(gt, pred)
            for root, ts in lineages.items():
                ious = [
                    matches.get((t, tid), (0, 0.0))[1] for tid in ts for t in range(tracks[tid][0], tracks[tid][1] + 1)
                ]
                correct = True
                for tid in ts:
                    daughters = children.get(tid, [])
                    if len(daughters) < 2:
                        continue
                    mother = matches.get((tracks[tid][1], tid), (0, 0))[0]
                    picked = [matches.get((tracks[d][0], d), (0, 0))[0] for d in daughters]
                    if mother == 0 or 0 in picked or len(set(picked)) < len(picked) or any(
                        pparent.get(p, 0) != mother for p in picked
                    ):
                        correct = False
                rows[root][f"iou_{run}_{budget}"] = float(np.mean(ious))
                rows[root][f"div_{run}_{budget}"] = correct
    return list(rows.values())


# Panel b: a small error-dense window of each movie over its first frame, so that the tracks are visible.
# Each entry is (dataset, sequence, Trackastra model, window share, time stretch).
PYRAMID_RENDERS = [
    ("DynamicNuclearNet", "3T3_nuc_s2_p1_1c", "general_2d_w_SAM2_features", 0.3, 1.0),
    ("WingDisc", "Ecd_20150608_P1", "general_2d_w_SAM2_features", 0.12, 1.0),
    ("Fluo-N3DH-SIM+", "01", "ctc", 0.35, 1.0),
]
PYRAMID_SIZE = (1400, 1400)


def pyr_shot(rgb, meshes, shape, top, elevation, window=None):
    """A space-time scene of the card 'rgb' (x right, y down the rows) and the tube meshes, without the studio.
    Return the cropped image, the image positions of the card corners (top left, top right, bottom right, bottom left)
    on its top face, and those of the 'window' (y0, x0, side) on the card."""
    bounds = (0, shape[1], -shape[0], 0, 0, top)
    view = d_view(elevation, -60)[0]
    plotter = d_studio(PYRAMID_SIZE, bounds, view, 900, 1.0)
    plotter.disable_shadows()
    thickness = ST_CARD["thickness"]
    d_card(plotter, (rgb * 255).astype("uint8"), (0, 0, 0), 1.0, thickness)
    for mesh, opacity in meshes:
        if mesh is not None:
            plotter.add_mesh(
                mesh, scalars="rgb", rgb=True, smooth_shading=True, specular=0.3, specular_power=30, ambient=0.35,
                diffuse=0.75, show_scalar_bar=False, opacity=opacity
            )
    z = thickness + 0.6
    h, w = shape
    if window is not None:
        y0, x0, side = window
        box = [(x0, -y0, z), (x0 + side, -y0, z), (x0 + side, -y0 - side, z), (x0, -y0 - side, z)]
        plotter.add_mesh(pv.lines_from_points(np.array(box + box[:1])), color=(0.05, 0.05, 0.05), line_width=6)
    image = d_screenshot(plotter, PYRAMID_SIZE)
    corners = np.array(d_project(plotter, [(0, 0, z), (w, 0, z), (w, -h, z), (0, -h, z)]))
    box_px = np.array(d_project(plotter, box)) if window is not None else None
    plotter.close()
    rows, cols = np.nonzero((image < 245).any(-1))
    r0, c0 = max(rows.min() - 10, 0), max(cols.min() - 10, 0)
    image = image[r0:rows.max() + 10, c0:cols.max() + 10]
    shift = np.array([c0, r0])
    return image, corners - shift, (None if box_px is None else box_px - shift)


def pyr_render(out_folder, dataset, sequence, tracking_model, share, stretch):
    """The pyramid view of one movie: 'pyramid_<name>_overview.png' and 'pyramid_<name>_<method>.png', and the card
    corners and the window square in 'pyramid_<name>.json'."""
    from traccuracy import EdgeFlag, NodeFlag
    os.makedirs(out_folder, exist_ok=True)
    gt_folder = os.path.join(tu.gt_folder_of(dataset, sequence), "TRA")
    n_frames = len([f for f in os.listdir(gt_folder) if f.endswith(".tif")])
    raw = tu.load_frames(dataset, sequence, 0, 1)[0].astype("float32")
    if raw.ndim == 3:
        raw = raw.max(axis=0)
    full_shape = raw.shape
    # the box of the image without black padding (as in DynamicNuclearNet)
    filled = np.nonzero(raw > raw.min())
    fy0, fy1, fx0, fx1 = filled[0].min(), filled[0].max() + 1, filled[1].min(), filled[1].max() + 1
    lo, hi = np.percentile(raw[fy0:fy1, fx0:fx1], [1, 99.8])
    gray = np.clip((raw - lo) / max(hi - lo, 1e-6), 0, 1)

    found = []
    for name, (key, _) in ST_METHODS.items():
        pred_folder = os.path.join(ST_PREDICTIONS, f"{dataset}_{sequence}_{key}_{tracking_model}_greedy")
        if dataset in tu.SPARSE_GT_DATASETS and os.path.isdir(pred_folder + "_annotated"):
            pred_folder += "_annotated"
        gt_graph, pred_graph, scores = st_match(gt_folder, pred_folder, n_frames)
        st_smooth_positions(gt_graph)
        st_smooth_positions(pred_graph)
        false_nodes = set(pred_graph.get_nodes_with_flag(NodeFlag.CTC_FALSE_POS))
        false_pos = set(pred_graph.get_edges_with_flag(EdgeFlag.CTC_FALSE_POS))
        false_pos |= {e for e in pred_graph.graph.edges if e[0] in false_nodes or e[1] in false_nodes}
        good = [e for e in pred_graph.graph.edges if e not in false_pos]
        missed = list(gt_graph.get_edges_with_flag(EdgeFlag.CTC_FALSE_NEG))
        found.append((key, pred_graph, gt_graph, good, list(false_pos), missed))
    # the window keeps half a window from the border, where most errors are tracks leaving the frame
    share = share * min(fy1 - fy0, fx1 - fx0) / min(full_shape)
    margin = int(share * min(full_shape) / 2)
    y0, x0, side = st_best_window(
        [[(f[1], f[4]), (f[2], f[5])] for f in found], full_shape, share,
        (fy0 + margin, fy1 - margin, fx0 + margin, fx1 - margin)
    )
    window = (y0, x0, side)
    name = f"{dataset}_{sequence}"
    layout = {"window": [int(y0), int(x0), int(side)]}

    # the overview: the first frame of the image box with the window on it, seen from higher up
    box_shape = (fy1 - fy0, fx1 - fx0)
    ST_CARD.update(thickness=3.0 * max(box_shape) / 512, base=6.0 * max(box_shape) / 512)
    rgb = np.repeat(gray[fy0:fy1, fx0:fx1, None], 3, -1)
    image, corners, box = pyr_shot(rgb, [], box_shape, ST_CARD["thickness"] + 1, 40, (y0 - fy0, x0 - fx0, side))
    Image.fromarray(image).save(os.path.join(out_folder, f"pyramid_{name}_overview.png"))
    layout["overview"] = {"corners": corners.tolist(), "box": box.tolist()}

    # the window of each method, as the space-time view
    ST_ORIGIN[:] = [y0, x0]
    ST_CARD.update(thickness=3.0 * side / 512, base=6.0 * side / 512)
    height = stretch * 0.8 * side / max(n_frames - 1, 1)
    radius = side / 200
    crop = np.repeat(gray[y0:y0 + side, x0:x0 + side, None], 3, -1)
    for key, pred_graph, gt_graph, good, false_pos, missed in found:
        good, false_pos, missed = (
            st_inside(pred_graph, good, window), st_inside(pred_graph, false_pos, window),
            st_inside(gt_graph, missed, window)
        )
        meshes = [
            (st_segments_mesh(pred_graph, good, [ST_GOOD_COLOR] * len(good), radius * 0.6, height), ST_GOOD_OPACITY),
            (st_segments_mesh(pred_graph, false_pos, [ST_FP_COLOR] * len(false_pos), radius * 1.6, height), 1.0),
            (st_segments_mesh(gt_graph, missed, [ST_FN_COLOR] * len(missed), radius * 1.6, height), 1.0),
        ]
        top = ST_CARD["base"] + (n_frames - 1) * height
        image, corners, _ = pyr_shot(crop, meshes, (side, side), top, 18)
        tag = "microsam2_apg" if key.startswith("hvit") else "cpsam"
        Image.fromarray(image).save(os.path.join(out_folder, f"pyramid_{name}_{tag}.png"))
        layout[tag] = {"corners": corners.tolist(), "false": len(false_pos), "missed": len(missed)}
        print("pyramid", name, tag, len(false_pos), "false", len(missed), "missed links in the window", flush=True)
    json.dump(layout, open(os.path.join(out_folder, f"pyramid_{name}.json"), "w"), indent=1)


def fv_render_floor(out_folder, dataset, sequence):
    """The floor view of a 3D+t movie for micro-sam2 APG and CellposeSAM: the first frame as a volume block, the
    nuclei of the first and the last frame as 3d surfaces, and the links between them in the colours of the
    space-time views."""
    os.makedirs(out_folder, exist_ok=True)
    gt_folder = os.path.join(tu.gt_folder_of(dataset, sequence), "TRA")
    n_frames = len([f for f in os.listdir(gt_folder) if f.endswith(".tif")])
    raw0 = tu.load_frames(dataset, sequence, 0, 1)[0].astype("float32")
    z_scale = z_scale_of(dataset, fv_load_frame(gt_folder, 0))
    ny, nx = raw0.shape[1:]
    big = max(ny, nx)
    ST_CARD.update(thickness=3.0 * big / 512, base=6.0 * big / 512)
    radius = big / 260
    for name, (key, _) in ST_METHODS.items():
        pred_folder = os.path.join(ST_PREDICTIONS, f"{dataset}_{sequence}_{key}_ctc_greedy")
        gt_graph, pred_graph, scores = st_match(gt_folder, pred_folder, n_frames)
        false_nodes = set(pred_graph.get_nodes_with_flag(NodeFlag.CTC_FALSE_POS))
        false_pos = set(pred_graph.get_edges_with_flag(EdgeFlag.CTC_FALSE_POS))
        false_pos |= {e for e in pred_graph.graph.edges if e[0] in false_nodes or e[1] in false_nodes}
        good = [e for e in pred_graph.graph.edges if e not in false_pos]
        missed = list(gt_graph.get_edges_with_flag(EdgeFlag.CTC_FALSE_NEG))
        tag = "microsam2_apg" if key.startswith("hvit") else "cpsam"
        height = 0.8 * big / max(n_frames - 1, 1)
        band = FV_FLOOR_BAND * big
        block = FV_BLOCK_THICKNESS * big
        start = block + band
        end = start + (n_frames - 1) * height
        nuclei = (
            fv_nuclei_meshes(fv_load_frame(pred_folder, 0), z_scale, squash=band, lift=block)
            + fv_nuclei_meshes(fv_load_frame(pred_folder, n_frames - 1), z_scale, squash=band, lift=end)
        )

        def pairs(graph, edges):
            nodes = graph.graph.nodes
            return [
                tuple(np.array([nodes[n]["x"], -nodes[n]["y"], start + nodes[n]["t"] * height]) for n in edge)
                for edge in edges
            ]

        def content(plotter):
            for mesh in nuclei:
                plotter.add_mesh(
                    mesh, color=FV_NUCLEUS_COLOR, smooth_shading=True, specular=0.2, ambient=0.35, diffuse=0.7
                )
            fv_add(plotter, fv_tubes(pairs(pred_graph, good), tuple(ST_GOOD_COLOR), radius * 0.6))
            fv_add(plotter, fv_tubes(pairs(pred_graph, false_pos), tuple(ST_FP_COLOR), radius * 1.6))
            fv_add(plotter, fv_tubes(pairs(gt_graph, missed), tuple(ST_FN_COLOR), radius * 1.6))

        def floor(plotter, camera_view):
            fv_volume_block(plotter, raw0, block, camera_view)
        image = fv_render_scene(content, (0, nx, -ny, 0, 0, end + band), floor, 0.0)
        Image.fromarray(image).save(os.path.join(out_folder, f"render_{dataset}_{sequence}_floor_{tag}.png"))
        print("rendered the floor view of", dataset, sequence, name, len(false_pos), len(missed), flush=True)


def ls_score_lineages(dataset):
    """Score all dividing lineages of the interactive runs of one dataset (see 'ls_sample_lineages')."""
    from multiprocessing import Pool
    jobs = []
    for folder in sorted(glob.glob(os.path.join(LS_PRED, f"{dataset}_*_{LS_RUNS['s2_box']}"))):
        sample = os.path.basename(folder)[len(dataset) + 1:-len(LS_RUNS["s2_box"]) - 1]
        jobs.append((dataset, sample))
    with Pool(16) as pool:
        rows = [row for part in pool.imap_unordered(ls_sample_lineages, jobs) for row in part]
    os.makedirs(LINEAGE_SCORES, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(LINEAGE_SCORES, f"{dataset}.csv"), index=False)
    print(len(rows), "dividing lineages of", dataset, "from", len(jobs), "movies")


RENDERS = ["lineage_scores", "lineages", "spacetime", "floor", "pyramid", "beke"]


def render(targets):
    """Make the renders (GPU node) or the lineage scores of the figure: 'lineage_scores' (panel d), 'lineages'
    (panel d), 'spacetime' and 'floor' (panel b) and 'beke' (panel e)."""
    global LR_OUT
    os.chdir(ROOT)
    if "beke" in targets:
        render_beke()
    if set(targets) & {"lineage_scores", "lineages", "spacetime", "floor", "pyramid"}:
        load_render_modules(LINEAGE_RENDERS)
    if "lineage_scores" in targets:
        for dataset, _, _, _ in INTERACTIVE_QUALITATIVE:
            ls_score_lineages(dataset)
    if "lineages" in targets:
        LR_OUT = LINEAGE_RENDERS
        os.makedirs(LR_OUT, exist_ok=True)
        for dataset, sample, root, _ in INTERACTIVE_QUALITATIVE:
            lr_render(dataset, sample, root, LINEAGE_PARTS)
    if "spacetime" in targets:
        ST_SEG.update(st_seg_scores())
        for dataset, sequence, model, window, stretch in SPACETIME_RENDERS:
            st_candidate(AUTOMATIC_RENDERS, dataset, sequence, model, window, stretch)
    if "pyramid" in targets:
        for dataset, sequence, model, share, stretch in PYRAMID_RENDERS:
            pyr_render(AUTOMATIC_RENDERS, dataset, sequence, model, share, stretch)
    if "floor" in targets:
        for dataset, sequence in FLOOR_RENDERS:
            fv_render_floor(AUTOMATIC_RENDERS, dataset, sequence)


# The combined figure: the four plots top to bottom, as panels a to d.
COMBINED_NAME = "figure_5"
COMBINED_ORDER = [
    "automatic_tracking", "automatic_tracking_qualitative", "interactive_tracking", "interactive_qualitative_beke",
]
COMBINED_DPI = 1200
# The gap above a plot, in inches, where it differs from Figure 2.
COMBINED_GAPS = {"interactive_tracking": 0.05}


def plot_figure():
    """The plots of COMBINED_ORDER in one figure, top to bottom, as 'plots_for_figure2.plot_figure', with the gaps of
    COMBINED_GAPS."""
    margins = {name: fig2.margins_of(name, make_plot) for name in COMBINED_ORDER}
    fig = plt.figure(figsize=(fig2.FIGURE_WIDTH, 40))
    total = fig.get_figheight()
    start = top = total - 1
    for k, name in enumerate(COMBINED_ORDER):
        if k > 0:
            top += fig2.PANEL_GAP - COMBINED_GAPS.get(name, fig2.PANEL_GAP)

        def region(height, name=name, top=top):
            upper = top + margins[name][0]
            ratios = [max(total - upper, 1e-9), height, max(upper - height, 1e-9)]
            grid = fig.add_gridspec(3, 1, height_ratios=ratios, hspace=0)
            return fig.add_subfigure(grid[1])
        sub = make_plot(name, region, compact=True)
        label_top = top
        if k == 0:
            renderer = fig.canvas.get_renderer()
            label_top = max(ax.get_window_extent(renderer).y1 for ax in sub.axes) / fig.dpi
        fig.text(
            0, label_top / total, "abcdefgh"[k], ha="left", va="top", fontsize=fig2.PANEL_LABEL_SIZE, fontweight="bold",
            color=TEXT_COLOR
        )
        top -= fig2.height_of(sub) - sum(margins[name]) + fig2.PANEL_GAP
    return fig, top + fig2.PANEL_GAP, start


def make_plot(name, region=None, compact=False):
    """Make one plot, as its own figure or in a region of the combined figure."""
    if name == "interactive_qualitative_beke":
        return plot_interactive_qualitative_beke(region=region)
    return PLOTS[name][2](region=region)


def main():
    names = list(PLOTS) + [COMBINED_NAME]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=names, default=names, help="Make only these plots.")
    parser.add_argument(
        "--render", nargs="+", choices=RENDERS, help="Make these renders (on a GPU node) instead of the plots."
    )
    args = parser.parse_args()
    if args.render:
        render(args.render)
        return
    set_style()
    fig2.OUTPUT_DIR = OUTPUT_DIR
    for name, (kind, style, plot) in PLOTS.items():
        if name in args.only:
            fig2.save_figure(plot(), name, kind, style)
    if COMBINED_NAME in args.only:
        fig2.save_combined(*plot_figure(), name=COMBINED_NAME, dpi=COMBINED_DPI)


if __name__ == "__main__":
    main()
