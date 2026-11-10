"""Plots for Figure 2 of the micro-sam2 paper.

The figure compares 2d automatic and interactive instance segmentation of micro-sam2 (hvit_l) with the baselines.
The script writes these plots to 'figures/figure2/<automatic|interactive>/<quantitative|qualitative>':
- 'automatic_2d': the mSA of each method per modality, for the in-domain and the out-of-domain datasets.
- 'interactive_2d': the mSA over the correction iterations, with a point or a box as the first prompt.
- 'automatic_2d_qualitative': one test image per modality and domain, with the reference and the predictions.
- 'interactive_2d_qualitative': the crops of one object per modality and domain, segmented from one prompt.
- 'figure_2' (in 'figures/final'): the four plots in one figure, as panels a to d.

Usage:

    python plots_for_figure2.py
    python plots_for_figure2.py --only automatic_2d_qualitative interactive_2d_qualitative

The inputs are:
- RESULTS_CSV and INTERACTIVE_CSV: the scores from the scripts in 'finetuning/v2/evaluation'.
- QUALITATIVE_PREDICTIONS: the predictions for the automatic qualitative plot. To make them, write the script with
  '--write_prediction_script run.sh' and run it on a GPU node.
- SINGLE_PROMPT_RUNS: the single-prompt predictions for the interactive qualitative plot, from
  'run_posthoc_scripts.py first_prompt'.

The text uses Liberation Sans, which has the letter widths of Arial (see plot_style.py).
"""

import io
import os
import re
import glob
import shutil
import argparse
import subprocess

import tifffile
import numpy as np
import pandas as pd
from PIL import Image
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_dilation, grey_dilation

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import font_manager
import matplotlib.transforms as transforms
import matplotlib.patheffects as patheffects

from elf.evaluation import mean_segmentation_accuracy

from plot_style import MAX_FONT_SIZE, METHOD_COLORS, TEXT_COLOR, set_style

# The labels that have little room (the values and ranks next to the bars, the mode tags in the bars and the scores
# on the images) are a half point smaller than all other text, which is MAX_FONT_SIZE (Constantin Pape, 2026-11-06:
# bigger fonts).
SMALL_FONT_SIZE = MAX_FONT_SIZE - 0.5


ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(ROOT, "figures", "cells_automatic_segmentation_2d_results.csv")
INTERACTIVE_CSV = os.path.join(ROOT, "figures", "cells_interactive_segmentation_2d_results.csv")
OUTPUT_DIR = os.path.join(ROOT, "figures", "figure2")
EXPERIMENTS = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments"
QUALITATIVE_PREDICTIONS = os.path.join(EXPERIMENTS, "qualitative_2d", "out")
SINGLE_PROMPT_RUNS = os.path.join(EXPERIMENTS, "qualitative_single_prompt_{dataset}")
EVALUATION = os.path.join(ROOT, "..", "..", "..", "..", "finetuning", "v2", "evaluation")

# The 'method', 'model' and 'mode' values of each method in the results. A mode of None selects the rows without one.
METHODS = {
    "StarDist": ("stardist", ("default",), None),
    "SAM": ("sam_amg", ("vit_l",), None),
    "micro-sam": ("micro-sam", ("vit_l_lm", "vit_l_histopathology"), "apg"),
    "CellSAM": ("cellsam", ("default",), None),
    "Cellpose3": ("cellpose", ("cyto3",), None),
    "SAM2": ("sam2_amg", ("hvit_l",), None),
    "InstanSeg": ("instanseg", ("fluorescence_nuclei_and_cells", "brightfield_nuclei"), None),
    "CellposeSAM": ("cellpose", ("cpsam",), None),
    "CellViT++": ("cellvitpp", ("sam_h_x40_amp",), None),
    "SAM3": ("sam3", ("sam3.1",), None),
    "micro-sam2": ("micro-sam2", ("hvit_l",), "apg"),
}

PANELS = {
    "label_free_cells": ("label-free", "cell", "Label-free (cell)"),
    "fluorescence_cells": ("fluorescence", "cell", "Fluorescence (cell)"),
    "fluorescence_nuclei": ("fluorescence", "nucleus", "Fluorescence (nucleus)"),
    "histopathology_nuclei": ("histopathology", "nucleus", "Histopathology (nucleus)"),
}

# The methods of each modality, in their order in the plots.
PANEL_METHODS = {
    "label_free_cells": ["SAM", "SAM2", "SAM3", "Cellpose3", "micro-sam", "CellSAM", "CellposeSAM", "micro-sam2"],
    "fluorescence_cells": [
        "SAM", "SAM2", "SAM3", "InstanSeg", "Cellpose3", "micro-sam", "CellSAM", "CellposeSAM", "micro-sam2",
    ],
    "fluorescence_nuclei": [
        "SAM", "SAM2", "SAM3", "StarDist", "InstanSeg", "Cellpose3", "micro-sam", "CellSAM", "CellposeSAM",
        "micro-sam2",
    ],
    "histopathology_nuclei": [
        "SAM", "SAM2", "SAM3", "StarDist", "InstanSeg", "micro-sam", "CellSAM", "CellViT++", "CellposeSAM",
        "micro-sam2",
    ],
}

# micro-sam shows only AIS, its default mode, in the automatic plot.
FIGURE_MODES = {"micro-sam": "ais"}

MODES = ["ais", "apg"]
MODE_METHODS = ["micro-sam", "micro-sam2"]

PANEL_MODELS = {
    "fluorescence_nuclei": {"Cellpose3": ("nuclei",)},
}

# These methods run a different model for some modalities, so they get a star.
STARRED_METHODS = {"Cellpose3", "InstanSeg", "StarDist"}

EXCLUDED_DATASETS = {"dic_hepg2", "xenium_nuclei", "xenium_cells"}

# The in-domain and out-of-domain datasets of each modality. They set the domain of each dataset.
PANEL_DATASETS = {
    "label_free_cells": {
        "ID": ["livecell", "neurips_cellseg_label_free", "cell_acdc", "vicar", "omnipose_bact_phase", "orgline"],
        # Only datasets kept out of the micro-sam2 training, see DATASETS_2D_LM_LABEL_FREE_OOD in 'common.py'.
        # dic_hepg2 is left out: all methods score close to 0 on it.
        "OOD": ["deepseas", "bbbc010", "yeastsam", "cellapp", "bac_mother", "ecoli_microcolony_lineage"],
    },
    "fluorescence_nuclei": {
        "ID": ["dsb", "dynamicnuclearnet", "bmgd", "ifnuclei", "u20s", "cellbindb"],
        # spatch_dapi is left out: its tumor samples also give the sPATCH H&E training data.
        "OOD": ["arvidsson", "hela_cytonuc", "covid_if_nuclei", "cardioblast_nuclei", "mif_tonsil", "micro_bench"],
    },
    "histopathology_nuclei": {
        "ID": ["glysac", "histo_miner", "monuseg", "cpm17", "puma", "pannuke"],
        # Only datasets kept out of the micro-sam2 training and densely annotated by hand or fully proofread, see
        # DATASETS_2D_HP_OOD in 'common.py'. cd47_ihcnusc is IHC, the others are H&E.
        "OOD": ["cytodark0", "pcns", "khoshdeli", "beacon", "cd47_ihcnusc", "amonuseg"],
    },
    "fluorescence_cells": {
        "ID": [
            "tissuenet", "neurips_cellseg_fluorescence", "flywing", "omnipose_bact_fluor", "cvz_fluo_cell",
            "deepbacs_fluorescence",
        ],
        "OOD": ["covid_if_cells", "hpa", "medussa", "mif_tonsil_cells", "fusionx", "wing_disc_timelapse"],
    },
}

DOMAIN_TITLES = {"ID": "In-domain", "OOD": "Out-of-domain"}
# The domain titles in the style of the dataset names: in-domain in italics, out-of-domain in bold.
DOMAIN_STYLES = {"ID": {"style": "italic"}, "OOD": {"fontweight": "bold"}}

# The value axis ends at the first multiple of VALUE_STEP that is at least VALUE_ROOM above the highest bar.
VALUE_STEP = 0.05
VALUE_ROOM = 0.01
# The room for the mean rank after the longest bar, and the number of bars that show their mean rank.
RANK_ROOM = 0.1

BAR_HEIGHT = 0.72
FOUNDATION_MODELS = {"SAM", "SAM2", "SAM3"}
# The gap between the method groups, in bar rows.
GROUP_GAP = 0.5
# This gray has a contrast of 5.3 on white. Nature asks for at least 4.5.
GROUP_COLOR = "#6b6b6b"

DATASET_NAMES = {
    "arvidsson": "Arvidsson",
    "cellapp": "Cell-APP",
    "cellbindb": "CellBinDB",
    "cd47_ihcnusc": "CD47 IHC-NuSC",
    "covid_if_nuclei": "Covid IF",
    "cpm17": "CPM17",
    "cytodark0": "CytoDArk0",
    "histo_miner": "Histo-Miner",
    "hpa": "HPA",
    "mcellseg": "mCellSeg",
    "medussa": "MEDUSSA",
    "neurips_cellseg_label_free": "NeurIPS CellSeg",
    "spatch_dapi": "SPATCH DAPI",
    "srsanet": "SRSA-Net",
    "tissuenet": "TissueNet",
    "vicar": "VICAR",
    "yeastsam": "YeastSAM",
}


def load_results(category, structure, mode, models=None, path=RESULTS_CSV, methods=None, metric="msa"):
    """Load the scores of all methods for one modality, with the methods of MODE_METHODS in 'mode'.

    Args:
        category: The category of the modality in the results.
        structure: The structure of the modality in the results.
        mode: The mode of the methods in MODE_METHODS.
        models: The models that replace the models in METHODS, per method name.
        path: The path to the results table.
        methods: The method table. The default is METHODS.
        metric: The score column.

    Returns:
        The results in long format.
    """
    df = pd.read_csv(path)
    df = df[(df["category"] == category) & (df["structure"] == structure)]
    df = df[~df["dataset"].isin(EXCLUDED_DATASETS)]
    results = []
    for name, (method, method_models, method_mode) in (methods or METHODS).items():
        method_models = (models or {}).get(name, method_models)
        if name in MODE_METHODS:
            method_mode = mode
        rows = df[(df["method"] == method) & (df["model"].isin(method_models))]
        rows = rows[rows["mode"].isna()] if method_mode is None else rows[rows["mode"] == method_mode]
        rows = rows[["dataset", "domain", metric]].rename(columns={metric: "score"})
        rows["method"] = name
        results.append(rows)
    return pd.concat(results, ignore_index=True)


def load_modes(category, structure, models=None, **kwargs):
    """Load the results of one modality with the methods of MODE_METHODS in every mode of MODES.

    The column 'mode' holds the mode of these methods, and an empty string for the other methods.
    """
    parts = []
    for mode in MODES:
        results = load_results(category, structure, mode, models, **kwargs)
        is_mode_method = results["method"].isin(MODE_METHODS)
        # Keep the other methods only once.
        if mode != MODES[-1]:
            results, is_mode_method = results[is_mode_method], is_mode_method[is_mode_method]
        parts.append(results.assign(mode=np.where(is_mode_method, mode, "")))
    return pd.concat(parts, ignore_index=True)


def one_mode(results, modes):
    """Keep one mode per method in 'modes', and clear the mode of its rows."""
    for method, mode in modes.items():
        rows = results["method"] == method
        results = results[~rows | (results["mode"] == mode)].copy()
        results.loc[results["method"] == method, "mode"] = ""
    return results


def names_of(methods, starred):
    """Return the plot names of the methods, such as "μSAM2" and "CPSAM", with a mark for the methods in 'starred'.

    'starred' is a set of methods, which get a star, or a dict from method to its mark (such as "†").
    """
    names = [method.replace("micro-sam", "μSAM").replace("Cellpose", "CP") for method in methods]
    marks = starred if isinstance(starred, dict) else {method: "*" for method in starred}
    return [f"{name}{marks.get(method, '')}" for name, method in zip(names, methods)]


def tick_formatter(step, zero=False):
    """Return the formatter of the value ticks. It leaves the 0 empty, unless 'zero' is set."""
    decimals = 1 if round(step * 10, 6).is_integer() else 2
    return plt.FuncFormatter(lambda value, _: "" if abs(value) < 1e-9 and not zero else f"{value:.{decimals}f}")


def method_positions(methods, n_bars, group_gap):
    """Return the bar position of each method, top to bottom, with a gap between the method groups.

    Args:
        methods: The methods.
        n_bars: The number of bars of each method with more than one bar.
        group_gap: The gap between the groups, in bar rows.

    Returns:
        The positions.
    """
    groups = np.array([0 if method in FOUNDATION_MODELS else 2 if method == "micro-sam2" else 1 for method in methods])
    extra = np.array([BAR_HEIGHT * (n_bars.get(method, 1) - 1) for method in methods])
    # The position of a method is the middle of its bars.
    offsets = np.cumsum(extra) - extra / 2
    return -(np.arange(len(methods)) + group_gap * groups + offsets)


def modes_per_method(results):
    """Return the number of modes of each method with results in more than one mode."""
    modes = {}
    for result in results:
        for method, mode in result[["method", "mode"]].drop_duplicates().itertuples(index=False):
            if mode:
                modes.setdefault(method, set()).add(mode)
    return {method: len(names) for method, names in modes.items()}


def mean_ranks(results, lower_is_better=False):
    """Return the rank of each (method, mode) per dataset, averaged over the datasets. The best rank is 1."""
    table = results.pivot_table(index="dataset", columns=["method", "mode"], values="score")
    return table.rank(axis=1, ascending=lower_is_better).mean()


def contrast(color_a, color_b):
    """The WCAG contrast ratio of two colors."""
    def luminance(color):
        r, g, b = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in colors.to_rgb(color)]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    low, high = sorted([luminance(color_a), luminance(color_b)])
    return (high + 0.05) / (low + 0.05)


def label_color(bar_color):
    """White or the text color for a label on a bar, whichever has the higher contrast."""
    return max(["white", TEXT_COLOR], key=lambda color: contrast(color, bar_color))


def plot_domain(ax, results, domain, methods, value_max, positions, tick_step, ranks, lower_is_better=False):
    """Plot one horizontal bar per method with its mean score over the datasets of one domain.

    Args:
        ax: The axes.
        results: The results of the modality.
        domain: The domain.
        methods: The methods to plot.
        value_max: The end of the value axis.
        positions: The bar position of each method.
        tick_step: The distance between the value ticks.
        ranks: Whether the bars show their mean rank.
        lower_is_better: Whether a lower score ranks better, as the CREMI score.
    """
    results = results[results["domain"] == domain]
    values = results.groupby(["method", "mode"])["score"].mean()
    rank_of = mean_ranks(results, lower_is_better) if ranks else {}
    y = np.asarray(positions)

    # The width of a mode name on a sub-bar, in data units.
    axes_width = ax.get_position().width * FIGURE_WIDTH
    tag_width = MODE_TAG_WIDTH / 72 / axes_width * value_max
    for yi, method in zip(y, methods):
        # A method with more than one mode gets one sub-bar per mode. The first modes get a lighter color.
        modes = [mode for mode in [""] + MODES if (method, mode) in values.index]
        for k, mode in enumerate(modes):
            position = yi + BAR_HEIGHT * ((len(modes) - 1) / 2 - k)
            color = METHOD_COLORS[method]
            if mode and mode != MODES[-1]:
                color = colors.to_hex([1 - SUB_BAR_TINT * (1 - c) for c in colors.to_rgb(color)])
            bar = ax.barh(position, values[method, mode], height=BAR_HEIGHT, color=color, linewidth=0, zorder=2)[0]
            if method == "micro-sam2":
                bar.set_hatch("///")
                bar.set_edgecolor("white")
            short = mode and values[method, mode] < tag_width
            if mode and not short:
                # The patch in the bar color covers the hatch under the mode name.
                ax.text(
                    0, position, MODE_NAMES[mode], ha="left", va="center", fontsize=SMALL_FONT_SIZE, fontweight="bold",
                    color=label_color(color), zorder=4,
                    bbox={"boxstyle": "square,pad=0.08", "facecolor": color, "linewidth": 0},
                    transform=transforms.offset_copy(ax.transData, fig=ax.figure, x=2, units="points"),
                )
            elif short:
                # The mode name does not fit on the bar, so it goes after the end of the bar.
                ax.text(
                    values[method, mode] + 0.01, position, MODE_NAMES[mode], ha="left", va="center",
                    fontsize=SMALL_FONT_SIZE, fontweight="bold", color=GROUP_COLOR,
                )
            # The value of the first mode is smaller and gray, so that the value of the last mode stands out.
            secondary = mode not in ("", MODES[-1])
            end = values[method, mode] + (tag_width if short else 0)
            value = ax.text(
                end + 0.01, position, f"{values[method, mode]:.3f}", ha="left", va="center",
                fontsize=SMALL_FONT_SIZE, color=GROUP_COLOR if secondary else TEXT_COLOR,
            )
            if ranks:
                ax.annotate(
                    f"({rank_of[method, mode]:.1f})", xy=(1, 0.5), xycoords=value, xytext=(1.5, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=SMALL_FONT_SIZE, color=GROUP_COLOR,
                    annotation_clip=False,
                )

    ax.set_xlim(0, value_max)
    ax.set_xticks(np.arange(0, value_max + 1e-6, tick_step))
    ax.xaxis.set_major_formatter(tick_formatter(tick_step))
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.5, linestyle=(0, (3, 3)), zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=MAX_FONT_SIZE)
    ax.set_title(DOMAIN_TITLES[domain], fontsize=MAX_FONT_SIZE, fontweight="bold", color=TEXT_COLOR, pad=4)


# The layout of the automatic plot, in inches.
AUTOMATIC_ROWS = [["label_free_cells", "fluorescence_cells"], ["fluorescence_nuclei", "histopathology_nuclei"]]
# The double column width of 183 mm.
FIGURE_WIDTH = 7.2
NAME_WIDTH = 0.6
RIGHT_MARGIN = 0.1
COLUMN_GAP = 0.3
BAR_ROW_HEIGHT = 0.2
ROW_BOTTOM = 0.45
DOMAIN_LABEL_WIDTH = 0.45
DOMAIN_LINE_OFFSET = 0.27
DOMAIN_ROW_TOP = 0.3
DOMAIN_ROW_SPACE = 0.1
DOMAIN_ROW_GAP_NO_VALUES = 0.08
# The value ticks use the smallest step of TICK_STEPS that leaves at least TICK_WIDTH between two ticks.
TICK_STEPS = [0.1, 0.2, 0.25, 0.5]
TICK_WIDTH = 0.3
# The band over the row of a method that has no model for a modality. The text has a contrast of 4.56 on the band.
NOT_APPLICABLE_COLOR = "#f7f7f7"
NOT_APPLICABLE_TEXT_COLOR = "#717171"
SCORE_LABEL = "Mean segmentation accuracy ↑"
SUB_BAR_TINT = 0.55
MODE_NAMES = {"ais": "AIS", "apg": "APG"}
# The width of a mode name on its sub-bar, in points.
MODE_TAG_WIDTH = 15

# micro-sam also gets a star, because it runs its histopathology model (PathoSAM) on H&E.
AUTOMATIC_STARRED_METHODS = STARRED_METHODS | {"micro-sam"}


def merge_orders(orders):
    """Merge method lists into one list that keeps the order of each list.

    A new method goes before the next method of its list that is already in the merged list.
    """
    merged = []
    for methods in orders:
        for i, method in enumerate(methods):
            if method in merged:
                continue
            after = [other for other in methods[i + 1:] if other in merged]
            before = [other for other in methods[:i] if other in merged]
            if after:
                merged.insert(merged.index(after[0]), method)
            elif before:
                merged.insert(merged.index(before[-1]) + 1, method)
            else:
                merged.append(method)
    return merged


def new_figure(height, region=None):
    """Return a new figure with the given height in inches, or a sub-figure from 'region' (see 'plot_figure')."""
    return plt.figure(figsize=(FIGURE_WIDTH, height)) if region is None else region(height)


def height_of(fig):
    """Return the height of a figure or a sub-figure, in inches."""
    return fig.bbox.height / fig.dpi


# The gaps of a modality title to the panels under its line and to the line, in inches.
TITLE_LINE_GAP = 0.06
TITLE_TEXT_GAP = 0.03


def column_grid(grid, n_columns):
    """Return the left edges and the width of the columns of a grid (left margin, right margin, gap), in inches."""
    left, right, gap = grid
    width = (FIGURE_WIDTH - left - right - gap * (n_columns - 1)) / n_columns
    return [left + k * (width + gap) for k in range(n_columns)], width


def draw_title(fig, left, right, y, title):
    """Write a modality title over a thin line from 'left' to 'right' at the height 'y', in inches."""
    height = height_of(fig)
    x0, x1, line = left / FIGURE_WIDTH, right / FIGURE_WIDTH, y / height
    fig.add_artist(plt.Line2D([x0, x1], [line, line], color=GROUP_COLOR, linewidth=0.6))
    fig.text(
        (x0 + x1) / 2, line + TITLE_TEXT_GAP / height, title, ha="center", va="bottom", fontsize=MAX_FONT_SIZE,
        fontweight="bold", color=TEXT_COLOR,
    )


def plot_row(
    fig, results, cells, methods, panels, starred, bottom, height, n_columns, scope, right_margin, grid, title_line,
    ranks, group_gap,
):
    """Plot one row of the automatic plot: one panel per cell, all with the same method rows.

    Args:
        fig: The figure.
        results: The results of each modality.
        cells: The (column, modality, domain, title) of each panel.
        methods: The methods of the row.
        panels: The (title, score label, methods) of each modality.
        starred: The methods with a star.
        bottom: The bottom of the bars, in inches.
        height: The height of the bars, in inches.
        n_columns: The number of columns.
        scope: The modalities that can share a value axis.
        right_margin: The room right of the panels, in inches.
        grid: The column grid, see 'column_grid'. The default layout is used for None.
        title_line: Whether the titles stand over a thin line.
        ranks: Whether the bars show their mean rank, see 'plot_domain'.
        group_gap: The gap between the method groups, in bar rows.

    Returns:
        The axes of each cell.
    """
    label_width = NAME_WIDTH
    gap = COLUMN_GAP
    if grid is not None:
        label_width, right_margin, gap = grid
    n_bars = modes_per_method([results[name] for name in scope])
    y = method_positions(methods, n_bars, group_gap)
    rows = dict(zip(methods, y))
    width = (FIGURE_WIDTH - label_width - right_margin - gap * (n_columns - 1)) / n_columns
    axes = []
    for column, name, domain, title in cells:
        panel_methods = panels[name][2]
        shown = [method for method in methods if method in panel_methods and method in set(results[name]["method"])]
        # The modalities with the same score share the value axis.
        shared = pd.concat([results[n].assign(modality=n) for n in scope if panels[n][1] == panels[name][1]])
        shared = shared[shared["method"].isin(shown)]
        highest = shared.groupby(["modality", "domain", "method", "mode"])["score"].mean().max()
        room = VALUE_ROOM + (RANK_ROOM if ranks else 0)
        value_max = np.ceil(round((highest + room) / VALUE_STEP, 6)) * VALUE_STEP
        left = label_width + column * (width + gap)
        ax = fig.add_axes([
            left / FIGURE_WIDTH, bottom / height_of(fig), width / FIGURE_WIDTH, height / height_of(fig),
        ], sharey=axes[0] if axes else None)
        tick_step = next((step for step in TICK_STEPS if width * step / value_max >= TICK_WIDTH), TICK_STEPS[-1])
        plot_domain(
            ax, results[name][results[name]["method"].isin(shown)], domain, shown, value_max,
            [rows[method] for method in shown], tick_step, ranks, panels[name][1].endswith("↓"),
        )
        if title_line:
            ax.set_title("")
            if title:
                draw_title(fig, left, left + width, bottom + height + TITLE_LINE_GAP, title)
        else:
            ax.set_title(title, fontsize=MAX_FONT_SIZE, fontweight="bold", color=TEXT_COLOR, pad=4)
        # The band shows that an empty row is not a score of zero.
        for method in methods:
            if method not in panel_methods:
                ax.barh(
                    rows[method], value_max, height=BAR_HEIGHT * n_bars.get(method, 1), color=NOT_APPLICABLE_COLOR,
                    linewidth=0, zorder=0,
                )
                # A narrow panel uses a smaller label, so that the label fits the band.
                ax.text(
                    value_max / 2, rows[method], "not applicable", ha="center", va="center",
                    fontsize=MAX_FONT_SIZE, style="italic", color=NOT_APPLICABLE_TEXT_COLOR,
                )
        if axes:
            ax.tick_params(axis="y", labelleft=False)
        axes.append(ax)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names_of(methods, starred), fontsize=MAX_FONT_SIZE)
    half = np.array([BAR_HEIGHT * n_bars.get(method, 1) / 2 for method in methods])
    axes[0].set_ylim((y - half).min() - 0.24, (y + half).max() + 0.24)
    return axes


def group_labels(panels):
    """Group neighbouring panels, given as (score label, bottom of the bars, axes), with the same label and bottom."""
    groups = []
    for label, bottom, ax in panels:
        if groups and groups[-1][:2] == (label, bottom):
            groups[-1][2].append(ax)
        else:
            groups.append((label, bottom, [ax]))
    return groups


def plot_automatic(
    results, rows, panels, starred=None, row_height=BAR_ROW_HEIGHT, region=None, domain_offset=DOMAIN_LINE_OFFSET,
    grid=None, title_line=False, ranks=False, group_gap=GROUP_GAP, row_gap=None,
):
    """Plot the automatic results of all modalities in one figure.

    The in-domain panels of all modalities are the first row, and the out-of-domain panels are the second row.

    Args:
        results: The results of each modality.
        rows: The modalities, in rows that are joined into one row.
        panels: The (title, score label, methods) of each modality. The score label ends with an arrow.
        starred: The methods with a star. The default is AUTOMATIC_STARRED_METHODS.
        row_height: The height of one bar row, in inches.
        region: The function that returns a sub-figure, see 'new_figure'.
        domain_offset: The distance from the panels to the line of the domain titles, in inches.
        grid: The column grid, see 'column_grid'.
        title_line: Whether the modality titles stand over a thin line.
        ranks: Whether the bars show their mean rank, see 'plot_domain'.
        group_gap: The gap between the method groups, in bar rows.
        row_gap: The room between the rows, in inches.

    Returns:
        The figure.
    """
    starred = AUTOMATIC_STARRED_METHODS if starred is None else starred
    modalities = [name for row in rows for name in row]
    layout = []
    for i, domain in enumerate(DOMAIN_TITLES):
        # Only the first row shows the modality titles.
        cells = [
            (k, name, domain, panels[name][0] if i == 0 else "") for k, name in enumerate(modalities)
            if domain in set(results[name]["domain"])
        ]
        layout.append((domain, cells))
    methods = merge_orders([panels[name][2] for name in modalities])
    n_bars = modes_per_method([results[name] for name in modalities])
    y = method_positions(methods, n_bars, group_gap)
    half = np.array([BAR_HEIGHT * n_bars.get(method, 1) / 2 for method in methods])
    height = ((y + half).max() - (y - half).min() + 0.48) * row_height
    heights = [height] * len(layout)
    tops = [DOMAIN_ROW_TOP] + [DOMAIN_ROW_SPACE if row_gap is None else 0] * (len(layout) - 1)
    gap = DOMAIN_ROW_GAP_NO_VALUES if row_gap is None else row_gap
    bottoms = [gap] * (len(layout) - 1) + [ROW_BOTTOM]
    right_margin = RIGHT_MARGIN + DOMAIN_LABEL_WIDTH - DOMAIN_LINE_OFFSET + domain_offset
    if grid is not None:
        right_margin = grid[1]
    figure_height = sum(heights) + sum(tops) + sum(bottoms)
    fig = new_figure(figure_height, region)

    # The lowest panel of each column, as (score label, bottom of the bars, axes).
    lowest = {}
    top = figure_height
    for i, ((domain, cells), room_top, room_bottom) in enumerate(zip(layout, tops, bottoms)):
        bottom = top - room_top - height
        axes = plot_row(
            fig, results, cells, methods, panels, starred, bottom, height, len(modalities), modalities,
            right_margin=right_margin, grid=grid, title_line=title_line, ranks=ranks,
            group_gap=group_gap,
        )
        for ax in axes:
            ticks = ax.get_xticks()
            ax.xaxis.set_major_formatter(tick_formatter(ticks[1] - ticks[0], zero=True))
        # Only the last row labels its value ticks.
        if i < len(layout) - 1:
            for ax in axes:
                ax.tick_params(axis="x", labelbottom=False)
        x = (FIGURE_WIDTH - right_margin + domain_offset) / FIGURE_WIDTH
        low, high = (bottom + 0.05) / figure_height, (bottom + height - 0.05) / figure_height
        fig.add_artist(plt.Line2D([x, x], [low, high], color=GROUP_COLOR, linewidth=0.6))
        fig.text(
            x + 0.05 / FIGURE_WIDTH, (low + high) / 2, DOMAIN_TITLES[domain], rotation=270, ha="left", va="center",
            fontsize=MAX_FONT_SIZE, color=TEXT_COLOR, **DOMAIN_STYLES[domain],
        )
        for cell, ax in zip(cells, axes):
            lowest[cell[0]] = (panels[cell[1]][1], bottom, ax)
        top = bottom - room_bottom
    score_labels = group_labels([lowest[column] for column in sorted(lowest)])

    for label, bottom, label_axes in score_labels:
        left, right = label_axes[0].get_position().x0, label_axes[-1].get_position().x1
        text, arrow = label[:-2], label[-1]
        score = fig.text(
            (left + right) / 2, (bottom - 0.24) / figure_height, text, ha="center", va="top", fontsize=MAX_FONT_SIZE,
            fontweight="bold", color=TEXT_COLOR,
        )
        label_axes[0].annotate(
            arrow, xy=(1, 0.5), xycoords=score, xytext=(1.5, 0), textcoords="offset points", ha="left",
            va="center", fontsize=MAX_FONT_SIZE, fontweight="bold", color=TEXT_COLOR,
            annotation_clip=False,
        )
    return fig


# The 'method', 'model' and optional 'mode' values of each interactive method in the results. Every method feeds its
# previous mask back with the correction clicks: SAM2, SAM3 and micro-sam2 always do, SAM and micro-sam use their
# 'with_masks' runs.
INTERACTIVE_METHODS = {
    "SAM": ("sam", ("vit_l",), "with_masks"),
    "SAM2": ("sam2", ("hvit_l",)),
    "SAM3": ("sam3", ("default",)),
    "micro-sam": ("micro-sam", ("vit_l_lm", "vit_l_histopathology"), "with_masks"),
    "micro-sam2": ("micro-sam2", ("hvit_l",)),
}
PROMPT_NAMES = {"box": "Box", "point": "Point"}


def load_interactive(category, structure, datasets, path=INTERACTIVE_CSV, methods=None, metric="msa"):
    """Load the scores of the interactive methods for one modality, per dataset, first prompt and iteration.

    Args:
        category: The category of the modality in the results.
        structure: The structure of the modality in the results.
        datasets: The datasets of each domain.
        path: The path to the results table.
        methods: The method table. The default is INTERACTIVE_METHODS.
        metric: The score column.

    Returns:
        The results in long format.
    """
    df = pd.read_csv(path)
    df = df[(df["category"] == category) & (df["structure"] == structure)]
    results = []
    for name, (method, models, *mode) in (methods or INTERACTIVE_METHODS).items():
        rows = df[(df["method"] == method) & (df["model"].isin(models))]
        if mode:
            # The table stores an empty mode as a missing value.
            rows = rows[rows["mode"].fillna("") == mode[0]]
        rows = rows[["dataset", "prompt", "iteration", metric]].rename(columns={metric: "score"})
        rows["method"] = name
        results.append(rows)
    return select_datasets(pd.concat(results, ignore_index=True), datasets)


# The layout of the interactive plot, in inches. The box runs are shifted by BOX_OFFSET iterations.
PANEL_HEIGHT = 1.5
LEGEND_HEIGHT = 0.08
LEGEND_BORDER_COLOR = "#cfcdc6"
# The first prompt and the first 4 corrections.
N_INTERACTIVE_ITERATIONS = 5
BOX_OFFSET = N_INTERACTIVE_ITERATIONS + 1
BOX_BAND_COLOR = "#f4f3ef"
AXIS_PRECISION = 0.05
INTERACTIVE_TICK_STEPS = [0.05, 0.1, 0.2, 0.25, 0.5]
MAX_TICKS = 8
INTERACTIVE_LABEL_WIDTH = 0.5
INTERACTIVE_SCORE_X = 0.12
INTERACTIVE_DOMAIN_WIDTH = 0.3
INTERACTIVE_DOMAIN_OFFSET = 0.1
# The share of a panel under its lowest point, for the panels in 'room_below'.
ROOM_BELOW = 0.12
X_LABEL_OFFSET = 0.15  # the iteration label under the panels, in inches
ITERATION_LABEL = "Correction iteration"
OWN_LABEL_OFFSET = 0.27  # the own score label of a modality, left of its panels, in inches


def first_tick(low, step):
    """Return the first value tick of an axis from 'low': a multiple of 'step', and not below 0."""
    return max(np.ceil(round(low / step, 6)) * step, 0.0)


def add_score_arrow(ax, label, arrow, fontsize=MAX_FONT_SIZE):
    """Add the arrow over a vertical score label. The arrow stays upright, so that it points along the axis."""
    ax.annotate(
        arrow, xy=(0.5, 1), xycoords=label, xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom",
        fontsize=fontsize, fontweight="bold", color=TEXT_COLOR, annotation_clip=False,
    )


def plot_interactive_by_domain(
    results, modalities, titles, starred=None, methods=None, score_labels=None, placeholder=False,
    panel_height=PANEL_HEIGHT, region=None, legend_panel=None, grid=None, title_line=False, legend_left=False,
    room_below=(), shared_rows=False, show_titles=True, score_gap=None, row_gap=None, band_color=BOX_BAND_COLOR,
    legend_wide=False, prompts_on_top=False, x_label=None,
):
    """Plot the mean score over the correction iterations of all modalities in one figure.

    The in-domain panels are the first row, and the out-of-domain panels are the second row. Each panel shows the
    point runs and the box runs side by side. The value axis of a panel spans its curves.

    Args:
        results: The interactive results of each modality.
        modalities: The modalities, one column each.
        titles: The title of each modality.
        starred: The methods with a star. The default is AUTOMATIC_STARRED_METHODS.
        methods: The methods, in their order. The default is INTERACTIVE_METHODS.
        score_labels: The score label of each modality. The default is SCORE_LABEL.
        placeholder: Whether all panels stay empty, with the note "Results pending".
        panel_height: The height of a panel, in inches.
        region: The function that returns a sub-figure, see 'new_figure'.
        legend_panel: The (domain, modality) of the panel with the legend.
        grid: The column grid, see 'column_grid'.
        title_line: Whether the modality titles stand over a thin line.
        legend_left: Whether the legend stands in the upper left corner, with the value axis raised for it.
        legend_wide: Whether the legend spans the top of its panel, with the value axis raised for it.
        prompts_on_top: Whether the prompt names stand once, over the halves of the first row, clear of the curves.
        x_label: The label of the iteration axis, once under all panels, as the score label of the automatic plot.
        room_below: The (domain, modality) of the panels with a lowered value axis.
        shared_rows: Whether the panels of a row with the shared score label share one value range.
        show_titles: Whether the panels show the modality titles.
        score_gap: The distance from the shared score label to the panels, in inches.
        row_gap: The room between the rows, in inches.
        band_color: The color of the box half of a panel.

    Returns:
        The figure.
    """
    starred = AUTOMATIC_STARRED_METHODS if starred is None else starred
    score_labels = {name: (score_labels or {}).get(name, SCORE_LABEL) for name in modalities}
    shared_label = max(set(score_labels.values()), key=list(score_labels.values()).count)
    methods = list(methods or INTERACTIVE_METHODS)
    averages = {
        name: results[name][results[name]["iteration"] < N_INTERACTIVE_ITERATIONS].groupby(
            ["domain", "prompt", "method", "iteration"]
        )["score"].mean()
        for name in modalities
    }
    domains = list(DOMAIN_TITLES)
    offsets = {"point": 0, "box": BOX_OFFSET}
    n_iterations = N_INTERACTIVE_ITERATIONS
    n_columns = len(modalities)
    grid = grid or (INTERACTIVE_LABEL_WIDTH, RIGHT_MARGIN + INTERACTIVE_DOMAIN_WIDTH, COLUMN_GAP)
    right_margin = grid[1]
    lefts, width = column_grid(grid, n_columns)
    tops = [DOMAIN_ROW_TOP if show_titles else DOMAIN_ROW_SPACE]
    tops += [DOMAIN_ROW_SPACE if row_gap is None else 0] * (len(domains) - 1)
    gap = DOMAIN_ROW_GAP_NO_VALUES if row_gap is None else row_gap
    bottoms = [gap] * (len(domains) - 1) + [ROW_BOTTOM]

    def curves_of(name, domain):
        return [
            averages[name][domain, prompt, method] for prompt in offsets for method in methods
            if (domain, prompt, method) in averages[name].index.droplevel("iteration")
        ]

    # The value range of each row over the panels with the shared score label, for 'shared_rows'.
    row_ranges = {}
    for domain in domains:
        curves = [
            curve for name in modalities if score_labels[name] == shared_label and not placeholder
            for curve in curves_of(name, domain)
        ]
        if curves:
            row_ranges[domain] = (min(c.min() for c in curves), max(c.max() for c in curves))
    height = LEGEND_HEIGHT + len(domains) * panel_height + sum(tops) + sum(bottoms)
    fig = new_figure(height, region)

    top = height - LEGEND_HEIGHT
    titled = set()
    own_label_axes = {}
    for domain, room_top, room_bottom in zip(domains, tops, bottoms):
        bottom = top - room_top - panel_height
        shared_axes = []
        for column, name in enumerate(modalities):
            pending = placeholder or domain not in set(results[name]["domain"])
            left = lefts[column]
            ax = fig.add_axes([left / FIGURE_WIDTH, bottom / height, width / FIGURE_WIDTH, panel_height / height])
            last = BOX_OFFSET + n_iterations - 1
            ax.axvspan(BOX_OFFSET - 0.6, last + 0.6, color=band_color, linewidth=0, zorder=0)
            average = averages[name]
            for method in [] if pending else methods:
                ours = method == "micro-sam2"
                for prompt, offset in offsets.items():
                    if (domain, prompt, method) not in average.index.droplevel("iteration"):
                        continue
                    curve = average[domain, prompt, method]
                    zorder = 3 if ours else 2
                    ax.plot(
                        curve.index + offset, curve.values, color=METHOD_COLORS[method], linewidth=1.0, marker="o",
                        markersize=2.2, markeredgecolor="white", markeredgewidth=0.4, zorder=zorder,
                    )
                    # The first prompt is an open ring.
                    ax.plot(
                        curve.index[:1] + offset, curve.values[:1], linestyle="none", marker="o", markersize=2.8,
                        markerfacecolor="white", markeredgecolor=METHOD_COLORS[method], markeredgewidth=0.7,
                        zorder=zorder + 0.5,
                    )
            # Each half labels the first prompt 0 in gray, and an arrow spans the corrections.
            ticks = [offset + i for offset in offsets.values() for i in range(n_iterations)]
            ax.set_xticks(ticks)
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=1.5)
            ax.tick_params(axis="both", labelsize=MAX_FONT_SIZE)
            transform = ax.get_xaxis_transform()
            # Only the last row labels its iterations.
            labelled = domain == domains[-1]
            for prompt, offset in offsets.items():
                for iteration in (0, 1, n_iterations - 1) if labelled else ():
                    ax.text(
                        offset + iteration, -0.065, str(iteration), ha="center", va="center", fontsize=MAX_FONT_SIZE,
                        color=GROUP_COLOR if iteration == 0 else TEXT_COLOR, transform=transform,
                    )
                if labelled:
                    ax.annotate(
                        "", xy=(offset + n_iterations - 1.55, -0.058), xytext=(offset + 1.45, -0.058),
                        xycoords=transform, textcoords=transform, annotation_clip=False,
                        arrowprops={
                            "arrowstyle": "-|>,head_length=0.35,head_width=0.15", "color": GROUP_COLOR,
                            "linewidth": 0.6, "shrinkA": 0, "shrinkB": 0,
                        },
                    )
                if not prompts_on_top:
                    ax.text(
                        offset + (n_iterations - 1) / 2, 0.015, PROMPT_NAMES[prompt], ha="center", va="bottom",
                        fontsize=MAX_FONT_SIZE, style="italic", color=GROUP_COLOR, transform=transform, zorder=4,
                    )
                elif domain == domains[0]:
                    ax.text(
                        offset + (n_iterations - 1) / 2, 1.02, PROMPT_NAMES[prompt], ha="center", va="bottom",
                        fontsize=MAX_FONT_SIZE, style="italic", color=GROUP_COLOR, transform=transform,
                    )
            ax.set_xlim(-0.6, last + 0.6)
            if pending:
                ax.set_yticks([])
                ax.text(
                    0.5, 0.55, "Results pending", ha="center", va="center", fontsize=MAX_FONT_SIZE, style="italic",
                    color=GROUP_COLOR, transform=ax.transAxes,
                )
            shown = [] if pending else curves_of(name, domain)
            lowest, highest = (0.0, 1.0) if pending else (min(c.min() for c in shown), max(c.max() for c in shown))
            in_shared_row = shared_rows and score_labels[name] == shared_label and domain in row_ranges
            if in_shared_row:
                lowest, highest = row_ranges[domain]
            # The axis ends are the nearest multiples of AXIS_PRECISION around the curves.
            low = np.floor(round(lowest / AXIS_PRECISION, 6)) * AXIS_PRECISION
            if (domain, name) in room_below and not pending:
                # Keep the lowest point over the 'Point' and 'Box' labels.
                axis_top = np.ceil(round((highest + 0.01) / AXIS_PRECISION, 6)) * AXIS_PRECISION
                low = (lowest - ROOM_BELOW * axis_top) / (1 - ROOM_BELOW)
                low = np.floor(round(low / AXIS_PRECISION, 6)) * AXIS_PRECISION
            high = np.ceil(round((highest + 0.01) / AXIS_PRECISION, 6)) * AXIS_PRECISION
            step = next(s for s in INTERACTIVE_TICK_STEPS if (high - low) / s <= MAX_TICKS)
            ax.set_ylim(low, high)
            if not pending:
                ax.set_yticks(np.arange(first_tick(low, step), high + 1e-6, step))
                ax.yaxis.set_major_formatter(tick_formatter(step))
            ax.grid(axis="y", color="#e6e5e0", linewidth=0.5, linestyle=(0, (3, 3)), zorder=0)
            ax.set_axisbelow(True)
            if in_shared_row:
                shared_axes.append((ax, step))
            if in_shared_row and column > 0:
                ax.tick_params(axis="y", labelleft=False)
            if name not in titled and show_titles:
                if title_line:
                    draw_title(fig, left, left + width, bottom + panel_height + TITLE_LINE_GAP, titles[name])
                else:
                    # With the prompt names over the first row, the title stands above them.
                    pad = 12 if prompts_on_top else 4
                    ax.set_title(titles[name], fontsize=MAX_FONT_SIZE, fontweight="bold", color=TEXT_COLOR, pad=pad)
                titled.add(name)
            if score_labels[name] != shared_label:
                own_label_axes.setdefault(name, []).append(ax)
            if legend_panel == (domain, name):
                legend = add_panel_legend(ax, methods, starred, left=legend_left, wide=legend_wide)
                if (legend_left or legend_wide) and not pending:
                    make_room_above(ax, legend, highest, step)
        # A panel raised for the legend raises the other panels of its shared row alike.
        if shared_axes:
            axis_low = min(ax.get_ylim()[0] for ax, _ in shared_axes)
            axis_top = max(ax.get_ylim()[1] for ax, _ in shared_axes)
            for ax, step in shared_axes:
                ax.set_ylim(axis_low, axis_top)
                ax.set_yticks(np.arange(first_tick(axis_low, step), axis_top - 0.5 * step + 1e-6, step))
        x = (FIGURE_WIDTH - right_margin + INTERACTIVE_DOMAIN_OFFSET) / FIGURE_WIDTH
        low, high = (bottom + 0.05) / height, (bottom + panel_height - 0.05) / height
        fig.add_artist(plt.Line2D([x, x], [low, high], color=GROUP_COLOR, linewidth=0.6))
        fig.text(
            x + 0.05 / FIGURE_WIDTH, (low + high) / 2, DOMAIN_TITLES[domain], rotation=270, ha="left", va="center",
            fontsize=MAX_FONT_SIZE, color=TEXT_COLOR, **DOMAIN_STYLES[domain],
        )
        top = bottom - room_bottom
    # A modality with its own score label gets it once, left of its column, centered on its panels.
    for name, axes in own_label_axes.items():
        upper, lower = axes[0].get_position(), axes[-1].get_position()
        ax = axes[-1]
        ax.set_ylabel(score_labels[name][:-2], fontsize=MAX_FONT_SIZE, fontweight="bold", color=TEXT_COLOR)
        ax.yaxis.set_label_coords(
            -OWN_LABEL_OFFSET / (lower.width * FIGURE_WIDTH), ((upper.y1 + lower.y0) / 2 - lower.y0) / lower.height
        )
        add_score_arrow(ax, ax.yaxis.label, score_labels[name][-1], fontsize=MAX_FONT_SIZE)
    middle = 0.5 * (height - LEGEND_HEIGHT) / height
    score_x = INTERACTIVE_SCORE_X if score_gap is None else grid[0] - score_gap
    score = fig.text(
        score_x / FIGURE_WIDTH, middle, shared_label[:-2], rotation=90, ha="left", va="center", fontsize=MAX_FONT_SIZE,
        fontweight="bold", color=TEXT_COLOR,
    )
    add_score_arrow(fig.axes[0], score, shared_label[-1])
    if x_label is not None:
        fig.text(
            (lefts[0] + lefts[-1] + width) / 2 / FIGURE_WIDTH, (bottom - X_LABEL_OFFSET) / height, x_label,
            ha="center", va="top", fontsize=MAX_FONT_SIZE, fontweight="bold", color=TEXT_COLOR,
        )
    return fig


def add_panel_legend(ax, methods, starred, left=False, wide=False):
    """Add the legend of the interactive plot to a panel, in two columns.

    Args:
        ax: The axes of the panel.
        methods: The methods.
        starred: The methods with a star.
        left: Whether the legend stands in the upper left corner. By default, it stands in the lower right corner.
        wide: Whether the legend spans the top of the panel in three columns.

    Returns:
        The legend.
    """
    names = dict(zip(methods, names_of(methods, starred)))
    handles = [
        plt.Line2D(
            [], [], color=METHOD_COLORS[method], marker="o", markersize=2.2, linewidth=1.0, markeredgecolor="white",
            markeredgewidth=0.4,
        )
        for method in methods
    ]
    if wide:
        place = dict(loc="upper left", bbox_to_anchor=(0.01, 0.985, 0.98, 0), mode="expand", ncol=3)
    elif left:
        place = dict(loc="upper left", bbox_to_anchor=(0.01, 0.985), ncol=2)
    else:
        place = dict(loc="lower right", bbox_to_anchor=(0.995, 0.09), ncol=2)
    legend = ax.legend(
        handles, [names[method] for method in methods], **place,
        fontsize=MAX_FONT_SIZE, handlelength=1.1, handletextpad=0.3, columnspacing=0.5, labelspacing=0.15,
        borderpad=0.25,
        frameon=True, framealpha=1, facecolor="white", edgecolor=LEGEND_BORDER_COLOR,
    )
    legend.get_frame().set_linewidth(0.6)
    legend.set_zorder(5)
    return legend


def make_room_above(ax, legend, highest, step):
    """Raise the top of the value axis until the highest point 'highest' stays under the legend.

    The function also adds the value ticks of 'step' up to the new top.
    """
    fig = ax.figure
    fig.canvas.draw()
    box = legend.get_window_extent().transformed(ax.transAxes.inverted())
    low, high = ax.get_ylim()
    # Leave a small gap under the legend.
    share = box.y0 - 0.03
    top = low + (highest - low) / share
    if top > high:
        top = np.ceil(round(top / AXIS_PRECISION, 6)) * AXIS_PRECISION
        ax.set_ylim(low, top)
        ax.set_yticks(np.arange(first_tick(low, step), top - 0.5 * step + 1e-6, step))


def save_figure(fig, name, kind, style="quantitative", dpi=None):
    """Save the figure as png, svg and pdf in the folder '<kind>/<style>' of OUTPUT_DIR.

    The files have no date and the svg ids use a fixed salt, so that the same figure gives the same files. 'dpi' sets
    the resolution of the images and of the png; the default keeps 'savefig.dpi'.
    """
    folder = os.path.join(OUTPUT_DIR, kind, style)
    os.makedirs(folder, exist_ok=True)
    metadata = {"png": None, "svg": {"Date": None}, "pdf": {"CreationDate": None}}
    with plt.rc_context({"svg.hashsalt": "micro-sam2"}):
        for extension, meta in metadata.items():
            fig.savefig(os.path.join(folder, f"{name}.{extension}"), bbox_inches="tight", metadata=meta, dpi=dpi)
    plt.close(fig)
    print(f"Saved '{name}' to '{folder}'.")


def select_datasets(results, datasets):
    """Keep the datasets of each domain and set their domain."""
    domains = {dataset: domain for domain, names in datasets.items() for dataset in names}
    results = results[results["dataset"].isin(domains)].copy()
    results["domain"] = results["dataset"].map(domains)
    return results


# The frame of the images in the qualitative plots.
# The datasets whose red channel is shown in magenta, so that their images are not red and green. TissueNet, with an
# empty blue channel, gets this in 'to_rgb' by itself.
RED_TO_MAGENTA = {"hpa"}
FRAME_COLOR = "#c8c8c8"
FRAME_WIDTH = 0.4


def to_rgb(image, shape, red_to_magenta=False):
    """Return an image as RGB in [0, 1], with the contrast stretched over the 1 to 99.7 percentiles.

    Args:
        image: The image.
        shape: The shape of the labels. It tells a channel-first image from a channel-last image.
        red_to_magenta: Whether to show the red channel in magenta, for an RGB image with a blue channel (as HPA).

    Returns:
        The RGB image. An image with two channels shows them in green and blue. An RGB image with an empty blue
        channel (as TissueNet) shows its red channel in magenta, so that it is not red and green, which readers with
        red-green color blindness cannot tell apart.
    """
    image = np.asarray(image, dtype="float32")
    if image.ndim == 3 and image.shape[:2] != shape:
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] == 2:
        image = np.stack([np.zeros(image.shape[:2]), image[..., 0], image[..., 1]], axis=-1)
    image = image[..., :3]
    if image.shape[-1] == 3 and not image[..., 2].any() and image[..., 0].any():
        image = np.stack([image[..., 0], image[..., 1], image[..., 0]], axis=-1)
    low, high = np.percentile(image, (1, 99.7))
    image = np.clip((image - low) / max(high - low, 1e-6), 0, 1)
    if red_to_magenta:
        image[..., 2] = np.maximum(image[..., 2], image[..., 0])
    return image


def framed_axes(fig, left, bottom, width, height):
    """Add axes for an image, in inches from the lower left corner, without ticks and with a hairline frame."""
    figure_width, figure_height = FIGURE_WIDTH, height_of(fig)
    ax = fig.add_axes([left / figure_width, bottom / figure_height, width / figure_width, height / figure_height])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        # The paper style hides the top and right spines, so the frame turns them on.
        spine.set_visible(True)
        spine.set_color(FRAME_COLOR)
        spine.set_linewidth(FRAME_WIDTH)
    return ax


def draw_score(ax, score, bold=False, fontsize=SMALL_FONT_SIZE):
    """Write the score in the lower right corner of a panel, in white with a dark halo, in bold if 'bold'."""
    text = ax.text(
        0.96, 0.04, f"{score:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=fontsize,
        color="white", fontweight="bold" if bold else "normal",
    )
    text.set_path_effects([patheffects.withStroke(linewidth=1.2, foreground="black")])


# The test images of the automatic qualitative plot: (dataset, sample index, modality, domain).
AUTOMATIC_QUALITATIVE = [
    ("vicar", 47, "label_free_cells", "ID"),
    ("yeastsam", 42, "label_free_cells", "OOD"),
    ("tissuenet", 1151, "fluorescence_cells", "ID"),
    ("hpa", 1, "fluorescence_cells", "OOD"),
    ("cellbindb", 10, "fluorescence_nuclei", "ID"),
    ("arvidsson", 9, "fluorescence_nuclei", "OOD"),
    ("cpm17", 2, "histopathology_nuclei", "ID"),
    ("cytodark0", 5, "histopathology_nuclei", "OOD"),
]

QUALITATIVE_METHODS = ["CellposeSAM", "micro-sam2"]
PREDICTION_NAMES = {"CellposeSAM": "cellposesam", "micro-sam2": "microsam2_apg"}

# The layout of the automatic qualitative plot, in inches.
QUALITATIVE_LABEL_WIDTH = 0.2
QUALITATIVE_NAME_GAP = 0.05
QUALITATIVE_COLUMN_GAP = 0.03
QUALITATIVE_MODALITY_GAP = 0.08
QUALITATIVE_HEADER_HEIGHT = 0.36
QUALITATIVE_TITLE_HEIGHT = 0.2
QUALITATIVE_ROW_GAP = 0.03
QUALITATIVE_FILL_ALPHA = 0.5
# This yellow stands out on dark, bright and stained images.
REFERENCE_COLOR = "#ffe100"
# The width of the reference outlines in pixels, for the datasets where a width of 1 is hard to see.
OUTLINE_WIDTHS = {"vicar": 2, "yeastsam": 3, "hpa": 5, "cellbindb": 3, "arvidsson": 3, "cpm17": 3, "cytodark0": 3}
PALETTE = [
    "#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff", "#ff8fd8", "#9b5de5", "#00d1c1", "#ff9f43", "#b8e05c", "#5dd9ff",
    "#ff5e9e", "#c8b6ff",
]
SEED = 0
REFERENCE_NAME = "Reference"


def load_segmentation(dataset, sample, name):
    """Return the labels ('labels') or the prediction of a method (see PREDICTION_NAMES) of a sample."""
    return tifffile.imread(os.path.join(QUALITATIVE_PREDICTIONS, dataset, f"{name}_{sample}.tif"))


def load_image(dataset, sample):
    """Return the image of a sample as RGB in [0, 1]."""
    image = tifffile.imread(os.path.join(QUALITATIVE_PREDICTIONS, dataset, f"image_{sample}.tif"))
    return to_rgb(image, load_segmentation(dataset, sample, "labels").shape, dataset in RED_TO_MAGENTA)


def distinct_colors(segmentation):
    """Return a color from PALETTE per object id, so that touching objects never share a color.

    The function colors the objects in a random order that SEED fixes. Each object takes the first palette color
    that no object within two pixels has.
    """
    near = grey_dilation(segmentation, size=(5, 5))
    pairs = np.stack([segmentation.ravel(), near.ravel()], axis=1)
    pairs = np.unique(pairs[(pairs[:, 0] > 0) & (pairs[:, 1] > 0) & (pairs[:, 0] != pairs[:, 1])], axis=0)
    neighbours = {}
    for a, b in pairs:
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)
    ids = np.setdiff1d(np.unique(segmentation), [0])
    np.random.default_rng(SEED).shuffle(ids)
    palette = np.array([colors.to_rgb(color) for color in PALETTE])
    choice = {}
    for i, label in enumerate(ids):
        taken = {choice[n] for n in neighbours.get(label, ()) if n in choice}
        free = [k for k in range(len(palette)) if k not in taken]
        # A rotating start spreads the colors over the palette.
        choice[label] = min(free, key=lambda k: (k - i) % len(palette)) if free else i % len(palette)
    lut = np.zeros((segmentation.max() + 1, 3))
    for label, k in choice.items():
        lut[label] = palette[k]
    return lut


def overlay(image, segmentation, fill=True, width=1):
    """Overlay an instance segmentation on an image.

    Args:
        image: The RGB image.
        segmentation: The instance segmentation.
        fill: Whether to fill each object in its own color. Otherwise, draw the outlines in REFERENCE_COLOR.
        width: The width of the outlines, in pixels.

    Returns:
        The RGB image with the overlay.
    """
    out = image.copy()
    if fill:
        lut = distinct_colors(segmentation)
        inside = segmentation > 0
        out[inside] = (1 - QUALITATIVE_FILL_ALPHA) * out[inside] + QUALITATIVE_FILL_ALPHA * lut[segmentation[inside]]
        return out
    outline = colors.to_rgb(REFERENCE_COLOR)
    if width == 1:
        outlines = find_boundaries(segmentation, mode="inner")
    elif width == 2:
        outlines = find_boundaries(segmentation, mode="thick")
    else:
        outlines = binary_dilation(find_boundaries(segmentation, mode="inner"), iterations=width // 2)
    out[outlines] = outline
    return out


def crop_to(image, aspect):
    """Return the central crop of an image with the given height to width ratio."""
    height, width = image.shape[:2]
    if height > aspect * width:
        rows = int(round(aspect * width))
        return image[(height - rows) // 2:(height - rows) // 2 + rows]
    columns = int(round(height / aspect))
    return image[:, (width - columns) // 2:(width - columns) // 2 + columns]


def plot_automatic_qualitative(methods_of, region=None, aspect=1.0, grid=None, show_titles=True):
    """Plot the chosen images in columns, with the reference and the methods in rows.

    Args:
        methods_of: The methods of each modality, in their order.
        region: The function that returns a sub-figure, see 'new_figure'.
        aspect: The height to width ratio of the images. Below 1, the images show their central crop.
        grid: The column grid of the modalities, see 'column_grid'.
        show_titles: Whether the columns show the modality titles.

    Returns:
        The figure.
    """
    starred = AUTOMATIC_STARRED_METHODS
    chosen_images = AUTOMATIC_QUALITATIVE
    header_height = QUALITATIVE_HEADER_HEIGHT - (0 if show_titles else QUALITATIVE_TITLE_HEIGHT)
    methods = merge_orders([methods_of[modality] for _, _, modality, *_ in chosen_images])
    rows = ["Ground truth"] + methods
    n_columns = len(chosen_images)
    n_gaps = sum(1 for a, b in zip(chosen_images, chosen_images[1:]) if a[2] != b[2])
    size = (
        FIGURE_WIDTH - QUALITATIVE_LABEL_WIDTH - QUALITATIVE_COLUMN_GAP * (n_columns - 1 - n_gaps)
        - QUALITATIVE_MODALITY_GAP * n_gaps
    ) / n_columns
    label_width = QUALITATIVE_LABEL_WIDTH
    if grid is not None:
        modalities = list(dict.fromkeys(chosen[2] for chosen in chosen_images))
        starts, width = column_grid(grid, len(modalities))
        per_column = n_columns // len(modalities)
        size = (width - QUALITATIVE_COLUMN_GAP * (per_column - 1)) / per_column
        label_width = grid[0]
    row_size = aspect * size
    height = header_height + len(rows) * row_size + (len(rows) - 1) * QUALITATIVE_ROW_GAP
    fig = new_figure(height, region)

    left, lefts = QUALITATIVE_LABEL_WIDTH, []
    for k, chosen in enumerate(chosen_images):
        if k and chosen[2] != chosen_images[k - 1][2]:
            left += QUALITATIVE_MODALITY_GAP - QUALITATIVE_COLUMN_GAP
            if grid is not None:
                left = starts[modalities.index(chosen[2])]
        elif k == 0 and grid is not None:
            left = starts[0]
        lefts.append(left)
        left += size + QUALITATIVE_COLUMN_GAP

    def center(row):
        return height - header_height - row * (row_size + QUALITATIVE_ROW_GAP) - row_size / 2

    for column, (dataset, sample, modality, domain) in enumerate(chosen_images):
        image = load_image(dataset, sample)
        labels = load_segmentation(dataset, sample, "labels")
        for row, method in enumerate(rows):
            ax = framed_axes(fig, lefts[column], center(row) - row_size / 2, size, row_size)
            if row == 0:
                outlined = overlay(image, labels, fill=False, width=OUTLINE_WIDTHS.get(dataset, 1))
                ax.imshow(crop_to(outlined, aspect), interpolation="none", aspect="auto")
                # The dataset name is in italics for in-domain and in bold for out-of-domain.
                in_domain = domain == "ID"
                ax.text(
                    0.5, 1.04, DATASET_NAMES.get(dataset, dataset), transform=ax.transAxes, ha="center",
                    va="bottom", fontsize=MAX_FONT_SIZE, color=TEXT_COLOR, style="italic" if in_domain else "normal",
                    fontweight="normal" if in_domain else "bold",
                )
            elif method in methods_of[modality]:
                segmentation = load_segmentation(dataset, sample, PREDICTION_NAMES[method])
                ax.imshow(crop_to(overlay(image, segmentation), aspect), interpolation="none", aspect="auto")
                draw_score(ax, mean_segmentation_accuracy(segmentation, labels))

    for row, method in enumerate(rows):
        name = names_of([method], starred)[0] if row else REFERENCE_NAME
        fig.text(
            (label_width - QUALITATIVE_NAME_GAP) / FIGURE_WIDTH, center(row) / height, name, ha="right", va="center",
            rotation=90, fontsize=MAX_FONT_SIZE, color=TEXT_COLOR,
        )

    for modality in dict.fromkeys(m for _, _, m, *_ in chosen_images) if show_titles else ():
        columns = [k for k, chosen in enumerate(chosen_images) if chosen[2] == modality]
        draw_title(fig, lefts[columns[0]], lefts[columns[-1]] + size, height - 0.2, PANELS[modality][2])
    return fig


def write_prediction_script(path):
    """Write the shell script that makes the predictions for the automatic qualitative plot."""
    python = "micromamba run -n super python"
    args = {
        "CellposeSAM": "evaluate_automatic_baselines.py --method cellpose -m cpsam",
        "micro-sam2": (
            "evaluate_automatic_segmentation.py -m hvit_l --mode apg --skip_tuning "
            "-c /user/archit/u12090/.cache/micro_sam/models/hvit_l_cells_decoder "
            "--interactive_checkpoint /user/archit/u12090/.cache/micro_sam/models/hvit_l_cells"
        ),
    }
    lines = ["#!/bin/bash", "export HF_HUB_OFFLINE=1", f"cd {os.path.abspath(EVALUATION)}"]
    experiments = os.path.join(os.path.dirname(QUALITATIVE_PREDICTIONS), "experiments")
    for dataset, sample, modality, *_ in AUTOMATIC_QUALITATIVE:
        env = f"SAMPLES={sample} OUT_DIR={os.path.join(QUALITATIVE_PREDICTIONS, dataset)}"
        for method in [method for method in PANEL_METHODS[modality] if method in QUALITATIVE_METHODS]:
            lines.append(
                f"{env} {python} {os.path.join(ROOT, 'run_posthoc_scripts.py')} predict_samples "
                f"{PREDICTION_NAMES[method]} {args[method]} -d {dataset} -e {experiments}"
            )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {len(lines) - 3} prediction runs to '{path}'.")


# The objects of the interactive qualitative plot, per modality: (dataset, image name, object id, domain).
INTERACTIVE_QUALITATIVE = {
    "label_free_cells": [
        ("neurips_cellseg_label_free", "00006", 70, "ID"),
        ("cellapp", "00013", 108, "OOD"),
    ],
    "fluorescence_cells": [
        ("tissuenet", "00013", 75, "ID"),
        ("medussa", "00017", 3, "OOD"),
    ],
    "fluorescence_nuclei": [
        ("cellbindb", "00007", 15, "ID"),
        ("covid_if_nuclei", "00017", 106, "OOD"),
    ],
    "histopathology_nuclei": [
        ("histo_miner", "00008", 62, "ID"),
        ("cd47_ihcnusc", "00015", 54, "OOD"),
    ],
}
# The gamma of the images of a dataset whose bright structures are saturated.
ZOOM_GAMMA = {}
# A crop is a square of ZOOM_CROP_SCALE times the longer side of the object box, and at least ZOOM_MIN_CROP pixels.
ZOOM_CROP_SCALE = 2.2
ZOOM_MIN_CROP = 40
# A tighter crop for large objects.
ZOOM_CROP_SCALES = {"neurips_cellseg_label_free": 1.5}
ZOOM_METHODS = ["micro-sam", "micro-sam2"]
ZOOM_STARRED = {"micro-sam"}
# The rows of the crops, top to bottom.
ZOOM_PROMPTS = ["point", "box"]
ZOOM_OBJECT_COLOR = "#ff4d6d"
ZOOM_FILL_ALPHA = 0.55
ZOOM_REFERENCE_WIDTH = 0.5
# The prompts are cyan, as in the micro-sam plots.
ZOOM_PROMPT_COLOR = (0.0, 1.0, 1.0)
# The size of the point prompt and the line width of the box prompt, in points.
ZOOM_POINT_SIZE = 3.0
ZOOM_BOX_WIDTH = 0.8
# The layout, in inches. The side of the full image is ZOOM_IMAGE_SIZE crops.
ZOOM_GAP = 0.04
ZOOM_IMAGE_SIZE = 1.5
ZOOM_BLOCK_GAP = 0.1
ZOOM_ROW_GAP = 0.04
ZOOM_DOMAIN_GAP = 0.08
ZOOM_CAPTION = 0.13
ZOOM_HEADER = 0.2
ZOOM_HEADER_NO_TITLES = 0.02


def single_prompt_folders(dataset, prompt):
    """Return the folders of the inputs and of the single-prompt predictions of micro-sam and micro-sam2."""
    root = SINGLE_PROMPT_RUNS.format(dataset=dataset)
    patterns = {
        "micro-sam2": f"{root}/microsam2/predictions/*hvit_l/{dataset}/{prompt}_with_masks_center",
        "micro-sam": f"{root}/microsam_v1/predictions/*/{dataset}/{prompt}_without_masks_center",
    }
    found = {method: glob.glob(pattern) for method, pattern in patterns.items()}
    for method, folders in found.items():
        if len(folders) != 1:
            raise FileNotFoundError(f"Expected one folder for '{patterns[method]}', found {len(folders)}.")
    return {
        "inputs": f"{found['micro-sam2'][0]}/inputs",
        "micro-sam2": f"{found['micro-sam2'][0]}/start_with_{prompt}/with_masks/iteration00",
        "micro-sam": f"{found['micro-sam'][0]}/predictions/iteration00",
    }


def load_single_prompt(dataset, image_name):
    """Return the image (RGB in [0, 1]), the labels and the predictions of ZOOM_METHODS with each prompt."""
    folders = {prompt: single_prompt_folders(dataset, prompt) for prompt in ZOOM_PROMPTS}
    inputs = folders[ZOOM_PROMPTS[0]]["inputs"]
    labels = tifffile.imread(f"{inputs}/labels/{image_name}.tif")
    image = to_rgb(tifffile.imread(f"{inputs}/images/{image_name}.tif"), labels.shape, dataset in RED_TO_MAGENTA)
    data = {
        (prompt, method): dict(np.load(f"{folders[prompt][method]}/{image_name}.npz"))
        for prompt in ZOOM_PROMPTS for method in ZOOM_METHODS
    }
    return image, labels, data


def crop_of(reference, scale):
    """Return the top, the left and the side of the square crop around an object, within the image."""
    ys, xs = np.nonzero(reference)
    side = int(max(ZOOM_MIN_CROP, scale * max(np.ptp(ys) + 1, np.ptp(xs) + 1)))
    side = min(side, *reference.shape)
    top = int(np.clip((ys.min() + ys.max()) / 2 - side / 2, 0, reference.shape[0] - side))
    left = int(np.clip((xs.min() + xs.max()) / 2 - side / 2, 0, reference.shape[1] - side))
    return top, left, side


def plot_interactive_qualitative(region=None, grid=None, show_titles=True):
    """Plot a column per modality with a block per object of INTERACTIVE_QUALITATIVE.

    A block is the full image with the crop marked, and the crops of the methods (columns) per prompt (rows).

    Args:
        region: The function that returns a sub-figure, see 'new_figure'.
        grid: The column grid of the modalities, see 'column_grid'.
        show_titles: Whether the columns show the modality titles.

    Returns:
        The figure.
    """
    chosen = INTERACTIVE_QUALITATIVE
    n_columns, n_rows = len(chosen), max(len(blocks) for blocks in chosen.values())
    header = ZOOM_HEADER if show_titles else ZOOM_HEADER_NO_TITLES
    grid = grid or (0, ZOOM_GAP, ZOOM_BLOCK_GAP + ZOOM_GAP)
    starts, column_width = column_grid(grid, n_columns)
    # The block height is the height of the two crops.
    size = (column_width - 2 * ZOOM_GAP) / (2 + ZOOM_IMAGE_SIZE)
    image_side = ZOOM_IMAGE_SIZE * size
    row_height = ZOOM_CAPTION + 2 * size + ZOOM_GAP
    domain_changes = [
        k for k in range(1, n_rows)
        if any(blocks[k][3] != blocks[k - 1][3] for blocks in chosen.values() if len(blocks) > k)
    ]
    height = (
        header + n_rows * row_height + (n_rows - 1) * ZOOM_ROW_GAP
        + len(domain_changes) * (ZOOM_DOMAIN_GAP - ZOOM_ROW_GAP) + 0.02
    )
    fig = new_figure(height, region)
    names = dict(zip(ZOOM_METHODS, names_of(ZOOM_METHODS, ZOOM_STARRED)))
    block_width = image_side + ZOOM_GAP + 2 * size + ZOOM_GAP

    for c, (modality, blocks) in enumerate(chosen.items()):
        left = starts[c]
        if show_titles:
            draw_title(fig, left, left + block_width - ZOOM_GAP, height - 0.17, PANELS[modality][2])
        top = height - header
        for col, method in enumerate(ZOOM_METHODS):
            x = left + image_side + ZOOM_GAP + col * (size + ZOOM_GAP) + size / 2
            y = top - ZOOM_CAPTION + 0.05
            name = names[method]
            fig.text(
                x / FIGURE_WIDTH, y / height, name, ha="center", va="baseline", fontsize=MAX_FONT_SIZE,
                color=TEXT_COLOR,
            )
        for r, (dataset, image_name, object_id, domain) in enumerate(blocks):
            top -= (ZOOM_DOMAIN_GAP if r in domain_changes else ZOOM_ROW_GAP) * (r > 0)
            plot_zoom_block(fig, left, top - row_height, size, image_side, dataset, image_name, object_id, domain)
            top -= row_height
    return fig


def plot_zoom_block(fig, left, bottom, size, image_side, dataset, image_name, object_id, domain):
    """Plot the block of one object of the interactive qualitative plot. The point crops are over the box crops."""
    image, labels, data = load_single_prompt(dataset, image_name)
    image = image ** ZOOM_GAMMA.get(dataset, 1.0)
    # The masks are in the order of the object ids.
    index = int(np.searchsorted(np.setdiff1d(np.unique(labels), [0]), object_id))
    reference = labels == object_id
    top, crop_left, side = crop_of(reference, ZOOM_CROP_SCALES.get(dataset, ZOOM_CROP_SCALE))
    window = np.s_[top:top + side, crop_left:crop_left + side]

    ax = framed_axes(fig, left, bottom + (2 * size + ZOOM_GAP - image_side) / 2, image_side, image_side)
    ax.imshow(image, interpolation="none")
    # Keep the crop mark inside the image, so that the frame does not hide it.
    inset = image.shape[1] / (72 * image_side) / 2
    x0, y0 = max(crop_left - 0.5, inset - 0.5), max(top - 0.5, inset - 0.5)
    x1 = min(crop_left + side - 0.5, image.shape[1] - 0.5 - inset)
    y1 = min(top + side - 0.5, image.shape[0] - 0.5 - inset)
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=ZOOM_OBJECT_COLOR, linewidth=1.0))
    in_domain = domain == "ID"
    name = ax.text(
        0.0, 1.0 + 0.05 / image_side, DATASET_NAMES.get(dataset, dataset), transform=ax.transAxes, ha="left",
        va="baseline", fontsize=MAX_FONT_SIZE, color=TEXT_COLOR, style="italic" if in_domain else "normal",
        fontweight="normal" if in_domain else "bold",
    )
    # A name wider than the image and the gap after it would run under the crops, so only such a name gets smaller.
    width = name.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    if width > image_side + ZOOM_GAP:
        name.set_fontsize(MAX_FONT_SIZE * (image_side + ZOOM_GAP) / width)

    # The prompt marks have a fixed size, so that they look the same in crops of any size.
    crops_left = left + image_side + ZOOM_GAP
    rgb = np.array(colors.to_rgb(ZOOM_OBJECT_COLOR))
    for row, prompt in enumerate(ZOOM_PROMPTS):
        y = bottom + (1 - row) * (size + ZOOM_GAP)
        for col, method in enumerate(ZOOM_METHODS):
            x = crops_left + col * (size + ZOOM_GAP)
            mask = data[prompt, method]["masks"][index]
            out = image.copy()
            out[mask] = (1 - ZOOM_FILL_ALPHA) * out[mask] + ZOOM_FILL_ALPHA * rgb
            ax = framed_axes(fig, x, y, size, size)
            ax.imshow(out[window], interpolation="none")
            ax.set_xlim(-0.5, side - 0.5)
            ax.set_ylim(side - 0.5, -0.5)
            contour = ax.contour(
                reference[window], levels=[0.5], colors=REFERENCE_COLOR, linewidths=ZOOM_REFERENCE_WIDTH
            )
            contour.set(path_effects=[patheffects.withStroke(linewidth=2 * ZOOM_REFERENCE_WIDTH, foreground="black")])
            if prompt == "point":
                px, py = data[prompt, method]["points"][index][0]
                ax.plot(
                    px - crop_left, py - top, "o", markersize=ZOOM_POINT_SIZE, color=ZOOM_PROMPT_COLOR,
                    markeredgecolor="black", markeredgewidth=0.3,
                )
            else:
                bx0, by0, bx1, by1 = data[prompt, method]["boxes"][index]
                ax.add_patch(plt.Rectangle(
                    (bx0 - crop_left - 0.5, by0 - top - 0.5), bx1 - bx0, by1 - by0, fill=False,
                    edgecolor=ZOOM_PROMPT_COLOR, linewidth=ZOOM_BOX_WIDTH,
                ))
            iou = np.logical_and(mask, reference).sum() / np.logical_or(mask, reference).sum()
            draw_score(ax, iou)


def check_style():
    """Print a warning when the text does not use an Arial-metric font."""
    font = font_manager.findfont(font_manager.FontProperties(family=["sans-serif"]))
    name = font_manager.FontProperties(fname=font).get_name()
    if name not in ("Arial", "Helvetica", "Liberation Sans"):
        print(
            f"Warning: the text uses '{name}', not Arial or Liberation Sans. Install Liberation Sans in "
            "'~/.local/share/fonts/liberation' to get the plots of the paper."
        )


def check_inputs(names):
    """Raise an error that lists the missing inputs of the plots in 'names'."""
    paths = []
    if "automatic_2d" in names:
        paths.append(RESULTS_CSV)
    if "interactive_2d" in names:
        paths.append(INTERACTIVE_CSV)
    if "automatic_2d_qualitative" in names:
        for dataset, sample, *_ in AUTOMATIC_QUALITATIVE:
            paths += [
                os.path.join(QUALITATIVE_PREDICTIONS, dataset, f"{name}_{sample}.tif") for name in ["image", "labels"]
            ]
    if "interactive_2d_qualitative" in names:
        paths += [
            SINGLE_PROMPT_RUNS.format(dataset=dataset)
            for blocks in INTERACTIVE_QUALITATIVE.values() for dataset, *_ in blocks
        ]
    missing = sorted({path for path in paths if not os.path.exists(path)})
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))


# The plots in their order in the combined figure, with their kind and style (see 'save_figure').
PLOTS = {
    "automatic_2d": ("automatic", "quantitative"),
    "automatic_2d_qualitative": ("automatic", "qualitative"),
    "interactive_2d": ("interactive", "quantitative"),
    "interactive_2d_qualitative": ("interactive", "qualitative"),
}
# The layout of the combined figure, in inches.
COMBINED_NAME = "figure_2"
COMBINED_DIR = os.path.join(ROOT, "figures", "final")
PANEL_GAP = 0.07
# The size of the panel letters in points, as Nature asks for.
PANEL_LABEL_SIZE = 8
COMBINED_PAD = 0.04
COMBINED_ROW_HEIGHT = 0.14
FIGURE_ROW_HEIGHT = 0.105
COMBINED_GROUP_GAP = 0.25
COMBINED_ROW_GAP = 0.1
# A lighter box half, so that the curves stand out.
COMBINED_BAND_COLOR = "#f8f7f3"
COMBINED_PANEL_HEIGHT = 0.85
COMBINED_ASPECT = 2 / 3
# The modality columns of all plots (left room, right room, gap), so that the columns align.
COMBINED_GRID = (0.55, 0.35, 0.25)
# The image plots widen the columns of COMBINED_GRID into the gaps, and stay centered under the titles.
IMAGE_GAP = 0.1
IMAGE_WIDENING = (COMBINED_GRID[2] - IMAGE_GAP) / 2
COMBINED_IMAGE_GRID = (COMBINED_GRID[0] - IMAGE_WIDENING, COMBINED_GRID[1] - IMAGE_WIDENING, IMAGE_GAP)
# A combined figure higher than A4 is scaled down.
A4 = (210 / 25.4, 297 / 25.4)
PNG_DPI = 600
# The resolution of the images in the saved plots and in the combined figure: above the resolution of the shown
# crops, so that no detail is lost. The default 'savefig.dpi' of 300 drops most of it.
IMAGE_DPI = 1200


def make_plot(name, region=None, compact=False):
    """Make one plot, as its own figure or in a region of the combined figure (see 'new_figure').

    With 'compact', the plots use the sizes of the combined figure, and only the first plot shows the modality titles.
    """
    modalities = [modality for row in AUTOMATIC_ROWS for modality in row]
    titles = {modality: title for modality, (_, _, title) in PANELS.items()}
    if name == "automatic_2d":
        panels = {modality: (title, SCORE_LABEL, PANEL_METHODS[modality]) for modality, title in titles.items()}
        results = {
            modality: one_mode(
                select_datasets(load_modes(category, structure, PANEL_MODELS.get(modality)), PANEL_DATASETS[modality]),
                FIGURE_MODES,
            )
            for modality, (category, structure, _) in PANELS.items()
        }
        return plot_automatic(
            results, AUTOMATIC_ROWS, panels, region=region, row_height=FIGURE_ROW_HEIGHT if compact else BAR_ROW_HEIGHT,
            domain_offset=0.1, grid=COMBINED_GRID if compact else None, title_line=compact, ranks=True,
            group_gap=COMBINED_GROUP_GAP if compact else GROUP_GAP, row_gap=COMBINED_ROW_GAP if compact else None,
        )
    if name == "interactive_2d":
        results = {
            modality: load_interactive(category, structure, PANEL_DATASETS[modality])
            for modality, (category, structure, _) in PANELS.items()
        }
        return plot_interactive_by_domain(
            results, modalities, titles, region=region,
            panel_height=COMBINED_PANEL_HEIGHT if compact else PANEL_HEIGHT,
            legend_panel=("OOD", "histopathology_nuclei"), grid=COMBINED_GRID if compact else None,
            title_line=compact, shared_rows=True, show_titles=not compact, score_gap=0.32 if compact else None,
            row_gap=COMBINED_ROW_GAP if compact else None,
            band_color=COMBINED_BAND_COLOR if compact else BOX_BAND_COLOR, prompts_on_top=True,
            x_label=ITERATION_LABEL,
        )
    if name == "automatic_2d_qualitative":
        methods_of = {
            modality: [method for method in methods if method in QUALITATIVE_METHODS]
            for modality, methods in PANEL_METHODS.items()
        }
        return plot_automatic_qualitative(
            methods_of, region=region, aspect=COMBINED_ASPECT if compact else 1.0,
            grid=COMBINED_IMAGE_GRID if compact else None, show_titles=not compact,
        )
    return plot_interactive_qualitative(
        region=region, grid=COMBINED_IMAGE_GRID if compact else None, show_titles=not compact
    )


def margins_of(name, make):
    """Return the empty room above and below the drawing of a compact plot, in inches.

    The combined figure removes this room, which holds for example the legend of the plot alone.
    """
    fig = make(name, compact=True)
    box = fig.get_tightbbox(fig.canvas.get_renderer())
    margins = fig.get_figheight() - box.y1, box.y0
    plt.close(fig)
    return margins


def plot_figure(plots=None, make=None):
    """Plot all plots in one figure, top to bottom, with PANEL_GAP between their drawings.

    Each plot is a sub-figure of its own height. The figure is higher than necessary, and 'save_combined' cuts it.

    Args:
        plots: The names of the plots, in their order. The default is PLOTS.
        make: The function that makes one plot. The default is 'make_plot'.

    Returns:
        The figure, and the bottom and the top of the drawing, in inches.
    """
    plots, make = list(plots or PLOTS), make or make_plot
    margins = {name: margins_of(name, make) for name in plots}
    fig = plt.figure(figsize=(FIGURE_WIDTH, 40))
    total = fig.get_figheight()
    # The top of the drawing of the next plot.
    start = top = total - 1

    def region(height):
        # A sub-figure fills a grid cell and ignores the grid margins, so the room above and below are cells too.
        upper = top + margins[name][0]
        ratios = [max(total - upper, 1e-9), height, max(upper - height, 1e-9)]
        grid = fig.add_gridspec(3, 1, height_ratios=ratios, hspace=0)
        return fig.add_subfigure(grid[1])

    for k, name in enumerate(plots):
        sub = make(name, region, compact=True)
        # The first plot carries the modality titles of all plots, so its letter stands below them, at the top of its
        # panels.
        label_top = top
        if k == 0:
            renderer = fig.canvas.get_renderer()
            label_top = max(ax.get_window_extent(renderer).y1 for ax in sub.axes) / fig.dpi
        fig.text(
            0, label_top / total, "abcdefgh"[k], ha="left", va="top", fontsize=PANEL_LABEL_SIZE, fontweight="bold",
            color=TEXT_COLOR,
        )
        top -= height_of(sub) - sum(margins[name]) + PANEL_GAP
    return fig, top + PANEL_GAP, start


def save_combined(fig, bottom, top, name=COMBINED_NAME, dpi=None):
    """Save the combined figure as svg, pdf and png in COMBINED_DIR.

    The function cuts the figure to its drawing between 'bottom' and 'top', in inches, and scales it down to fit A4.
    rsvg-convert makes the pdf and the png from the svg, so that all three files show the same figure. 'dpi' sets the
    resolution of the images in the svg and of the png. The default keeps 'savefig.dpi' for the images and PNG_DPI
    for the png.
    """
    converter = shutil.which("rsvg-convert")
    if converter is None:
        raise RuntimeError("The combined figure needs 'rsvg-convert' (librsvg) to write its pdf and png.")
    os.makedirs(COMBINED_DIR, exist_ok=True)
    bottom, top = bottom - COMBINED_PAD, top + COMBINED_PAD
    height = top - bottom
    box = transforms.Bbox([[0, bottom], [FIGURE_WIDTH, top]])
    buffer = io.StringIO()
    with plt.rc_context({"svg.hashsalt": "micro-sam2"}):
        fig.savefig(buffer, format="svg", bbox_inches=box, metadata={"Date": None}, dpi=dpi)
    plt.close(fig)
    scale = min(1.0, A4[0] / FIGURE_WIDTH, A4[1] / height)

    def scaled(match):
        return f'{match.group(1)}="{float(match.group(2)) * scale:.3f}pt"'

    # The first width and height are the page size. The view box keeps the drawing.
    svg = re.sub(r'(width|height)="([0-9.]+)pt"', scaled, buffer.getvalue(), count=2)
    path = os.path.join(COMBINED_DIR, name)
    with open(f"{path}.svg", "w") as f:
        f.write(svg)
    # SOURCE_DATE_EPOCH fixes the date in the pdf.
    env = dict(os.environ, SOURCE_DATE_EPOCH="0")
    subprocess.run([converter, "-f", "pdf", "-o", f"{path}.pdf", f"{path}.svg"], check=True, env=env)
    png_dpi = dpi or PNG_DPI
    png_args = ["-f", "png", "-d", str(png_dpi), "-p", str(png_dpi), "-b", "white", "-o", f"{path}.png"]
    subprocess.run([converter, *png_args, f"{path}.svg"], check=True, env=env)
    # rsvg-convert writes no resolution into the png, so we add it.
    with Image.open(f"{path}.png") as image:
        image.load()
    image.save(f"{path}.png", dpi=(png_dpi, png_dpi))
    width_mm, height_mm = FIGURE_WIDTH * scale * 25.4, height * scale * 25.4
    print(f"Saved '{name}' ({width_mm:.0f} x {height_mm:.0f} mm, scaled by {scale:.2f}) to '{COMBINED_DIR}'.")


def main():
    names = list(PLOTS) + [COMBINED_NAME]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=names, default=names, help="Make only these plots.")
    parser.add_argument("--write_prediction_script", help="Write the prediction script to this path, and stop.")
    args = parser.parse_args()
    if args.write_prediction_script:
        write_prediction_script(args.write_prediction_script)
        return

    check_inputs(list(PLOTS) if COMBINED_NAME in args.only else args.only)
    set_style()
    check_style()
    for name, (kind, style) in PLOTS.items():
        if name in args.only:
            save_figure(make_plot(name), name, kind, style, dpi=IMAGE_DPI)
    if COMBINED_NAME in args.only:
        save_combined(*plot_figure(), dpi=IMAGE_DPI)


if __name__ == "__main__":
    main()
