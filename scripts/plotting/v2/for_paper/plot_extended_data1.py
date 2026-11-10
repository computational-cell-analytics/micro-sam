"""Plots for Extended Data Figure 1 of the micro-sam2 paper: the quantitative panels of Figures 2 and 3 for all model
sizes of micro-sam2.

The figure repeats panels a and c of Figure 2 and panels a and c of Figure 3, on the same datasets, with the four
Hiera sizes of micro-sam2 (T, S, B, L) and of SAM2. The script writes these plots to
'figures/extendeddata1/<automatic|interactive>/quantitative':
- 'automatic_2d' (panel a): the mSA of CellposeSAM and of micro-sam2 (APG) per modality, for the in-domain and the
  out-of-domain datasets.
- 'interactive_2d' (panel b): the mSA of SAM2 and micro-sam2 over the correction iterations, with a point or a box as
  the first prompt.
- 'automatic_3d' (panel c): as panel a for 3d. The EM neurites compare SegNeuron instead of CellposeSAM.
- 'interactive_3d' (panel d): as panel b for 3d, with nnInteractive.
- 'extended_data_1' (in 'figures/final'): the four plots in one figure, as panels a to d.

A model size is a tint of the color of its method: T is the lightest, L has the full color, as in Figures 2 and 3.

Usage:

    python plot_extended_data1.py
    python plot_extended_data1.py --only extended_data_1
"""

import os
import argparse

import pandas as pd
import matplotlib.colors as colors
import matplotlib.pyplot as plt

import plots_for_figure2 as fig2
import plots_for_figure3 as fig3
from plot_style import METHOD_COLORS, set_style

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "figures", "extendeddata1")
COMBINED_NAME = "extended_data_1"

# The Hiera sizes, from the smallest to the largest, with their short names and the share of the method color.
SIZES = ["hvit_t", "hvit_s", "hvit_b", "hvit_l"]
SIZE_NAMES = {"hvit_t": "T", "hvit_s": "S", "hvit_b": "B", "hvit_l": "L"}
SIZE_TINTS = {"hvit_t": 0.5, "hvit_s": 0.67, "hvit_b": 0.83, "hvit_l": 1.0}

# The ('method', 'model') of the automatic baselines, per dimension. micro-sam2 runs APG.
AUTOMATIC_BASELINES_2D = {"CellposeSAM": ("cellpose", "cpsam")}
AUTOMATIC_BASELINES_3D = {"CellposeSAM": ("cellpose", "cpsam_do3d"), "SegNeuron": ("segneuron", "default")}
# The baseline of each 3d modality: SegNeuron for the EM neurites, where CellposeSAM has no results.
BASELINE_3D = {"fluorescence_cells": "CellposeSAM", "fluorescence_nuclei": "CellposeSAM", "em_neurites": "SegNeuron"}
MICROSAM2_MODE = "apg"

# The interactive methods with one size, as (method, model), and those with all sizes, as their method.
INTERACTIVE_SINGLE_3D = {"nnInteractive": ("nninteractive", "default")}
INTERACTIVE_SIZED = {"SAM2": "sam2", "micro-sam2": "micro-sam2"}


def tinted(color, tint):
    """Return the color mixed with white, with the share 'tint' of the color."""
    return colors.to_hex([1 - tint * (1 - c) for c in colors.to_rgb(color)])


def size_label(method, size):
    """The name of a method with one model size in the plots, such as 'μSAM2 T'."""
    return f"{fig2.names_of([method], set())[0]} {SIZE_NAMES[size]}"


def interactive_methods(single):
    """Return the interactive methods, as name -> ('method', models, mode), and their colors.

    Each size of a sized method is a method of its own, named '<method> <size>'.
    """
    methods, method_colors = {}, {}
    for name, (method, model) in single.items():
        methods[name] = (method, (model,), "")
        method_colors[name] = METHOD_COLORS[name]
    for name, method in INTERACTIVE_SIZED.items():
        for size in SIZES:
            key = f"{name} {size}"
            methods[key] = (method, (size,), "")
            method_colors[key] = tinted(METHOD_COLORS[name], SIZE_TINTS[size])
    return methods, method_colors


def load_automatic(path, category, structure, datasets, baselines, metric="msa"):
    """Load the automatic scores of the baselines and of each size of micro-sam2 for one modality.

    The column 'mode' holds the size of micro-sam2, and an empty string for the baselines.
    """
    df = pd.read_csv(path)
    df = df[(df["category"] == category) & (df["structure"] == structure)]
    parts = []
    for name, (method, model) in baselines.items():
        rows = df[(df["method"] == method) & (df["model"] == model) & df["mode"].isna()]
        parts.append(rows.assign(method=name, mode=""))
    rows = df[(df["method"] == "micro-sam2") & df["model"].isin(SIZES) & (df["mode"] == MICROSAM2_MODE)]
    parts.append(rows.assign(mode=rows["model"]))
    results = pd.concat(parts, ignore_index=True)[["dataset", "method", "mode", metric]]
    return fig2.select_datasets(results.rename(columns={metric: "score"}), datasets)


def plot_automatic(dimension, region=None, compact=False):
    """Plot the automatic results of one dimension, with micro-sam2 in all sizes, in the layout of Figure 2 or 3."""
    results, panels = {}, {}
    if dimension == "2d":
        for name, (category, structure, title) in fig2.PANELS.items():
            results[name] = load_automatic(
                fig2.RESULTS_CSV, category, structure, fig2.PANEL_DATASETS[name], AUTOMATIC_BASELINES_2D,
            )
            panels[name] = (title, fig2.SCORE_LABEL, ["CellposeSAM", "micro-sam2"])
        rows, grid, domain_offset = fig2.AUTOMATIC_ROWS, fig2.COMBINED_GRID, 0.1
    else:
        for name, (category, structure, title, metric, score_label) in fig3.PANELS.items():
            baseline = BASELINE_3D[name]
            results[name] = load_automatic(
                fig3.RESULTS_CSV, category, structure, fig3.PANEL_DATASETS[name],
                {baseline: AUTOMATIC_BASELINES_3D[baseline]}, metric,
            )
            panels[name] = (title, score_label, [baseline, "micro-sam2"])
        rows, grid, domain_offset = fig3.AUTOMATIC_ROWS, fig3.COMBINED_GRID, 0.27
    return fig2.plot_automatic(
        results, rows, panels, starred=set(), domain_offset=domain_offset, region=region,
        row_height=fig2.FIGURE_ROW_HEIGHT if compact else fig2.BAR_ROW_HEIGHT,
        grid=grid if compact else None, title_line=compact, ranks=True,
        group_gap=fig2.COMBINED_GROUP_GAP if compact else fig2.GROUP_GAP,
        row_gap=fig2.COMBINED_ROW_GAP if compact else None,
    )


def add_size_legend(fig, methods, method_colors):
    """Add the legend of the interactive plot under its panels, in one row."""
    handles, labels = [], []
    for key in methods:
        name, _, size = key.partition(" ")
        handles.append(plt.Line2D(
            [], [], color=method_colors[key], marker="o", markersize=2.2, linewidth=1.0, markeredgecolor="white",
            markeredgewidth=0.4,
        ))
        labels.append(size_label(name, size) if size else fig2.names_of([name], set())[0])
    bottom = min(ax.get_position().y0 for ax in fig.axes)
    offset = (fig2.X_LABEL_OFFSET + 0.14) / fig2.height_of(fig)
    legend = fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, bottom - offset), ncol=len(labels),
        fontsize=fig2.MAX_FONT_SIZE, handlelength=1.1, handletextpad=0.3, columnspacing=0.8, borderpad=0.25,
        frameon=True, framealpha=1, facecolor="white", edgecolor=fig2.LEGEND_BORDER_COLOR,
    )
    legend.get_frame().set_linewidth(0.6)


def plot_interactive(dimension, region=None, compact=False):
    """Plot the interactive results of one dimension, with SAM2 and micro-sam2 in all sizes."""
    if dimension == "2d":
        modalities = [name for row in fig2.AUTOMATIC_ROWS for name in row]
        specs = {name: (category, structure, "msa", fig2.SCORE_LABEL)
                 for name, (category, structure, _) in fig2.PANELS.items()}
        titles = {name: title for name, (_, _, title) in fig2.PANELS.items()}
        datasets, path, grid, single = fig2.PANEL_DATASETS, fig2.INTERACTIVE_CSV, fig2.COMBINED_GRID, {}
    else:
        modalities = [name for row in fig3.AUTOMATIC_ROWS for name in row]
        specs = {name: (category, structure, metric, score_label)
                 for name, (category, structure, _, metric, score_label) in fig3.PANELS.items()}
        titles = {name: title for name, (_, _, title, _, _) in fig3.PANELS.items()}
        datasets, path, grid = fig3.PANEL_DATASETS, fig3.INTERACTIVE_CSV, fig3.COMBINED_GRID
        single = INTERACTIVE_SINGLE_3D
    methods, method_colors = interactive_methods(single)
    results = {
        name: fig2.load_interactive(
            category, structure, datasets[name], path=path, methods=methods, metric=metric,
        ).dropna(subset=["score"])
        for name, (category, structure, metric, _) in specs.items()
    }
    score_labels = {name: score_label for name, (_, _, _, score_label) in specs.items()}
    # The curves look up their colors in the color table of Figure 2.
    fig2.METHOD_COLORS = {**METHOD_COLORS, **method_colors}
    try:
        fig = fig2.plot_interactive_by_domain(
            results, modalities, titles, starred=set(), methods=methods, score_labels=score_labels,
            region=region, panel_height=fig2.COMBINED_PANEL_HEIGHT if compact else fig2.PANEL_HEIGHT,
            grid=grid if compact else None, title_line=compact, shared_rows=True, show_titles=not compact,
            score_gap=0.32 if compact else None, row_gap=fig2.COMBINED_ROW_GAP if compact else None,
            band_color=fig2.COMBINED_BAND_COLOR if compact else fig2.BOX_BAND_COLOR, prompts_on_top=True,
            x_label=fig2.ITERATION_LABEL,
        )
    finally:
        fig2.METHOD_COLORS = METHOD_COLORS
    add_size_legend(fig, methods, method_colors)
    return fig


# The plots in their order in the combined figure, with their kind and style (see 'fig2.save_figure').
PLOTS = {
    "automatic_2d": ("automatic", "quantitative"),
    "interactive_2d": ("interactive", "quantitative"),
    "automatic_3d": ("automatic", "quantitative"),
    "interactive_3d": ("interactive", "quantitative"),
}


def make_plot(name, region=None, compact=False):
    """Make one plot, as its own figure or in a region of the combined figure (see 'fig2.plot_figure')."""
    kind, dimension = name.split("_")
    plot = plot_automatic if kind == "automatic" else plot_interactive
    return plot(dimension, region=region, compact=compact)


def set_size_modes():
    """Let the bar plots of Figure 2 show the sizes of micro-sam2 as its sub-bars, in place of the modes."""
    fig2.MODES = SIZES
    fig2.MODE_NAMES = SIZE_NAMES
    fig2.MODE_TINTS = SIZE_TINTS


def main():
    names = list(PLOTS) + [COMBINED_NAME]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=names, default=names, help="Make only these plots.")
    args = parser.parse_args()
    set_style()
    fig2.check_style()
    set_size_modes()
    fig2.OUTPUT_DIR = OUTPUT_DIR
    for name, (kind, style) in PLOTS.items():
        if name in args.only:
            fig2.save_figure(make_plot(name), name, kind, style)
    if COMBINED_NAME in args.only:
        fig2.save_combined(*fig2.plot_figure(PLOTS, make_plot), name=COMBINED_NAME)


if __name__ == "__main__":
    main()
