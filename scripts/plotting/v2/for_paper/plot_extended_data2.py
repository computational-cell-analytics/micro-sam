"""Plots for Extended Data Figure 2 of the micro-sam2 paper: the runtimes of micro-sam2 on CPU, GPU and Apple silicon.

The numbers come from 'finetuning/v2/evaluation/benchmark_runtimes.py' (run 'collect' there first): one timed run
per sample of LIVECell (2d LM), GoNuclear (3d LM) and CREMI (3d EM), for the four sizes of micro-sam2 (T, S, B, L)
and the references CellposeSAM (LIVECell, GoNuclear) and SegNeuron (CREMI). The script writes these plots to
'figures/extendeddata2/<automatic|interactive>/quantitative':
- 'automatic_runtimes' (panel a): the mean time per image (2d) or per volume (3d) of AIS and APG, split into the
  encoder, the decoder and the postprocessing (for APG also the prompting of SAM2 with the decoder seeds), and of the
  references, split into their network and their postprocessing. The error bars are the standard deviation of the
  total over the samples.
- 'interactive_runtimes' (panel b): the mean time of one object from its first prompt (a point or a box), per
  object in 2d and per object and slice in 3d, with the standard deviation over the objects, and the time of the
  encoder per image or volume (in 3d with the set-up of the video predictor).
- 'extended_data_2' (in 'figures/final'): both plots in one figure.

A device without results (e.g. the Mac before it is run) shows 'Results pending'.

Usage:

    python plot_extended_data2.py
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import plots_for_figure2 as fig2
from plot_style import METHOD_COLORS, TEXT_COLOR, set_style

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "figures", "extendeddata2")
COMBINED_NAME = "extended_data_2"
BENCHMARK_ROOT = os.environ.get(
    "RUNTIME_BENCHMARK_ROOT", "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/runtime_benchmark"
)
SAMPLE_CSV = os.path.join(BENCHMARK_ROOT, "runtimes_per_sample.csv")
OBJECT_CSV = os.path.join(BENCHMARK_ROOT, "runtimes_per_object.csv")

DEVICES = {"cpu": "CPU (16 cores)", "a100": "GPU (A100)", "m1": "Apple M1"}
DATASETS = {"livecell": "LIVECell (2d, LM)", "gonuclear": "GoNuclear (3d, LM)", "cremi": "CREMI (3d, EM)"}
REFERENCES = {"cellposesam": "CellposeSAM", "segneuron": "SegNeuron"}
SIZES = ["hvit_t", "hvit_s", "hvit_b", "hvit_l"]
SIZE_NAMES = {"hvit_t": "T", "hvit_s": "S", "hvit_b": "B", "hvit_l": "L"}
MODES = {"ais": "AIS", "apg": "APG"}
PROMPTS = {"interactive_point": "Point", "interactive_box": "Box"}
# The number of slices of the 3d samples (all evaluation crops are 32 x 512 x 512), for the time per slice.
N_SLICES = {"gonuclear": 32, "cremi": 32}
# The share of the micro-sam2 color for the interactive bars: a tint that no part of panel a uses.
INTERACTIVE_TINT = 0.8

# The share of the method color per part: the encoder has the full color, the later parts are lighter.
PART_TINTS = {"encoder": 1.0, "network": 1.0, "decoder": 0.6, "postprocessing": 0.3}
PART_NAMES = {"encoder": "Encoder", "decoder": "Decoder", "network": "Network", "postprocessing": "Postprocessing"}

BAR_HEIGHT = 0.72
ROW_HEIGHT = 0.12
PANEL_GRID = (0.75, 0.15, 0.25)


def tinted(color, tint):
    return colors.to_hex([1 - tint * (1 - c) for c in colors.to_rgb(color)])


def load():
    samples = pd.read_csv(SAMPLE_CSV) if os.path.exists(SAMPLE_CSV) else pd.DataFrame(
        columns=["device_tag", "dataset", "method", "model", "sample", "part", "seconds"]
    )
    objects = pd.read_csv(OBJECT_CSV) if os.path.exists(OBJECT_CSV) else pd.DataFrame(
        columns=["device_tag", "dataset", "method", "model", "seconds_per_slice", "n_slices"]
    )
    return samples, objects


def automatic_rows(dataset):
    """The bars of one automatic panel, top to bottom, as (label, method, model, method color name)."""
    reference = "segneuron" if dataset == "cremi" else "cellposesam"
    rows = [("Reference", reference, "default", REFERENCES[reference])]
    for mode, mode_name in MODES.items():
        rows += [(f"μSAM2 {mode_name} {SIZE_NAMES[size]}", mode, size, "micro-sam2") for size in SIZES]
    return rows


def draw_pending(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, "Results pending", ha="center", va="center", style="italic", color=fig2.GROUP_COLOR,
            transform=ax.transAxes, fontsize=fig2.MAX_FONT_SIZE)


def style_axes(ax):
    ax.grid(axis="x", color="#e6e5e0", linewidth=0.5, linestyle=(0, (3, 3)), zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def grid_axes(fig, n_rows, n_columns, panel_height, top, row_gap=0.25):
    """Axes in a grid of devices (rows) and datasets (columns), with sizes in inches from the top of the figure."""
    lefts, width = fig2.column_grid(PANEL_GRID, n_columns)
    height = fig2.height_of(fig)
    axes = np.empty((n_rows, n_columns), dtype=object)
    for i in range(n_rows):
        bottom = height - top - (i + 1) * panel_height - i * row_gap
        for j, left in enumerate(lefts):
            axes[i, j] = fig.add_axes([
                left / fig2.FIGURE_WIDTH, bottom / height, width / fig2.FIGURE_WIDTH, panel_height / height,
            ])
    return axes


def label_grid(fig, axes):
    for j, title in enumerate(DATASETS.values()):
        axes[0, j].set_title(title, fontsize=fig2.MAX_FONT_SIZE, fontweight="bold", color=TEXT_COLOR, pad=4)
    for i, device in enumerate(DEVICES.values()):
        box = axes[i, -1].get_position()
        fig.text(box.x1 + 0.01, (box.y0 + box.y1) / 2, device, rotation=270, ha="left", va="center",
                 fontsize=fig2.MAX_FONT_SIZE, color=TEXT_COLOR, fontweight="bold")


def plot_automatic(region=None, compact=False):
    samples, _ = load()
    n_rows = len(automatic_rows("livecell"))
    panel_height = (n_rows + 0.6) * ROW_HEIGHT
    legend_height = 0.3
    height = 0.3 + len(DEVICES) * (panel_height + 0.25) + 0.5 + legend_height
    fig = fig2.new_figure(height, region)
    axes = grid_axes(fig, len(DEVICES), len(DATASETS), panel_height, top=0.3)
    for i, device in enumerate(DEVICES):
        for j, dataset in enumerate(DATASETS):
            ax = axes[i, j]
            rows = automatic_rows(dataset)
            data = samples[(samples["device_tag"] == device) & (samples["dataset"] == dataset)]
            y = -np.arange(len(rows)) - np.array([0 if k == 0 else 0.4 if k <= 4 else 0.8 for k in range(len(rows))])
            if data.empty:
                draw_pending(ax)
            else:
                means = data.groupby(["method", "model", "part"])["seconds"].mean()
                totals = data[data["part"] == "total"].groupby(["method", "model"])["seconds"].agg(["mean", "std"])
                end = 0.0
                for yi, (label, method, model, color_name) in zip(y, rows):
                    if (method, model) not in totals.index:
                        continue
                    parts = ["network", "postprocessing"] if method in REFERENCES else [
                        "encoder", "decoder", "postprocessing"]
                    left = 0.0
                    for part in parts:
                        value = means.get((method, model, part), 0.0)
                        bar = ax.barh(yi, value, left=left, height=BAR_HEIGHT, linewidth=0, zorder=2,
                                      color=tinted(METHOD_COLORS[color_name], PART_TINTS[part]))[0]
                        if color_name == "micro-sam2":
                            bar.set_hatch("///")
                            bar.set_edgecolor("white")
                        left += value
                    mean, std = totals.loc[(method, model)]
                    ax.errorbar(mean, yi, xerr=0 if np.isnan(std) else std, color=fig2.GROUP_COLOR, linewidth=0.6,
                                capsize=1.2, zorder=3)
                    note = f" {mean:.2f}" if mean < 100 else f" {mean:.0f}"
                    if dataset in N_SLICES:
                        note += f" ({mean / N_SLICES[dataset]:.2g} s/slice)"
                    if method in REFERENCES:
                        note += f"  {REFERENCES[method]}"
                    ax.text(mean + (0 if np.isnan(std) else std), yi, note, va="center", ha="left",
                            fontsize=fig2.SMALL_FONT_SIZE, color=TEXT_COLOR)
                    end = max(end, mean + (0 if np.isnan(std) else std))
                ax.set_xlim(0, end * (1.9 if dataset in N_SLICES else 1.45))
                style_axes(ax)
            ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
            ax.set_yticks(y)
            ax.set_yticklabels([row[0] for row in rows] if j == 0 else [], fontsize=fig2.SMALL_FONT_SIZE)
    label_grid(fig, axes)
    bottom_box = axes[-1, 0].get_position()
    fig.text(0.5, bottom_box.y0 - 0.25 / height, "Time per image (2d) or volume (3d) [s] ↓", ha="center", va="top",
             fontsize=fig2.MAX_FONT_SIZE, fontweight="bold", color=TEXT_COLOR)
    fig.text(0.5, bottom_box.y0 - 0.4 / height, "Reference: CellposeSAM (LIVECell, GoNuclear), SegNeuron (CREMI)",
             ha="center", va="top", fontsize=fig2.SMALL_FONT_SIZE, color=TEXT_COLOR)
    handles = [plt.Rectangle((0, 0), 1, 1, color=tinted(METHOD_COLORS["micro-sam2"], PART_TINTS[part]))
               for part in ["encoder", "decoder", "postprocessing"]]
    labels = ["μSAM2 encoder", "μSAM2 decoder", "μSAM2 postprocessing"]
    for name in REFERENCES.values():
        handles += [plt.Rectangle((0, 0), 1, 1, color=tinted(METHOD_COLORS[name], PART_TINTS[part]))
                    for part in ["network", "postprocessing"]]
        labels += [f"{name} network", f"{name} postprocessing"]
    legend = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4,
                        fontsize=fig2.MAX_FONT_SIZE, frameon=True, edgecolor=fig2.LEGEND_BORDER_COLOR,
                        handlelength=1.1, columnspacing=0.8)
    legend.get_frame().set_linewidth(0.6)
    return fig


def plot_interactive(region=None, compact=False):
    samples, objects = load()
    rows = [(f"{PROMPTS[method]} {SIZE_NAMES[size]}", method, size) for method in PROMPTS for size in SIZES]
    panel_height = (len(rows) + 0.6) * ROW_HEIGHT
    height = 0.3 + len(DEVICES) * (panel_height + 0.25) + 0.6
    fig = fig2.new_figure(height, region)
    axes = grid_axes(fig, len(DEVICES), len(DATASETS), panel_height, top=0.3)
    y = -np.arange(len(rows)) - np.where(np.arange(len(rows)) >= len(SIZES), 0.4, 0)
    for i, device in enumerate(DEVICES):
        for j, dataset in enumerate(DATASETS):
            ax = axes[i, j]
            obj = objects[(objects["device_tag"] == device) & (objects["dataset"] == dataset)]
            enc = samples[(samples["device_tag"] == device) & (samples["dataset"] == dataset)
                          & samples["part"].isin(["encoder", "other"])]
            if obj.empty:
                draw_pending(ax)
            else:
                # 2d: per object; 3d: per object and slice, in ms.
                stats = obj.groupby(["method", "model"])["seconds_per_slice"].agg(["mean", "std"]) * 1000
                encoder = enc.groupby(["method", "model", "sample"])["seconds"].sum().groupby(["method", "model"])
                encoder = encoder.mean()
                end = 0.0
                for yi, (label, method, model) in zip(y, rows):
                    if (method, model) not in stats.index:
                        continue
                    mean, std = stats.loc[(method, model)]
                    color = tinted(METHOD_COLORS["micro-sam2"], INTERACTIVE_TINT)
                    bar = ax.barh(yi, mean, height=BAR_HEIGHT, color=color, linewidth=0, zorder=2)[0]
                    bar.set_hatch("///")
                    bar.set_edgecolor("white")
                    ax.errorbar(mean, yi, xerr=std, color=fig2.GROUP_COLOR, linewidth=0.6, capsize=1.2, zorder=3)
                    note = f" {mean:.1f}"
                    if (method, model) in encoder.index:
                        note += f"  (enc. {encoder[(method, model)]:.2f} s)"
                    ax.text(mean + std, yi, note, va="center", ha="left", fontsize=fig2.SMALL_FONT_SIZE,
                            color=TEXT_COLOR)
                    end = max(end, mean + std)
                ax.set_xlim(0, end * 1.9)
                style_axes(ax)
            ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
            ax.set_yticks(y)
            ax.set_yticklabels([row[0] for row in rows] if j == 0 else [], fontsize=fig2.SMALL_FONT_SIZE)
    label_grid(fig, axes)
    bottom_box = axes[-1, 0].get_position()
    fig.text(0.5, bottom_box.y0 - 0.25 / height,
             "Time per object (2d) or per object and slice (3d) after one prompt [ms] ↓", ha="center", va="top",
             fontsize=fig2.MAX_FONT_SIZE, fontweight="bold", color=TEXT_COLOR)
    fig.text(0.5, bottom_box.y0 - 0.4 / height,
             "In brackets: the time of the encoder, once per image (2d) or volume (3d)", ha="center", va="top",
             fontsize=fig2.SMALL_FONT_SIZE, color=TEXT_COLOR)
    return fig


PLOTS = {
    "automatic_runtimes": ("automatic", "quantitative"),
    "interactive_runtimes": ("interactive", "quantitative"),
}


def make_plot(name, region=None, compact=False):
    return (plot_automatic if name == "automatic_runtimes" else plot_interactive)(region=region, compact=compact)


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
            fig2.save_figure(make_plot(name), name, kind, style)
    if COMBINED_NAME in args.only:
        fig2.save_combined(*fig2.plot_figure(PLOTS, make_plot), name=COMBINED_NAME)


if __name__ == "__main__":
    main()
