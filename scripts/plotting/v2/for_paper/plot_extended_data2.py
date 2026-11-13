"""Plots for Extended Data Figure 2 of the micro-sam2 paper: the runtimes of micro-sam2 on CPU, GPU and Apple silicon.

The numbers come from 'finetuning/v2/evaluation/runtime_benchmarking_script.py' (run 'collect' there first): one
timed run per sample of LIVECell (2d LM) and GoNuclear (3d LM), for the four sizes of micro-sam2 (T, S, B, L) and the
reference CellposeSAM. AIS and APG share their first stage: the embeddings of the image or of every slice (untiled),
then the decoder on them. The script writes these plots to
'figures/extendeddata2/<automatic|interactive>/quantitative':
- 'automatic_runtimes' (panel a): the mean time per image (2d) or per volume (3d) of AIS and APG, split into the
  encoder, the decoder and the postprocessing (for APG also the prompting of SAM2 with the decoder seeds), and of the
  references, split into their network and their postprocessing. The error bars are the standard deviation of the
  total over the samples.
- 'interactive_runtimes' (panel b): the mean time of one object from its first prompt (a point or a box), per
  object in 2d and per object and slice in 3d, with the standard deviation over the objects, and the time of the
  encoder per image or volume (in 3d with the set-up of the video predictor), as two bars per row on a log axis.
  The CPU row of panel a has a log axis too.
- 'extended_data_2' (in 'figures/final'): both plots in one figure.

In 3d, all methods on all devices are timed on the same three volumes per dataset (see 'VOLUMES_3D' in
'runtime_benchmarking_script.py'). A device without results (e.g. the Mac before it is run) shows 'Results pending'.

Usage:

    python plot_extended_data2.py
"""

import glob
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.transforms as transforms
from matplotlib.text import Annotation

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
# The tables of the devices timed off the cluster (the Mac) are in one folder per device tag, next to the main
# tables. They are separate, because 'runtime_benchmarking_script.py collect' rewrites the main tables.
DEVICE_CSV_ROOT = BENCHMARK_ROOT

DEVICES = {"cpu": "CPU (16 cores)", "a100": "GPU (A100)", "m2": "Apple M2"}
# CREMI (3d EM, reference SegNeuron) is timed too, but not shown: APG L takes more than 12 h per volume on CPU.
DATASETS = {"livecell": "LIVECell (2d, LM)", "gonuclear": "GoNuclear (3d, LM)"}
REFERENCES = {"cellposesam": "CellposeSAM"}
# The text over a mode that was not run in a panel, because it takes too long there.
NOT_RUN_TEXT = "not run (hours per volume)"
# The short names of the references in the figure.
REFERENCE_NAMES = {"cellposesam": "CPSAM"}
SIZES = ["hvit_t", "hvit_s", "hvit_b", "hvit_l"]
SIZE_NAMES = {"hvit_t": "T", "hvit_s": "S", "hvit_b": "B", "hvit_l": "L"}
MODES = {"ais": "AIS", "apg": "APG"}
PROMPTS = {"interactive_point": "Point", "interactive_box": "Box"}
# The number of slices of the 3d samples (all evaluation crops are 32 x 512 x 512), for the time per slice.
N_SLICES = {"gonuclear": 32, "cremi": 32}
# The devices whose automatic runtimes span orders of magnitude get a log time axis.
LOG_DEVICES = {"cpu", "a100", "m2"}
# The gap between the stacked parts and the smallest width of a part, as fractions of the axis width.
PART_GAP = 0.006
# The space between the end of a bar (or its error bar) and its number, in points.
LABEL_PAD = 2
# The share of the axis width taken by the longest bar; the rest is for its number.
BAR_SHARE_2D, BAR_SHARE_3D = 0.8, 0.7
# The largest share of a block that its number may fill.
TEXT_FILL = 0.8
# The color of the time per slice next to the totals (contrast 5.3 on white).
SLICE_COLOR = "#6b6b6b"
# The panels (device, dataset) whose totals carry the unit.
SECONDS_PANELS = {(device, dataset) for device in ("cpu", "a100", "m2")
                  for dataset in ("livecell", "gonuclear", "cremi")}
# The panels (device, dataset) that write times over MINUTES_FROM seconds in minutes.
MINUTES_PANELS = {("cpu", "gonuclear"), ("cpu", "cremi"), ("m2", "gonuclear")}
MINUTES_FROM = 200
MIN_PART_WIDTH = 0.012
# The share of the micro-sam2 color for the interactive bars: a tint that no part of panel a uses.

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
    # The tables of a device folder replace the rows of this device in the main tables.
    for folder in sorted(glob.glob(os.path.join(DEVICE_CSV_ROOT, "runtimes_*", ""))):
        device_tag = os.path.basename(os.path.dirname(folder))[len("runtimes_"):]
        for name, table in (("samples", "runtimes_per_sample.csv"), ("objects", "runtimes_per_object.csv")):
            path = os.path.join(folder, table)
            if not os.path.exists(path):
                continue
            current = samples if name == "samples" else objects
            current = pd.concat([current[current["device_tag"] != device_tag], pd.read_csv(path)], ignore_index=True)
            if name == "samples":
                samples = current
            else:
                objects = current
    # Only the selected samples: all images in 2d, the volumes of VOLUMES_3D per dataset in 3d (one on the M2).
    if "selected" in samples:
        samples = samples[samples["selected"].astype(bool)]
        objects = objects[objects["selected"].astype(bool)]
    return samples, objects


def automatic_rows(dataset):
    """The bars of one automatic panel, top to bottom, as (label, method, model, method color name)."""
    reference = "segneuron" if dataset == "cremi" else "cellposesam"
    rows = [(REFERENCE_NAMES.get(reference, REFERENCES[reference]), reference, "default", REFERENCES[reference])]
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


def format_minutes(value):
    """Minutes, or hours from 60 minutes on."""
    minutes = value / 60
    if minutes >= 60:
        return f"{minutes / 60:.1f} h"
    return f"{minutes:.1f} min" if minutes < 10 else f"{minutes:.0f} min"


def format_tick(value, _):
    """Log axis ticks in seconds; from 1000 s on as powers of ten, which are shorter."""
    if value >= 1000:
        # Unicode superscripts instead of mathtext, which would embed a second font.
        return "10" + str(int(round(np.log10(value)))).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")) + " s"
    return f"{value:g} s"


def format_seconds(value):
    if value < 0.1:
        return f"{value:.2g}"
    return f"{value:.0f}" if value >= 10 else f"{value:.1f}" if value >= 1 else f"{value:.2f}"


def contrast(first, second):
    """The WCAG contrast ratio of two colors."""
    def luminance(color):
        rgb = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in colors.to_rgb(color)]
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    high, low = sorted([luminance(first), luminance(second)], reverse=True)
    return (high + 0.05) / (low + 0.05)


def text_color_on(color):
    """White text on dark colors, dark text on light colors; black where neither reaches a contrast of 4.5."""
    red, green, blue = colors.to_rgb(color)
    preferred = "white" if 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.5 else TEXT_COLOR
    if contrast(preferred, color) >= 4.5:
        return preferred
    return max(["white", TEXT_COLOR, "black"], key=lambda text: contrast(text, color))


def draw_segments(ax, segments, log, minutes=False, seconds=False):
    """Draw the stacked parts with a small gap between them, so that each part shows as a block.

    The bar ends at the total. A part shorter than MIN_PART_WIDTH of the axis gets that width, taken from the
    parts before it, so it is not to scale.
    """
    x0, x1 = ax.get_xlim()
    forward, back = (np.log10, lambda u: 10.0 ** u) if log else (lambda x: x, lambda u: u)
    span = forward(x1) - forward(x0)
    min_width, gap = MIN_PART_WIDTH * span, PART_GAP * span
    for yi, start, values, parts, color_name, whisker in segments:
        keep = [(part, value) for part, value in zip(parts, values) if value > 0]
        if not keep:
            continue
        # The boundaries of the parts on the (log) axis, from the start to the total.
        bounds = [forward(max(start, x0))]
        total = 0.0
        for _, value in keep:
            total += value
            bounds.append(forward(max(total, start)))
        # Going back from the end, every part gets at least the minimum width.
        for k in range(len(keep) - 1, 0, -1):
            bounds[k] = min(bounds[k], bounds[k + 1] - min_width)
        for k, (part, _) in enumerate(keep):
            low = bounds[k]
            high = bounds[k + 1] - (gap if k < len(keep) - 1 else 0.0)
            if high <= low:
                continue
            color = tinted(METHOD_COLORS[color_name], PART_TINTS[part])
            ax.barh(yi, back(high) - back(low), left=back(low), height=BAR_HEIGHT, linewidth=0, zorder=2, color=color)
            # The time of the part is written in its block, if it fits; in the last block, left of the error bar.
            right = min(high, forward(whisker)) if k == len(keep) - 1 and whisker > 0 else high
            value = keep[k][1]
            if minutes and value > MINUTES_FROM:
                label = format_minutes(value)
            else:
                label = format_seconds(value) + (" s" if seconds else "")
            text = ax.text(back((low + right) / 2), yi, label, ha="center", va="center",
                           fontsize=fig2.SMALL_FONT_SIZE - 1, color=text_color_on(color), zorder=4)
            renderer = ax.figure.canvas.get_renderer()
            block = ax.transData.transform([(back(low), yi), (back(right), yi)])
            if text.get_window_extent(renderer).width > TEXT_FILL * (block[1, 0] - block[0, 0]):
                text.remove()


def write_total(ax, x, y, seconds, device, dataset, per_slice=None):
    """Write a total right of 'x', followed by the time per slice in gray italics (3d)."""
    if (device, dataset) in MINUTES_PANELS and seconds > MINUTES_FROM:
        note = f" {format_minutes(seconds)}"
    else:
        note = " " + (f"{seconds:.2g}" if seconds < 0.1 else f"{seconds:.2f}" if seconds < 100 else f"{seconds:.0f}")
        if (device, dataset) in SECONDS_PANELS:
            note += " s"
    text = ax.text(x, y, note, va="center", ha="left", fontsize=fig2.SMALL_FONT_SIZE, color=TEXT_COLOR,
                   transform=transforms.offset_copy(ax.transData, fig=ax.figure, x=LABEL_PAD, units="points"))
    if per_slice is not None:
        if per_slice > MINUTES_FROM:
            per_slice = f"{per_slice / 60:.1f} min"
        else:
            per_slice = f"{per_slice:.2g} s" if per_slice < 100 else f"{per_slice:.0f} s"
        ax.annotate(f" ({per_slice}/slice)", xy=(1, 0.5), xycoords=text, va="center", ha="left",
                    fontsize=fig2.SMALL_FONT_SIZE, color=SLICE_COLOR, fontstyle="italic")


def log_start(first):
    """The power of ten under the shortest first part; one lower if that part would be a sliver."""
    start = 10.0 ** np.floor(np.log10(first.min()))
    return start / 10 if first.min() < 2 * start else start


def set_log_axis(ax, start, end, share):
    """A log time axis on which the longest bar takes 'share' of the width."""
    ax.set_xscale("log")
    ax.set_xlim(start, start * (end / start) ** (1 / share))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_tick))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())


def title_on_edge(fig, legend, title):
    """Write the title of a legend card on the top left of its frame, over the frame line."""
    fig.add_artist(Annotation(title, xy=(0, 1), xycoords=legend, xytext=(6, 0), textcoords="offset points",
                              ha="left", va="center", fontsize=fig2.MAX_FONT_SIZE, color=TEXT_COLOR,
                              zorder=legend.get_zorder() + 1, bbox=dict(facecolor="white", edgecolor="none", pad=1.5)))


def plot_automatic(region=None, compact=False):
    samples, _ = load()
    n_rows = len(automatic_rows("livecell"))
    panel_height = (n_rows + 0.6) * ROW_HEIGHT
    legend_height = 0.3
    height = 0.3 + len(DEVICES) * (panel_height + 0.25) + 0.3 + legend_height
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
                log = device in LOG_DEVICES
                # On a log axis, the bars start at the power of ten under the fastest first part (encoder or
                # network), so that every part stays visible.
                first = means[means.index.get_level_values("part").isin(["encoder", "network"]) & (means > 0)]
                start = 10.0 ** np.floor(np.log10(first.min())) if log else 0.0
                # A first part just above the power of ten would be a sliver, so start one power lower.
                if log and first.min() < 2 * start:
                    start /= 10
                end, segments = 0.0, []
                for yi, (label, method, model, color_name) in zip(y, rows):
                    if (method, model) not in totals.index:
                        continue
                    parts = ["network", "postprocessing"] if method in REFERENCES else [
                        "encoder", "decoder", "postprocessing"]
                    # The parts are stacked; on a log axis, the first part starts at 'start'.
                    values = [means.get((method, model, part), 0.0) for part in parts]
                    segments.append((yi, start, values, parts, color_name))
                    mean, std = totals.loc[(method, model)]
                    std = 0.0 if np.isnan(std) else std
                    # On a log axis, the lower error bar stops at the start of the axis.
                    lower = min(std, mean - start * 1.01) if log else std
                    segments[-1] += (mean - lower,)
                    ax.errorbar(mean, yi, xerr=[[lower], [std]], color=fig2.GROUP_COLOR, linewidth=0.6,
                                capsize=1.2, zorder=3)
                    write_total(ax, mean + std, yi, mean, device, dataset,
                                mean / N_SLICES[dataset] if dataset in N_SLICES else None)
                    end = max(end, mean + std)
                # The bars take the same share of the axis on a linear and on a log axis; the rest is for the notes.
                share = BAR_SHARE_3D if dataset in N_SLICES else BAR_SHARE_2D
                if log:
                    ax.set_xscale("log")
                    ax.set_xlim(start, start * (end / start) ** (1 / share))
                    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_tick))
                    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
                else:
                    ax.set_xlim(0, end / share)
                draw_segments(ax, segments, log, minutes=(device, dataset) in MINUTES_PANELS,
                              seconds=(device, dataset) in SECONDS_PANELS)
                # A mode without any result in a panel (not run, too slow) gets one band over its rows, as the
                # 'not applicable' bands of Figure 2.
                for mode in MODES:
                    mode_y = [yi for yi, row in zip(y, rows) if row[1] == mode]
                    if any((mode, size) in totals.index for size in SIZES):
                        continue
                    ax.axhspan(min(mode_y) - BAR_HEIGHT / 2, max(mode_y) + BAR_HEIGHT / 2,
                               color=fig2.NOT_APPLICABLE_COLOR, linewidth=0, zorder=0)
                    ax.text(0.5, np.mean(mode_y), NOT_RUN_TEXT, transform=ax.get_yaxis_transform(),
                            ha="center", va="center", fontsize=fig2.MAX_FONT_SIZE, style="italic",
                            color=fig2.NOT_APPLICABLE_TEXT_COLOR)
                style_axes(ax)
            ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
            ax.set_yticks(y)
            ax.set_yticklabels([row[0] for row in rows] if j == 0 else [], fontsize=fig2.SMALL_FONT_SIZE)
    label_grid(fig, axes)
    bottom_box = axes[-1, 0].get_position()
    fig.text(0.5, bottom_box.y0 - 0.25 / height, "Time per image (2d) or volume (3d) [s, log scale] ↓",
             ha="center", va="top",
             fontsize=fig2.MAX_FONT_SIZE, fontweight="bold", color=TEXT_COLOR)
    # One legend card per method, side by side: micro-sam2 left of the center, the references right of it.
    cards = [("μSAM2", "micro-sam2", ["encoder", "decoder", "postprocessing"])]
    cards += [(REFERENCE_NAMES.get(method, name), name, ["network", "postprocessing"])
              for method, name in REFERENCES.items()]
    for k, (title, color_name, parts) in enumerate(cards):
        handles = [plt.Rectangle((0, 0), 1, 1, color=tinted(METHOD_COLORS[color_name], PART_TINTS[part]))
                   for part in parts]
        legend = fig.legend(handles, [PART_NAMES[part] for part in parts], ncol=len(parts),
                            loc="lower right" if k == 0 else "lower left",
                            bbox_to_anchor=(0.49 if k == 0 else 0.51, 0.0),
                            fontsize=fig2.MAX_FONT_SIZE, frameon=True,
                            edgecolor=fig2.LEGEND_BORDER_COLOR, handlelength=1.1, columnspacing=0.8, borderpad=0.9)
        legend.get_frame().set_linewidth(0.6)
        title_on_edge(fig, legend, title)
    return fig


def plot_interactive(region=None, compact=False):
    samples, objects = load()
    rows = [(f"{PROMPTS[method]} {SIZE_NAMES[size]}", method, size) for method in PROMPTS for size in SIZES]
    panel_height = (len(rows) + 0.6) * ROW_HEIGHT
    legend_height = 0.3
    height = 0.3 + len(DEVICES) * (panel_height + 0.25) + 0.3 + legend_height
    fig = fig2.new_figure(height, region)
    axes = grid_axes(fig, len(DEVICES), len(DATASETS), panel_height, top=0.3)
    # Each row is one stacked bar on a log time axis, as in panel a: the image encoder, once per image (2d) or
    # volume (3d), then the prompt encoder with the mask decoder for one object after its first prompt, in 3d through
    # all slices. The gray italic number is the decoder time per slice.
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
                encoder = enc.groupby(["method", "model", "sample"])["seconds"].sum() \
                    .groupby(["method", "model"]).agg(["mean", "std"])
                decoder = obj.groupby(["method", "model"])["seconds"].agg(["mean", "std"])
                per_slice = obj.groupby(["method", "model"])["seconds_per_slice"].mean()
                start = log_start(encoder["mean"][encoder["mean"] > 0])
                segments, end = [], 0.0
                for yi, (label, method, model) in zip(y, rows):
                    if (method, model) not in encoder.index or (method, model) not in decoder.index:
                        continue
                    enc_mean, enc_std = encoder.loc[(method, model)]
                    dec_mean, dec_std = decoder.loc[(method, model)]
                    mean = enc_mean + dec_mean
                    std = float(np.hypot(np.nan_to_num(enc_std), np.nan_to_num(dec_std)))
                    lower = min(std, mean - start * 1.01)
                    ax.errorbar(mean, yi, xerr=[[lower], [std]], color=fig2.GROUP_COLOR, linewidth=0.6,
                                capsize=1.2, zorder=3)
                    segments.append((yi, start, [enc_mean, dec_mean], ["encoder", "decoder"], "micro-sam2",
                                     mean - lower))
                    write_total(ax, mean + std, yi, mean, device, dataset,
                                per_slice.loc[(method, model)] if dataset in N_SLICES else None)
                    end = max(end, mean + std)
                set_log_axis(ax, start, end, BAR_SHARE_3D if dataset in N_SLICES else BAR_SHARE_2D)
                draw_segments(ax, segments, True, minutes=(device, dataset) in MINUTES_PANELS,
                              seconds=(device, dataset) in SECONDS_PANELS)
                style_axes(ax)
            ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
            ax.set_yticks(y)
            ax.set_yticklabels([row[0] for row in rows] if j == 0 else [], fontsize=fig2.SMALL_FONT_SIZE)
    label_grid(fig, axes)
    bottom_box = axes[-1, 0].get_position()
    fig.text(0.5, bottom_box.y0 - 0.25 / height, "Time per object [s, log scale] ↓", ha="center", va="top",
             fontsize=fig2.MAX_FONT_SIZE, fontweight="bold", color=TEXT_COLOR)
    handles = [plt.Rectangle((0, 0), 1, 1, color=tinted(METHOD_COLORS["micro-sam2"], PART_TINTS[part]))
               for part in ("encoder", "decoder")]
    legend = fig.legend(handles, ["Encoder", "Prompt encoder & Mask decoder"],
                        loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=fig2.MAX_FONT_SIZE,
                        frameon=True, edgecolor=fig2.LEGEND_BORDER_COLOR, handlelength=1.1,
                        columnspacing=0.8, borderpad=0.9)
    legend.get_frame().set_linewidth(0.6)
    title_on_edge(fig, legend, "μSAM2")
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
