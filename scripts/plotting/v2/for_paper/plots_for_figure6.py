"""Plots for Figure 6 of the micro-sam2 paper.

The figure compares 3d automatic and interactive segmentation of EM mitochondria and nuclei with micro-sam2
(the 'hvit_s_em_mito' and 'hvit_s_em_nuclei' models) and the baselines. It uses the layout and the style of Figure 3
(see plots_for_figure3.py). The script writes these plots to 'figures/figure6/<automatic|interactive>/quantitative':
- 'automatic_3d': the score of each method per organelle, for the in-domain and the out-of-domain datasets.
- 'automatic_3d_qualitative': per organelle, a large region of one test volume in 3d with its planes, and the 3d
  objects of the reference, of the specialist model and of micro-sam2 (in 'automatic/qualitative').
- 'interactive_3d': the score over the correction iterations, with a point or a box as the first prompt.
- 'interactive_3d_qualitative': per organelle and domain, two objects (one nucleus out of domain) in 3d, segmented by
  nnInteractive and micro-sam2 from the first point and the first box, in the layout of panel d of Figure 3.

The combined figure 'figure_6' puts the four plots together as panels a-d, as Figure 3, in 'figures/final'.

Usage:

    python plots_for_figure6.py
    python plots_for_figure6.py --only automatic_3d
    python plots_for_figure6.py --only figure_6

The inputs are the EM organelle tables in 'figures', see 'collect_em_organelle_results.py' and
'collect_em_interactive_results.py' in the control folder of the baselines_2d experiment, and the regions,
predictions and renders of 'predict_region.py' and 'render_regions.py' in QUALITATIVE.
"""

import os
import argparse

import h5py
import numpy as np
import pandas as pd
from PIL import Image

import plots_for_figure2 as fig2
import plots_for_figure3 as fig3
from plot_style import set_style


ROOT = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(ROOT, "figures")
OUTPUT_DIR = os.path.join(FIGURES, "figure6")
QUALITATIVE = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/qualitative_em"

# The (results table, interactive table, structure, title) of each organelle. The automatic tables have no NucMM
# (user, 2026-10-30), so the interactive plot leaves it out too: both plots show the same datasets.
PANELS = {
    "em_mito": (
        "em_mito_automatic_segmentation_3d_results.csv", "em_mito_interactive_segmentation_3d_results.csv",
        "mitochondrion", "Electron Microscopy (Mitochondrion)",
    ),
    "em_nuclei": (
        "em_nuc_automatic_segmentation_3d_results.csv", "em_nuc_interactive_segmentation_3d_results.csv",
        "nucleus", "Electron Microscopy (Nucleus)",
    ),
}
SCORE_LABEL = "Mean Segmentation Accuracy ↑"

# The 'method', 'model' and 'mode' values of each automatic method in the 3d results. micro-sam shows AIS, its
# default mode (see fig2.FIGURE_MODES), and micro-sam2 AIS and APG.
METHODS = {
    "MitoNet": ("mitonet", ("MitoNet_v1",), None),
    "NucleoNet": ("nucleonet", ("NucleoNet_base_v2",), None),
    "micro-sam": ("micro-sam", ("vit_l_em_organelles",), "apg"),
    "micro-sam2": ("micro-sam2", ("hvit_s",), "apg"),
}
PANEL_METHODS = {
    "em_mito": ["MitoNet", "micro-sam", "micro-sam2"],
    "em_nuclei": ["NucleoNet", "micro-sam", "micro-sam2"],
}
# micro-sam2 runs a different model per organelle ('hvit_s_em_mito', 'hvit_s_em_nuclei'), so it gets a star, in all
# plots (user, 2026-11-01). The interactive plot also stars micro-sam, with its EM model (user, 2026-11-01).
STARRED_METHODS = {"micro-sam2"}
INTERACTIVE_STARRED_METHODS = {"micro-sam2", "micro-sam"}

# The largest model of each method, as in Figure 3. SAM and micro-sam correct with clicks only, as the others do.
INTERACTIVE_METHODS = {
    "SAM": ("sam", ("vit_l",), "points"),
    "SAM2": ("sam2", ("hvit_l",), ""),
    "SAM3": ("sam3", ("default",), ""),
    "nnInteractive": ("nninteractive", ("default",), ""),
    "micro-sam": ("micro-sam", ("vit_l_em_organelles",), "points"),
    "micro-sam2": ("micro-sam2", ("hvit_s",), ""),
}

AUTOMATIC_ROWS = [["em_mito", "em_nuclei"]]


def panel_datasets(name):
    """Return the in-domain and out-of-domain datasets of an organelle, in the order of its automatic table."""
    table = pd.read_csv(os.path.join(FIGURES, PANELS[name][0])).drop_duplicates("dataset")
    return {domain: list(table[table["domain"] == domain]["dataset"]) for domain in fig2.DOMAIN_TITLES}


def plot_automatic(region=None, compact=False):
    """Plot the automatic results of both organelles, with micro-sam2 in AIS and APG.

    With 'compact', the plot uses the sizes of the combined figure, as in plots_for_figure3.py.
    """
    results = {}
    for name, (table, _, structure, _) in PANELS.items():
        rows = fig2.load_modes("EM", structure, path=os.path.join(FIGURES, table), methods=METHODS)
        results[name] = fig2.one_mode(fig2.select_datasets(rows, panel_datasets(name)), fig2.FIGURE_MODES)
    panels = {name: (title, SCORE_LABEL, PANEL_METHODS[name]) for name, (_, _, _, title) in PANELS.items()}
    return fig2.plot_automatic(
        results, AUTOMATIC_ROWS, panels, STARRED_METHODS, domain_offset=0.27, ranks=True, region=region,
        row_height=fig2.FIGURE_ROW_HEIGHT if compact else fig2.BAR_ROW_HEIGHT,
        grid=fig3.COMBINED_GRID if compact else None, title_line=compact,
        group_gap=fig2.COMBINED_GROUP_GAP if compact else fig2.GROUP_GAP,
        row_gap=fig2.COMBINED_ROW_GAP if compact else None,
    )


def plot_interactive(region=None, compact=False):
    """Plot the interactive results of both organelles over the correction iterations.

    The legend stands in the point half of the out-of-domain mitochondria panel, whose curves stay low. The value
    axes end at 1, the highest mSA: the shared row would otherwise be raised over the nucleus curves for the legend.
    """
    results = {
        name: fig2.load_interactive(
            "EM", structure, panel_datasets(name), path=os.path.join(FIGURES, table), methods=INTERACTIVE_METHODS,
        ).dropna(subset=["score"])
        for name, (_, table, structure, _) in PANELS.items()
    }
    titles = {name: title for name, (_, _, _, title) in PANELS.items()}
    fig = fig2.plot_interactive_by_domain(
        results, list(PANELS), titles, INTERACTIVE_STARRED_METHODS, methods=INTERACTIVE_METHODS,
        legend_panel=("OOD", "em_mito"), legend_left=True, shared_rows=True, prompts_on_top=compact, region=region,
        panel_height=fig2.COMBINED_PANEL_HEIGHT if compact else fig2.PANEL_HEIGHT,
        grid=fig3.COMBINED_GRID if compact else None, title_line=compact, show_titles=not compact,
        score_gap=0.32 if compact else None, row_gap=fig2.COMBINED_ROW_GAP if compact else None,
        band_color=fig2.COMBINED_BAND_COLOR if compact else fig2.BOX_BAND_COLOR,
    )
    for ax in fig.axes:
        low, high = ax.get_ylim()
        if high > 1:
            ticks = ax.get_yticks()
            ax.set_ylim(low, 1)
            ax.set_yticks(ticks[ticks <= 1 + 1e-6])
    return fig


# The region of each organelle in the qualitative plot: (dataset, domain, the micro-sam2 prediction, the baseline).
# micro-sam2 tiles the mitochondria into 384x384 tiles with a halo of 64, which gives them the scale of the test
# crops, and does not tile the nuclei, which are large in a slice already (user, 2026-11-01). micro-sam is left out.
AUTOMATIC_QUALITATIVE = {
    "em_mito": ("cmito_l2", "OOD", "microsam2_apg_t384h64", "MitoNet"),
    "em_nuclei": ("cellmap_nucleus", "ID", "microsam2_apg", "NucleoNet"),
}
QUALITATIVE_NAMES = {"cmito_l2": "CMito L2", "cellmap_nucleus": "CellMap Nucleus"}
# The (z, y, x) steps of the downsampling of the regions and their z/xy voxel ratio, as in 'render_regions.py'.
DOWNSAMPLE = {"cmito_l2": ((1, 1, 1), 30 / 8), "cellmap_nucleus": ((2, 2, 2), 1.0)}
# The regions whose panels are cut around the objects of all methods, not only around the reference, so that every
# fragment of NucleoNet shows (user, 2026-11-01). All panels of a region keep one scale.
FIT_ALL = {"cellmap_nucleus"}
# The room around the objects in these panels, as a share of the render size, so that no fragment touches the frame.
FIT_ALL_PAD = 0.03


def load_region_render(dataset, name):
    """Return a render of 'render_regions.py' as RGBA in [0, 1]."""
    path = os.path.join(QUALITATIVE, "renders", f"{dataset}_{name}.png")
    return np.asarray(Image.open(path).convert("RGBA")) / 255


def region_planes(dataset):
    """Return the image and the reference labels of a region, downsampled as for its renders, and their z/xy ratio.

    The image is the mean of each block of voxels, so that it does not alias; the labels keep every step-th voxel.
    """
    steps, z_ratio = DOWNSAMPLE[dataset]
    window = tuple(np.s_[::step] for step in steps)
    with h5py.File(os.path.join(QUALITATIVE, "predictions", dataset, "region.h5"), "r") as f:
        labels = f["labels"][window]
        raw = f["raw"]
        blocks = []
        # A block of slices at a time, so that the whole region is never in memory as float.
        for z in range(0, labels.shape[0] * steps[0], steps[0]):
            block = raw[z:z + steps[0]].astype("float32")
            ny, nx = block.shape[1] // steps[1], block.shape[2] // steps[2]
            block = block[:, :ny * steps[1], :nx * steps[2]].reshape(-1, ny, steps[1], nx, steps[2])
            blocks.append(block.mean(axis=(0, 2, 4)))
    return np.stack(blocks)[:, :labels.shape[1], :labels.shape[2]], labels, z_ratio * steps[0] / steps[1]


def region_score(dataset, prediction):
    """Return the mSA of a prediction on its whole region, as 'predict_region.py' stores it."""
    with h5py.File(os.path.join(QUALITATIVE, "predictions", dataset, f"{prediction}.h5"), "r") as f:
        return float(f.attrs["msa"])


def label_panel(ax, name, bold=False):
    """Write a name in the upper left corner of a panel."""
    ax.text(
        0.04, 0.96, name, transform=ax.transAxes, ha="left", va="top", fontsize=5, color=fig3.TEXT_COLOR,
        fontweight="bold" if bold else "normal",
        bbox={"boxstyle": "square,pad=0.1", "facecolor": "white", "alpha": 0.8, "linewidth": 0},
    )


def plot_automatic_qualitative(region=None):
    """Plot a column per organelle with the panels of its region in two rows, in the panel size of Figure 3.

    The upper row has the region in 3d with its xy, xz and yz planes, and the reference objects; the lower row the
    objects of the baseline and of micro-sam2, at the scale of the reference.
    """
    starts, column_width = fig2.column_grid(fig3.COMBINED_IMAGE_GRID, len(AUTOMATIC_QUALITATIVE))
    gap = fig3.QUALITATIVE_GAP
    width = (column_width - gap) / 2
    size = fig3.QUALITATIVE_ASPECT * width
    height = fig3.QUALITATIVE_HEADER + 2 * size + gap
    fig = fig2.new_figure(height, region)
    upper, lower = size + gap, 0.0
    for start, (dataset, domain, prediction, baseline) in zip(starts, AUTOMATIC_QUALITATIVE.values()):
        volume = load_region_render(dataset, "volume")
        volume_width = fig3.PLANE_SPLIT * width
        fig3.render_axes(fig, start, upper, volume_width, size, fig3.trim([volume], fig3.TRACING_PAD)[0], framed=False)
        fig3.plot_planes(
            fig, start + volume_width + gap, upper, width - volume_width - gap, size, dataset, None,
            volume=region_planes(dataset),
        )
        in_domain = domain == "ID"
        fig.text(
            (start + column_width / 2) / fig2.FIGURE_WIDTH, (upper + size + 0.04) / height, QUALITATIVE_NAMES[dataset],
            ha="center", va="baseline", fontsize=6, color=fig3.TEXT_COLOR, style="italic" if in_domain else "normal",
            fontweight="normal" if in_domain else "bold",
        )

        reference = load_region_render(dataset, "reference")
        renders = [load_region_render(dataset, name) for name in ("empanada", prediction)]
        if dataset in FIT_ALL:
            reference, *images = fig3.trim([reference] + renders, FIT_ALL_PAD)
        else:
            reference = fig3.trim([reference], fig3.QUALITATIVE_PAD)[0]
            images = fig3.centered_windows(renders, reference.shape[:2])
        label_panel(fig3.render_axes(fig, start + width + gap, upper, width, size, reference), fig2.REFERENCE_NAME)
        for k, (method, image) in enumerate(zip([baseline, "micro-sam2"], images)):
            ours = method == "micro-sam2"
            ax = fig3.render_axes(fig, start + k * (width + gap), lower, width, size, image)
            label_panel(ax, fig2.names_of([method], STARRED_METHODS)[0], bold=ours)
            score = region_score(dataset, prediction if ours else "empanada")
            fig2.draw_score(ax, score, ours, fontsize=5)
    return fig


# The (dataset, sample, ids of the objects, domain) of each volume of the interactive qualitative plot, from the
# candidates of 'interactive_pairs.py' (user, 2026-11-01). The out-of-domain nucleus crops hold one nucleus each, so
# that block shows one object.
INTERACTIVE_QUALITATIVE = {
    "em_mito": [("mitoemv2_stem", 2, (94, 105), "ID"), ("mito_anf", 3, (5, 6), "OOD")],
    "em_nuclei": [("fafb_nuclei", 19, (10, 15), "ID"), ("etchacell_nucleus", 41, (1,), "OOD")],
}
INTERACTIVE_PAIRS = os.path.join(QUALITATIVE, "interactive")
TRACING_METHODS = ["nnInteractive", "micro-sam2"]
# The volumes whose renders are cut around the objects of all methods, so that the objects of nnInteractive, which
# reach far beyond the reference, show whole (user, 2026-11-01).
TRACING_FIT_ALL = {"mitoemv2_stem", "mito_anf"}
TRACING_NAMES = {
    "mitoemv2_stem": "MitoEM v2 Stem", "mito_anf": "Mito ANF", "fafb_nuclei": "FAFB Nuclei",
    "etchacell_nucleus": "EtchACell Nucleus",
}


def pair_score(dataset, sample, objects, method, prompt):
    """Return the mean IoU of the objects after the first prompt, as 'interactive_pairs.py' stores it."""
    table = pd.read_csv(os.path.join(INTERACTIVE_PAIRS, "pair_candidates.csv"))
    row = table[(table["dataset"] == dataset) & (table["sample"] == sample)].iloc[0]
    return float(np.mean([row[f"iou{n}_{method}_{prompt}"] for n in range(1, len(objects) + 1)]))


def plot_interactive_qualitative(region=None, compact=False):
    """Plot the interactive qualitative plot with the code of Figure 3 and the volumes of INTERACTIVE_QUALITATIVE.

    The Figure 3 functions read their volumes, renders and scores from settings of their module, so these settings
    point to the ones of this figure while the plot is drawn, and are restored after.
    """
    settings = {
        "INTERACTIVE_QUALITATIVE": INTERACTIVE_QUALITATIVE, "RENDERS": INTERACTIVE_PAIRS,
        "TRACING_METHODS": {name: TRACING_METHODS for name in INTERACTIVE_QUALITATIVE},
        "PANELS": {name: (None, None, title) for name, (_, _, _, title) in PANELS.items()},
        "DATASET_NAMES": TRACING_NAMES, "OBJECT_GAPS": {}, "FIT_ALL": TRACING_FIT_ALL, "tracing_score": pair_score,
    }
    saved = {name: getattr(fig3, name) for name in settings}
    try:
        for name, value in settings.items():
            setattr(fig3, name, value)
        return fig3.plot_interactive_qualitative(
            region, grid=fig3.COMBINED_TRACING_GRID if compact else None, show_titles=not compact,
            starred=STARRED_METHODS,
        )
    finally:
        for name, value in saved.items():
            setattr(fig3, name, value)


# The plots in their order in the combined figure, with their kind and style.
PLOTS = {
    "automatic_3d": ("automatic", "quantitative", plot_automatic),
    "automatic_3d_qualitative": ("automatic", "qualitative", plot_automatic_qualitative),
    "interactive_3d": ("interactive", "quantitative", plot_interactive),
    "interactive_3d_qualitative": ("interactive", "qualitative", plot_interactive_qualitative),
}
COMBINED_NAME = "figure_6"


def make_plot(name, region=None, compact=False):
    """Make one plot, as its own figure or, with 'compact', in a region of the combined figure."""
    plot = PLOTS[name][2]
    return plot(region=region) if plot is plot_automatic_qualitative else plot(region=region, compact=compact)


def main():
    names = list(PLOTS) + [COMBINED_NAME]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=names, default=names, help="Make only these plots.")
    args = parser.parse_args()
    set_style()
    fig2.check_style()
    fig2.OUTPUT_DIR = OUTPUT_DIR
    for name, (kind, style, _) in PLOTS.items():
        if name in args.only:
            fig2.save_figure(make_plot(name), name, kind, style)
    if COMBINED_NAME in args.only:
        fig2.save_combined(*fig2.plot_figure(PLOTS, make_plot), name=COMBINED_NAME, dpi=fig3.COMBINED_DPI)


if __name__ == "__main__":
    main()
