"""Plots for Figure 6 of the micro-sam2 paper.

The figure compares 3d automatic and interactive segmentation of EM mitochondria and nuclei with micro-sam2
(the 'hvit_s_em_mito' and 'hvit_s_em_nuclei' models) and the baselines. It uses the layout and the style of Figure 3
(see plots_for_figure3.py). The script writes these plots to 'figures/figure6/<automatic|interactive>/quantitative':
- 'automatic_3d': the score of each method per organelle, for the in-domain and the out-of-domain datasets.
- 'automatic_3d_qualitative': per organelle, a large region of one test volume in 3d with its planes, and the 3d
  objects of the reference, of the specialist model and of micro-sam2 (in 'automatic/qualitative').
- 'interactive_3d': the score over the correction iterations, with a point or a box as the first prompt.
- 'interactive_3d_qualitative': two mitochondria volumes (in and out of domain) and one nucleus volume (in domain),
  two objects each in 3d, segmented by nnInteractive and micro-sam2 from the first point and the first box, in the
  layout of panel d of Figure 3. The room of the fourth volume holds panel e: the nuclei that micro-sam2 segments
  automatically in a FIB-SEM spheroid of the Beghin lab, in 3d and on one slice of the raw.

The combined figure 'figure_6' puts the four plots together as panels a-e, as Figure 3, in 'figures/final'.

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

# The (results table, interactive table, structure, title) of each organelle. Both plots show the datasets of
# PANEL_DATASETS.
PANELS = {
    "em_mito": (
        "em_mito_automatic_segmentation_3d_results.csv", "em_mito_interactive_segmentation_3d_results.csv",
        "mitochondrion", "Electron microscopy (mitochondrion)",
    ),
    "em_nuclei": (
        "em_nuc_automatic_segmentation_3d_results.csv", "em_nuc_interactive_segmentation_3d_results.csv",
        "nucleus", "Electron microscopy (nucleus)",
    ),
}
SCORE_LABEL = "Mean segmentation accuracy ↑"

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
# plots (user, 2026-11-01). micro-sam has no star in the interactive plot (user, 2026-11-05). MitoNet and NucleoNet get
# a cross: their inference scale is tuned per dataset on its tuning crops (Constantin Pape, 2026-11-06).
STARRED_METHODS = {"micro-sam2": "*", "MitoNet": "\u2020", "NucleoNet": "\u2020"}
INTERACTIVE_STARRED_METHODS = {"micro-sam2"}

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


# The six in-domain and six out-of-domain datasets of each organelle (user, 2026-11-05 and 2026-11-06). They set the
# domain of each dataset: parlakgul_liver_eval is in domain, as it comes from the study of the trained Parlakgul
# fasting-liver volume. kasthuri is left out: it comes from the Kasthuri et al. 2015 data of the trained SNEMI Mito.
# mito_anf replaces it out of domain; only its mitochondria of traced nerve fibers are labelled (user, 2026-11-06).
# The nucleus sets of 2026-11-05 are in the tables by 'collect_em_organelle_results.py'.
# etchacell_nucleus and placenta_nucleus score only the predictions on labelled nuclei (common.py).
PANEL_DATASETS = {
    "em_mito": {
        "ID": ["mitoemv2", "mitoem", "lucchi", "asem_mito", "snemi_mito", "parlakgul_liver_eval"],
        "OOD": ["mitolab", "mito_anf", "human_liver_em", "cerebellum_fibsem_mito", "macrophage_a431_mito", "cmito"],
    },
    "em_nuclei": {
        "ID": [
            "cellmap_nucleus", "fafb_nuclei", "microns_nuclei", "platynereis_nuclei", "dauer_larva_nucleus",
            "human_organoids_nucleus",
        ],
        "OOD": [
            "human_liver_em_nucleus", "etchacell_nucleus", "aive_nucleus", "malecns_nuclei", "placenta_nucleus",
            "zheng_ca3_nuclei",
        ],
    },
}
# The sub-volumes of one dataset are pooled into one dataset, with the score of each weighted by its number of scored
# crops (user, 2026-11-05). Separate datasets are never pooled. mitolab_tem is 2d, so it is not in MitoLab.
POOLED = {
    "mitoemv2": [
        "mitoemv2_beta", "mitoemv2_jurkat", "mitoemv2_macro", "mitoemv2_podo", "mitoemv2_sperm", "mitoemv2_stem",
        "mitoemv2_mossy",
    ],
    "mitoem": ["mitoem_human", "mitoem_rat"],
    "mitolab": [
        "mitolab_c_elegans", "mitolab_fly_brain", "mitolab_glycolytic_muscle", "mitolab_hela_cell",
        "mitolab_salivary_gland",
    ],
    "cmito": ["cmito_l1", "cmito_l2", "cmito_l3", "cmito_adult", "cmito_dauer"],
}
# The interactive evaluation scores at most this many crops of a 3d dataset (INTERACTIVE_MAX_CROPS_3D in common.py).
INTERACTIVE_MAX_CROPS = 8


def panel_datasets(name):
    """Return the in-domain and out-of-domain datasets of an organelle."""
    return PANEL_DATASETS[name]


def crop_counts(name, interactive=False):
    """Return the number of scored crops of each dataset of an organelle, from its automatic table."""
    table = pd.read_csv(os.path.join(FIGURES, PANELS[name][0])).drop_duplicates("dataset")
    counts = dict(zip(table["dataset"], table["n_samples"]))
    return {d: min(n, INTERACTIVE_MAX_CROPS) for d, n in counts.items()} if interactive else counts


def pool(results, counts):
    """Pool the sub-volumes of POOLED into one dataset, with the crop-weighted mean of their scores.

    A pooled score is left out unless every sub-volume has a score for that method and setting.
    """
    members = {member: group for group, names in POOLED.items() for member in names}
    pooled = results[results["dataset"].isin(members)].copy()
    keys = [column for column in pooled.columns if column not in ("dataset", "domain", "score")]
    pooled["group"] = pooled["dataset"].map(members)
    pooled["weight"] = pooled["dataset"].map(counts)
    rows = []
    for values, part in pooled.groupby(["group"] + keys, dropna=False):
        part = part.dropna(subset=["score"])
        if set(part["dataset"]) != set(POOLED[values[0]]):
            continue
        rows.append({
            "dataset": values[0], **dict(zip(keys, values[1:])),
            "score": float(np.average(part["score"], weights=part["weight"])),
        })
    return pd.concat([results[~results["dataset"].isin(members)], pd.DataFrame(rows)], ignore_index=True)


def plot_automatic(region=None, compact=False):
    """Plot the automatic results of both organelles, with micro-sam2 in AIS and APG.

    With 'compact', the plot uses the sizes of the combined figure, as in plots_for_figure3.py.
    """
    results = {}
    for name, (table, _, structure, _) in PANELS.items():
        rows = pool(
            fig2.load_modes("EM", structure, path=os.path.join(FIGURES, table), methods=METHODS), crop_counts(name)
        )
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

    The legend spans the out-of-domain mitochondria panel in two rows (user, 2026-11-10); its curves stay low. The value
    axes end at 1, the highest mSA: the shared row would otherwise be raised over the nucleus curves for the legend.
    """
    results = {}
    for name, (_, table, structure, _) in PANELS.items():
        datasets = panel_datasets(name)
        members = [m for d in sum(datasets.values(), []) for m in POOLED.get(d, [d])]
        rows = fig2.load_interactive(
            "EM", structure, {"all": members}, path=os.path.join(FIGURES, table), methods=INTERACTIVE_METHODS,
        )
        rows = pool(rows, crop_counts(name, interactive=True))
        results[name] = fig2.select_datasets(rows, datasets).dropna(subset=["score"])
    titles = {name: title for name, (_, _, _, title) in PANELS.items()}
    fig = fig2.plot_interactive_by_domain(
        results, list(PANELS), titles, INTERACTIVE_STARRED_METHODS, methods=INTERACTIVE_METHODS,
        legend_panel=("OOD", "em_mito"), legend_wide=True, shared_rows=True, prompts_on_top=compact, region=region,
        panel_height=fig2.COMBINED_PANEL_HEIGHT if compact else fig2.PANEL_HEIGHT,
        grid=fig3.COMBINED_GRID if compact else None, title_line=compact, show_titles=not compact,
        score_gap=0.32 if compact else None, row_gap=fig2.COMBINED_ROW_GAP if compact else None,
        band_color=fig2.COMBINED_BAND_COLOR if compact else fig2.BOX_BAND_COLOR, x_label=fig2.ITERATION_LABEL,
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


def label_panel(ax, name):
    """Write a name in the upper left corner of a panel."""
    ax.text(
        0.04, 0.96, name, transform=ax.transAxes, ha="left", va="top", fontsize=fig2.MAX_FONT_SIZE,
        color=fig3.TEXT_COLOR,
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
            ha="center", va="baseline", fontsize=fig2.MAX_FONT_SIZE, color=fig3.TEXT_COLOR,
            style="italic" if in_domain else "normal",
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
            label_panel(ax, fig2.names_of([method], STARRED_METHODS)[0])
            score = region_score(dataset, prediction if ours else "empanada")
            fig2.draw_score(ax, score)
    return fig


# The (dataset, sample, ids of the objects, domain) of each volume of the interactive qualitative plot, from the
# candidates of 'interactive_pairs.py' (user, 2026-11-01). The out-of-domain mitochondria are from MitoANF, one of
# the datasets of the panels (user, 2026-11-06). The out-of-domain nucleus crops hold one nucleus each, so
# that block shows one object.
INTERACTIVE_QUALITATIVE = {
    "em_mito": [("mitoemv2_stem", 2, (94, 105), "ID"), ("mito_anf", 3, (5, 6), "OOD")],
    "em_nuclei": [("fafb_nuclei", 19, (10, 15), "ID")],
}
# The out-of-domain nucleus block of the interactive qualitative plot gives its room to panel e (user, 2026-11-10):
# the nuclei that micro-sam2 segments automatically in a FIB-SEM spheroid of the Beghin lab, which has no reference.
# The renders are from 'render_section' in 'explore_beghin_em_data.py': the 3d nuclei with the outline of one slice,
# and the nuclei that cross the slice, cut at it and standing on its raw. 'layout.json' holds the corners of both.
SPHEROID_RENDERS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "user_study", "v2", "beghin_lab", "renders",
    "section",
)
SPHEROID_GAP = 0.08
# The distance of the panel letter left of the images, in inches.
SPHEROID_LETTER_OFFSET = 0.1
INTERACTIVE_PAIRS = os.path.join(QUALITATIVE, "interactive")
TRACING_METHODS = ["nnInteractive", "micro-sam2"]
# The volumes whose renders are cut around the objects of all methods, so that the objects of nnInteractive, which
# reach far beyond the reference, show whole (user, 2026-11-01).
TRACING_FIT_ALL = {"mitoemv2_stem", "mito_anf"}
TRACING_NAMES = {
    "mitoemv2_stem": "MitoEM v2 Stem", "mito_anf": "Mito ANF", "fafb_nuclei": "FAFB Nuclei",
    "etchacell_nucleus": "EtchACell",
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
        fig = fig3.plot_interactive_qualitative(
            region, grid=fig3.COMBINED_TRACING_GRID if compact else None, show_titles=not compact,
            starred=STARRED_METHODS,
        )
    finally:
        for name, value in saved.items():
            setattr(fig3, name, value)
    plot_spheroid(fig, compact)
    return fig


def tracing_layout(compact):
    """Return the left edge of the nucleus column, its width and the height of a block row of the interactive
    qualitative plot, in inches, as 'plots_for_figure3.plot_interactive_qualitative' computes them."""
    gap = fig3.TRACING_GAP
    grid = fig3.COMBINED_TRACING_GRID if compact else (0, gap, fig2.ZOOM_BLOCK_GAP + gap)
    starts, column_width = fig2.column_grid(grid, len(INTERACTIVE_QUALITATIVE))
    n_methods = len(TRACING_METHODS)
    size = (column_width - fig3.TRACING_LABEL_WIDTH - fig3.TRACING_PROMPT_WIDTH - n_methods * gap) / (n_methods + 1)
    return starts[-1], column_width, 2 * fig3.TRACING_ASPECT * size + gap


def plot_spheroid(fig, compact):
    """Plot panel e in the room of the second block of the nucleus column: the 3d nuclei of the spheroid with the
    outline of a slice, and the nuclei of the slice on its raw, joined by lines from the right corners of the outline
    to the top and the bottom corner of the slice, like a pyramid."""
    import json
    from matplotlib.patches import ConnectionPatch, Polygon

    layout = json.load(open(os.path.join(SPHEROID_RENDERS, "layout.json")))
    studio, studio_offset = crop_white(load_rgb(os.path.join(SPHEROID_RENDERS, "section_studio.png")))
    section, section_offset = crop_white(load_rgb(os.path.join(SPHEROID_RENDERS, "section_nuclei_on_raw.png")))
    outline = np.array(layout["studio"]["plane_outline"]) - studio_offset
    corners = np.array(layout["section"]["card_corners"]) - section_offset

    height = fig2.height_of(fig)
    header = fig2.ZOOM_HEADER_NO_TITLES if compact else fig2.ZOOM_HEADER
    left, column_width, row_height = tracing_layout(compact)
    top = height - header - fig2.ZOOM_CAPTION - row_height - fig2.ZOOM_DOMAIN_GAP
    bottom = top - row_height
    x = left + fig3.TRACING_LABEL_WIDTH
    # Both images also take most of the gap above the block (user, 2026-11-10: a bit bigger), unless they would be
    # wider than the column.
    ratios = [image.shape[1] / image.shape[0] for image in (studio, section)]
    room = row_height + fig2.ZOOM_DOMAIN_GAP - 0.02
    image_height = min(room, (left + column_width - x - SPHEROID_GAP) / sum(ratios))
    widths = [image_height * ratio for ratio in ratios]
    y = bottom + (room - image_height) / 2

    def axes_at(x0, width):
        ax = fig.add_axes([x0 / fig2.FIGURE_WIDTH, y / height, width / fig2.FIGURE_WIDTH, image_height / height])
        ax.axis("off")
        return ax

    ax_studio = axes_at(x, widths[0])
    ax_section = axes_at(x + widths[0] + SPHEROID_GAP, widths[1])
    ax_studio.imshow(studio)
    ax_section.imshow(section)
    ax_studio.add_patch(Polygon(outline, closed=True, fill=False, edgecolor=fig3.TEXT_COLOR, lw=0.5, clip_on=False))
    right = outline[np.argsort(outline[:, 0])[-2:]]
    upper, lower = right[np.argmin(right[:, 1])], right[np.argmax(right[:, 1])]
    top_corner, bottom_corner = corners[np.argmin(corners[:, 1])], corners[np.argmax(corners[:, 1])]
    for point_a, point_b in ((upper, top_corner), (lower, bottom_corner)):
        fig.add_artist(ConnectionPatch(
            point_a, point_b, "data", "data", axesA=ax_studio, axesB=ax_section, color=fig3.TEXT_COLOR, lw=0.4,
        ))
    # The letter of panel e stands next to the spheroid, without a dataset name or a caption (user, 2026-11-10).
    if compact:
        fig.text(
            (x - SPHEROID_LETTER_OFFSET) / fig2.FIGURE_WIDTH, (y + image_height) / height, "e", ha="left",
            va="top", fontsize=fig2.PANEL_LABEL_SIZE, fontweight="bold", color=fig3.TEXT_COLOR,
        )


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def crop_white(image, margin=8):
    """Crop an image on white to its content plus a margin, and return the offset (column, row) of the crop."""
    rows, columns = np.nonzero((image < 245).any(-1))
    r0, c0 = max(rows.min() - margin, 0), max(columns.min() - margin, 0)
    return image[r0:rows.max() + margin, c0:columns.max() + margin], np.array([c0, r0])


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
