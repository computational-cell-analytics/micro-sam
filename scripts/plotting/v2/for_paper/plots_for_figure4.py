"""Plots for Figure 4 of the micro-sam2 paper.

The figure compares micro-sam2 with other foundation models for object and pixel classification. A random forest
learns the classes from the features of each model. An end-to-end network (ResNet18 for objects, U-Net for pixels)
is the supervised baseline.

The script writes these plots to 'figures/figure4/classification/<style>':
- 'classification' (style 'quantitative'): the score of each method per dataset and label count, and the mean rank
  of the feature models with the fewest labels and over all label counts.
- 'classification_qualitative' (style 'qualitative'): one test crop per task (BCSS pixels, PanNuke objects) with the
  reference, the micro-sam2 prediction and the strokes of a training tile.
- 'classification_slide' (style 'slide'): pixel classification on BCSS, from the whole slide to the test crop and
  back to the micro-sam2 class map of the whole slide.
- 'object_classification_slide' (style 'slide'): the same for object classification on PUMA. PUMA has no whole
  slides, so the panel uses the 5120 x 5120 context image of the test ROI.
- 'figure_4' (in 'figures/final'): panel a, the BCSS panel as b and the PUMA panel as c.

Usage:

    python plots_for_figure4.py
    python plots_for_figure4.py --only classification
    python plots_for_figure4.py --only figure_4

Panel a reads 'object_classification_results.csv' and 'pixel_classification_results.csv' in 'figures'. These are
the scores of the few-shot-sam runs of Carolin Teuber ('classifiication_object.csv' and 'classifiication_pixel.csv').
Panels b and c read the outputs of 'run_posthoc_classification.py'. Run its steps in this order. The steps with a *
need a GPU.

    # Panel b: BCSS pixel classification with 1000 labels, on the whole slide of the test crop.
    python run_posthoc_classification.py prepare --dataset bcss
    python run_posthoc_classification.py annotate --dataset bcss
  * python run_posthoc_classification.py predict --dataset bcss --budget 1000 --methods micro-sam2
    python run_posthoc_classification.py download --dataset bcss --slide TCGA-E2-A14X-01Z-00-DX1
  * python run_posthoc_classification.py slide --dataset bcss --budget 1000 --methods micro-sam2 \
        --slide TCGA-E2-A14X-01Z-00-DX1 [--part k n]
    python run_posthoc_classification.py slide --dataset bcss --budget 1000 --methods micro-sam2 \
        --slide TCGA-E2-A14X-01Z-00-DX1 --merge

    # Panel c: PUMA object classification with all labels (budget 100000), on the context image of the test ROI.
    python run_posthoc_classification.py prepare --dataset puma
    python run_posthoc_classification.py annotate --dataset puma
  * python run_posthoc_classification.py predict --dataset puma --budget 100000 --methods micro-sam2
    python run_posthoc_classification.py download --dataset puma
  * python run_posthoc_classification.py context --dataset puma --roi training_set_primary_roi_045 --step segment
  * python run_posthoc_classification.py context --dataset puma --roi training_set_primary_roi_045 --step classify \
        --budget 100000 --methods micro-sam2

    # All plots and the combined figure.
    python plots_for_figure4.py

The slide run takes about 20 GPU hours. Use '--part k n' to split it into n blocks of tile rows for parallel jobs.
The 'predict' step saves the random forests in WORK_DIR/forests, and the 'slide' and 'context' steps load them.
"""

import os
import re
import argparse
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager, transforms

import plots_for_figure2 as fig2
from plot_style import METHOD_COLORS, MAX_FONT_SIZE, TEXT_COLOR, set_style


ROOT = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(ROOT, "figures")
OUTPUT_DIR = os.path.join(FIGURES, "figure4")

# The (table, title, datasets) of each row. Each dataset maps to its panel title.
ROWS = {
    "object": (
        "object_classification_results.csv", "Object Classification",
        {"pannuke": "PanNuke", "puma": "PUMA (Nuclei)", "bbbc041": "BBBC041"},
    ),
    "pixel": (
        "pixel_classification_results.csv", "Pixel Classification",
        {"bcss": "BCSS", "puma_tissue": "PUMA (Tissue)", "ignite": "IGNITE"},
    ),
}

# The method name of each 'features' value. micro-sam stands for PathoSAM on the histopathology datasets.
FEATURES = {
    "SAM": "SAM",
    "SAM2": "SAM2",
    "SAM3": "SAM3",
    "DINOv3": "DINOv3",
    "UNIv2": "UNIv2",
    "microSAM1 (LM)": "micro-sam",
    "microSAM1 (histopathology)": "micro-sam",
    "microSAM2": "micro-sam2",
    "-": "End-to-end",
}
METHOD_ORDER = ["SAM", "SAM2", "SAM3", "DINOv3", "UNIv2", "micro-sam", "End-to-end", "micro-sam2"]
# The legend name of each method. The star marks a domain-specific model (μSAM is PathoSAM, or the LM model on BBBC041).
METHOD_LABELS = {
    "micro-sam": "\u03bcSAM*", "micro-sam2": "\u03bcSAM2", "UNIv2": "UNIv2*", "End-to-end": "Baseline",
}
# The slot of a method: its width, the background, and the shade of the fewest labels (0 white, 1 full color).
SLOT_WIDTH = 0.82
# The slots are a subtle grey, and a "not applicable" slot a darker grey, as the rows of Figure 2. The text on it
# is darker than in Figure 2 to keep a contrast of 4.9 on the darker grey.
SLOT_COLOR = "#fafafa"
NOT_APPLICABLE_COLOR = "#ececec"
NOT_APPLICABLE_TEXT_COLOR = "#666666"
# The half width of the slot background; the white gap between two slots is 1 - 2 * SLOT_SPAN.
SLOT_SPAN = 0.47
LIGHTEST_SHADE = 0.4
# The dot area in points^2 for the fewest and the most labels; the area grows linearly with the label step.
DOT_SIZES = (4, 16)
# The key right of each row names the number of labels of each dot, drawn in a neutral grey.
KEY_WIDTH = 0.2
KEY_COLOR = "#52514E"
KEY_FONT_SIZE = 5.5
# The key moves this far left (in figure width) into the gap after the third panel, to stand close to it.
KEY_SHIFT = 0.025
# The rank charts start this much further left (in figure width), closer to the keys, and keep their right edge.
RANK_SHIFT = 0.03
OUR_DOT_SIZES = (7, 26)

# The legend has one card per group of methods. The gap between two cards is in figure width.
# The (title, methods) of each card. The title breaks the top border of its card.
LEGEND_GROUPS = [
    ("General Purpose", ["SAM", "SAM2", "SAM3", "DINOv3"]), ("Domain-Specific", ["UNIv2", "micro-sam"]),
    ("Supervised", ["End-to-end"]), ("Ours", ["micro-sam2"]),
]
LEGEND_Y = 0.093
CARD_GAP = 0.01
# The card border is the legend border of Figure 2.
CARD_EDGE_COLOR = fig2.LEGEND_BORDER_COLOR
# The card titles use the grey of Figure 2 (contrast 5.3 on white). They start this many points right of the corner.
CARD_TITLE_COLOR = fig2.GROUP_COLOR
CARD_TITLE_INSET = 5

# The panel size in inches, for three panels in the double column width of 183 mm.
PANEL_HEIGHT = 1.45

# The ranks compare the feature models per dataset and label count, without UNIv2 and the supervised baseline.
RANK_EXCLUDED = {"UNIv2", "End-to-end"}
RANK_COLUMNS = ["Few Labels", "All Labels"]
# The width of the slope chart relative to a line panel, and the smallest vertical distance between two names.
SLOPE_WIDTH = 1.1
LABEL_GAP = 0.3
GUIDE_COLOR = "#e6e5e0"
# The rank values next to the names: smaller and in the grey of the ranks of Figure 2 (contrast 5.3 on white).
RANK_VALUE_COLOR = fig2.GROUP_COLOR
RANK_VALUE_SIZE = 5


def load_results(table):
    """Load a classification table in long form: one row per dataset, method and number of labels."""
    results = pd.read_csv(os.path.join(FIGURES, table))
    results["method"] = results["features"].map(FEATURES)
    budgets = [column[:-len("_mean")] for column in results.columns if column.endswith("_mean")]
    rows = []
    for budget in budgets:
        part = results[["dataset", "method", f"{budget}_mean", f"{budget}_std"]].copy()
        part.columns = ["dataset", "method", "mean", "std"]
        part["budget"] = budget
        rows.append(part)
    return pd.concat(rows).dropna(subset=["mean"]), budgets


def shade(color, fraction):
    """Mix the color with white: the fewest labels get LIGHTEST_SHADE, the most labels the full color."""
    rgb = np.array(colors.to_rgb(color))
    return 1 - (LIGHTEST_SHADE + (1 - LIGHTEST_SHADE) * fraction) * (1 - rgb)


def budget_label(budget):
    """Return the name of a number of labels, such as 1k for 1000."""
    if budget == "all":
        return "All"
    value = int(budget)
    return f"{value // 1000}k" if value >= 1000 else str(value)


def plot_key(ax, budgets):
    """Plot a card with the dot of each number of labels from top (most) to bottom (fewest), with its name.

    The card stands in the vertical middle of its row.
    """
    fractions = np.arange(len(budgets)) / (len(budgets) - 1)
    smallest, largest = DOT_SIZES
    handles = [
        plt.Line2D(
            [], [], linestyle="none", marker="o", markersize=np.sqrt(smallest + (largest - smallest) * fraction),
            markerfacecolor=shade(KEY_COLOR, fraction), markeredgecolor=KEY_COLOR, markeredgewidth=0.7,
        )
        for fraction in fractions
    ]
    labels = [budget_label(budget) for budget in budgets]
    legend = ax.legend(
        handles[::-1], labels[::-1], loc="center left", bbox_to_anchor=(0, 0.5), frameon=True, fancybox=True,
        fontsize=KEY_FONT_SIZE, handlelength=0.9, handletextpad=0.15, labelspacing=0.9, borderpad=0.4,
        borderaxespad=0,
    )
    legend.get_frame().set(edgecolor=CARD_EDGE_COLOR, linewidth=0.6, facecolor="white")
    legend.get_frame().set_boxstyle("round", pad=0.2, rounding_size=0.4)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    ax.axis("off")


def plot_panel(ax, results, budgets):
    """Plot each method in its own slot, with its label counts from left (fewest) to right (most)."""
    # Every panel has a slot for each method, so that the slots keep the legend order.
    for slot, method in enumerate(METHOD_ORDER):
        curve = results[results["method"] == method].set_index("budget").reindex(budgets).dropna(subset=["mean"])
        if curve.empty:
            ax.axvspan(slot - SLOT_SPAN, slot + SLOT_SPAN, color=NOT_APPLICABLE_COLOR, linewidth=0, zorder=0)
            ax.text(
                slot, 0.5, "not applicable", transform=ax.get_xaxis_transform(), rotation=90, ha="center",
                va="center", fontsize=5, style="italic", color=NOT_APPLICABLE_TEXT_COLOR,
            )
            continue
        ax.axvspan(slot - SLOT_SPAN, slot + SLOT_SPAN, color=SLOT_COLOR, linewidth=0, zorder=0)
        steps = np.array([budgets.index(budget) for budget in curve.index])
        x = slot - SLOT_WIDTH / 2 + SLOT_WIDTH * steps / (len(budgets) - 1)
        ours = method == "micro-sam2"
        color = METHOD_COLORS[method]
        ax.errorbar(
            x, curve["mean"], yerr=curve["std"], color=color, linewidth=1.4 if ours else 0.7, elinewidth=0.5,
            capsize=0, linestyle=(0, (3, 1.5)) if method == "End-to-end" else "-", zorder=2,
        )
        fractions = steps / (len(budgets) - 1)
        smallest, largest = OUR_DOT_SIZES if ours else DOT_SIZES
        ax.scatter(
            x, curve["mean"], s=smallest + (largest - smallest) * fractions,
            color=[shade(color, fraction) for fraction in fractions], edgecolor=color, linewidth=0.7, zorder=3,
            clip_on=False,
        )
    ax.set_xticks([])
    ax.set_xlim(-0.6, len(METHOD_ORDER) - 0.4)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.5, linestyle=(0, (3, 3)), zorder=1)
    ax.set_axisbelow(True)


def legend_handle(method):
    """Return the legend handle of a method: its line with one dot.

    The short handle uses shorter dashes than the plot, so that the dashes of the baseline stay visible.
    """
    ours = method == "micro-sam2"
    return plt.Line2D(
        [], [], color=METHOD_COLORS[method], marker="o", markersize=4 if ours else 3, linewidth=1.4 if ours else 0.7,
        markeredgecolor=METHOD_COLORS[method], linestyle=(0, (1.2, 0.8)) if method == "End-to-end" else "-",
    )


def add_card(fig, group, width=None):
    """Add the legend card of a group of methods. A card with a width spreads its entries over that width."""
    anchor = (0, LEGEND_Y) if width is None else (0, LEGEND_Y, width, 0)
    legend = fig.legend(
        [legend_handle(method) for method in group], [METHOD_LABELS.get(method, method) for method in group],
        loc="lower left", bbox_to_anchor=anchor, ncol=len(group), frameon=True, fancybox=True,
        fontsize=MAX_FONT_SIZE, handlelength=1.4, handletextpad=0.4, columnspacing=0.75, borderpad=0.4,
        mode=None if width is None else "expand",
    )
    legend.get_frame().set(edgecolor=CARD_EDGE_COLOR, linewidth=0.6, facecolor="white")
    legend.get_frame().set_boxstyle("round", pad=0.2, rounding_size=0.4)
    for text in legend.get_texts():
        if text.get_text() == METHOD_LABELS["micro-sam2"]:
            text.set_fontweight("bold")
    return legend


def place_cards(fig, legends, starts):
    """Move the legend cards to these left edges, in figure width."""
    for legend, start in zip(legends, starts):
        anchor = legend.get_bbox_to_anchor().transformed(fig.transFigure.inverted())
        legend.set_bbox_to_anchor((start, LEGEND_Y, anchor.width, 0), transform=fig.transFigure)


def add_legend(fig, left, keys):
    """Add the legend as one card per group of methods, side by side and centered between left and right.

    Left is the left edge of the dataset panels in figure width, and right the right edge of the key cards.
    """
    # The cards get their size in a draw at the dpi of the png, because text widths change with the dpi.
    dpi = fig.dpi
    fig.set_dpi(plt.rcParams["savefig.dpi"])
    renderer = fig.canvas.get_renderer()
    title_font = font_manager.FontProperties(size=6, weight="bold")
    legends = [add_card(fig, group) for _, group in LEGEND_GROUPS]
    fig.canvas.draw()
    # A card that is narrower than its title, with an inset on both sides, is made as wide as that.
    for index, (title, group) in enumerate(LEGEND_GROUPS):
        width = legends[index].get_window_extent(renderer).width
        needed = renderer.get_text_width_height_descent(title, title_font, ismath=False)[0]
        needed += 2 * CARD_TITLE_INSET * renderer.dpi / 72
        if needed > width:
            # The expanded card loses its border padding, so its width is set again with the measured shortfall.
            legends[index].remove()
            legends[index] = add_card(fig, group, width=needed / fig.bbox.width)
            fig.canvas.draw()
            shortfall = needed - legends[index].get_frame().get_window_extent(renderer).width
            legends[index].remove()
            legends[index] = add_card(fig, group, width=(needed + shortfall) / fig.bbox.width)
    fig.canvas.draw()
    widths = [legend.get_window_extent(renderer).width / fig.bbox.width for legend in legends]
    right = max(key.get_legend().get_frame().get_window_extent(renderer).x1 for key in keys) / fig.bbox.width
    starts = np.cumsum([0] + [width + CARD_GAP for width in widths[:-1]])
    place_cards(fig, legends, starts)
    # The rounded borders stand out of the measured cards, so the row is centered again on the drawn borders.
    fig.canvas.draw()
    frames = [legend.get_frame().get_window_extent(renderer) for legend in legends]
    drawn = (min(frame.x0 for frame in frames) + max(frame.x1 for frame in frames)) / 2 / fig.bbox.width
    place_cards(fig, legends, starts + (left + right) / 2 - drawn)
    # The title stands on the top border near the left corner, on a white box that hides the border under it.
    fig.canvas.draw()
    for legend, (title, _) in zip(legends, LEGEND_GROUPS):
        frame = legend.get_frame().get_window_extent(renderer).transformed(fig.transFigure.inverted())
        fig.text(
            frame.x0, frame.y1, title, ha="left", va="center", fontsize=6, fontweight="bold", color=CARD_TITLE_COLOR,
            bbox={"boxstyle": "square,pad=0.15", "facecolor": "white", "linewidth": 0}, zorder=10,
            transform=transforms.offset_copy(fig.transFigure, fig=fig, x=CARD_TITLE_INSET, units="points"),
        )
    fig.set_dpi(dpi)


def mean_ranks(tables, lowest):
    """Return the mean rank of each feature model over the datasets and label counts of all tables."""
    ranks = []
    for results, budgets in tables:
        results = results[~results["method"].isin(RANK_EXCLUDED)]
        results = results[results["budget"] == budgets[0]] if lowest else results
        rank = results.groupby(["dataset", "budget"])["mean"].rank(ascending=False)
        ranks.append(results.assign(rank=rank))
    return pd.concat(ranks).groupby("method")["rank"].mean()


def spread(values, gap):
    """Move the label positions apart until two neighbors are at least 'gap' apart, keeping their order."""
    order = sorted(values, key=values.get)
    positions = {method: values[method] for method in order}
    for _ in range(50):
        moved = False
        for upper, lower in zip(order, order[1:]):
            overlap = gap - (positions[lower] - positions[upper])
            if overlap > 1e-6:
                positions[upper] -= overlap / 2
                positions[lower] += overlap / 2
                moved = True
        if not moved:
            break
    return positions


def plot_rank_slope(ax, tables, title=None, columns_named=True):
    """Plot the mean rank of each feature model with the fewest labels and over all label counts."""
    columns = [mean_ranks(tables, lowest) for lowest in (True, False)]
    methods = [method for method in METHOD_ORDER if method in columns[0].index]
    for x in (0, 1):
        ax.axvline(x, color=GUIDE_COLOR, linewidth=0.8, zorder=0)
    for method in methods:
        ours = method == "micro-sam2"
        ax.plot(
            [0, 1], [columns[0][method], columns[1][method]], color=METHOD_COLORS[method],
            linewidth=2.0 if ours else 1.0, marker="o", markersize=4.5 if ours else 3, markeredgecolor="white",
            markeredgewidth=0.6, solid_capstyle="round", zorder=3 if ours else 2,
        )
    renderer = ax.figure.canvas.get_renderer()
    value_font = font_manager.FontProperties(size=RANK_VALUE_SIZE)
    for x, column, side in ((0, columns[0], "right"), (1, columns[1], "left")):
        positions = spread({method: column[method] for method in methods}, LABEL_GAP)
        offset = -0.07 if side == "right" else 0.07
        for method in methods:
            # The value stands next to the line and the name next to the value, with a gap of 1.5 points.
            anchor = (x + offset, positions[method])
            value = f"({column[method]:.1f})"
            ax.annotate(value, anchor, ha=side, va="center", fontsize=RANK_VALUE_SIZE, color=RANK_VALUE_COLOR)
            width = renderer.get_text_width_height_descent(value, value_font, ismath=False)[0]
            width = width * 72 / renderer.dpi + 1.5
            ax.annotate(
                METHOD_LABELS.get(method, method), anchor, xytext=(-width if side == "right" else width, 0),
                textcoords="offset points", ha=side, va="center", fontsize=6, color=TEXT_COLOR,
                fontweight="bold" if method == "micro-sam2" else "normal",
            )
    ax.invert_yaxis()
    ax.set_xlim(-0.75, 1.75)
    ax.set_xticks([0, 1], RANK_COLUMNS if columns_named else ["", ""])
    ax.tick_params(axis="x", length=0, labelsize=6, pad=2)
    if title:
        ax.set_title(title, fontsize=MAX_FONT_SIZE, fontweight="bold", color=TEXT_COLOR, pad=4)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_classification():
    """Plot the object classification row over the pixel classification row."""
    fig = plt.figure(figsize=(fig2.FIGURE_WIDTH, len(ROWS) * PANEL_HEIGHT + 0.4))
    grid = fig.add_gridspec(
        len(ROWS), 5, hspace=0.35, wspace=0.22, width_ratios=[1, 1, 1, KEY_WIDTH, SLOPE_WIDTH]
    )
    tables = [load_results(table) for table, _, _ in ROWS.values()]
    for row, ((_, (_, row_title, datasets)), (results, budgets)) in enumerate(zip(ROWS.items(), tables)):
        row_axes = [fig.add_subplot(grid[row, column]) for column in range(3)]
        panels = row_axes
        row_axes[0].set_ylabel(f"{row_title}\nScore", fontsize=6, fontweight="bold", color=TEXT_COLOR)
        for ax, (dataset, title) in zip(row_axes, datasets.items()):
            plot_panel(ax, results[results["dataset"] == dataset], budgets)
            ax.set_title(title, fontsize=MAX_FONT_SIZE, fontweight="bold", color=TEXT_COLOR, pad=4)
    keys = [fig.add_subplot(grid[row, 3]) for row in range(len(tables))]
    ranks = []
    for row, table in enumerate(tables):
        # Only the top chart has the title; both charts name the columns.
        ranks.append(fig.add_subplot(grid[row, 4]))
        plot_rank_slope(ranks[-1], [table], title="Mean Rank" if row == 0 else None, columns_named=True)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.2)
    for key in keys:
        position = key.get_position()
        key.set_position([position.x0 - KEY_SHIFT, position.y0, position.width, position.height])
    for rank in ranks:
        position = rank.get_position()
        rank.set_position([position.x0 - RANK_SHIFT, position.y0, position.width + RANK_SHIFT, position.height])
    for key, (_, budgets) in zip(keys, tables):
        plot_key(key, budgets)
    add_legend(fig, panels[0].get_position().x0, keys)
    return fig


# The (dataset, budget, test crop, training tile, row title) of each row of the qualitative plot.
QUALITATIVE = [
    (
        "bcss", 1000, "TCGA-E2-A14X-DX1_xmin88836_ymin66393_MPP-0.2500_2_1",
        "TCGA-AQ-A54N-DX1_xmin32440_ymin26558_MPP-0.2500_0_3", "Pixel Classification\n(BCSS)",
    ),
    ("pannuke", 100, "fold_3_1675", "fold_1_0782", "Object Classification\n(PanNuke)"),
]
QUALITATIVE_COLUMNS = ["Training Strokes", "Training Reference", "Test Image", "Test Reference"]
# The methods shown on the test crop, after the reference columns, with their column title.
QUALITATIVE_METHODS = {"micro-sam2": "\u03bcSAM2"}
# The strokes are drawn thicker with a white edge, so that they stay visible on dark tissue. This is only the display.
STROKE_DISPLAY_DILATION = 4
PIXEL_STROKE_EDGE = 2
OBJECT_STROKE_EDGE = "#ffffff"


# The part of the long axis of an object that its stroke line covers.
OBJECT_STROKE_LENGTH = 0.7


def object_stroke_overlay(image, strokes, instances):
    """Draw the stroke of each stroked object as a thin line along its long axis, with a white edge."""
    from skimage.draw import line
    from skimage.measure import regionprops
    from skimage.morphology import dilation, disk
    center = np.zeros(strokes.shape, dtype="uint8")
    for prop in regionprops(instances):
        values = strokes[instances == prop.label]
        values = values[values > 0]
        if len(values) == 0:
            continue
        cy, cx = prop.centroid
        # The long axis points along (cos, sin) of the orientation in (row, column) order.
        dr, dc = np.cos(prop.orientation), np.sin(prop.orientation)
        half = OBJECT_STROKE_LENGTH * prop.axis_major_length / 2
        start = (int(round(cy - half * dr)), int(round(cx - half * dc)))
        end = (int(round(cy + half * dr)), int(round(cx + half * dc)))
        rr, cc = line(*start, *end)
        keep = (rr >= 0) & (rr < center.shape[0]) & (cc >= 0) & (cc < center.shape[1])
        center[rr[keep], cc[keep]] = np.bincount(values).argmax()
    core = dilation(center, disk(1))
    edge = dilation(core > 0, disk(2)) & (core == 0)
    out = image.copy()
    out[edge] = np.array(colors.to_rgb(OBJECT_STROKE_EDGE)) * 255
    return class_overlay(out, core, alpha=1.0)


def pixel_stroke_overlay(image, strokes):
    """Draw the pixel strokes thicker in their class colors, with a white edge."""
    from skimage.morphology import dilation, disk
    core = dilation(strokes, disk(STROKE_DISPLAY_DILATION))
    edge = dilation(core > 0, disk(PIXEL_STROKE_EDGE)) & (core == 0)
    out = image.copy()
    out[edge] = np.array(colors.to_rgb(OBJECT_STROKE_EDGE)) * 255
    return class_overlay(out, core, alpha=1.0)


# The class colors and the opacity of the class fill over the image. The last two are for the PUMA nuclei.
CLASS_COLORS = [
    "#E69F00", "#0072B2", "#009E73", "#CC79A7", "#F0E442", "#7F3C8D", "#56B4E9", "#8C564B", "#7F7F7F", "#11A579",
]
CLASS_ALPHA = 0.45


def class_overlay(image, labels, instances=None, alpha=CLASS_ALPHA):
    """Blend the class colors over the image; objects get an opaque outline in their class color."""
    from skimage.segmentation import find_boundaries
    palette = np.array([[0, 0, 0]] + [colors.to_rgb(c) for c in CLASS_COLORS]) * 255
    out = image.astype("float").copy()
    mask = (labels > 0) & (labels <= len(CLASS_COLORS))
    out[mask] = (1 - alpha) * out[mask] + alpha * palette[labels[mask]]
    if instances is not None:
        edges = find_boundaries(instances, mode="inner") & mask
        out[edges] = palette[labels[edges]]
    return out.astype("uint8")


def crop_macro_f1(prediction, semantic, instances, task):
    """The macro F1 of a prediction on a crop: over the labeled pixels, or over the objects (one class each)."""
    from skimage.measure import regionprops
    if task == "pixel":
        valid = semantic > 0
        truth, predicted = semantic[valid].astype(int), prediction[valid].astype(int)
    else:
        truth, predicted = [], []
        for prop in regionprops(instances, intensity_image=semantic):
            values = prop.image_intensity[prop.image].astype(int)
            values = values[values > 0]
            if len(values):
                truth.append(np.bincount(values).argmax())
                predicted.append(int(prediction[tuple(prop.coords[0])]))
        truth, predicted = np.array(truth), np.array(predicted)
    scores = []
    for c in np.unique(truth):
        tp = np.sum((truth == c) & (predicted == c))
        scores.append(2 * tp / max(np.sum(truth == c) + np.sum(predicted == c), 1))
    return float(np.mean(scores))


# The sizes of the layout of the qualitative plot, in inches.
QUALITATIVE_LABEL_WIDTH = 0.42
QUALITATIVE_COLUMN_GAP = 0.04
QUALITATIVE_TITLE_HEIGHT = 0.2
QUALITATIVE_LEGEND_HEIGHT = 0.27
QUALITATIVE_ROW_GAP = 0.1


def plot_classification_qualitative():
    """Plot one row per task: a training tile with its strokes and labels, the test crop, its labels, the predictions.

    Each prediction shows its macro F1 on the crop, and each row has its class legend in a card under it.
    """
    import h5py
    from matplotlib.patches import Patch
    import run_posthoc_classification as posthoc
    titles = QUALITATIVE_COLUMNS + list(QUALITATIVE_METHODS.values())
    n_columns = len(titles)
    width = fig2.FIGURE_WIDTH
    side = (width - QUALITATIVE_LABEL_WIDTH - (n_columns - 1) * QUALITATIVE_COLUMN_GAP) / n_columns
    row_height = side + QUALITATIVE_LEGEND_HEIGHT + QUALITATIVE_ROW_GAP
    height = QUALITATIVE_TITLE_HEIGHT + len(QUALITATIVE) * row_height - QUALITATIVE_ROW_GAP
    fig = plt.figure(figsize=(width, height))
    for row, (name, budget, test_stem, train_stem, title) in enumerate(QUALITATIVE):
        task, class_names = posthoc.DATASETS[name]
        raw, semantic, instances = posthoc.read_tile(name, "test", test_stem)
        train_raw, train_semantic, train_instances, strokes = posthoc.read_tile(
            name, "train", train_stem, ("raw", "semantic", "instances", "annotation")
        )
        predictions = {}
        for method in QUALITATIVE_METHODS:
            with h5py.File(posthoc.prediction_path(name, budget, method), "r") as f:
                predictions[method] = f[test_stem][:]
        objects = instances if task == "object" else None
        train_objects = train_instances if task == "object" else None
        strokes = np.where(strokes == posthoc.IGNORE, 0, strokes)
        if task == "object":
            stroke_panel = object_stroke_overlay(train_raw, strokes, train_instances)
        else:
            stroke_panel = pixel_stroke_overlay(train_raw, strokes)
        panels = [
            (stroke_panel, None), (class_overlay(train_raw, train_semantic, train_objects), None), (raw, None),
            (class_overlay(raw, semantic, objects), None),
        ] + [
            (class_overlay(raw, prediction, objects), (method, crop_macro_f1(prediction, semantic, instances, task)))
            for method, prediction in predictions.items()
        ]
        top = height - QUALITATIVE_TITLE_HEIGHT - row * row_height
        for column, (panel, score) in enumerate(panels):
            left = QUALITATIVE_LABEL_WIDTH + column * (side + QUALITATIVE_COLUMN_GAP)
            ax = fig.add_axes([left / width, (top - side) / height, side / width, side / height])
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set(color=fig2.FRAME_COLOR, linewidth=fig2.FRAME_WIDTH)
            if score is not None:
                fig2.draw_score(ax, score[1], score[0] == "micro-sam2")
            if row == 0:
                ours = titles[column] == QUALITATIVE_METHODS["micro-sam2"]
                ax.set_title(
                    titles[column], fontsize=MAX_FONT_SIZE, color=TEXT_COLOR, pad=3,
                    fontweight="bold" if ours else "normal",
                )
            if column == 0:
                ax.set_ylabel(title, fontsize=6, fontweight="bold", color=TEXT_COLOR, labelpad=3)
        # The legend card spans the image columns under the row.
        handles = [Patch(color=CLASS_COLORS[k], label=n) for k, n in enumerate(class_names)]
        legend = fig.legend(
            handles=handles, loc="center", ncol=len(handles), fontsize=6, frameon=True, fancybox=True,
            handlelength=1.0, handleheight=0.8, columnspacing=1.6, borderpad=0.35, labelcolor=TEXT_COLOR,
            bbox_to_anchor=(
                (QUALITATIVE_LABEL_WIDTH + (width - QUALITATIVE_LABEL_WIDTH) / 2) / width,
                (top - side - QUALITATIVE_LEGEND_HEIGHT / 2) / height,
            ),
        )
        legend.get_frame().set(edgecolor=CARD_EDGE_COLOR, linewidth=0.6, facecolor="white")
        legend.get_frame().set_boxstyle("round", pad=0.2, rounding_size=0.4)
    return fig


# The (dataset, budget, slide) of the pixel classification panel. The slide holds the BCSS test crop.
SLIDE = ("bcss", 1000, "TCGA-E2-A14X-01Z-00-DX1")
# The BCSS test crop (tile 2_1) on the slide as (x, y, size) at the base level. BCSS images are 1.215 times smaller.
SLIDE_CROP = (89458, 67637, 622)
# The whole slide is shown at 1/4 of the class map, with the majority class of each block.
SLIDE_DISPLAY_BLOCK = 4
# Tissue components smaller than this many class map pixels (4 windows of 48 pixels) are dust or debris outside the
# tissue; they are not shown. This changes only the display, not the prediction.
SLIDE_MIN_COMPONENT = 4 * 48 * 48


def block_majority(classes, block, n_classes):
    """Downscale a class map by 'block' with the most frequent class of each block (0 where no pixel has a class)."""
    rows, cols = classes.shape[0] // block, classes.shape[1] // block
    blocks = classes[:rows * block, :cols * block].reshape(rows, block, cols, block)
    counts = np.stack([(blocks == c).sum(axis=(1, 3)) for c in range(1, n_classes + 1)])
    out = counts.argmax(axis=0).astype("uint8") + 1
    out[counts.max(axis=0) == 0] = 0
    return out


# The (row, column) of each tile in the 2 x 2 grid of a slide panel.
SLIDE_TILES = {"Training Strokes": (0, 0), "Test Image": (1, 0), "Test Reference": (0, 1), "μSAM2": (1, 1)}
# The gaps of the layout and the height of the titles, in inches.
SLIDE_COLUMN_GAP = 0.2
SLIDE_ROW_GAP = 0.17
SLIDE_CONE_GAP = 0.27
SLIDE_TITLE_HEIGHT = 0.16
# The tile titles of a slide panel are a point smaller than the other titles, so that they fit over the tiles.
SLIDE_TILE_FONT_SIZE = MAX_FONT_SIZE - 1
# The padding of the legend card around its rows, in inches: (sides, top and bottom). The sides get more, since the
# legend width is measured before saving, a little narrower than drawn.
LEGEND_CARD_PAD = (0.12, 0.05)
# The smallest crop box on a slide image, in inches. A crop that is too small to see gets this box at its center.
SLIDE_BOX_SIZE = 0.09
# The zoom cone from the crop box to a tile: a light fill with a thin edge, over the slide image and under the tiles.
CONE_FILL = "#bdbcb8"
CONE_ALPHA = 0.3
CONE_EDGE = "#8f8e8a"
CONE_WIDTH = 0.5
BOX_COLOR = TEXT_COLOR
# The frame of the training tile: a pastel red, as the annotation the random forest learns from.
TRAINING_FRAME_COLOR = "#e8898a"
TRAINING_FRAME_WIDTH = 1.2
# The images of a slide panel are scaled by this factor; the text keeps its size, and the zoom cones take the free
# width. Figure 4 fits a vertical A4 page at full size.
SLIDE_PANEL_SCALE = 1.0


def slide_tiles(name, budget, test_stem, train_stem):
    """Read the tiles of a slide panel and return them with the prediction and the macro F1 on the test crop."""
    import h5py
    import run_posthoc_classification as posthoc
    task = posthoc.DATASETS[name][0]
    raw, semantic, instances = posthoc.read_tile(name, "test", test_stem)
    train_raw, train_instances, strokes = posthoc.read_tile(
        name, "train", train_stem, ("raw", "instances", "annotation")
    )
    strokes = np.where(strokes == posthoc.IGNORE, 0, strokes)
    # Classes outside the class list of the task (PUMA epithelium and apoptosis) are not labeled for it.
    n_classes = len(posthoc.DATASETS[name][1])
    semantic = np.where(semantic <= n_classes, semantic, 0)
    strokes = np.where(strokes <= n_classes, strokes, 0)
    with h5py.File(posthoc.prediction_path(name, budget, "micro-sam2"), "r") as f:
        prediction = f[test_stem][:]
    objects = instances if task == "object" else None
    if task == "object":
        stroke_tile = object_stroke_overlay(train_raw, strokes, train_instances)
    else:
        stroke_tile = pixel_stroke_overlay(train_raw, strokes)
    tiles = {
        "Training Strokes": stroke_tile, "Test Image": raw, "Test Reference": class_overlay(raw, semantic, objects),
        "μSAM2": class_overlay(raw, prediction, objects),
    }
    shown = set(np.unique(semantic)) | set(np.unique(prediction)) | set(np.unique(strokes))
    return tiles, crop_macro_f1(prediction, semantic, instances, task), shown


def plot_slide_panel(panel):
    """Plot a slide panel from the slide image to the tiles and back to the class map.

    The panel has these parts:
    - left: the test slide image (H&E) with the box of the test crop;
    - middle: a 2 x 2 grid with the training tile over the test crop, and the reference over the prediction;
    - right: the micro-sam2 classes of the whole slide image.
    Zoom cones join the two boxes to the test crop and to the prediction.

    Args:
        panel: The dictionary with the tiles, the score and the two slide images. It also holds the extent of the
            slide images in source pixels, the crop (x, y, size), the titles, the row label and the classes.

    Returns:
        The figure.
    """
    from matplotlib.patches import Patch, Rectangle, Polygon
    tiles, score = panel["tiles"], panel["score"]
    extent, crop = panel["extent"], panel["crop"]
    class_names = panel["class_names"]
    shown = [c for c in range(1, len(class_names) + 1) if c in panel["classes"]]
    # The legend stands under the tile grid, between the two slide images, in two rows.
    legend_height = QUALITATIVE_LEGEND_HEIGHT * 1.6

    # The sizes: the slide image on the left and the class map on the right span the two grid rows and the legend.
    width = fig2.FIGURE_WIDTH
    aspect = extent[1] / extent[0]
    # The 0.25 inch on the right leaves room for the save padding and for the title, which is wider than the map.
    free = width - 0.25 - QUALITATIVE_LABEL_WIDTH - 2 * SLIDE_CONE_GAP - SLIDE_COLUMN_GAP
    side = SLIDE_PANEL_SCALE * (free - 2 * aspect * (SLIDE_ROW_GAP + legend_height)) / (4 * aspect + 2)
    grid_height = 2 * side + SLIDE_ROW_GAP
    map_height = grid_height + legend_height
    map_width = aspect * map_height
    # The free width left by the scale widens the two zoom cones, so that the panel spans the figure width.
    cone_gap = SLIDE_CONE_GAP + (free - 2 * map_width - 2 * side) / 2
    bottom = legend_height
    height = map_height + SLIDE_TITLE_HEIGHT
    fig = plt.figure(figsize=(width, height))

    def add_panel(image, left, low, panel_width, panel_height, title, zorder=0, fontsize=MAX_FONT_SIZE):
        ax = fig.add_axes([left / width, low / height, panel_width / width, panel_height / height], zorder=zorder)
        ax.imshow(image, aspect="auto", interpolation="nearest" if image.shape[0] > 1000 else "antialiased")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set(color=fig2.FRAME_COLOR, linewidth=fig2.FRAME_WIDTH, visible=True)
        weight = "bold" if "μSAM2" in title else "normal"
        ax.set_title(title, fontsize=fontsize, color=TEXT_COLOR, pad=2.5, fontweight=weight)
        return ax

    left_slide = QUALITATIVE_LABEL_WIDTH
    grid_left = left_slide + map_width + cone_gap
    map_left = grid_left + 2 * side + SLIDE_COLUMN_GAP + cone_gap
    add_panel(panel["image"], left_slide, 0, map_width, map_height, panel["image_title"])
    add_panel(panel["map"], map_left, 0, map_width, map_height, panel["map_title"])
    fig.text(
        (left_slide - QUALITATIVE_LABEL_WIDTH / 2) / width, (map_height / 2) / height, panel["label"],
        rotation=90, ha="center", va="center", fontsize=6, fontweight="bold", color=TEXT_COLOR,
    )
    corners = {}
    for title, (row, column) in SLIDE_TILES.items():
        x = grid_left + column * (side + SLIDE_COLUMN_GAP)
        y = bottom + (1 - row) * (side + SLIDE_ROW_GAP)
        ax = add_panel(tiles[title], x, y, side, side, title, zorder=3, fontsize=SLIDE_TILE_FONT_SIZE)
        if title == "Training Strokes":
            for spine in ax.spines.values():
                spine.set(color=TRAINING_FRAME_COLOR, linewidth=TRAINING_FRAME_WIDTH, visible=True)
        if title == "μSAM2":
            fig2.draw_score(ax, score, True)
        corners[title] = (x, y)

    half = max(SLIDE_BOX_SIZE, crop[2] / extent[1] * map_width) / 2
    # The shapes are placed in figure fractions: the inch transform is not moved by the tight bounding box at saving.
    to_figure = transforms.Affine2D().scale(1 / width, 1 / height) + fig.transFigure

    def box_center(left):
        """The center of the crop box in inches of the figure, on the slide image at 'left'."""
        x, y, size = crop
        return (left + (x + size / 2) / extent[1] * map_width, (1 - (y + size / 2) / extent[0]) * map_height)

    def cone(center, box_side, tile_x, tile_y):
        """A zoom cone from the side of the crop box (box_side = +1 right, -1 left) to the side of a tile at tile_x."""
        box_x = center[0] + box_side * half
        points = [(box_x, center[1] + half), (tile_x, tile_y + side), (tile_x, tile_y), (box_x, center[1] - half)]
        fig.add_artist(Polygon(
            points, closed=True, facecolor=CONE_FILL, alpha=CONE_ALPHA, edgecolor="none", transform=to_figure, zorder=1
        ))
        for (x0, y0), (x1, y1) in [(points[0], points[1]), (points[3], points[2])]:
            fig.add_artist(plt.Line2D(
                [x0, x1], [y0, y1], color=CONE_EDGE, linewidth=CONE_WIDTH, transform=to_figure, zorder=2,
                solid_capstyle="round",
            ))

    def box(center):
        fig.add_artist(Rectangle(
            (center[0] - half, center[1] - half), 2 * half, 2 * half, fill=False, edgecolor=BOX_COLOR, linewidth=0.7,
            transform=to_figure, zorder=5,
        ))

    center = box_center(left_slide)
    cone(center, 1, *corners["Test Image"])
    box(center)
    center = box_center(map_left)
    ours_x, ours_y = corners["μSAM2"]
    cone(center, -1, ours_x + side, ours_y)
    box(center)

    # The legend: two rows, each centered on its own, in one card around both.
    from matplotlib.patches import FancyBboxPatch
    handles = [Patch(color=CLASS_COLORS[c - 1], label=class_names[c - 1]) for c in shown]
    first = int(np.ceil(len(handles) / 2))
    center_x = (grid_left + side + SLIDE_COLUMN_GAP / 2) / width
    legends = []
    for row, row_handles in enumerate([handles[:first], handles[first:]]):
        legend = fig.legend(
            handles=row_handles, loc="center", ncol=len(row_handles), fontsize=6, frameon=False,
            handlelength=1.0, handleheight=0.8, columnspacing=1.2, borderpad=0.0, labelcolor=TEXT_COLOR,
            bbox_to_anchor=(center_x, (legend_height * (0.64 - 0.32 * row)) / height),
        )
        legends.append(legend)
    renderer = fig.canvas.get_renderer()
    boxes = [legend.get_window_extent(renderer).transformed(fig.transFigure.inverted()) for legend in legends]
    x0, x1 = min(b.x0 for b in boxes), max(b.x1 for b in boxes)
    y0, y1 = min(b.y0 for b in boxes), max(b.y1 for b in boxes)
    # The card is LEGEND_CARD_PAD around the legend rows, with rounded corners of 0.04 inch.
    pad_x, pad_y = LEGEND_CARD_PAD[0] / width, LEGEND_CARD_PAD[1] / height
    card = FancyBboxPatch(
        (x0 - pad_x, y0 - pad_y), x1 - x0 + 2 * pad_x, y1 - y0 + 2 * pad_y,
        boxstyle=f"round,pad=0,rounding_size={0.04 / height:.4f}", transform=fig.transFigure, facecolor="white",
        edgecolor=CARD_EDGE_COLOR, linewidth=0.6, zorder=4,
    )
    fig.add_artist(card)
    for legend in legends:
        legend.set_zorder(5)
    return fig


def plot_classification_slide():
    """Plot the pixel classification panel: BCSS, from the whole test slide to the test crop and back."""
    import h5py
    from scipy import ndimage
    from skimage.morphology import remove_small_objects
    import run_posthoc_classification as posthoc
    name, budget, slide = SLIDE
    class_names = posthoc.DATASETS[name][1]
    tiles, score, shown = slide_tiles(name, budget, QUALITATIVE[0][2], QUALITATIVE[0][3])
    with h5py.File(os.path.join(posthoc.SLIDE_DIR, f"{slide}_{name}_micro-sam2_budget{budget}.h5"), "r") as f:
        scale = int(f.attrs["downscale"])
        rows, cols = (int(np.ceil(v / scale)) for v in f.attrs["shape"])
        classes, thumbnail = f["classes"][:rows, :cols], f["thumbnail"][:rows, :cols]
    tissue = remove_small_objects(ndimage.binary_fill_holes(classes > 0), max_size=SLIDE_MIN_COMPONENT - 1)
    classes = np.where(tissue, classes, 0)
    palette = (np.array([[1, 1, 1]] + [colors.to_rgb(c) for c in CLASS_COLORS]) * 255).astype("uint8")
    block = SLIDE_DISPLAY_BLOCK
    small_classes = block_majority(classes, block, len(class_names))
    r, c = small_classes.shape
    small_thumbnail = thumbnail[:r * block, :c * block].reshape(r, block, c, block, 3).mean(axis=(1, 3))
    return plot_slide_panel({
        "tiles": tiles, "score": score, "image": small_thumbnail.astype("uint8"), "map": palette[small_classes],
        "extent": (r * block * scale, c * block * scale), "crop": SLIDE_CROP, "class_names": class_names,
        "classes": set(range(1, len(class_names) + 1)), "image_title": "Whole Slide Image (Test)",
        "map_title": "μSAM2 (Tissue Segmentation)", "label": "Pixel Classification\n(BCSS)",
    })


# The PUMA panel: the test crop (x, y, size) on the context image. The budget is all labels, because 100 labels give
# a mean tile macro F1 of only 0.20.
PUMA_PANEL = {
    "budget": 100000, "roi": "training_set_primary_roi_045", "test_tile": "0_1",
    "train_stem": "training_set_primary_roi_053_1_1", "crop": (2560, 2048, 512),
}
# The context image is shown 2 times smaller. The class map fades the H&E by this fraction, so the nuclei stand out.
PUMA_DISPLAY_BLOCK = 2
PUMA_MAP_FADE = 0.7
# A class is in the legend if it is in the tiles or in this fraction of the nuclei of the context.
PUMA_MIN_FRACTION = 0.005


def plot_object_classification_slide():
    """Plot the object classification panel: PUMA, from the context of the test ROI to the test crop and back."""
    import h5py
    import tifffile
    import run_posthoc_classification as posthoc
    name, budget, roi = "puma", PUMA_PANEL["budget"], PUMA_PANEL["roi"]
    class_names = posthoc.DATASETS[name][1]
    tiles, score, shown = slide_tiles(name, budget, f"{roi}_{PUMA_PANEL['test_tile']}", PUMA_PANEL["train_stem"])
    context = tifffile.imread(os.path.join(posthoc.CONTEXT_DIR, f"{roi}_context.tif"))[..., :3]
    with h5py.File(posthoc.context_segmentation_path(roi), "r") as f:
        segmentation, classes = f["segmentation"][:], f[f"classes_budget{budget}"][:]
    block = PUMA_DISPLAY_BLOCK

    def small(image):
        r, c = image.shape[0] // block, image.shape[1] // block
        return image[:r * block, :c * block].reshape(r, block, c, block, 3).mean(axis=(1, 3)).astype("uint8")

    ids, first = np.unique(segmentation, return_index=True)
    nucleus_classes = classes.ravel()[first[ids > 0]]
    counts = np.bincount(nucleus_classes, minlength=len(class_names) + 1)
    shown |= {c for c in range(1, len(class_names) + 1) if counts[c] >= PUMA_MIN_FRACTION * counts[1:].sum()}
    faded = ((1 - PUMA_MAP_FADE) * context + PUMA_MAP_FADE * 255).astype("uint8")
    return plot_slide_panel({
        "tiles": tiles, "score": score, "image": small(context),
        "map": small(class_overlay(faded, classes, segmentation, alpha=1.0)), "extent": context.shape[:2],
        "crop": PUMA_PANEL["crop"], "class_names": class_names, "classes": shown - {0},
        "image_title": "Context Image (Test)", "map_title": "μSAM2 (Nucleus Classification)",
        "label": "Object Classification\n(PUMA)",
    })


def check_inputs(names):
    """Raise an error that lists the missing inputs of the plots in 'names'."""
    paths = []
    if "classification" in names:
        paths += [os.path.join(FIGURES, table) for table, _, _ in ROWS.values()]
    missing = sorted({path for path in paths if not os.path.exists(path)})
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))


# The (kind, style, function) of each plot, see 'save_figure' in 'plots_for_figure2.py'.
PLOTS = {
    "classification": ("classification", "quantitative", plot_classification),
    "classification_qualitative": ("classification", "qualitative", plot_classification_qualitative),
    "classification_slide": ("classification", "slide", plot_classification_slide),
    "object_classification_slide": ("classification", "slide", plot_object_classification_slide),
}


# The panels of the combined figure, top to bottom: (plot name, kind, style).
COMBINED_PANELS = [
    ("classification", "classification", "quantitative"),
    ("classification_slide", "classification", "slide"),
    ("object_classification_slide", "classification", "slide"),
]
COMBINED_NAME = "figure_4"
# The resolution of the images and of the combined png. It is above the 1100 dpi of the shown tiles.
IMAGE_DPI = 1200
# The gap between two panels, in points. The panel letter stands in the top left corner of its panel.
COMBINED_GAP = 7


def panel_svg(path, prefix, x, y):
    """Return the svg of a panel as a nested svg at (x, y), with its ids prefixed, and its width and height."""
    with open(path) as f:
        svg = f.read()
    svg = svg[svg.index("<svg"):]
    width, height = (float(v) for v in re.search(r'width="([0-9.]+)pt" height="([0-9.]+)pt"', svg).groups())
    # Both panels come from matplotlib with the same id names, so the ids of each panel get their own prefix.
    svg = re.sub(r'id="([^"]+)"', rf'id="{prefix}\1"', svg)
    svg = re.sub(r'url\(#([^)]+)\)', rf'url(#{prefix}\1)', svg)
    svg = re.sub(r'href="#([^"]+)"', rf'href="#{prefix}\1"', svg)
    # The combined svg has one user unit per point, so the nested size is given in user units, not in 'pt'.
    svg = re.sub(r'width="[0-9.]+pt" height="[0-9.]+pt"', f'width="{width}" height="{height}"', svg, count=1)
    svg = re.sub(r"<svg ", f'<svg x="{x}" y="{y:.3f}" ', svg, count=1)
    return svg, width, height


def plot_combined():
    """Stack the saved svgs of the panels into one svg with the panel letters, and convert it to pdf and png.

    The panels keep their editable text, because they are nested as svg and not drawn again.
    """
    width = fig2.FIGURE_WIDTH * 72
    parts, y = [], 0.0
    for k, (name, kind, style) in enumerate(COMBINED_PANELS):
        path = os.path.join(OUTPUT_DIR, kind, style, f"{name}.svg")
        svg, panel_width, panel_height = panel_svg(path, f"panel{k}_", 0, y)
        # A panel narrower than the figure is centered.
        if panel_width < width - 1:
            svg, panel_width, panel_height = panel_svg(path, f"panel{k}_", (width - panel_width) / 2, y)
        if panel_width > width + 1:
            raise ValueError(f"The panel '{name}' is {panel_width:.0f} pt wide, more than the figure ({width:.0f} pt).")
        parts.append(svg)
        parts.append(
            f'<text x="0" y="{y + fig2.PANEL_LABEL_SIZE:.3f}" font-family="Arial, Liberation Sans, Helvetica, '
            f'sans-serif" font-size="{fig2.PANEL_LABEL_SIZE}" font-weight="bold" fill="{TEXT_COLOR}">'
            f'{"abcdefgh"[k]}</text>'
        )
        y += panel_height + COMBINED_GAP
    height = y - COMBINED_GAP
    # As 'save_combined' of Figures 2 and 3: the page is scaled down to fit A4 if it is larger; the view box keeps the
    # drawing.
    scale = min(1.0, fig2.A4[0] * 72 / width, fig2.A4[1] * 72 / height)
    combined = (
        f'<?xml version="1.0" encoding="utf-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" width="{width * scale:.3f}pt" height="{height * scale:.3f}pt" '
        f'viewBox="0 0 {width:.3f} {height:.3f}" version="1.1">\n<rect width="100%" height="100%" fill="white"/>\n'
        + "\n".join(parts) + "\n</svg>\n"
    )
    os.makedirs(fig2.COMBINED_DIR, exist_ok=True)
    path = os.path.join(fig2.COMBINED_DIR, COMBINED_NAME)
    with open(f"{path}.svg", "w") as f:
        f.write(combined)
    env = dict(os.environ, SOURCE_DATE_EPOCH="0")
    subprocess.run(["rsvg-convert", "-u", "-f", "pdf", "-o", f"{path}.pdf", f"{path}.svg"], check=True, env=env)
    dpi = str(IMAGE_DPI)
    subprocess.run(
        ["rsvg-convert", "-u", "-f", "png", "-d", dpi, "-p", dpi, "-b", "white", "-o", f"{path}.png", f"{path}.svg"],
        check=True, env=env,
    )
    # rsvg-convert writes no resolution into the png, so we add it: Google Docs and other programs take the print size
    # of the png from it.
    from PIL import Image
    with Image.open(f"{path}.png") as image:
        image.load()
    image.save(f"{path}.png", dpi=(IMAGE_DPI, IMAGE_DPI))
    width_mm, height_mm = width * scale / 72 * 25.4, height * scale / 72 * 25.4
    size = f"{width_mm:.0f} x {height_mm:.0f} mm, scaled by {scale:.2f}"
    print(f"Saved '{COMBINED_NAME}' ({size}) to '{fig2.COMBINED_DIR}'.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    names = list(PLOTS) + [COMBINED_NAME]
    parser.add_argument("--only", nargs="+", choices=names, default=names, help="Make only these plots.")
    args = parser.parse_args()
    check_inputs(args.only)
    set_style()
    fig2.check_style()
    fig2.OUTPUT_DIR = OUTPUT_DIR
    for name in args.only:
        if name in PLOTS:
            kind, style, plot = PLOTS[name]
            fig2.save_figure(plot(), name, kind, style, dpi=IMAGE_DPI)
    if COMBINED_NAME in args.only:
        plot_combined()


if __name__ == "__main__":
    main()
