"""The cell-nucleus volume ratio of the Beke time-lapse (classic_lsm/timelapse_3-6hpf) on the level of the lineage.

The ratio of each cell comes from the cell table of 'analyze_huisken_volume_ratio.py', only the cells that it keeps
are used. The lineage is the 'lineage' key (track_id, t_start, t_end, parent_id) of the tracked nuclei. The nucleus id
of the cell table is the track id. There are two plots:

- The lineage tree (default): one tree per root track over time. Each track is colored by the ratio of its cell at
  each time point, gray where there is no ratio (the outliers, the first division round, the gaps of the track).
  The lineages that start at the first time point come first, the ones that start later are broken lineages.
- The cell cycles (--cycles): the ratio over the cell cycle of each track that is born and divides, normalized to
  0 at the birth and 1 at the division. One panel per generation (the depth in the lineage tree), only for the
  lineages that start at the first time point. The last panel shows the median ratio in the middle of each cycle
  per generation, with the sisters connected. Cycles shorter than --min_cycle_length time points are left out,
  because they are mostly false divisions.

Usage:
    python plot_huisken_lineage_ratio.py  # the lineage tree of the old defaults
    python plot_huisken_lineage_ratio.py --cycles --version new_defaults  # the cell cycles of the new defaults
"""

import os
import argparse

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from analyze_huisken_volume_ratio import (
    DATA_DIR, RESULT_DIR, VERSIONS, START_HPF, MINUTES_PER_TIMEPOINT, MITOSIS_TIMEPOINTS,
)

STEP_HPF = MINUTES_PER_TIMEPOINT / 60
# The phase window of the mid-cycle ratio, the minimal number of ratios in it, and the bins of the median line.
MID_CYCLE, MIN_MID_VALUES = (0.25, 0.75), 3
N_PHASE_BINS, MIN_BIN_VALUES = 10, 3

# A one-hue sequential ramp for the ratio and the neutral colors.
RATIO_CMAP = LinearSegmentedColormap.from_list(
    "ratio", ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
)
MEDIAN_COLOR, NO_VALUE_COLOR, CYCLE_COLOR, MITOSIS_COLOR = "#2a78d6", "#bdbcb6", "#a3a29c", "#f0efec"
TEXT_COLOR = "#52514e"


def to_hpf(t):
    return START_HPF + np.asarray(t) * STEP_HPF


def load_data(version):
    """Load the kept cells of the cell table and the lineage, with the root and the generation of each track."""
    table_path = os.path.join(RESULT_DIR, version, "volume_ratio_cells.tsv")
    if not os.path.exists(table_path):
        raise FileNotFoundError(
            f"The cell table {table_path} does not exist. "
            f"Create it with 'python analyze_huisken_volume_ratio.py --version {version}'."
        )
    lineage_path = os.path.join(DATA_DIR, version, "C02_nuclei_tracks_fixed.h5")
    if not os.path.exists(lineage_path):
        raise FileNotFoundError(f"The tracked nuclei {lineage_path} do not exist.")
    with h5py.File(lineage_path, "r") as f:
        if "lineage" not in f:
            raise KeyError(f"{lineage_path} has no 'lineage' key, track the nuclei with 'explore_huisken_data.py'.")
        lineage = pd.DataFrame(f["lineage"][:], columns=["track_id", "t_start", "t_end", "parent_id"])

    table = pd.read_csv(table_path, sep="\t", keep_default_na=False)
    if "outlier" not in table.columns:
        raise ValueError(f"{table_path} has no outlier column, rerun 'analyze_huisken_volume_ratio.py'.")
    unknown = set(table["nucleus_id"]) - set(lineage["track_id"])
    if unknown:
        raise ValueError(
            f"{len(unknown)} nucleus ids of {table_path} are not in the lineage of {lineage_path}. "
            "The cell table is from other tracks, rerun 'analyze_huisken_volume_ratio.py'."
        )
    table = table[table["outlier"] == ""]

    lineage = lineage.set_index("track_id")
    roots, generations = {}, {}
    for track_id in lineage.index:
        root, generation = track_id, 0
        while lineage.at[root, "parent_id"] > 0:
            root, generation = lineage.at[root, "parent_id"], generation + 1
        roots[track_id], generations[track_id] = root, generation
    lineage["root"], lineage["generation"] = pd.Series(roots), pd.Series(generations)
    first_t = lineage["t_start"].min()
    lineage["from_start"] = lineage["t_start"][lineage["root"]].to_numpy() == first_t
    return table, lineage


def layout_trees(lineage):
    """The y position of each track: the leaves are stacked, a parent sits between its daughters."""
    children = lineage.reset_index().groupby("parent_id")["track_id"].apply(sorted).to_dict()
    roots = lineage[lineage["parent_id"] == 0]
    roots = roots.sort_values(["from_start", "t_start"], ascending=[False, True]).index
    y_pos, next_y = {}, [0.0]

    def place(track_id):
        daughters = children.get(track_id, [])
        if daughters:
            y_pos[track_id] = np.mean([place(daughter) for daughter in daughters])
        else:
            y_pos[track_id] = next_y[0]
            next_y[0] += 1
        return y_pos[track_id]

    separator = None
    for root in roots:
        if separator is None and not lineage.at[root, "from_start"]:
            separator = next_y[0] + 0.5
            next_y[0] += 2
        place(root)
        next_y[0] += 1  # A gap between the trees.
    return y_pos, roots, separator, next_y[0]


def plot_tree(table, lineage, version, path):
    y_pos, roots, separator, n_rows = layout_trees(lineage)
    fig, ax = plt.subplots(figsize=(10, 1.5 + 0.11 * n_rows))
    for t in MITOSIS_TIMEPOINTS:
        ax.axvspan(to_hpf(t) - STEP_HPF / 2, to_hpf(t) + STEP_HPF / 2, color=MITOSIS_COLOR, linewidth=0, zorder=0)

    # The track spans in gray, starting at the division of the parent, and the division connectors.
    division_x = pd.Series(to_hpf(lineage["t_end"]) + STEP_HPF / 2, index=lineage.index)
    segments = []
    for track_id, row in lineage.iterrows():
        x_start = division_x[row["parent_id"]] if row["parent_id"] > 0 else to_hpf(row["t_start"]) - STEP_HPF / 2
        segments.append([(x_start, y_pos[track_id]), (division_x[track_id], y_pos[track_id])])
    for parent_id, daughters in lineage.groupby("parent_id"):
        if parent_id > 0:
            ys = [y_pos[track_id] for track_id in daughters.index]
            segments.append([(division_x[parent_id], min(ys)), (division_x[parent_id], max(ys))])
    ax.add_collection(LineCollection(segments, colors=NO_VALUE_COLOR, linewidths=1, zorder=1))

    # The ratio of each kept cell, one segment per time point.
    x = to_hpf(table["t"].to_numpy())
    y = table["nucleus_id"].map(y_pos).to_numpy()
    ratio_segments = np.stack([np.stack([x - STEP_HPF / 2, y], 1), np.stack([x + STEP_HPF / 2, y], 1)], 1)
    vmin, vmax = np.percentile(table["ratio"], [2, 98])
    norm = LogNorm(vmin=vmin, vmax=vmax)
    ratios = LineCollection(ratio_segments, cmap=RATIO_CMAP, norm=norm, linewidths=4, capstyle="butt", zorder=2)
    ratios.set_array(table["ratio"].to_numpy())
    ax.add_collection(ratios)

    for root in roots:
        ax.text(
            to_hpf(lineage.at[root, "t_start"]) - STEP_HPF, y_pos[root], str(root),
            ha="right", va="center", fontsize=5, color=TEXT_COLOR
        )
    label_x = to_hpf(lineage["t_end"].max()) + 2 * STEP_HPF
    ax.text(
        label_x, -0.5, "lineages from the first time point", rotation=-90, ha="left", va="top",
        fontsize=8, color=TEXT_COLOR
    )
    if separator is not None:
        ax.axhline(separator, color=NO_VALUE_COLOR, linewidth=0.8, linestyle="--")
        ax.text(
            label_x, separator + 0.5, "broken lineages", rotation=-90, ha="left", va="top", fontsize=8, color=TEXT_COLOR
        )

    ax.set_xlim(START_HPF - 6 * STEP_HPF, label_x + 2 * STEP_HPF)
    ax.set_ylim(n_rows, -1)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlabel("Time (hpf)")
    ax.set_title(
        f"Cell / nucleus volume on the lineage ({version.replace('_', ' ')}); "
        "gray: no ratio, shaded: first division round", fontsize=9, color=TEXT_COLOR
    )
    colorbar = fig.colorbar(ratios, ax=ax, extend="both", fraction=0.03, pad=0.06)
    colorbar.set_label("Cell / nucleus volume")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"Saved the plot to {path}")


def get_cycles(table, lineage, min_cycle_length):
    """The kept cells of the complete cycles from the lineages of the first time point, with their phase."""
    has_daughters = lineage.index.isin(lineage["parent_id"])
    complete = lineage[(lineage["parent_id"] > 0) & has_daughters]
    length = complete["t_end"] - complete["t_start"] + 1
    cycles = complete[(length >= min_cycle_length) & complete["from_start"]]
    print(
        f"{len(complete)} complete cycles, {(length < min_cycle_length).sum()} shorter than {min_cycle_length}, "
        f"{(~complete['from_start'] & (length >= min_cycle_length)).sum()} in broken lineages, "
        f"{len(cycles)} used"
    )
    if len(cycles) == 0:
        raise ValueError("There are no complete cycles to plot, try a smaller --min_cycle_length.")

    cells = table[table["nucleus_id"].isin(cycles.index)].copy()
    t_start = cells["nucleus_id"].map(cycles["t_start"])
    t_end = cells["nucleus_id"].map(cycles["t_end"])
    cells["phase"] = (cells["t"] - t_start) / (t_end - t_start)
    cells["generation"] = cells["nucleus_id"].map(cycles["generation"])
    return cycles, cells.sort_values(["nucleus_id", "t"])


def plot_cycles(table, lineage, version, min_cycle_length, path):
    cycles, cells = get_cycles(table, lineage, min_cycle_length)
    generations = sorted(cycles["generation"].unique())
    fig, axes = plt.subplots(1, len(generations) + 1, figsize=(3 * (len(generations) + 1), 3.5), sharey=True)
    bins = np.linspace(0, 1, N_PHASE_BINS + 1)
    for ax, generation in zip(axes, generations):
        gen_cells = cells[cells["generation"] == generation]
        for _, track in gen_cells.groupby("nucleus_id"):
            ax.plot(track["phase"], track["ratio"], color=CYCLE_COLOR, linewidth=0.8, marker="o", markersize=2)
        binned = gen_cells.groupby(pd.cut(gen_cells["phase"], bins, include_lowest=True), observed=False)["ratio"]
        median = binned.median().where(binned.size() >= MIN_BIN_VALUES)
        ax.plot((bins[:-1] + bins[1:]) / 2, median, color=MEDIAN_COLOR, linewidth=2, label="median")
        n_cycles = (cycles["generation"] == generation).sum()
        ax.set_title(f"Generation {generation} ({n_cycles} cycles)", fontsize=9)
        ax.set_xlabel("Cell cycle (birth to division)")
        ax.set_xlim(0, 1)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Cell / nucleus volume")
    axes[0].legend(frameon=False, fontsize=8)

    # The median ratio in the middle of each cycle, with the sisters connected.
    ax = axes[-1]
    mid = cells[cells["phase"].between(*MID_CYCLE)].groupby("nucleus_id")["ratio"]
    mid = mid.median()[mid.size() >= MIN_MID_VALUES]
    rng = np.random.default_rng(0)
    x = cycles["generation"] + rng.uniform(-0.15, 0.15, len(cycles))
    for _, sisters in cycles[cycles.index.isin(mid.index)].groupby("parent_id"):
        if len(sisters) == 2:
            ax.plot(x[sisters.index], mid[sisters.index], color=CYCLE_COLOR, linewidth=0.8, zorder=1)
    ax.scatter(x[mid.index], mid, s=24, color=MEDIAN_COLOR, edgecolors="white", linewidths=1, zorder=2)
    ax.set_xticks(generations)
    ax.set_xlabel("Generation")
    ax.set_title(f"Mid-cycle median ({len(mid)} cycles); sisters joined", fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Cell / nucleus volume over the cell cycle ({version.replace('_', ' ')})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"Saved the plot to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="old_defaults", choices=VERSIONS, help="The results to plot.")
    parser.add_argument("--cycles", action="store_true", help="Plot the cell cycles instead of the lineage tree.")
    parser.add_argument(
        "--min_cycle_length", type=int, default=10,
        help="The minimal number of time points of a cell cycle (for --cycles)."
    )
    args = parser.parse_args()
    table, lineage = load_data(args.version)
    folder = os.path.join(RESULT_DIR, args.version)
    if args.cycles:
        plot_cycles(table, lineage, args.version, args.min_cycle_length, os.path.join(folder, "lineage_cycles.png"))
    else:
        plot_tree(table, lineage, args.version, os.path.join(folder, "lineage_tree.png"))


if __name__ == "__main__":
    main()
