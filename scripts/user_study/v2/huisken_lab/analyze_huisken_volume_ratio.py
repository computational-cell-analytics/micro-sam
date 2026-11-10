"""The cell-nucleus volume ratio over the time-lapse of Beke (classic_lsm/timelapse_3-6hpf).

Each nucleus is mapped to the membrane cell that holds at least MIN_NUCLEUS_OVERLAP of its voxels. Only the cells
with exactly one nucleus are used. The ratio is the cell volume over the nucleus volume, where the cell includes its
nucleus. The ratio of each cell is averaged over all cells of a time point.

The outliers come from the segmentation and are left out of the mean. The nucleus id is its track id. A nucleus that
drops below MIN_NUCLEUS_FRACTION of its largest volume in the PREVIOUS_TIMEPOINTS before is in mitosis or a fragment.
A nucleus grows 3-5x during a cell cycle, so only a sudden drop counts. The drops split each track into its cell
cycles, because a track can continue through a division. A cell above MAX_CELL_FACTOR of the median of its cycle is
merged with its neighbours or the background. A ratio below MIN_RATIO means the cell is little more than its nucleus.
In the first division round (MITOSIS_TIMEPOINTS) the nuclei are the chromosomes that 'segment_huisken_timelapse.py'
adds by a threshold, which have no nuclear volume, so these time points are left out. The plot leaves out the time
points with fewer than MIN_CELLS kept cells.

The second panel shows the number of cells at each time point: the tracked nuclei, which stand for all cells of the
embryo, and the cells that are kept for the ratio.

Usage:
    python analyze_huisken_volume_ratio.py  # the tables and the plot, next to the segmentations
    python analyze_huisken_volume_ratio.py --version new_defaults  # the same for the results of the new defaults
    python analyze_huisken_volume_ratio.py --plot  # the filters and the plot, from the saved table
    python analyze_huisken_volume_ratio.py --compare  # the old and the new defaults side by side
"""

import os
import argparse

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/huisken_data/platynereis_timelapse_3-6hpf"
RESULT_DIR = "./results"
VERSIONS = ["old_defaults", "new_defaults"]

START_HPF, MINUTES_PER_TIMEPOINT = 3.0, 2.0
MIN_NUCLEUS_OVERLAP = 0.5
MIN_NUCLEUS_FRACTION, PREVIOUS_TIMEPOINTS = 0.5, 3
MAX_CELL_FACTOR, MIN_RATIO = 3.0, 1.5
MIN_CELLS = 5
# The SPOT_TIMEPOINTS of 'segment_huisken_timelapse.py'.
MITOSIS_TIMEPOINTS = (9, 10, 11, 12, 13, 14)


def map_cells_to_nuclei(cells, nuclei):
    """Map each nucleus of one time point to its cell and return the cells with exactly one nucleus.

    Returns (cell_id, nucleus_id, cell_voxels, nucleus_voxels) rows.
    """
    cell_voxels = np.bincount(cells.ravel())
    nucleus_voxels = np.bincount(nuclei.ravel())
    mask = (nuclei > 0) & (cells > 0)
    pairs, overlaps = np.unique(np.stack([nuclei[mask], cells[mask]]), axis=1, return_counts=True)
    # The cell that holds most of each nucleus, if it holds enough of it.
    order = np.lexsort((overlaps, pairs[0]))
    pairs, overlaps = pairs[:, order], overlaps[order]
    last = np.r_[pairs[0, 1:] != pairs[0, :-1], True]
    nucleus_ids, cell_ids, overlaps = pairs[0, last], pairs[1, last], overlaps[last]
    keep = overlaps >= MIN_NUCLEUS_OVERLAP * nucleus_voxels[nucleus_ids]
    nucleus_ids, cell_ids = nucleus_ids[keep], cell_ids[keep]

    ids, n_nuclei = np.unique(cell_ids, return_counts=True)
    single = set(ids[n_nuclei == 1])
    return [
        (int(c), int(n), int(cell_voxels[c]), int(nucleus_voxels[n]))
        for c, n in zip(cell_ids, nucleus_ids) if c in single
    ]


def compute_cell_table(cell_path, nucleus_path):
    """The volumes and the ratio of each cell with one nucleus, for each time point."""
    rows = []
    with h5py.File(cell_path, "r") as f_cells, h5py.File(nucleus_path, "r") as f_nuclei:
        for t in range(f_cells["segmentation"].shape[0]):
            cells, nuclei = f_cells["segmentation"][t], f_nuclei["segmentation"][t]
            pairs = map_cells_to_nuclei(cells, nuclei) if nuclei.any() else []
            rows.extend((t, *pair) for pair in pairs)
            print(f"t{t}: {len(pairs)} cells with one nucleus", flush=True)
    table = pd.DataFrame(rows, columns=["t", "cell_id", "nucleus_id", "cell_voxels", "nucleus_voxels"])
    table.insert(1, "hpf", START_HPF + table["t"] * MINUTES_PER_TIMEPOINT / 60)
    table["ratio"] = table["cell_voxels"] / table["nucleus_voxels"]
    return table


def find_outliers(table):
    """Name the filter that removes each cell, or leave it empty for the cells that are kept."""
    table = table.sort_values(["nucleus_id", "t"])
    previous_max = table.groupby("nucleus_id")["nucleus_voxels"].transform(
        lambda volumes: volumes.shift(1).rolling(PREVIOUS_TIMEPOINTS, min_periods=1).max()
    )
    nucleus_drop = table["nucleus_voxels"] < MIN_NUCLEUS_FRACTION * previous_max
    cycle = nucleus_drop.groupby(table["nucleus_id"]).cumsum()
    cell_median = table.groupby([table["nucleus_id"], cycle])["cell_voxels"].transform("median")
    outlier = pd.Series("", index=table.index)
    outlier[table["ratio"] < MIN_RATIO] = "small_ratio"
    outlier[table["cell_voxels"] > MAX_CELL_FACTOR * cell_median] = "large_cell"
    outlier[nucleus_drop] = "nucleus_drop"
    outlier[table["t"].isin(MITOSIS_TIMEPOINTS)] = "mitosis"
    return outlier.sort_index()


def summarize(table):
    """The ratio averaged over the kept cells of each time point."""
    kept = table[table["outlier"] == ""]
    summary = kept.groupby(["t", "hpf"])["ratio"].agg(n_cells="size", mean_ratio="mean")
    summary["n_outliers"] = (table["outlier"] != "").groupby([table["t"], table["hpf"]]).sum()
    return summary.reset_index()


def count_cells(table, nucleus_path):
    """The number of tracked nuclei and of the cells kept for the ratio at each time point."""
    with h5py.File(nucleus_path, "r") as f:
        n_timepoints = f["segmentation"].shape[0]
        track_timepoints = f["tracks"][:, 1].astype("int64")
    counts = pd.DataFrame({"t": np.arange(n_timepoints)})
    counts["hpf"] = START_HPF + counts["t"] * MINUTES_PER_TIMEPOINT / 60
    counts["n_nuclei"] = np.bincount(track_timepoints, minlength=n_timepoints)
    n_kept = (table["outlier"] == "").groupby(table["t"]).sum()
    counts["n_cells"] = n_kept.reindex(counts["t"], fill_value=0).to_numpy()
    return counts


def draw_ratio(ax, summary):
    # The time points with too few cells are gaps in the line, bridged by a dashed line between their ends.
    reliable = summary[summary["n_cells"] >= MIN_CELLS]
    ax.plot(reliable["hpf"], reliable["mean_ratio"], color="#2a78d6", linewidth=1, linestyle="--")
    ax.plot(summary["hpf"], summary["mean_ratio"].where(summary["n_cells"] >= MIN_CELLS), color="#2a78d6", linewidth=2)
    ax.set_ylabel("Mean cell / nucleus volume")
    ax.spines[["top", "right"]].set_visible(False)


def draw_counts(ax, counts):
    series = [("n_nuclei", "tracked nuclei", "#2a78d6"), ("n_cells", "cells in the ratio", "#eb6834")]
    for column, label, color in series:
        ax.plot(counts["hpf"], counts[column], color=color, linewidth=2, label=label)
        ax.text(counts["hpf"].iloc[-1], counts[column].iloc[-1], f" {label}", va="center", fontsize=8,
                color="#52514e")
    ax.axhline(MIN_CELLS, color="#bdbcb6", linewidth=1, linestyle=":")
    ax.set_xlabel("Time (hpf)")
    ax.set_ylabel("Number of cells")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)


def plot_ratio(summary, counts, path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    draw_ratio(axes[0], summary)
    draw_counts(axes[1], counts)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"Saved the plot to {path}")


def plot_comparison(path):
    """Plot the ratio and the cell numbers of the old and the new defaults side by side, from the saved tables."""
    fig, axes = plt.subplots(2, len(VERSIONS), figsize=(13, 7), sharex=True, sharey="row")
    for column, version in enumerate(VERSIONS):
        folder = os.path.join(RESULT_DIR, version)
        counts_path = os.path.join(folder, "cell_counts.tsv")
        if not os.path.exists(counts_path):
            raise FileNotFoundError(
                f"The cell numbers {counts_path} do not exist. "
                f"Create them with 'python analyze_huisken_volume_ratio.py --plot --version {version}'."
            )
        draw_ratio(axes[0, column], pd.read_csv(os.path.join(folder, "volume_ratio_summary.tsv"), sep="\t"))
        draw_counts(axes[1, column], pd.read_csv(counts_path, sep="\t"))
        axes[0, column].set_title(version.replace("_", " "))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"Saved the plot to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="old_defaults", choices=VERSIONS, help="The results to analyze.")
    parser.add_argument("--plot", action="store_true", help="Only plot the saved table.")
    parser.add_argument("--compare", action="store_true", help="Plot the old and the new defaults side by side.")
    args = parser.parse_args()
    if args.compare:
        plot_comparison(os.path.join(RESULT_DIR, "volume_ratio_old_vs_new.png"))
        return
    folder = os.path.join(RESULT_DIR, args.version)
    os.makedirs(folder, exist_ok=True)
    cell_table_path = os.path.join(folder, "volume_ratio_cells.tsv")
    summary_path = os.path.join(folder, "volume_ratio_summary.tsv")
    counts_path = os.path.join(folder, "cell_counts.tsv")
    plot_path = os.path.join(folder, "volume_ratio.png")
    nucleus_path = os.path.join(DATA_DIR, args.version, "C02_nuclei_tracks_fixed.h5")

    if args.plot:
        table = pd.read_csv(cell_table_path, sep="\t").drop(columns="outlier")
    else:
        cell_path = os.path.join(DATA_DIR, args.version, "C01_membrane_seg.h5")
        table = compute_cell_table(cell_path, nucleus_path)
    table["outlier"] = find_outliers(table)
    table.to_csv(cell_table_path, sep="\t", index=False)
    print(f"Saved the table of the cells to {cell_table_path}")
    counts = table["outlier"].replace("", "kept").value_counts()
    print("Cells:", ", ".join(f"{key}: {value}" for key, value in counts.items()))
    summary = summarize(table)
    summary.to_csv(summary_path, sep="\t", index=False, float_format="%.3f")
    print(f"Saved the summary to {summary_path}")
    cell_counts = count_cells(table, nucleus_path)
    cell_counts.to_csv(counts_path, sep="\t", index=False, float_format="%.3f")
    print(f"Saved the cell numbers to {counts_path}")
    plot_ratio(summary, cell_counts, plot_path)


if __name__ == "__main__":
    main()
