"""The nucleus-to-cytoplasm (N/C) ratio of the cells of the EM spheroid of the Beghin lab.

Cytoplasm volume = cell volume - nucleus volume, N/C ratio = nucleus volume / cytoplasm volume (see nc_ratio.py in
the parent folder). The cells and nuclei are those of the exported OME-Zarr (explore_beghin_em_data.py
export_segmentations): the updated nuclei and their cells on the grid of the raw data, where a cell has the id of
its nucleus. The volumes are in um^3.

The objects of the attribute 'excluded' of the export are not full nuclei or cells (update_partial_nuclei.py). They
are in the table with the column 'excluded' ('nucleus and cell' or 'nucleus'), but they are not part of the
histogram and the summary: without a full nucleus the ratio of a cell is not valid, also if the cell is a full cell.

Outputs, in NC_RATIO_DIR:
- nc_ratio.csv: one row per cell, with the columns of compute_nc_ratio and 'excluded'.
- nc_ratio_histogram.png: the histogram of the N/C ratio of the cells that are not excluded.

Example:
    python compute_beghin_em_nc_ratio.py  # the table and the plot
    python compute_beghin_em_nc_ratio.py --plot  # only the plot, from the saved table
"""
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import zarr
import matplotlib.pyplot as plt

import explore_beghin_em_data as explore

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
from nc_ratio import compute_nc_ratio  # noqa: E402

NC_RATIO_DIR = os.path.join(explore.VOLUME_PREDICTION_DIR, "nc_ratio")
TABLE_PATH = os.path.join(NC_RATIO_DIR, "nc_ratio.csv")
PLOT_PATH = os.path.join(NC_RATIO_DIR, "nc_ratio_histogram.png")
VOLUME_COLUMNS = ("cell_volume", "nucleus_volume", "cytoplasm_volume")
BIN_WIDTH = 0.025


def compute_table(level, num_workers):
    """Compute the N/C ratio of all cells of the export at a scale level and mark the excluded objects."""
    root = zarr.open_group(explore.EXPORT_PATH, mode="r")
    cells, nuclei = root["cells"][f"s{level}"], root["nuclei"][f"s{level}"]
    voxel_size = [vs * 1e-3 for vs in explore.get_level_voxel_size(level)]
    assert np.allclose(root.attrs["voxel_size_s0"], explore.VOXEL_SIZE)

    t0 = time.time()
    table = compute_nc_ratio(cells, nuclei, voxel_size=voxel_size, unit="um", num_workers=num_workers)
    print(f"Computed the volumes of {len(table)} cells at s{level} {cells.shape} in {time.time() - t0:.0f} s")
    assert table.attrs["volume_unit"] == "um^3"
    table = table.rename(columns={name: f"{name}_um3" for name in VOLUME_COLUMNS})

    # The cells are seeded by the nuclei, so each cell has the id of its nucleus.
    mismatch = table[(table["n_nuclei"] != 1) | (table["nucleus_id"] != table["cell_id"])]
    if len(mismatch):
        print(f"Cells that do not have exactly the nucleus with their id:\n{mismatch.to_string(index=False)}")

    excluded = {obj["id"]: obj["exclude"] for obj in root.attrs["excluded"]}
    assert set(excluded) <= set(table["cell_id"]), sorted(set(excluded) - set(table["cell_id"]))
    table["excluded"] = table["cell_id"].map(excluded).fillna("")
    return table


def plot_histogram(table, plot_path):
    """Plot the histogram of the N/C ratio of the cells that are not excluded."""
    used = table[(table["excluded"] == "") & table["nc_ratio"].notna()]
    ratio = used["nc_ratio"].to_numpy()
    median, (q25, q75) = np.median(ratio), np.percentile(ratio, [25, 75])
    bins = np.arange(0, np.ceil(ratio.max() / BIN_WIDTH) * BIN_WIDTH + BIN_WIDTH / 2, BIN_WIDTH)

    text, grid = "#191919", "#d9d9d9"
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=200)
    ax.hist(ratio, bins=bins, color="#257de0", edgecolor="white", linewidth=1.0)
    ax.axvline(median, color=text, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.annotate(
        f"median {median:.2f}\n(quartiles {q25:.2f} - {q75:.2f})", xy=(median, 1), xycoords=("data", "axes fraction"),
        xytext=(6, -4), textcoords="offset points", va="top", ha="left", fontsize=9, color=text,
    )
    ax.set_xlabel("N/C ratio (nucleus volume / cytoplasm volume)", color=text)
    ax.set_ylabel("Number of cells", color=text)
    ax.set_title(
        f"N/C ratio of the cells of the EM spheroid (n = {len(ratio)})", loc="left", fontsize=11, color=text,
    )
    ax.set_xlim(0, bins[-1])
    ax.yaxis.grid(True, color=grid, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#666666")
    ax.tick_params(colors="#666666", length=3)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"Histogram of {len(ratio)} cells: {plot_path}")


def print_summary(table):
    used = table[table["excluded"] == ""]
    print(f"{len(table)} cells, {len(table) - len(used)} excluded, {len(used)} used")
    columns = ["cell_volume_um3", "nucleus_volume_um3", "cytoplasm_volume_um3", "nc_ratio", "nucleus_fraction_in_cell"]
    print(used[columns].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(3).to_string())
    print(f"On the border of the volume: {int(used['cell_on_border'].sum())} cells, "
          f"{int(used['nucleus_on_border'].sum())} nuclei")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", type=int, default=0, help="The scale level of the export, by default s0.")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--plot", action="store_true", help="Only plot, from the saved table.")
    parser.add_argument("--output_dir", default=NC_RATIO_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    table_path = os.path.join(args.output_dir, os.path.basename(TABLE_PATH))
    if args.plot:
        table = pd.read_csv(table_path, keep_default_na=False, na_values=["nan", "NaN"])
    else:
        table = compute_table(args.level, args.num_workers)
        table.to_csv(table_path, index=False)
        print(f"Table: {table_path}")
    print_summary(table)
    plot_histogram(table, os.path.join(args.output_dir, os.path.basename(PLOT_PATH)))


if __name__ == "__main__":
    main()
