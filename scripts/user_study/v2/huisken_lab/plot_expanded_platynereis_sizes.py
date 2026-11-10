"""Plot the size histogram of the nuclei of the whole expanded Platynereis.

The sizes are read from the table of 'export_expanded_platynereis.py', which holds the number of voxels of each nucleus
at full resolution (s0). By default the sizes are plotted in voxels. With '--voxel_size' they are converted to volumes:
it takes one (isotropic) or three (z, y, x) voxel sizes and optionally their unit as the last value. The sizes span
several orders of magnitude, so the bins are log-spaced by default.

Usage:
    python plot_expanded_platynereis_sizes.py  # the sizes in voxels
    python plot_expanded_platynereis_sizes.py --voxel_size 1.2547 1.2556 1.2556 um  # the volumes in µm³
    python plot_expanded_platynereis_sizes.py --voxel_size 0.5 nm  # an isotropic voxel size of 0.5 nm
    python plot_expanded_platynereis_sizes.py --linear  # linear bins and axis
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TABLE_PATH = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/expanded_platynereis_export/nucleus_sizes_s0.csv"
UNIT_NAMES = {"um": "µm", "micrometer": "µm", "nanometer": "nm", "millimeter": "mm"}

BAR_COLOR = "#2a78d6"
SURFACE_COLOR = "#fcfcfb"
TEXT_PRIMARY, TEXT_SECONDARY = "#0b0b0b", "#52514e"


def parse_voxel_size(values):
    """Parse '--voxel_size' into the volume of one voxel and the unit (None if no unit is given)."""
    unit = None
    try:
        float(values[-1])
    except ValueError:
        unit, values = values[-1], values[:-1]
    if len(values) not in (1, 3):
        raise ValueError(f"Expected one or three voxel sizes (and optionally a unit), got {values}")
    sizes = [float(value) for value in values]
    if any(size <= 0 for size in sizes):
        raise ValueError(f"The voxel sizes must be positive, got {sizes}")
    voxel_volume = sizes[0] ** 3 if len(sizes) == 1 else float(np.prod(sizes))
    return voxel_volume, (None if unit is None else UNIT_NAMES.get(unit, unit))


def get_axis_label(voxel_volume, unit):
    """The x-axis label: the size in voxels by default and a volume otherwise."""
    if unit is not None:
        return f"Nucleus volume [{unit}³]"
    if voxel_volume != 1:
        return "Nucleus volume"
    return "Nucleus size [voxels]"


def format_size(value, unit, voxel_volume):
    """Format a size for the median label, with thousands separators and the unit."""
    text = f"{value:,.0f}" if value >= 100 else f"{value:.3g}"
    if unit is not None:
        return f"{text} {unit}³"
    return text if voxel_volume != 1 else f"{text} voxels"


def plot_histogram(sizes, axis_label, median_label, output_path, n_bins, linear):
    """Plot the histogram of the sizes with the median marked."""
    if linear:
        bins = np.linspace(sizes.min(), sizes.max(), n_bins + 1)
    else:
        bins = np.logspace(np.log10(sizes.min()), np.log10(sizes.max()), n_bins + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE_COLOR)
    ax.set_facecolor(SURFACE_COLOR)
    # A thin surface-colored edge separates the adjacent bars.
    ax.hist(sizes, bins=bins, color=BAR_COLOR, edgecolor=SURFACE_COLOR, linewidth=1.0, zorder=2)
    if not linear:
        ax.set_xscale("log")

    median = np.median(sizes)
    ax.axvline(median, color=TEXT_SECONDARY, linewidth=1.5, linestyle="--", zorder=3)
    ax.annotate(
        f"median {median_label(median)}", xy=(median, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -4),
        textcoords="offset points", ha="left", va="top", color=TEXT_SECONDARY, fontsize=9,
    )

    ax.set_title(f"Nucleus sizes (n = {len(sizes)})", color=TEXT_PRIMARY, loc="left")
    ax.set_xlabel(axis_label, color=TEXT_PRIMARY)
    ax.set_ylabel("Number of nuclei", color=TEXT_PRIMARY)
    ax.tick_params(colors=TEXT_SECONDARY)
    ax.grid(axis="y", color=TEXT_SECONDARY, alpha=0.15, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=SURFACE_COLOR)
    plt.close(fig)
    print(f"Saved the histogram to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table_path", default=TABLE_PATH, help="The csv table with the 'n_voxels' of each nucleus.")
    parser.add_argument(
        "--voxel_size", nargs="+", default=["1"], metavar="VALUE",
        help="One (isotropic) or three (z y x) voxel sizes, optionally followed by the unit, e.g. '1.25 um'. "
        "The default is a voxel size of 1 and no unit, i.e. the size in voxels.",
    )
    parser.add_argument("--output_path", help="The plot file. Default: 'nucleus_size_histogram.png' next to the table.")
    parser.add_argument("--n_bins", type=int, default=50)
    parser.add_argument("--linear", action="store_true", help="Linear instead of log-spaced bins.")
    args = parser.parse_args()

    voxel_volume, unit = parse_voxel_size(args.voxel_size)
    table = pd.read_csv(args.table_path)
    sizes = table["n_voxels"].to_numpy() * voxel_volume

    output_path = args.output_path
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(args.table_path)), "nucleus_size_histogram.png")
    plot_histogram(
        sizes, get_axis_label(voxel_volume, unit), lambda value: format_size(value, unit, voxel_volume),
        output_path, args.n_bins, args.linear,
    )


if __name__ == "__main__":
    main()
