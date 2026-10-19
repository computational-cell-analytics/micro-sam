"""Montages of nucleus-centered boxes in the BetaSeg volumes.

They check that no unlabeled neighbouring nucleus enters the boxes.

Only the nucleus of the central target cell is labeled in each volume. Each montage shows every ~16th z-slice of the box
around the labeled nucleus (plus a margin), the labeled nucleus outlined in blue, and the part of the box outside the
target cell mask (the 'membrane' label, a filled cell mask) tinted red: a nucleus inside a red area is unlabeled.
"""
import argparse
import glob
import os

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa
from skimage.segmentation import find_boundaries  # noqa

from figure_utils import DATA_ROOT, get_output_dir, normalize  # noqa


N_COLS = 6


def nucleus_box(nucleus, margin):
    idx = [np.where(nucleus.any(axis=tuple(a for a in range(3) if a != ax)))[0] for ax in range(3)]
    return tuple(
        slice(max(0, int(i.min()) - margin), min(size, int(i.max()) + 1 + margin))
        for i, size in zip(idx, nucleus.shape)
    )


def render_volume(path, margin, out_dir):
    name = os.path.basename(path).replace("_source.h5", "")
    with h5py.File(path, "r") as f:
        nucleus = f["labels/nucleus"][:] > 0
        box = nucleus_box(nucleus, margin)
        raw = f["raw"][box]
        cell = f["labels/membrane"][box] > 0
    nucleus = nucleus[box]
    z_ids = np.unique(np.linspace(0, raw.shape[0] - 1, max(2, raw.shape[0] // 16)).astype(int))
    n_rows = int(np.ceil(len(z_ids) / N_COLS))
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(3.2 * N_COLS, 3.2 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, z in zip(axes.flat, z_ids):
        image = normalize(raw[z][::2, ::2])
        overlay = np.zeros(image.shape + (4,), dtype="float32")
        overlay[~cell[z][::2, ::2]] = (1.0, 0.2, 0.2, 0.3)
        overlay[find_boundaries(nucleus[z][::2, ::2], mode="inner")] = (0.2, 0.6, 1.0, 1.0)
        ax.imshow(image, cmap="gray")
        ax.imshow(overlay)
        ax.set_title(f"z={z}", fontsize=8)
    outside = 1 - cell.mean()
    shape = tuple(s.stop - s.start for s in box)
    fig.suptitle(
        f"betaseg / {name} - nucleus box with margin {margin}, shape {shape}, {outside:.0%} outside the target cell\n"
        "blue: labeled nucleus; red: outside the target cell (a nucleus there is unlabeled)", fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.7 / fig.get_figheight()))
    out_path = os.path.join(out_dir, f"{name}_margin{margin}.png")
    fig.savefig(out_path, dpi=90)
    plt.close(fig)
    box_str = [(s.start, s.stop) for s in box]
    print(f"{name}: box {box_str}, {outside:.0%} outside the cell, {len(z_ids)} slices -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin", type=int, default=32, help="Margin around the nucleus bounding box in voxels.")
    args = parser.parse_args()
    out_dir = os.path.join(get_output_dir("betaseg"), "nucleus_boxes")
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(DATA_ROOT, "betaseg", "data", "*.h5"))):
        render_volume(path, args.margin, out_dir)


if __name__ == "__main__":
    main()
