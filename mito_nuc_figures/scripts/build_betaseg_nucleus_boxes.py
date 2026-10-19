"""Write the BetaSeg nucleus training boxes.

Only the nucleus of the central target cell is labeled in each volume. Each box is the bounding box of the labeled
nucleus plus a margin. Neighbouring cells, which can bring unlabeled nuclei into the box (for example in high_c4 and
low_c2, see betaseg_nucleus_check.py), are excluded from the loss: the labels carry IGNORE_LABEL outside the target
cell mask (the 'membrane' label, a filled cell mask) and in the regions without image data.
"""
import argparse
import glob
import json
import os

import h5py
import numpy as np
from scipy import ndimage

from betaseg_nucleus_boxes import nucleus_box
from figure_utils import DATA_ROOT


IGNORE_LABEL = np.iinfo("uint32").max
MISSING_DATA_MIN_AREA = 1000


def missing_data_mask(raw):
    """Connected raw == 0 or raw == 255 regions of a slice above MISSING_DATA_MIN_AREA."""
    mask = np.zeros(raw.shape, dtype=bool)
    for z in range(raw.shape[0]):
        for value in (0, 255):
            components, _ = ndimage.label(raw[z] == value)
            areas = np.bincount(components.ravel())
            mask[z] |= np.isin(components, np.where(areas > MISSING_DATA_MIN_AREA)[0]) & (components > 0)
    return mask


def write_box(path, margin, out_dir):
    name = os.path.basename(path).replace("_source.h5", "")
    with h5py.File(path, "r") as f:
        nucleus = f["labels/nucleus"][:] > 0
        box = nucleus_box(nucleus, margin)
        raw = f["raw"][box]
        cell = f["labels/membrane"][box] > 0
    nucleus = nucleus[box]
    missing = missing_data_mask(raw)
    ignore = (~cell & ~nucleus) | missing

    labels = nucleus.astype("uint32")
    labels[ignore] = IGNORE_LABEL
    out_path = os.path.join(out_dir, f"{name}.h5")
    tmp_path = f"{out_path}.incomplete"
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip", chunks=(32, 256, 256))
        f.create_dataset("labels/nucleus", data=labels, compression="gzip", chunks=(32, 256, 256))
        f.attrs["box"] = [[b.start, b.stop] for b in box]
        f.attrs["ignore_label"] = int(IGNORE_LABEL)
    os.replace(tmp_path, out_path)

    info = {
        "box": [[b.start, b.stop] for b in box], "shape": list(raw.shape),
        "nucleus_fraction": float(nucleus.mean()), "outside_cell_fraction": float((~cell).mean()),
        "missing_data_fraction": float(missing.mean()), "ignore_fraction": float(ignore.mean()),
        "nucleus_outside_cell_fraction": float((nucleus & ~cell).sum() / nucleus.sum()),
    }
    print(f"{name}: {info}")
    return name, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin", type=int, default=32, help="Margin around the nucleus bounding box in voxels.")
    parser.add_argument("--output", default=os.path.join(DATA_ROOT, "betaseg_nucleus_boxes"), help="Output folder.")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    boxes = dict(
        write_box(path, args.margin, args.output)
        for path in sorted(glob.glob(os.path.join(DATA_ROOT, "betaseg", "data", "*.h5")))
    )
    with open(os.path.join(args.output, "boxes.json"), "w") as f:
        json.dump({"margin": args.margin, "ignore_label": int(IGNORE_LABEL), "boxes": boxes}, f, indent=2)


if __name__ == "__main__":
    main()
