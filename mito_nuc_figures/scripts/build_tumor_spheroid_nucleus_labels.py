"""Write the tumor spheroid nucleus labels for the nucleus EM generalist, with the cells without a nucleus ignored.

The 20 manually annotated slices (50 nm) label cells and nuclei. A few large cells per slice carry a cell label but no
nucleus label, although their dark, mottled content looks like nucleoplasm. It is unclear whether their nucleus is
missing from the annotation or not in the section. Thus, these cells get IGNORE_LABEL and the loss ignores them. The
nuclei keep their ids. The labels are stored as uint32, so that the ignore label survives torch-em's cast of the labels
back to their stored dtype.
"""
import os
import glob
import json
import argparse

import h5py
import numpy as np

from figure_utils import DATA_ROOT


SOURCE = os.path.join(DATA_ROOT, "tumor_spheroid_em", "2d_manual", "50-50-50")
IGNORE_LABEL = np.iinfo("uint32").max


def write_slice(path, out_dir):
    with h5py.File(path, "r") as f:
        raw = f["raw"][:]
        nuclei = f["labels/nuclei"][:].astype("uint32")
        cells = f["labels/cells"][:]
    cell_ids = np.unique(cells[cells > 0])
    with_nucleus = np.unique(cells[(cells > 0) & (nuclei > 0)])
    without_nucleus = np.setdiff1d(cell_ids, with_nucleus)
    ignore = np.isin(cells, without_nucleus) & (nuclei == 0)
    labels = nuclei.copy()
    labels[ignore] = IGNORE_LABEL
    out_path = os.path.join(out_dir, os.path.basename(path))
    tmp_path = f"{out_path}.incomplete"
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("labels", data=labels, compression="gzip")
        f.attrs["ignore_label"] = int(IGNORE_LABEL)
    os.replace(tmp_path, out_path)
    info = {
        "n_nuclei": int(len(np.unique(nuclei)) - 1), "n_cells": int(len(cell_ids)),
        "n_cells_without_nucleus": int(len(without_nucleus)), "ignore_fraction": float(ignore.mean()),
    }
    print(os.path.basename(path), info)
    return os.path.basename(path), info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(DATA_ROOT, "tumor_spheroid_nucleus"), help="Output folder.")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    infos = dict(write_slice(path, args.output) for path in sorted(glob.glob(os.path.join(SOURCE, "*.h5"))))
    with open(os.path.join(args.output, "info.json"), "w") as f:
        json.dump({"ignore_label": int(IGNORE_LABEL), "slices": infos}, f, indent=2)


if __name__ == "__main__":
    main()
