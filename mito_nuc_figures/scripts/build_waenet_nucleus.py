"""Write the WAE-Net nucleus training data for the nucleus EM generalist.

- Dataset 1 (FIB-SEM of pancreatic carcinoid cells, all nuclei labeled): the 323 slices, stacked in numeric file
  order into one volume. The nuclei are 3D connected components. The last VAL_FRACTION of z validates.
- Dataset 3 (unregistered serial ssTEM sections of one interphase cell): one 2D image per section. The nuclei are 2D
  connected components, and the labels on the black rotation padding are removed. A contiguous block of sections
  validates, since neighbouring sections are near-duplicates.
- Datasets 4-7 (single TEM images of Jurkat, primary T, J558L B and expanded T cells): only the central cell is
  labeled. Its mask is the union of the 'cytoplasm' and 'nucleus' labels. The 'background' label covers everything
  else, including the neighbour cells, whose nuclei are not labeled. The loss is restricted to the central cell mask
  dilated by CELL_BAND pixels. All other pixels (neighbour cells, distant resin, burned-in scale bars) get
  IGNORE_LABEL. Images without a nucleus label and the images in DROPPED_IMAGES are not used. Images that are not
  512 x 512 (6 in dataset 6) are cropped around the central cell or padded with IGNORE_LABEL to 512 x 512. The last
  VAL_FRACTION of the images of each dataset (by file number) validates, since consecutive images can be serial
  sections of one cell.

The 2D datasets are written as stacks of 512 x 512 images, one h5 file per split. Each file has a temporary name that
is moved in place when complete.
"""
import os
import re
import json
import argparse

import h5py
import numpy as np
from scipy import ndimage

from figure_utils import DATA_ROOT


IGNORE_LABEL = np.iinfo("uint32").max
VAL_FRACTION = 0.1
CELL_BAND = 10
MIN_NUCLEUS_AREA = 100
PADDING_MIN_AREA = 1000
SIZE = 512
DS3_VAL_SECTIONS = range(38, 46)
TEM_DATASETS = (4, 5, 6, 7)
# Images whose masks do not follow the cell (found by screening every image).
DROPPED_IMAGES = {
    "dataset_5/49.h5": "nucleus and cytoplasm masks are offset from the cell and overlap a neighbour",
    "dataset_6/75.h5": "nucleus outline does not match the nucleus in the raw",
    "dataset_7/108.h5": "nucleus boundary is not visible in the low-contrast raw, the mask cannot be verified",
}


def file_number(name):
    return int(re.sub(r"\D", "", os.path.splitext(name)[0]))


def sorted_names(folder):
    return sorted(os.listdir(folder), key=file_number)


def write_h5(out_path, raw, labels, attrs):
    tmp_path = f"{out_path}.incomplete"
    with h5py.File(tmp_path, "w") as f:
        chunks = (1, SIZE, SIZE) if raw.ndim == 3 and raw.shape[1:] == (SIZE, SIZE) else True
        f.create_dataset("raw", data=raw, compression="gzip", chunks=chunks)
        f.create_dataset("labels", data=labels, compression="gzip", chunks=chunks)
        for key, value in attrs.items():
            f.attrs[key] = value
    os.replace(tmp_path, out_path)


def padding_mask(raw):
    components, _ = ndimage.label(raw == 0)
    areas = np.bincount(components.ravel())
    return np.isin(components, np.where(areas > PADDING_MIN_AREA)[0]) & (components > 0)


def instances_2d(nucleus, ignore=None):
    labels, _ = ndimage.label(nucleus)
    labels = labels.astype("uint32")
    sizes = np.bincount(labels.ravel())
    small = np.isin(labels, np.where(sizes < MIN_NUCLEUS_AREA)[0]) & (labels > 0)
    labels[small] = IGNORE_LABEL
    if ignore is not None:
        labels[ignore & (labels == 0)] = IGNORE_LABEL
    return labels


def build_dataset_1(root, out_dir):
    names = sorted_names(os.path.join(root, "dataset_1"))
    raw, nucleus = [], []
    for name in names:
        with h5py.File(os.path.join(root, "dataset_1", name), "r") as f:
            raw.append(f["raw"][:])
            nucleus.append(f["labels/nucleus"][:] > 0)
    raw, nucleus = np.stack(raw), np.stack(nucleus)
    labels, n_ids = ndimage.label(nucleus)
    write_h5(os.path.join(out_dir, "dataset_1.h5"), raw, labels.astype("uint32"), {"slices": names})
    return {"shape": list(raw.shape), "n_nuclei": int(n_ids), "n_val_slices": int(round(raw.shape[0] * VAL_FRACTION))}


def build_dataset_3(root, out_dir):
    images = {"train": [], "val": []}
    n_removed = n_empty = 0
    for name in sorted_names(os.path.join(root, "dataset_3")):
        with h5py.File(os.path.join(root, "dataset_3", name), "r") as f:
            raw, nucleus = f["raw"][:], f["labels/nucleus"][:] > 0
        pad = padding_mask(raw)
        n_removed += int((nucleus & pad).sum())
        nucleus &= ~pad
        if not nucleus.any():
            n_empty += 1
            continue
        split = "val" if file_number(name) in DS3_VAL_SECTIONS else "train"
        images[split].append((name, raw, instances_2d(nucleus)))
    info = {"n_without_nucleus": n_empty, "padding_pixels_removed": n_removed}
    for split, items in images.items():
        write_h5(
            os.path.join(out_dir, f"dataset_3_{split}.h5"), np.stack([it[1] for it in items]),
            np.stack([it[2] for it in items]), {"images": [it[0] for it in items]},
        )
        info[f"n_{split}"] = len(items)
    return info


def fit_to_size(raw, labels):
    """Crop around the labeled cell or pad with IGNORE_LABEL to SIZE x SIZE."""
    cell = labels != IGNORE_LABEL
    out_raw = np.zeros((SIZE, SIZE), dtype=raw.dtype)
    out_labels = np.full((SIZE, SIZE), IGNORE_LABEL, dtype=labels.dtype)
    src, dst = [], []
    for ax in range(2):
        n = raw.shape[ax]
        if n >= SIZE:
            center = int(np.mean(np.where(cell.any(axis=1 - ax))[0]))
            start = min(max(center - SIZE // 2, 0), n - SIZE)
            src.append(slice(start, start + SIZE))
            dst.append(slice(0, SIZE))
        else:
            src.append(slice(0, n))
            dst.append(slice(0, n))
    out_raw[tuple(dst)] = raw[tuple(src)]
    out_labels[tuple(dst)] = labels[tuple(src)]
    return out_raw, out_labels


def build_tem_datasets(root, out_dir):
    images = {"train": [], "val": []}
    info = {}
    for dataset in TEM_DATASETS:
        folder = f"dataset_{dataset}"
        kept, n_empty, n_dropped, n_resized = [], 0, 0, 0
        for name in sorted_names(os.path.join(root, folder)):
            with h5py.File(os.path.join(root, folder, name), "r") as f:
                raw = f["raw"][:]
                nucleus = f["labels/nucleus"][:] > 0
                cytoplasm = f["labels/cytoplasm"][:] > 0
            if not nucleus.any():
                n_empty += 1
                continue
            if f"{folder}/{name}" in DROPPED_IMAGES:
                n_dropped += 1
                continue
            band = ndimage.binary_dilation(nucleus | cytoplasm, iterations=CELL_BAND)
            labels = instances_2d(nucleus, ignore=~band)
            if raw.shape != (SIZE, SIZE):
                raw, labels = fit_to_size(raw, labels)
                n_resized += 1
            kept.append((f"{folder}/{name}", raw, labels))
        n_val = int(round(len(kept) * VAL_FRACTION))
        images["train"].extend(kept[:len(kept) - n_val])
        images["val"].extend(kept[len(kept) - n_val:])
        info[folder] = {
            "n_without_nucleus": n_empty, "n_dropped": n_dropped, "n_fitted_to_512": n_resized,
            "n_train": len(kept) - n_val, "n_val": n_val,
        }
    for split, items in images.items():
        labels = np.stack([it[2] for it in items])
        write_h5(
            os.path.join(out_dir, f"tem_{split}.h5"), np.stack([it[1] for it in items]), labels,
            {"images": [it[0] for it in items], "ignore_label": int(IGNORE_LABEL)},
        )
        info[f"ignore_fraction_{split}"] = float((labels == IGNORE_LABEL).mean())
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(DATA_ROOT, "waenet"), help="The WAE-Net data folder.")
    parser.add_argument("--output", default=os.path.join(DATA_ROOT, "waenet_nucleus"), help="Output folder.")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    info = {
        "dataset_1": build_dataset_1(args.input, args.output),
        "dataset_3": build_dataset_3(args.input, args.output),
        "tem": build_tem_datasets(args.input, args.output),
        "dropped_images": DROPPED_IMAGES, "cell_band": CELL_BAND, "ignore_label": int(IGNORE_LABEL),
    }
    print(json.dumps(info, indent=2))
    with open(os.path.join(args.output, "info.json"), "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
