"""Stack the SNEMI Mito slices into one volume for 3D training.

The 100 slices of the SNEMI Mito dataset are consecutive sections of the SNEMI3D volume (6 x 6 x 30 nm). They are
split into contiguous ranges: train 0-49, valid 50-69, test 70-99. They are stacked into one h5 file with 'raw' and
binary 'labels' (the masks are stored as 0 / 255). Thus, 3D patches can come from the train and valid ranges. The test
range stays untouched by training, which reads the train and valid ranges only. Nine label slices are stored as RGBA
PNGs (gray values in every color channel), so the script uses the first channel.
"""
import os
import argparse
from glob import glob

import h5py
import numpy as np
import imageio.v3 as imageio

from figure_utils import DATA_ROOT


SPLITS = ("train", "valid", "test")


def read_slice(path):
    """One slice; a few slices are stored as RGBA PNGs with the gray values in every color channel."""
    image = imageio.imread(path)
    return image[..., 0] if image.ndim == 3 else image


def read_split(root, split, folder):
    paths = sorted(glob(os.path.join(root, split, folder, "*.png")))
    return [os.path.basename(p) for p in paths], np.stack([read_slice(p) for p in paths])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(DATA_ROOT, "snemi_mito_stack"), help="Output folder.")
    args = parser.parse_args()
    root = os.path.join(DATA_ROOT, "snemi_mito", "SNEMI3D_mito")
    names, raws, labels = [], [], []
    for split in SPLITS:
        split_names, raw = read_split(root, split, "images")
        label_names, label = read_split(root, split, "labels")
        assert split_names == label_names, split
        names += split_names
        raws.append(raw)
        labels.append((label > 0).astype("uint8"))
    assert names == [f"{i:04}.png" for i in range(len(names))], "the slices are not consecutive"
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "snemi_mito.h5")
    tmp_path = f"{out_path}.incomplete"
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("raw", data=np.concatenate(raws), compression="gzip", chunks=(8, 256, 256))
        f.create_dataset("labels", data=np.concatenate(labels), compression="gzip", chunks=(8, 256, 256))
        f.attrs["splits"] = "train 0-49, valid 50-69, test 70-99"
        f.attrs["voxel_size_nm"] = [30, 6, 6]
    os.replace(tmp_path, out_path)
    print(out_path, len(names), "slices")


if __name__ == "__main__":
    main()
