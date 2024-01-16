import os
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path

import h5py
import imageio.v3 as imageio

from skimage.measure import label

from util import download_em_dataset, ROOT


def has_foreground(label):
    if len(np.unique(label)) > 1:
        return True
    else:
        return False


def for_lucchi():
    lucchi_paths = glob(os.path.join(ROOT, "lucchi", "*.h5"))

    # paths to save the raw and label slices
    raw_dir = os.path.join(ROOT, "lucchi", "slices", "raw")
    label_dir = os.path.join(ROOT, "lucchi", "slices", "labels")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for vol_path in lucchi_paths:
        # using the split name to save the slices
        _split = Path(vol_path).stem.split("_")[-1]

        with h5py.File(vol_path, "r") as _file:
            raw = _file["raw"][:]
            labels = _file["labels"][:]

            for i, (_raw, _label) in tqdm(enumerate(zip(raw, labels)), total=raw.shape[0]):
                # we only save labels with foreground
                if has_foreground(_label):
                    instances = label(_label)
                    raw_path = os.path.join(raw_dir, f"lucchi_{_split}_{i+1:05}.tif")
                    label_path = os.path.join(label_dir, f"lucchi_{_split}_{i+1:05}.tif")
                    imageio.imwrite(raw_path, _raw, compression="zlib")
                    imageio.imwrite(label_path, instances, compression="zlib")


def main():
    # let's ensure all the data is downloaded
    download_em_dataset(ROOT)

    # now let's save the slices as tif
    for_lucchi()


if __name__ == "__main__":
    main()
