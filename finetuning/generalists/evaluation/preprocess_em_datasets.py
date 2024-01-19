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


def for_lucchi(save_dir):
    lucchi_paths = glob(os.path.join(ROOT, "lucchi", "*.h5"))

    os.makedirs(os.path.join(save_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "labels"), exist_ok=True)

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
                    raw_path = os.path.join(save_dir, "raw", f"lucchi_{_split}_{i+1:05}.tif")
                    label_path = os.path.join(save_dir, "labels", f"lucchi_{_split}_{i+1:05}.tif")
                    imageio.imwrite(raw_path, _raw, compression="zlib")
                    imageio.imwrite(label_path, instances, compression="zlib")


def for_snemi(save_dir):
    snemi_vol_path = os.path.join(ROOT, "snemi", "snemi_train.h5")

    # creating the sub-directories
    for _split in ["val", "test"]:
        for sample_type in ["raw", "labels"]:
            os.makedirs(os.path.join(save_dir, _split, sample_type), exist_ok=True)

    with h5py.File(snemi_vol_path, "r") as _file:
        raw = _file["volumes"]["raw"][:]
        labels = _file["volumes"]["labels"]["neuron_ids"][:]

        for i, (_raw, _label) in tqdm(enumerate(zip(raw, labels)), total=raw.shape[0]):
            val_choice = int(raw.shape[0] * 0.2)  # making a 20% val split, rest is test
            split = "val" if i < val_choice else "test"
            # we only save labels with foreground
            if has_foreground(_label):
                instances = label(_label)
                raw_path = os.path.join(save_dir, split, "raw", f"snemi_train_{i+1:05}.tif")
                label_path = os.path.join(save_dir, split, "labels", f"snemi_train_{i+1:05}.tif")
                imageio.imwrite(raw_path, _raw, compression="zlib")
                imageio.imwrite(label_path, instances, compression="zlib")


def for_nuc_mm(save_dir):
    species = ["mouse", "zebrafish"]
    for one_spec in species:
        nuc_mm_vol_paths = glob(os.path.join(ROOT, "nuc_mm", one_spec, "*", "*"))
        print(f"Preprocessing {one_spec}")

        i = 0
        for _one_vol_path in nuc_mm_vol_paths:
            _config = _one_vol_path.split("/")[-2]
            os.makedirs(os.path.join(save_dir, one_spec, "raw"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, one_spec, "labels"), exist_ok=True)

            with h5py.File(_one_vol_path, "r") as _file:
                raw = _file["raw"][:]
                labels = _file["labels"][:]

                for _raw, _label in tqdm(zip(raw, labels), total=raw.shape[0], desc=f"{_one_vol_path}"):
                    # we only save labels with foreground
                    if has_foreground(_label):
                        instances = label(_label)
                        raw_path = os.path.join(save_dir, one_spec, "raw", f"nuc_mm_{_config}_{i+1:05}.tif")
                        label_path = os.path.join(save_dir, one_spec, "labels", f"nuc_mm_{_config}_{i+1:05}.tif")
                        imageio.imwrite(raw_path, _raw, compression="zlib")
                        imageio.imwrite(label_path, instances, compression="zlib")
                        i += 1


def for_platy_cilia(save_dir):
    vol_paths = sorted(glob(os.path.join(ROOT, "platynereis", "cilia", "train_*")))

    os.makedirs(os.path.join(save_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "labels"), exist_ok=True)

    for vol_path in vol_paths:
        vol_id = os.path.split(vol_path)[-1].split(".")[0][-1]
        split = "val" if vol_id == "3" else "test"  # volumes 01 and 02 are for test, 03 for val

        with h5py.File(vol_path, "r") as _file:
            raw = _file["volumes"]["raw"][:]
            labels = _file["volumes"]["labels"]["segmentation"][:]

            for i, (_raw, _label) in tqdm(enumerate(zip(raw, labels)), total=raw.shape[0]):
                # we only save labels with foreground
                if has_foreground(_label):
                    instances = label(_label)
                    raw_path = os.path.join(save_dir, "raw", f"platy_cilia_{split}_{i+1:05}.tif")
                    label_path = os.path.join(save_dir, "labels", f"platy_cilia_{split}_{i+1:05}.tif")
                    imageio.imwrite(raw_path, _raw, compression="zlib")
                    imageio.imwrite(label_path, instances, compression="zlib")


def for_uro_cell(save_dir):
    raise NotImplementedError


def main():
    # let's ensure all the data is downloaded
    download_em_dataset(ROOT)

    dataset_name = "platynereis"

    # paths to save the raw and label slices
    save_dir = os.path.join(ROOT, dataset_name, "slices")

    # now let's save the slices as tif
    for_platy_cilia(save_dir)


if __name__ == "__main__":
    main()
