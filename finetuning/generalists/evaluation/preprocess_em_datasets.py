import os
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path

import h5py
import z5py
import imageio.v3 as imageio

from skimage.measure import label

from util import download_em_dataset, ROOT


def has_foreground(label):
    if len(np.unique(label)) > 1:
        return True
    else:
        return False


def for_lucchi(save_dir):
    """
    for val: we take the train volume
    for test: we take the test volume
    """
    lucchi_paths = glob(os.path.join(ROOT, "lucchi", "*.h5"))

    os.makedirs(os.path.join(save_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "labels"), exist_ok=True)

    for vol_path in lucchi_paths:
        # using the split name to save the slices
        _split = Path(vol_path).stem.split("_")[-1]

        with h5py.File(vol_path, "r") as f:
            raw = f["raw"][:]
            labels = f["labels"][:]

            for i, (_raw, _label) in tqdm(enumerate(zip(raw, labels)), total=raw.shape[0]):
                # we only save labels with foreground
                if has_foreground(_label):
                    instances = label(_label)
                    raw_path = os.path.join(save_dir, "raw", f"lucchi_{_split}_{i+1:05}.tif")
                    label_path = os.path.join(save_dir, "labels", f"lucchi_{_split}_{i+1:05}.tif")
                    imageio.imwrite(raw_path, _raw, compression="zlib")
                    imageio.imwrite(label_path, instances, compression="zlib")


def for_snemi(save_dir):
    """
    for validation: we make a 20% split, and take the first 20% slices from the train volume
    for testing: we take the rest 80% slices from the train volume
    """
    snemi_vol_path = os.path.join(ROOT, "snemi", "snemi_train.h5")

    # creating the sub-directories
    for _split in ["val", "test"]:
        for sample_type in ["raw", "labels"]:
            os.makedirs(os.path.join(save_dir, _split, sample_type), exist_ok=True)

    with h5py.File(snemi_vol_path, "r") as f:
        raw = f["volumes"]["raw"][:]
        labels = f["volumes"]["labels"]["neuron_ids"][:]

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
    """
    (for both mouse and zebrafish)
    for validation: we use the `val` volumes for validation
    for test: we use the `train` volumes for testing
    """
    species = ["mouse", "zebrafish"]
    for one_spec in species:
        nuc_mm_vol_paths = glob(os.path.join(ROOT, "nuc_mm", one_spec, "*", "*"))
        print(f"Preprocessing {one_spec}")

        i = 0
        for _one_vol_path in nuc_mm_vol_paths:
            _config = _one_vol_path.split("/")[-2]
            os.makedirs(os.path.join(save_dir, one_spec, "raw"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, one_spec, "labels"), exist_ok=True)

            with h5py.File(_one_vol_path, "r") as f:
                raw = f["raw"][:]
                labels = f["labels"][:]

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
    """
    the training volumes have labels only
    for validation: we take volume 03
    for test: we take volume 01 and 02
    """
    vol_paths = sorted(glob(os.path.join(ROOT, "platynereis", "cilia", "train_*")))

    os.makedirs(os.path.join(save_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "labels"), exist_ok=True)

    for vol_path in vol_paths:
        vol_id = os.path.split(vol_path)[-1].split(".")[0][-1]
        split = "val" if vol_id == "3" else "test"  # volumes 01 and 02 are for test, 03 for val

        with h5py.File(vol_path, "r") as f:
            raw = f["volumes"]["raw"][:]
            labels = f["volumes"]["labels"]["segmentation"][:]

            for i, (_raw, _label) in tqdm(enumerate(zip(raw, labels)), total=raw.shape[0]):
                # we only save labels with foreground
                if has_foreground(_label):
                    instances = label(_label)
                    raw_path = os.path.join(save_dir, "cilia", "raw", f"platy_cilia_{split}_{i+1:05}.tif")
                    label_path = os.path.join(save_dir, "cilia", "labels", f"platy_cilia_{split}_{i+1:05}.tif")
                    imageio.imwrite(raw_path, _raw, compression="zlib")
                    imageio.imwrite(label_path, instances, compression="zlib")


def make_center_crop(image, desired_shape):
    center_coords = (int(image.shape[0] / 2), int(image.shape[1] / 2))
    tolerances = (int(desired_shape[0] / 2), int(desired_shape[1] / 2))

    # let's take the center crop from the image
    cropped_image = image[
        center_coords[0] - tolerances[0]: center_coords[0] + tolerances[0],
        center_coords[1] - tolerances[1]: center_coords[1] + tolerances[1]
    ]
    return cropped_image


def for_mitoem(save_dir, desired_shape=(768, 768)):
    """
    (for both rat and human)
    for validation: we take the first 10 slices (while training, we only use the first 5 slices)
    for test: we take the next 90 slices in the `val` volume
    """
    val_vol_paths = sorted(glob(os.path.join(ROOT, "mitoem", "*_val.n5")))

    os.makedirs(os.path.join(save_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "labels"), exist_ok=True)

    for vol_path in val_vol_paths:
        species = os.path.split(vol_path)[-1].split("_")[0]

        with z5py.File(vol_path, "r") as f:
            raw = f["raw"][:]
            labels = f["labels"][:]
            for i, (_raw, _label) in tqdm(enumerate(zip(raw, labels)), total=raw.shape[0]):
                split = "val" if i < 10 else "test"

                _raw = make_center_crop(_raw, desired_shape)
                _label = make_center_crop(_label, desired_shape)

                # we only save labels with foreground
                if has_foreground(_label):
                    instances = label(_label)
                    raw_path = os.path.join(save_dir, "raw", f"mitoem_{species}_{split}_{i+1:05}.tif")
                    label_path = os.path.join(save_dir, "labels", f"mitoem_{species}_{split}_{i+1:05}.tif")
                    imageio.imwrite(raw_path, _raw, compression="zlib")
                    imageio.imwrite(label_path, instances, compression="zlib")


def main():
    # let's ensure all the data is downloaded
    download_em_dataset(ROOT)

    dataset_name = "mitoem"

    # paths to save the raw and label slices
    save_dir = os.path.join(ROOT, dataset_name, "slices")

    # now let's save the slices as tif
    for_mitoem(save_dir)


if __name__ == "__main__":
    main()
