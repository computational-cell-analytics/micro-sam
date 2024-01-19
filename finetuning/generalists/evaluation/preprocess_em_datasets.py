import os
import shutil
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


def make_center_crop(image, desired_shape):
    center_coords = (int(image.shape[0] / 2), int(image.shape[1] / 2))
    tolerances = (int(desired_shape[0] / 2), int(desired_shape[1] / 2))

    # let's take the center crop from the image
    cropped_image = image[
        center_coords[0] - tolerances[0]: center_coords[0] + tolerances[0],
        center_coords[1] - tolerances[1]: center_coords[1] + tolerances[1]
    ]
    return cropped_image


def make_custom_splits(val_samples, save_dir):
    # make an external splitting logic
    # 1. move to val dir
    os.makedirs(os.path.join(save_dir, "val", "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "val", "labels"), exist_ok=True)

    for raw_path, label_path in (
        zip(
            sorted(glob(os.path.join(save_dir, "raw", "*")))[:val_samples],
            sorted(glob(os.path.join(save_dir, "labels", "*")))[:val_samples]
        )
    ):
        # let's move the raw slice
        slice_id = os.path.split(raw_path)[-1]
        dst = os.path.join(save_dir, "val", "raw", slice_id)
        shutil.move(raw_path, dst)

        # let's move the label slice
        slice_id = os.path.split(label_path)[-1]
        dst = os.path.join(save_dir, "val", "labels", slice_id)
        shutil.move(label_path, dst)

    # 2. move to test dir
    os.makedirs(os.path.join(save_dir, "test", "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "test", "labels"), exist_ok=True)

    for raw_path, label_path in (
        zip(sorted(glob(os.path.join(save_dir, "raw", "*"))), sorted(glob(os.path.join(save_dir, "labels", "*"))))
    ):
        # let's move the raw slice
        slice_id = os.path.split(raw_path)[-1]
        dst = os.path.join(save_dir, "test", "raw", slice_id)
        shutil.move(raw_path, dst)

        # let's move the label slice
        slice_id = os.path.split(label_path)[-1]
        dst = os.path.join(save_dir, "test", "labels", slice_id)
        shutil.move(label_path, dst)

    # let's remove the left-overs
    shutil.rmtree(os.path.join(save_dir, "raw"))
    shutil.rmtree(os.path.join(save_dir, "labels"))


def from_h5_to_tif(
    h5_vol_path, raw_key, raw_dir, labels_key, labels_dir,
    slice_prefix_name, do_connected_components=True, interface=h5py, crop_shape=None
):
    with interface.File(h5_vol_path, "r") as f:
        raw = f[raw_key][:]
        labels = f[labels_key][:]

        for i, (_raw, _label) in tqdm(enumerate(zip(raw, labels)), total=raw.shape[0]):
            if crop_shape is not None:
                _raw = make_center_crop(_raw, crop_shape)
                _label = make_center_crop(_label, crop_shape)

            # we only save labels with foreground
            if has_foreground(_label):
                if do_connected_components:
                    instances = label(_label)
                else:
                    instances = _label

                raw_path = os.path.join(raw_dir, f"{slice_prefix_name}_{i+1:05}.tif")
                labels_path = os.path.join(labels_dir, f"{slice_prefix_name}_{i+1:05}.tif")

                imageio.imwrite(raw_path, _raw, compression="zlib")
                imageio.imwrite(labels_path, instances, compression="zlib")


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

        from_h5_to_tif(
            h5_vol_path=vol_path,
            raw_key="raw",
            raw_dir=os.path.join(save_dir, "raw"),
            labels_key="labels",
            labels_dir=os.path.join(save_dir, "labels"),
            slice_prefix_name=f"lucchi_{_split}"
        )


def for_snemi(save_dir):
    """
    for validation: we make a 20% split, and take the first 20% slices from the train volume
    for testing: we take the rest 80% slices from the train volume
    """
    snemi_vol_path = os.path.join(ROOT, "snemi", "snemi_train.h5")

    os.makedirs(os.path.join(save_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "labels"), exist_ok=True)

    from_h5_to_tif(
        h5_vol_path=snemi_vol_path,
        raw_key="volumes/raw",
        raw_dir=os.path.join(save_dir, "raw",),
        labels_key="volumes/labels/neuron_ids",
        labels_dir=os.path.join(save_dir, "labels"),
        slice_prefix_name="snemi_train"
    )

    make_custom_splits(val_samples=20, save_dir=save_dir)


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

        for _one_vol_path in nuc_mm_vol_paths:
            _config = _one_vol_path.split("/")[-2]
            _vol_id = Path(_one_vol_path).stem

            os.makedirs(os.path.join(save_dir, one_spec, "raw"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, one_spec, "labels"), exist_ok=True)

            from_h5_to_tif(
                h5_vol_path=_one_vol_path,
                raw_key="raw",
                raw_dir=os.path.join(save_dir, one_spec, "raw"),
                labels_key="labels",
                labels_dir=os.path.join(save_dir, one_spec, "labels"),
                slice_prefix_name=f"nuc_mm_{_config}_{_vol_id}"
            )


def for_platynereis(save_dir, choice):
    """
    for cilia:
        the training volumes have labels only
        for validation: we take volume 03
        for test: we take volume 01 and 02
    for nuclei:
        for validation: we take volume 01
        for testing: we take volume [02-12]
    """
    os.makedirs(os.path.join(save_dir, choice, "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, choice, "labels"), exist_ok=True)

    if choice == "cilia":
        vol_paths = sorted(glob(os.path.join(ROOT, "platynereis", "cilia", "train_*")))

        for vol_path in vol_paths:
            vol_id = os.path.split(vol_path)[-1].split(".")[0][-2:]
            split = "val" if vol_id == "03" else "test"  # volumes 01 and 02 are for test, 03 for val

            from_h5_to_tif(
                h5_vol_path=vol_path,
                raw_key="volumes/raw",
                raw_dir=os.path.join(save_dir, "cilia", "raw"),
                labels_key="volumes/labels/segmentation",
                labels_dir=os.path.join(save_dir, "cilia", "labels"),
                slice_prefix_name=f"platy_cilia_{vol_id}_{split}"
            )

    elif choice == "nuclei":
        vol_paths = sorted(glob(os.path.join(ROOT, "platynereis", "nuclei", "*")))

        for vol_path in vol_paths:
            vol_id = os.path.split(vol_path)[-1].split(".")[0][-2:]
            split = "val" if vol_id == "01" else "test"  # volumes 01 for val, rest for test

            from_h5_to_tif(
                h5_vol_path=vol_path,
                raw_key="volumes/raw",
                raw_dir=os.path.join(save_dir, "nuclei", "raw"),
                labels_key="volumes/labels/nucleus_instance_labels",
                labels_dir=os.path.join(save_dir, "nuclei", "labels"),
                slice_prefix_name=f"platy_nuclei_{vol_id}_{split}"
            )

    else:
        raise ValueError("Choose from `nuclei` or `cilia`")


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

        from_h5_to_tif(
            h5_vol_path=vol_path,
            raw_key="raw",
            raw_dir=os.path.join(save_dir, "raw"),
            labels_key="labels",
            labels_dir=os.path.join(save_dir, "labels"),
            slice_prefix_name=f"mitoem_{species}",
            interface=z5py,
            crop_shape=desired_shape
        )

    make_custom_splits(val_samples=10, save_dir=save_dir)


def main():
    # let's ensure all the data is downloaded
    download_em_dataset(ROOT)

    dataset_name = "nuc_mm"

    # paths to save the raw and label slices
    save_dir = os.path.join(ROOT, dataset_name, "slices")

    # now let's save the slices as tif
    for_nuc_mm(save_dir)


if __name__ == "__main__":
    main()
