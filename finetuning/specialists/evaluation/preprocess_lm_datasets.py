import os
import h5py
import shutil
import numpy as np
from tqdm import tqdm
from glob import glob
from pathlib import Path

import imageio.v3 as imageio
from skimage.measure import label

from util import ROOT, download_lm_dataset


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
    def move_samples(split, all_raw_files, all_label_files):
        for raw_path, label_path in (zip(all_raw_files, all_label_files)):
            # let's move the raw slice
            slice_id = os.path.split(raw_path)[-1]
            dst = os.path.join(save_dir, split, "raw", slice_id)
            shutil.move(raw_path, dst)

            # let's move the label slice
            slice_id = os.path.split(label_path)[-1]
            dst = os.path.join(save_dir, split, "labels", slice_id)
            shutil.move(label_path, dst)

    # make a custom splitting logic
    # 1. move to val dir
    os.makedirs(os.path.join(save_dir, "val", "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "val", "labels"), exist_ok=True)

    move_samples(
        split="val",
        all_raw_files=sorted(glob(os.path.join(save_dir, "raw", "*")))[:val_samples],
        all_label_files=sorted(glob(os.path.join(save_dir, "labels", "*")))[:val_samples]
    )

    # 2. move to test dir
    os.makedirs(os.path.join(save_dir, "test", "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "test", "labels"), exist_ok=True)

    move_samples(
        split="test",
        all_raw_files=sorted(glob(os.path.join(save_dir, "raw", "*"))),
        all_label_files=sorted(glob(os.path.join(save_dir, "labels", "*")))
    )

    # let's remove the left-overs
    shutil.rmtree(os.path.join(save_dir, "raw"))
    shutil.rmtree(os.path.join(save_dir, "labels"))


def from_h5_to_tif(
    h5_vol_path, raw_key, raw_dir, labels_key, labels_dir, slice_prefix_name,
    do_connected_components=True, interface=h5py, crop_shape=None
):
    with interface.File(h5_vol_path, "r") as f:
        raw = f[raw_key][:]
        labels = f[labels_key][:]

        if raw.ndim == 2 and labels.ndim == 2:
            raw, labels = raw[None], labels[None]

        assert raw.ndim == 3 and labels.ndim == 3

        for i, (_raw, _label) in tqdm(enumerate(zip(raw, labels)), total=raw.shape[0], desc=h5_vol_path):
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


def for_covid_if(save_dir):
    """
    the first 5 samples as val, rest for test
    """
    covid_if_paths = sorted(glob(os.path.join(ROOT, "covid_if", "*.h5")))

    os.makedirs(os.path.join(save_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "labels"), exist_ok=True)

    for vol_path in covid_if_paths:
        # using the split name to save the slices
        _split = Path(vol_path).stem

        from_h5_to_tif(
            h5_vol_path=vol_path,
            raw_key="raw/serum_IgG/s0",
            raw_dir=os.path.join(save_dir, "raw"),
            labels_key="labels/cells/s0",
            labels_dir=os.path.join(save_dir, "labels"),
            slice_prefix_name=f"covid_if_{_split}"
        )

    make_custom_splits(val_samples=5, save_dir=save_dir)


def for_tissuenet(save_dir):
    pass


def for_deepbacs(save_dir):
    pass


def for_plantseg_root(save_dir):
    pass


def for_hpa(save_dir):
    pass


def for_lizard(save_dir):
    pass


def for_mouse_embryo(save_dir):
    pass


def plantseg_ovules(save_dir):
    pass


# TODO:
# checkout CTC datasets (HeLa and potentially others as well)
# checkout NeurIPS CellSeg val set


def main():
    # let's ensure all the data is downloaded
    download_lm_dataset(ROOT)

    # name of the dataset - to get the volumes and save the slices
    dataset_name = "covid_if"

    # paths to save the raw and label slices
    save_dir = os.path.join(ROOT, dataset_name, "slices")

    # now let's save the slices as tif
    for_covid_if(save_dir)


if __name__ == "__main__":
    main()
