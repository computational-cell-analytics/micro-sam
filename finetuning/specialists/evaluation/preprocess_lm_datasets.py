import os
import z5py
import h5py
import shutil
import numpy as np
from tqdm import tqdm
from glob import glob
from pathlib import Path

import imageio.v3 as imageio
from skimage.measure import label

from torch_em.transform.raw import normalize, normalize_percentile

from util import ROOT, download_lm_dataset


def has_foreground(label):
    if len(np.unique(label)) > 1:
        return True
    else:
        return False


def make_center_crop(image, desired_shape):
    if image.shape > desired_shape:
        return image

    center_coords = (int(image.shape[0] / 2), int(image.shape[1] / 2))
    tolerances = (int(desired_shape[0] / 2), int(desired_shape[1] / 2))

    # let's take the center crop from the image
    cropped_image = image[
        center_coords[0] - tolerances[0]: center_coords[0] + tolerances[0],
        center_coords[1] - tolerances[1]: center_coords[1] + tolerances[1]
    ]
    return cropped_image


def convert_rgb(raw):
    raw = normalize_percentile(raw, axis=(1, 2))
    raw = np.mean(raw, axis=0)
    raw = normalize(raw)
    raw = raw * 255
    return raw


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


def save_to_tif(i, _raw, _label, crop_shape, raw_dir, labels_dir, do_connected_components, slice_prefix_name):
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


def from_h5_to_tif(
    h5_vol_path, raw_key, raw_dir, labels_key, labels_dir, slice_prefix_name,
    do_connected_components=True, interface=h5py, crop_shape=None, to_one_channel=False
):
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    if h5_vol_path.split(".")[-1] == "zarr":
        kwargs = {"use_zarr_format": True}
    else:
        kwargs = {}

    with interface.File(h5_vol_path, "r", **kwargs) as f:
        raw = f[raw_key][:]
        labels = f[labels_key][:]

        if raw.ndim == 3 and raw.shape[0] == 3 and to_one_channel:  # the case in tissuenet
            print("Got an RGB image, converting it to one-channel.")
            raw = convert_rgb(raw)

        if raw.ndim == 2 and labels.ndim == 2:
            raw, labels = raw[None], labels[None]

        if raw.ndim == 3 and labels.ndim == 3:  # when we have a volume or mono-channel slice
            for i, (_raw, _label) in tqdm(enumerate(zip(raw, labels)), total=raw.shape[0], desc=h5_vol_path):
                save_to_tif(
                    i, _raw, _label, crop_shape, raw_dir, labels_dir, do_connected_components, slice_prefix_name
                )

        elif raw.ndim == 3 and labels.ndim == 2:  # when we have a multi-channel input (rgb)
            if raw.shape[0] == 4:  # hpa has 4 channel inputs (0: microtubules, 1: protein, 2: nuclei, 3: er)
                raw = raw[1:]

            save_to_tif(0, raw, labels, crop_shape, raw_dir, labels_dir, do_connected_components, slice_prefix_name)


def for_covid_if(save_dir):
    """
    the first 5 samples as val, rest for test (all in org. resolution)
    """
    covid_if_paths = sorted(glob(os.path.join(ROOT, "covid_if", "*.h5")))

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
    """
    all test volumes to the val set, 100 volumes from val to val slices (all in org. resolution)
    """
    tissuenet_val_paths = sorted(glob(os.path.join(ROOT, "tissuenet", "val", "*.zarr")))
    tissuenet_test_paths = sorted(glob(os.path.join(ROOT, "tissuenet", "test", "*.zarr")))

    def save_slices_per_split(all_vol_paths, split):
        for vol_path in all_vol_paths:
            vol_id = Path(vol_path).stem

            from_h5_to_tif(
                h5_vol_path=vol_path,
                raw_key="raw/rgb",
                raw_dir=os.path.join(save_dir, split, "raw"),
                labels_key="labels/cell",
                labels_dir=os.path.join(save_dir, split, "labels"),
                slice_prefix_name=f"tissuenet_{split}_{vol_id}",
                interface=z5py,
                to_one_channel=True
            )

    save_slices_per_split(tissuenet_val_paths[:100], "val")
    save_slices_per_split(tissuenet_test_paths, "test")


def for_plantseg(save_dir):
    """
    (for both `ovules` and `root`)
    for validation: one volume's all slices from `val`
    for testing: all volume's all slices from `test`
    """
    def do_slicing(_choice):
        plantseg_val_vols = sorted(glob(os.path.join(ROOT, "plantseg", f"{_choice}_val", "*.h5")))
        plantseg_test_vols = sorted(glob(os.path.join(ROOT, "plantseg", f"{_choice}_test", "*.h5")))

        def save_slices_per_split(all_vol_paths, split):
            for vol_path in all_vol_paths:
                vol_id = Path(vol_path).stem

                from_h5_to_tif(
                    h5_vol_path=vol_path,
                    raw_key="raw",
                    raw_dir=os.path.join(save_dir, _choice, split, "raw"),
                    labels_key="label",
                    labels_dir=os.path.join(save_dir, _choice, split, "labels"),
                    slice_prefix_name=f"plantseg_{_choice}_{split}_{vol_id}"
                )

        save_slices_per_split([plantseg_val_vols[0]], "val")
        save_slices_per_split(plantseg_test_vols, "test")

    do_slicing("root")
    do_slicing("ovules")


def for_hpa(save_dir):
    """
    take all train volumes for test inference, and all val volumes for validation
    """
    hpa_val_vols = sorted(glob(os.path.join(ROOT, "hpa", "val", "*.h5")))
    hpa_test_vols = sorted(glob(os.path.join(ROOT, "hpa", "train", "*.h5")))

    def save_slices_per_split(all_vol_paths, split):
        for vol_path in all_vol_paths:
            vol_id = Path(vol_path).stem

            from_h5_to_tif(
                h5_vol_path=vol_path,
                raw_key="raw",
                raw_dir=os.path.join(save_dir, split, "raw"),
                labels_key="labels",
                labels_dir=os.path.join(save_dir, split, "labels"),
                slice_prefix_name=f"hpa_{split}_{vol_id}",
                crop_shape=(768, 768)
            )

    save_slices_per_split(hpa_val_vols, "val")
    save_slices_per_split(hpa_test_vols, "test")


def for_lizard(save_dir):
    """
    for validation: first 10 slices
    for testing: rest slices
    """
    lizard_vols = sorted(glob(os.path.join(ROOT, "lizard", "*.h5")))

    for vol_path in lizard_vols:
        vol_id = Path(vol_path).stem

        from_h5_to_tif(
            h5_vol_path=vol_path,
            raw_key="image",
            raw_dir=os.path.join(save_dir, "raw"),
            labels_key="labels/segmentation",
            labels_dir=os.path.join(save_dir, "labels"),
            slice_prefix_name=f"lizard_{vol_id}"
        )

    make_custom_splits(10, save_dir)


def for_mouse_embryo(save_dir):
    """
    for validation: one volume's all slices from `test`
    for testing: all volume's all slices from `train`
    """
    mouse_embryo_val_vols = sorted(glob(os.path.join(ROOT, "mouse-embryo", "Nuclei", "test", "*.h5")))
    mouse_embryo_test_vols = sorted(glob(os.path.join(ROOT, "mouse-embryo", "Nuclei", "train", "*.h5")))

    def save_slices_per_split(all_vols, split):
        for vol_path in all_vols:
            vol_id = Path(vol_path).stem

            from_h5_to_tif(
                h5_vol_path=vol_path,
                raw_key="raw",
                raw_dir=os.path.join(save_dir, split, "raw"),
                labels_key="label",
                labels_dir=os.path.join(save_dir, split, "labels"),
                slice_prefix_name=f"mouse_embryo_{split}_{vol_id}"
            )

    save_slices_per_split([mouse_embryo_val_vols[0]], "val")
    save_slices_per_split(mouse_embryo_test_vols, "test")


def for_ctc(save_dir):
    """
    We only infer on DIC-HeLa for now.
    for validation: first 10 slices from `train`
    for testing: rest slices from `train`

    TODO: add all datasets later for inference, ideally get dataloaders in torch-em for all of them.
    """
    all_hela_image_paths = sorted(
        glob(os.path.join(ROOT, "ctc", "hela_samples", "DIC-C2DH-HeLa.zip.unzip", "DIC-C2DH-HeLa", "01", "*"))
    )
    all_hela_label_paths = sorted(
        glob(os.path.join(ROOT, "ctc", "hela_samples", "hela-ctc-01-gt.zip.unzip", "masks", "*"))
    )

    os.makedirs(os.path.join(save_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "labels"), exist_ok=True)

    for image_path, label_path in tqdm(
        zip(all_hela_image_paths, all_hela_label_paths), total=len(all_hela_image_paths)
    ):
        image_id = os.path.split(image_path)[-1]
        label_id = os.path.split(label_path)[-1]

        dst_image_path = os.path.join(save_dir, "raw", image_id)
        dst_label_path = os.path.join(save_dir, "labels", label_id)

        shutil.copy(image_path, dst_image_path)
        shutil.copy(label_path, dst_label_path)

    make_custom_splits(10, save_dir)


def for_neurips_cellseg(save_dir, use_tuning_set=False):
    """
    we infer on the `TuningSet` - the data for open-evaluation on grand-challenge

    val set info is here: /home/nimanwai/torch-em/torch_em/data/datasets/split_0.1.json

    for validation: use the true val set used
    for testing: use the `TuningSet` and `TestForSam`
    """
    # let's get the val slices
    from torch_em.data.datasets.neurips_cell_seg import _get_image_and_label_paths
    val_image_paths, val_label_paths = _get_image_and_label_paths(
        os.path.join(ROOT, "neurips-cell-seg"), split="val", val_fraction=0.1
    )

    os.makedirs(os.path.join(save_dir, "val", "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "val", "labels"), exist_ok=True)

    for image_path, label_path in tqdm(zip(val_label_paths, val_label_paths), total=len(val_image_paths)):
        image_id = os.path.split(image_path)[-1]
        label_id = os.path.split(label_path)[-1]

        dst_image_path = os.path.join(save_dir, "val", "raw", image_id)
        dst_label_path = os.path.join(save_dir, "val", "labels", label_id)

        shutil.copy(image_path, dst_image_path)
        shutil.copy(label_path, dst_label_path)

    # now, let's get the test slices
    test_image_paths = sorted(glob(os.path.join(ROOT, "neurips-cell-seg", "TestForSam", "images", "*")))
    test_label_paths = sorted(glob(os.path.join(ROOT, "neurips-cell-seg", "TestForSam", "labels", "*")))

    if use_tuning_set:
        test_image_paths.extend(
            sorted(glob(os.path.join(ROOT, "neurips-cell-seg", "new", "Tuning", "images", "*")))
        )
        test_label_paths.extend(
            sorted(glob(os.path.join(ROOT, "neurips-cell-seg", "new", "Tuning", "labels", "*")))
        )

    os.makedirs(os.path.join(save_dir, "test", "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "test", "lanels"), exist_ok=True)

    for image_path, label_path in tqdm(zip(test_image_paths, test_label_paths), total=len(test_image_paths)):
        image_id = os.path.split(image_path)[-1]
        label_id = os.path.split(label_path)[-1]

        dst_image_path = os.path.join(save_dir, "val", "raw", image_id)
        dst_label_path = os.path.join(save_dir, "val", "labels", label_id)

        shutil.copy(image_path, dst_image_path)
        shutil.copy(label_path, dst_label_path)


def main():
    # let's ensure all the data is downloaded
    download_lm_dataset(ROOT)

    # name of the dataset - to get the volumes and save the slices
    dataset_name = "neurips-cell-seg"

    # now let's save the slices as tif
    for_neurips_cellseg(os.path.join(ROOT, dataset_name, "slices"))


if __name__ == "__main__":
    main()
