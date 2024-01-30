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
    h5_vol_path, raw_key, raw_dir, labels_key, labels_dir,
    slice_prefix_name, do_connected_components=True, interface=h5py, crop_shape=None
):
    with interface.File(h5_vol_path, "r") as f:
        raw = f[raw_key][:]
        labels = f[labels_key][:]

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

    for vol_path in val_vol_paths:
        species = os.path.split(vol_path)[-1].split("_")[0]

        os.makedirs(os.path.join(save_dir, species, "raw"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, species, "labels"), exist_ok=True)

        from_h5_to_tif(
            h5_vol_path=vol_path,
            raw_key="raw",
            raw_dir=os.path.join(save_dir, species, "raw"),
            labels_key="labels",
            labels_dir=os.path.join(save_dir, species, "labels"),
            slice_prefix_name=f"mitoem_{species}",
            interface=z5py,
            crop_shape=desired_shape
        )

        make_custom_splits(val_samples=5, save_dir=os.path.join(save_dir, species))


def for_mitolab(save_dir):
    """
    NOTE: we do not reduce the resolution for the benchmark samples to make a fair comparison with empanada
    for all 7 benckmarks (6 vEM mito datasets, 1 TEM dataset)
    *we get 10% of all samples into val
    - c_elegans (256, 256, 256)
    - fly_brain (256, 255, 255)
    - glycotic_muscle (302, 383, 765)
    - hela_cell (256, 256, 256)
    - lucchi_pp (165, 768, 1024)
    - salivary gland (1260, 1081, 1200) (TODO: take a closer look later)
    - tem:
    """
    all_dataset_ids = []

    # first, we look at the 6 vEM mito datasets
    _roi_vol_paths = sorted(glob(os.path.join(ROOT, "mitolab", "10982", "data", "mito_benchmarks", "*")))
    assert len(_roi_vol_paths) == 6, "The mito datasets have not been downloaded correctly. Please redownload them."

    for vol_path in _roi_vol_paths:
        # let's take a closer look at per dataset
        dataset_id = os.path.split(vol_path)[-1]
        all_dataset_ids.append(dataset_id)

        os.makedirs(os.path.join(save_dir, dataset_id, "raw"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, dataset_id, "labels"), exist_ok=True)

        em_path = glob(os.path.join(vol_path, "*_em.tif"))[0]
        mito_path = glob(os.path.join(vol_path, "*_mito.tif"))[0]

        vem = imageio.imread(em_path)
        vmito = imageio.imread(mito_path)

        for i, (slice_em, slice_mito) in tqdm(
            enumerate(zip(vem, vmito)), total=vem.shape[0], desc=f"Processing {dataset_id}"
        ):
            if has_foreground(slice_mito):
                instances = label(slice_mito)

                raw_path = os.path.join(save_dir, dataset_id, "raw", f"{dataset_id}_{i+1:05}.tif")
                labels_path = os.path.join(save_dir, dataset_id, "labels", f"{dataset_id}_{i+1:05}.tif")

                imageio.imwrite(raw_path, slice_em, compression="zlib")
                imageio.imwrite(labels_path, instances, compression="zlib")

    # now, let's work on the tem dataset
    image_paths = sorted(glob(os.path.join(ROOT, "mitolab", "10982", "data", "tem_benchmark", "images", "*")))
    mask_paths = sorted(glob(os.path.join(ROOT, "mitolab", "10982", "data", "tem_benchmark", "masks", "*")))

    os.makedirs(os.path.join(save_dir, "tem", "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "tem", "labels"), exist_ok=True)

    # let's move the tem data to slices
    for image_path, mask_path in tqdm(zip(image_paths, mask_paths), desc="Preprocessimg tem", total=len(image_paths)):
        sample_id = os.path.split(image_path)[-1]
        image_dst = os.path.join(save_dir, "tem", "raw", sample_id)
        mask_dst = os.path.join(save_dir, "tem", "labels", sample_id)

        shutil.copy(image_path, image_dst)
        shutil.copy(mask_path, mask_dst)

    all_dataset_ids.append("tem")

    for dataset_id in all_dataset_ids:
        make_custom_splits(val_samples=10, save_dir=os.path.join(save_dir, dataset_id))


def for_uro_cell(save_dir):
    """
    """
    os.makedirs(os.path.join(save_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "labels"), exist_ok=True)

    # only the 4 volumes below have mito labels
    all_vol_with_labels = ["fib1-0-0-0.h5", "fib1-1-0-3.h5", "fib1-3-2-1.h5", "fib1-3-3-0.h5", "fib1-4-3-0.h5"]

    for one_vol_with_labels in all_vol_with_labels:
        vol_path = os.path.join(ROOT, "uro_cell", one_vol_with_labels)
        vol_id = Path(vol_path).stem

        from_h5_to_tif(
            h5_vol_path=vol_path,
            raw_key="raw",
            raw_dir=os.path.join(save_dir, "raw"),
            labels_key="labels/mito",
            labels_dir=os.path.join(save_dir, "labels"),
            slice_prefix_name=f"uro_cell_{vol_id}"
        )

    make_custom_splits(val_samples=100, save_dir=save_dir)


def for_sponge_em(save_dir):
    """
    using all train volumes
    we take only 10 samples for validation, rest for test
    """
    vol_paths = sorted(glob(os.path.join(ROOT, "sponge_em", "*.h5")))

    os.makedirs(os.path.join(save_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "labels"), exist_ok=True)

    for vol_path in vol_paths:
        vol_id = Path(vol_path).stem

        from_h5_to_tif(
            h5_vol_path=vol_path,
            raw_key="volumes/raw",
            raw_dir=os.path.join(save_dir, "raw"),
            labels_key="volumes/labels/instances",
            labels_dir=os.path.join(save_dir, "labels"),
            slice_prefix_name=f"sponge_em_{vol_id}"
        )

    make_custom_splits(val_samples=10, save_dir=save_dir)


# TODO:
# CREMI, ISBI, AxonDeepSeg?, Platy Cells, VNC?


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
