import os
import shutil
from glob import glob
from tqdm import tqdm
from pathlib import Path

import h5py
import z5py
import numpy as np
import imageio.v3 as imageio
from scipy.ndimage import binary_closing
from skimage.measure import label as connected_components

from elf.wrapper import RoiWrapper

from torch_em.data import datasets
from torch_em.data import MinForegroundSampler
from torch_em.util.segmentation import size_filter
from torch_em.transform.raw import normalize, normalize_percentile

from micro_sam.training import identity

from util import download_all_datasets, ROOT


def convert_rgb(raw):
    raw = normalize_percentile(raw, axis=(1, 2))
    raw = np.mean(raw, axis=0)
    raw = normalize(raw)
    raw = raw * 255
    return raw


def has_foreground(label):
    if len(np.unique(label)) > 1:
        return True
    else:
        return False


def make_center_crop(image, desired_shape):
    if image.shape < desired_shape:
        return image

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


def save_to_tif(i, _raw, _label, crop_shape, raw_dir, labels_dir, do_connected_components, slice_prefix_name):
    if crop_shape is not None:
        _raw = make_center_crop(_raw, crop_shape)
        _label = make_center_crop(_label, crop_shape)

    # we only save labels with foreground
    if has_foreground(_label):
        if do_connected_components:
            instances = connected_components(_label)
        else:
            instances = _label

        raw_path = os.path.join(raw_dir, f"{slice_prefix_name}_{i+1:05}.tif")
        labels_path = os.path.join(labels_dir, f"{slice_prefix_name}_{i+1:05}.tif")

        imageio.imwrite(raw_path, _raw, compression="zlib")
        imageio.imwrite(labels_path, instances, compression="zlib")


def from_h5_to_tif(
    h5_vol_path, raw_key, raw_dir, labels_key, labels_dir, slice_prefix_name, do_connected_components=True,
    interface=h5py, crop_shape=None, roi_slices=None, to_one_channel=False
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

        if raw.ndim == 3 and raw.shape[0] == 3 and to_one_channel:  # for tissuenet
            print("Got an RGB image, converting it to one-channel.")
            raw = convert_rgb(raw)

        if roi_slices is not None:  # for cremi
            raw = RoiWrapper(raw, roi_slices)[:]
            labels = RoiWrapper(labels, roi_slices)[:]

        if raw.ndim == 2 and labels.ndim == 2:  # for axondeepseg tem modality
            raw, labels = raw[None], labels[None]

        if raw.ndim == 3 and labels.ndim == 3:  # when we have a volume or mono-channel image
            for i, (_raw, _label) in tqdm(enumerate(zip(raw, labels)), total=raw.shape[0], desc=h5_vol_path):
                save_to_tif(
                    i, _raw, _label, crop_shape, raw_dir, labels_dir, do_connected_components, slice_prefix_name
                )

        elif raw.ndim == 3 and labels.ndim == 2:  # when we have a multi-channel input (rgb)
            if raw.shape[0] == 4:  # hpa has 4 channel inputs (0: microtubules, 1: protein, 2: nuclei, 3: er)
                raw = raw[1:]

            # making channels last (to make use of 3-channel inputs)
            raw = raw.transpose(1, 2, 0)

            save_to_tif(0, raw, labels, crop_shape, raw_dir, labels_dir, do_connected_components, slice_prefix_name)


#
# EM DATASETS
#


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
    for nuclei: (in-domain case)
        for validation: we take volume 01
        for testing: we take volume [09-12]
    for cells (membrane):
        for validation: we take volume 07 and 08
        for testing: we take volume 09
    """
    if choice == "cilia":
        vol_paths = sorted(glob(os.path.join(ROOT, "platynereis", "cilia", "train_*")))
        for vol_path in vol_paths:
            vol_id = os.path.split(vol_path)[-1].split(".")[0][-2:]
            split = "val" if vol_id == "03" else "test"  # volumes 01 and 02 are for test, 03 for val
            from_h5_to_tif(
                h5_vol_path=vol_path,
                raw_key="volumes/raw",
                raw_dir=os.path.join(save_dir, choice, "raw"),
                labels_key="volumes/labels/segmentation",
                labels_dir=os.path.join(save_dir, choice, "labels"),
                slice_prefix_name=f"platy_{choice}_{split}_{vol_id}"
            )

    elif choice == "nuclei":
        chosen_test_ids = ["09", "10", "11", "12"]
        chosen_val_ids = ["01"]
        vol_paths = sorted(glob(os.path.join(ROOT, "platynereis", "nuclei", "*")))
        for vol_path in vol_paths:
            vol_id = os.path.split(vol_path)[-1].split(".")[0][-2:]
            if vol_id in chosen_val_ids:
                split = "val"
            elif vol_id in chosen_test_ids:
                split = "test"
            else:
                continue

            print("Creating slices from volume:", vol_id)
            from_h5_to_tif(
                h5_vol_path=vol_path,
                raw_key="volumes/raw",
                raw_dir=os.path.join(save_dir, choice, "raw"),
                labels_key="volumes/labels/nucleus_instance_labels",
                labels_dir=os.path.join(save_dir, choice, "labels"),
                slice_prefix_name=f"platy_{choice}_{split}_{vol_id}"
            )

    elif choice == "cells":
        vol_paths = sorted(glob(os.path.join(ROOT, "platynereis", "membrane", "*")))[-3:]
        for vol_path in vol_paths:
            vol_id = os.path.split(vol_path)[-1].split(".")[0][-2:]
            split = "test" if vol_id == "09" else "val"  # volume 09 for test, 07 & 08 for val
            from_h5_to_tif(
                h5_vol_path=vol_path,
                raw_key="volumes/raw/s1",
                raw_dir=os.path.join(save_dir, choice, "raw"),
                labels_key="volumes/labels/segmentation/s1",
                labels_dir=os.path.join(save_dir, choice, "labels"),
                slice_prefix_name=f"platy_{choice}_{split}_{vol_id}",
                interface=z5py
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
    - salivary gland (1260, 1081, 1200) (cropped to: 1024, 1024)
    - tem: crop to (768, 768)
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
            if Path(em_path).stem.startswith("salivary_gland"):
                slice_em = make_center_crop(slice_em, (1024, 1024))
                slice_mito = make_center_crop(slice_mito, (1024, 1024))

            if has_foreground(slice_mito):
                instances = connected_components(slice_mito)

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

        tem_image = make_center_crop(imageio.imread(image_path), (768, 768))
        tem_mask = make_center_crop(imageio.imread(mask_path), (768, 768))

        if has_foreground(tem_mask):
            imageio.imwrite(image_dst, tem_image)
            imageio.imwrite(mask_dst, tem_mask)

    all_dataset_ids.append("tem")

    for dataset_id in all_dataset_ids:
        make_custom_splits(val_samples=10, save_dir=os.path.join(save_dir, dataset_id))


def for_uro_cell(save_dir):
    """
    for validation: first volumes' 100 slices
    for testing: rest all
    """
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


def for_cremi(save_dir):
    """
    """
    vol_paths = sorted(glob(os.path.join(ROOT, "cremi", "sample*.h5")))

    all_roi_slices = {
        "val": np.s_[75:100, :, :],
        "test": np.s_[100:, :, :]
    }

    for vol_path in vol_paths:
        vol_id = Path(vol_path).stem

        for split_name, roi_slice in all_roi_slices.items():
            from_h5_to_tif(
                h5_vol_path=vol_path,
                raw_key="volumes/raw",
                raw_dir=os.path.join(save_dir, "raw"),
                labels_key="volumes/labels/neuron_ids",
                labels_dir=os.path.join(save_dir, "labels"),
                slice_prefix_name=f"cremi_{split_name}_{vol_id}",
                roi_slices=roi_slice,
                crop_shape=(768, 768)
            )


def for_isbi(save_dir):
    """
    (we only have 1 volume with 30 slices)
    for validation: first 5/30 slices
    for testing: rest 25/30 slices
    """
    from_h5_to_tif(
        h5_vol_path=os.path.join(ROOT, "isbi", "isbi.h5"),
        raw_key="raw",
        raw_dir=os.path.join(save_dir, "raw"),
        labels_key="labels/gt_segmentation",
        labels_dir=os.path.join(save_dir, "labels"),
        slice_prefix_name="isbi"
    )

    make_custom_splits(5, save_dir)


def for_axondeepseg(save_dir):
    """
    (we use the tem modality)
    for validation: first 15 slices
    for testing: rest all
    """
    vol_paths = sorted(glob(os.path.join(ROOT, "axondeepseg", "tem", "*.h5")))

    for vol_path in vol_paths:
        vol_id = Path(vol_path).stem

        from_h5_to_tif(
            h5_vol_path=vol_path,
            raw_key="raw",
            raw_dir=os.path.join(save_dir, "raw"),
            labels_key="labels",
            labels_dir=os.path.join(save_dir, "labels"),
            slice_prefix_name=f"axondeepseg_{vol_id}",
            crop_shape=(768, 768)
        )

    make_custom_splits(15, save_dir)


def for_vnc(save_dir):
    """
    for validation: first 5 slices
    for testing: rest
    """
    vol_path = os.path.join(ROOT, "vnc", "vnc_train.h5")

    from_h5_to_tif(
        h5_vol_path=vol_path,
        raw_key="raw",
        raw_dir=os.path.join(save_dir, "raw"),
        labels_key="labels/mitochondria",
        labels_dir=os.path.join(save_dir, "labels"),
        slice_prefix_name="vnc_train",
    )

    make_custom_splits(5, save_dir)


class FilterObjectsLabelTrafo:
    def __init__(self, min_size=0, gap_closing=0):
        self.min_size = min_size
        self.gap_closing = gap_closing

    def __call__(self, labels):
        if self.gap_closing > 0:
            labels = binary_closing(labels, iterations=self.gap_closing)

        if self.min_size > 0:
            labels = size_filter(labels, min_size=self.min_size)

        labels = connected_components(labels)

        return labels


def for_asem(save_dir, choice):
    """
    for simplicity, I will use the dataloader to get the samples.
    for validation: 50 samples
    for testing: 181 samples
    """
    save_dir = os.path.join(save_dir, choice)
    loader = datasets.get_asem_loader(
        os.path.join(ROOT, "asem"),
        patch_shape=(1, 512, 512),
        batch_size=1,
        ndim=2,
        organelles=choice,
        volume_ids="cell_6",
        sampler=MinForegroundSampler(min_fraction=0.01),
        raw_transform=identity,
        num_workers=16,
        shuffle=True,
        label_transform=FilterObjectsLabelTrafo(min_size=100, gap_closing=3)
    )

    n_desired = 500
    counter = 0
    for x, y in tqdm(loader):
        image = x.squeeze().numpy()
        labels = y.squeeze().numpy()

        image_path = os.path.join(save_dir, "raw", f"asem_{choice}_{counter:05}.tif")
        label_path = os.path.join(save_dir, "labels", f"asem_{choice}_{counter:05}.tif")

        os.makedirs(os.path.split(image_path)[0], exist_ok=True)
        os.makedirs(os.path.split(label_path)[0], exist_ok=True)

        imageio.imwrite(image_path, image, compression="zlib")
        imageio.imwrite(label_path, labels, compression="zlib")
        counter += 1

        if counter > n_desired:
            # it's not needed to run the loader further, hence ciao
            break

    make_custom_splits(50, save_dir)


#
# LM DATASETS
#


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


def for_tissuenet(save_dir, to_one_chan):
    """
    all test volumes to the val set, 100 volumes from val to val slices (all in org. resolution)
    """
    tissuenet_val_paths = sorted(glob(os.path.join(ROOT, "tissuenet", "val", "*.zarr")))
    tissuenet_test_paths = sorted(glob(os.path.join(ROOT, "tissuenet", "test", "*.zarr")))

    def save_slices_per_split(all_vol_paths, split, to_one_chan):
        for vol_path in all_vol_paths:
            vol_id = Path(vol_path).stem

            _choice = "one_chan" if to_one_chan else "multi_chan"

            from_h5_to_tif(
                h5_vol_path=vol_path,
                raw_key="raw/rgb",
                raw_dir=os.path.join(save_dir, _choice, split, "raw"),
                labels_key="labels/cell",
                labels_dir=os.path.join(save_dir, _choice, split, "labels"),
                slice_prefix_name=f"tissuenet_{split}_{vol_id}",
                interface=z5py,
                to_one_channel=to_one_chan
            )

    save_slices_per_split(tissuenet_val_paths[:100], "val", to_one_chan=to_one_chan)
    save_slices_per_split(tissuenet_test_paths, "test", to_one_chan=to_one_chan)


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
            slice_prefix_name=f"lizard_{vol_id}",
            crop_shape=(768, 768)
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


def for_neurips_cellseg(save_dir, chosen_set, desired_shape=(1024, 1024)):
    """
    we infer on the `TuningSet` - the data for open-evaluation on grand-challenge

    val set info is here: /home/nimanwai/torch-em/torch_em/data/datasets/split_0.1.json

    for validation: use the true val set used
    for testing: use the `TuningSet` and `TestForSam`
    """
    def neurips_raw_trafo(raw):
        if raw.ndim == 3 and raw.shape[-1] == 3:  # make channels first
            raw = raw.transpose(2, 0, 1)

        raw = datasets.neurips_cell_seg.to_rgb(raw)  # ensures 3 channels for the neurips data
        raw = normalize_percentile(raw)
        raw = np.mean(raw, axis=0)
        raw = normalize(raw)
        raw = raw * 255
        return raw

    # let's get the val slices
    from torch_em.data.datasets.neurips_cell_seg import _get_image_and_label_paths
    val_image_paths, val_label_paths = _get_image_and_label_paths(
        os.path.join(ROOT, "neurips-cell-seg"), split="val", val_fraction=0.1
    )

    os.makedirs(os.path.join(save_dir, chosen_set, "val", "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, chosen_set, "val", "labels"), exist_ok=True)

    for image_path, label_path in tqdm(zip(val_image_paths, val_label_paths), total=len(val_image_paths)):
        image_id = Path(image_path).stem
        label_id = os.path.split(label_path)[-1]
        dst_image_path = os.path.join(save_dir, chosen_set, "val", "raw", f"val_{image_id}.tif")
        dst_label_path = os.path.join(save_dir, chosen_set, "val", "labels", f"val_{label_id}")

        raw = neurips_raw_trafo(imageio.imread(image_path))
        gt = imageio.imread(label_path)

        if gt.shape > desired_shape:
            if has_foreground(make_center_crop(gt, desired_shape)):
                imageio.imwrite(dst_image_path, make_center_crop(raw, desired_shape))
                imageio.imwrite(dst_label_path, make_center_crop(gt, desired_shape))
        else:
            # converting all images to one channel image - same as generalist training logic
            imageio.imwrite(dst_image_path, neurips_raw_trafo(imageio.imread(image_path)))
            shutil.copy(label_path, dst_label_path)

    # now, let's get the test slices
    self_test_image_paths = sorted(glob(os.path.join(ROOT, "neurips-cell-seg", "TestForSam", "images", "*")))
    self_test_label_paths = sorted(glob(os.path.join(ROOT, "neurips-cell-seg", "TestForSam", "labels", "*")))

    tuning_image_paths = sorted(glob(os.path.join(ROOT, "neurips-cell-seg", "new", "Tuning", "images", "*")))
    tuning_label_paths = sorted(glob(os.path.join(ROOT, "neurips-cell-seg", "new", "Tuning", "labels", "*")))

    if chosen_set == "all":
        test_image_paths = [*self_test_image_paths, *tuning_image_paths]
        test_label_paths = [*self_test_label_paths, *tuning_label_paths]
    elif chosen_set == "self":
        test_image_paths = self_test_image_paths
        test_label_paths = self_test_label_paths
    elif chosen_set == "tuning":
        test_image_paths = tuning_image_paths
        test_label_paths = tuning_label_paths
    else:
        raise ValueError("Choose from 'all' / 'self' / 'tuning'.")

    os.makedirs(os.path.join(save_dir, chosen_set, "test", "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, chosen_set, "test", "labels"), exist_ok=True)

    for image_path, label_path in tqdm(zip(test_image_paths, test_label_paths), total=len(test_image_paths)):
        image_id = Path(image_path).stem
        label_id = os.path.split(label_path)[-1]
        dst_image_path = os.path.join(save_dir, chosen_set, "test", "raw", f"test_{image_id}.tif")
        dst_label_path = os.path.join(save_dir, chosen_set, "test", "labels", f"test_{label_id}")

        raw = neurips_raw_trafo(imageio.imread(image_path))
        gt = imageio.imread(label_path)

        if gt.shape > desired_shape:
            if has_foreground(make_center_crop(gt, desired_shape)):
                imageio.imwrite(dst_image_path, make_center_crop(raw, desired_shape))
                imageio.imwrite(dst_label_path, make_center_crop(gt, desired_shape))
        else:
            # converting all images to one channel image - same as generalist training logic
            imageio.imwrite(dst_image_path, neurips_raw_trafo(imageio.imread(image_path)))
            shutil.copy(label_path, dst_label_path)


def for_deepbacs(save_dir):
    "Move the datasets from the internal split (provided by default in deepbacs) to our `slices` logic"
    for split in ["val", "test"]:
        image_paths = sorted(glob(os.path.join(ROOT, "deepbacs", "mixed", split, "source", "*")))
        label_paths = sorted(glob(os.path.join(ROOT, "deepbacs", "mixed", split, "target", "*")))

        os.makedirs(os.path.join(save_dir, split, "raw"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, split, "labels"), exist_ok=True)

        for src_image_path, src_label_path in tqdm(zip(image_paths, label_paths), total=len(image_paths)):
            image_id = os.path.split(src_image_path)[-1]
            label_id = os.path.split(src_label_path)[-1]

            dst_image_path = os.path.join(save_dir, split, "raw", image_id)
            dst_label_path = os.path.join(save_dir, split, "labels", label_id)

            shutil.copy(src_image_path, dst_image_path)
            shutil.copy(src_label_path, dst_label_path)


def for_dynamicnuclearnet(save_dir):
    """
    for validation: take first 100 images from the validation set
    for testing: take all images from the test set
    """
    all_val_paths = sorted(glob(os.path.join(ROOT, "dynamicnuclearnet", "val", "*.zarr")))
    all_test_paths = sorted(glob(os.path.join(ROOT, "dynamicnuclearnet", "test", "*.zarr")))

    def save_slices_per_split(all_vol_paths, split):
        for vol_path in all_vol_paths:
            vol_id = Path(vol_path).stem

            from_h5_to_tif(
                h5_vol_path=vol_path,
                raw_key="raw",
                raw_dir=os.path.join(save_dir, split, "raw"),
                labels_key="labels",
                labels_dir=os.path.join(save_dir, split, "labels"),
                slice_prefix_name=f"dynamicnuclearnet_{split}_{vol_id}",
                interface=z5py
            )

    save_slices_per_split(all_val_paths[:100], "val")
    save_slices_per_split(all_test_paths, "test")


def for_pannuke(save_dir, make_one_chan=True):
    """
    take 500 samples per fold (total - 1500)
    for validation: first 50 images per fold
    for testing: rest all
    """
    all_vol_paths = sorted(glob(os.path.join(ROOT, "pannuke", "*.h5")))

    for vol_path in tqdm(all_vol_paths, desc="Processing the pannuke folds"):
        fold_name = Path(vol_path).stem

        with h5py.File(vol_path, "r") as f:
            raw = f["images"][:]
            labels = f["labels/instances"][:]

            raw = raw.transpose(1, 2, 3, 0)  # transpose it to B * H * W * C

            def _save_per_split(raw_here, labels_here, split, offset=0):
                for i, (s_raw, s_labels) in enumerate(zip(raw_here, labels_here)):
                    i += offset
                    fname = f"{fold_name}_{i:04}.tif"
                    image_path = os.path.join(save_dir, split, "raw", fname)
                    labels_path = os.path.join(save_dir, split, "labels", fname)

                    os.makedirs(os.path.split(image_path)[0], exist_ok=True)
                    os.makedirs(os.path.split(labels_path)[0], exist_ok=True)

                    if make_one_chan:
                        s_raw = s_raw.mean(axis=-1)

                    if has_foreground(s_labels):
                        imageio.imwrite(image_path, s_raw, compression="zlib")
                        imageio.imwrite(labels_path, s_labels, compression="zlib")

            _save_per_split(raw[:50], labels[:50], "val")  # choose first 50 samples per fold
            _save_per_split(raw[50: 500], labels[50: 500], "test", offset=50)  # choose next 450 samples per fold


def preprocess_lm_datasets():
    for_covid_if(os.path.join(ROOT, "covid_if", "slices"))
    for_tissuenet(os.path.join(ROOT, "tissuenet", "slices"), to_one_chan=True)
    for_tissuenet(os.path.join(ROOT, "tissuenet", "slices"), to_one_chan=False)
    for_plantseg((os.path.join(ROOT, "plantseg", "slices")))
    for_hpa(os.path.join(ROOT, "hpa", "slices"))
    for_lizard(os.path.join(ROOT, "lizard", "slices"))
    for_mouse_embryo(os.path.join(ROOT, "mouse-embryo", "slices"))
    for_ctc((os.path.join(ROOT, "ctc/hela_samples", "slices")))
    for_neurips_cellseg(os.path.join(ROOT, "neurips-cell-seg", "slices"), chosen_set="all")
    for_neurips_cellseg(os.path.join(ROOT, "neurips-cell-seg", "slices"), chosen_set="self")
    for_neurips_cellseg(os.path.join(ROOT, "neurips-cell-seg", "slices"), chosen_set="tuning")
    for_deepbacs(os.path.join(ROOT, "deepbacs", "slices"))
    for_dynamicnuclearnet(os.path.join(ROOT, "dynamicnuclearnet", "slices"))
    for_pannuke(os.path.join(ROOT, "pannuke", "slices"))


def preprocess_em_datasets():
    for_lucchi(os.path.join(ROOT, "lucchi", "slices"))
    for_snemi(os.path.join(ROOT, "snemi", "slices"))
    for_nuc_mm(os.path.join(ROOT, "nuc_mm", "slices"))
    for_platynereis(os.path.join(ROOT, "platynereis", "slices"), choice="nuclei")
    for_platynereis(os.path.join(ROOT, "platynereis", "slices"), choice="cilia")
    for_platynereis(os.path.join(ROOT, "platynereis", "slices"), choice="cells")
    for_mitoem(os.path.join(ROOT, "mitoem", "slices"))
    for_mitolab(os.path.join(ROOT, "mitolab", "slices"))
    for_uro_cell(os.path.join(ROOT, "uro_cell", "slices"))
    for_sponge_em(os.path.join(ROOT, "sponge_em", "slices"))
    for_isbi(os.path.join(ROOT, "isbi", "slices"))
    for_axondeepseg(os.path.join(ROOT, "axondeepseg", "slices"))
    for_cremi(os.path.join(ROOT, "cremi", "slices"))
    for_vnc(os.path.join(ROOT, "vnc", "slices"))
    for_asem(os.path.join(ROOT, "asem", "slices"), choice="mito")
    for_asem(os.path.join(ROOT, "asem", "slices"), choice="er")


def main():
    print("The preprocessing has been done.")
    return

    # let's ensure all the data is downloaded
    download_all_datasets(ROOT)

    # now let's save the slices as tif
    preprocess_lm_datasets()
    preprocess_em_datasets()


if __name__ == "__main__":
    main()
