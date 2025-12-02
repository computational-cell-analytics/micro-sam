import os
from tqdm import tqdm
from glob import glob
from natsort import natsorted

import numpy as np
import imageio.v3 as imageio

from torch_em.data import datasets

from elf.io import open_file

from micro_sam.evaluation.livecell import _get_livecell_paths


DATA_DIR = "/mnt/vast-nhr/projects/cidas/cca/data"


def _process_images(image_paths, label_paths, split, base_dir, dataset_name, limiter=None):

    if os.path.exists(os.path.join(base_dir, split)):
        return _find_paths(base_dir, split, dataset_name)

    im_folder = os.path.join(base_dir, split, "images")
    label_folder = os.path.join(base_dir, split, "labels")
    os.makedirs(im_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)

    curr_image_paths, curr_label_paths = [], []
    for i, (im, label) in tqdm(
        enumerate(zip(image_paths, label_paths)), desc=f"Store '{dataset_name}' images for '{split}' split",
        total=len(image_paths) if limiter is None else limiter,
    ):
        if im.ndim == 3 and im.shape[0] == 3:  # eg. for PanNuke
            im = im.transpose(1, 2, 0)  # Make channels last for RGB images.

        # If there are no labels in ground-truth, no point in storing it
        if len(np.unique(label)) == 1:
            continue

        # Store images in a folder.
        curr_image_path = os.path.join(im_folder, f"{dataset_name}_{i:04}.tif")
        curr_label_path = os.path.join(label_folder, f"{dataset_name}_{i:04}.tif")
        imageio.imwrite(curr_image_path, im, compression="zlib")
        imageio.imwrite(curr_label_path, label, compression="zlib")
        curr_image_paths.append(curr_image_path)
        curr_label_paths.append(curr_label_path)

        if limiter and i == limiter:  # When 'n' number of images are done, that's enough.
            break

    return curr_image_paths, curr_label_paths


def _find_paths(base_dir, split, dataset_name):
    image_paths = natsorted(glob(os.path.join(base_dir, split, "images", f"{dataset_name}_*.tif")))
    label_paths = natsorted(glob(os.path.join(base_dir, split, "labels", f"{dataset_name}_*.tif")))
    return image_paths, label_paths


def prepare_data_paths(dataset_name, split, base_dir):
    """This script converts all images to expected 2d images.
    """
    base_dir = os.path.join(base_dir, "benchmark_apg")

    if dataset_name == "pannuke":
        # This needs to be done as the PanNuke images are stacked together.
        if split == "val":
            volume_path = datasets.histopathology.pannuke.get_pannuke_paths(
                path=os.path.join(DATA_DIR, "pannuke"), folds=["fold_2"], download=True,
            )[0]
        else:
            volume_path = datasets.histopathology.pannuke.get_pannuke_paths(
                path=os.path.join(DATA_DIR, "pannuke"), folds=["fold_3"], download=True,
            )[0]

        f = open_file(volume_path)
        raw_stack, label_stack = f["images"][:], f["labels/instances"][:]
        raw_stack = raw_stack.transpose(1, 0, 2, 3)

        image_paths, label_paths = _process_images(
            raw_stack, label_stack, split, base_dir, dataset_name, 100 if split == "val" else None,
        )

    elif dataset_name == "tissuenet":
        # This needs to be done as these are zarr images.
        fpaths = datasets.light_microscopy.tissuenet.get_tissuenet_paths(
            path=os.path.join(DATA_DIR, "tissuenet"), split=split,
        )
        fpaths = natsorted(fpaths)
        images = [open_file(p)["raw/rgb"][:].transpose(1, 2, 0) for p in fpaths]
        labels = [open_file(p)["labels/cell"][:] for p in fpaths]

        image_paths, label_paths = _process_images(
            images, labels, split, base_dir, dataset_name, 100 if split == "val" else None,
        )

    else:
        raise ValueError

    return image_paths, label_paths


def get_image_label_paths(dataset_name, split):
    assert split in ["val", "test"]

    # Label-free
    if dataset_name == "livecell":
        image_paths, label_paths = _get_livecell_paths(
            input_folder=os.path.join(DATA_DIR, dataset_name), split=split,
        )
    elif dataset_name == "omnipose":
        if split == "val":  # NOTE: Since 'val' does not exist for this data.
            split = "test"

        image_paths, label_paths = datasets.light_microscopy.omnipose.get_omnipose_paths(
            os.path.join(DATA_DIR, dataset_name), split, data_choice=["bact_phase", "worm"],
        )
    elif dataset_name == "deepbacs":
        image_dir, label_dir = datasets.light_microscopy.deepbacs.get_deepbacs_paths(
            os.path.join(DATA_DIR, dataset_name), split=split, bac_type="mixed",
        )
        image_paths = natsorted(glob(os.path.join(image_dir, "*.tif")))
        label_paths = natsorted(glob(os.path.join(label_dir, "*.tif")))
    elif dataset_name == "yeaz":
        image_paths, label_paths = datasets.light_microscopy.yeaz.get_yeaz_paths(
            os.path.join(DATA_DIR, dataset_name), choice="bf", split=split,
        )
    elif dataset_name == "orgasegment":
        if split == "test":
            split = "eval"  # Simple name matching
        image_paths, label_paths = datasets.light_microscopy.orgasegment.get_orgasegment_paths(
            os.path.join(DATA_DIR, dataset_name), split,
        )
    elif dataset_name == "deepseas":
        if split == "val":  # NOTE: Since 'val' does not exist for this data.
            split = "test"
        datasets.light_microscopy.deepseas.get_deepseas_paths(
            os.path.join(DATA_DIR, dataset_name), split,
        )

    # Histopathology
    elif dataset_name == "pannuke":
        image_paths, label_paths = prepare_data_paths(
            dataset_name=dataset_name, split=split, base_dir=os.path.join(DATA_DIR, dataset_name),
        )

    # Fluoroscence (Nuclei)
    elif dataset_name == "dsb":
        if split == "val":  # NOTE: Since 'val' does not exist for this data.
            split = "test"

        image_paths, label_paths = datasets.light_microscopy.dsb.get_dsb_paths(
            os.path.join(DATA_DIR, "dsb"), source="reduced", split=split,
        )

    # Fluorescence (Cells)
    elif dataset_name == "tissuenet":
        image_paths, label_paths = prepare_data_paths(
            dataset_name=dataset_name, split=split, base_dir=os.path.join(DATA_DIR, dataset_name),
        )

    else:
        raise ValueError

    return image_paths, label_paths
