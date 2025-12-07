import os
from tqdm import tqdm
from glob import glob
from typing import Literal
from natsort import natsorted

import numpy as np
import imageio.v3 as imageio

from torch_em.data import datasets

from elf.io import open_file

from micro_sam.evaluation.livecell import _get_livecell_paths


DATA_DIR = "/mnt/vast-nhr/projects/cidas/cca/data"


def _process_images(
    image_paths,
    label_paths,
    split,
    base_dir,
    dataset_name,
    cell_count_criterion=None,
    limiter=None
):
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

        cell_count = len(np.unique(label))

        # If there are no labels in ground-truth, no point in storing it
        if cell_count == 1:
            continue

        # Check for minimum cells per image.
        if cell_count < cell_count_criterion:
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
            image_paths=raw_stack,
            label_paths=label_stack,
            split=split,
            base_dir=base_dir,
            dataset_name=dataset_name,
            limiter=100 if split == "val" else None,
            cell_count_criterion=5,
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
            image_paths=images,
            label_paths=labels, 
            split=split,
            base_dir=base_dir,
            dataset_name=dataset_name,
            limiter=100 if split == "val" else None,
            cell_count_criterion=10,
        )

    else:
        raise ValueError

    return image_paths, label_paths


def get_image_label_paths(dataset_name: str, split: Literal["val", "test"]):
    """Returns the available / prepared 2d image and corresponding labels for APG benchmarking.
    """
    assert split in ["val", "test"]

    # Label-free
    if dataset_name == "livecell":
        image_paths, label_paths = _get_livecell_paths(
            input_folder=os.path.join(DATA_DIR, dataset_name),
            split=split,
            n_val_per_cell_type=5 if split == "val" else None,

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
        image_paths, label_paths = datasets.light_microscopy.deepseas.get_deepseas_paths(
            os.path.join(DATA_DIR, dataset_name), split, download=True,
        )
    elif dataset_name == "bacmother":   # TODO: Double-check
        image_paths, label_paths = datasets.light_microscopy.bac_mother.get_bac_mother_paths(
            os.path.join(DATA_DIR, dataset_name), split=split, download=True,
        )
    elif dataset_name == "dic_hepg2":  # TODO: Double-check
        image_paths, label_paths = datasets.light_microscopy.dic_hepg2.get_dic_hepg2_paths(
            os.path.join(DATA_DIR, dataset_name), split=split, download=True,
        )
    elif dataset_name == "toiam":  # TODO: Double check and make splits on the fly
        image_paths, label_paths = datasets.light_microscopy.toiam.get_toiam_paths(
            os.path.join(DATA_DIR, dataset_name), download=True,
        )
    elif dataset_name == "usiigaci":  # TODO: Double check
        split = "train" if split == "test" else split
        image_paths, label_paths = datasets.light_microscopy.usiigaci.get_usiigaci_paths(
            os.path.join(DATA_DIR, dataset_name), split=split, download=True,
        )
    elif dataset_name == "vicar":  # TODO: Double check and make splits on the fly.
        image_paths, label_paths = datasets.light_microscopy.vicar.get_vicar_paths(
            os.path.join(DATA_DIR, dataset_name), download=True,
        )

    # NOTE: Can we add other organoid segmentation data?

    # Histopathology
    elif dataset_name == "pannuke":
        image_paths, label_paths = prepare_data_paths(
            dataset_name=dataset_name, split=split, base_dir=os.path.join(DATA_DIR, dataset_name),
        )
    elif dataset_name == "monuseg":  # TODO: Double check and make splits on the fly.
        image_paths, label_paths = datasets.histopathology.monuseg.get_monuseg_paths(
            os.path.join(DATA_DIR, dataset_name), split=split, download=True,
        )

    # Fluoroscence (Nuclei)
    elif dataset_name == "dsb":
        if split == "val":  # NOTE: Since 'val' does not exist for this data.
            split = "test"

        image_paths, label_paths = datasets.light_microscopy.dsb.get_dsb_paths(
            os.path.join(DATA_DIR, "dsb"), source="reduced", split=split,
        )
    elif dataset_name == "aisegcell":
        # TODO: Container format, implement stuff
        ...
    elif dataset_name == "arvidsson":  # TODO: Double check
        image_paths, label_paths = datasets.light_microscopy.arvidsson.get_arvidsson_paths(
            os.path.join(DATA_DIR, dataset_name), split, download=True,
        )
    elif dataset_name == "bitdepth_nucseg":  # TODO: Double check and make splits.
        image_paths, label_paths = datasets.light_microscopy.bitdepth_nucseg.get_bitdepth_nucseg_paths(
            os.path.join(DATA_DIR, dataset_name), download=True,
        )
    elif dataset_name == "blastospim":  # TODO: Double check data and make splits on the fly.
        # TODO: Container format, implement stuff
        ...
    elif dataset_name == "celegans_atlas":  # TODO: Double check data and make 2d crops.
        image_paths, label_paths = datasets.light_microscopy.celegans_atlas.get_celegans_atlas_paths(
            os.path.join(DATA_DIR, dataset_name), split=split, download=True,
        )
    elif dataset_name == "cellseg_3d":  # TODO: Make splits on the fly and make 2d crops.
        image_paths, label_paths = datasets.light_microscopy.cellseg_3d.get_cellseg_3d_paths(
            os.path.join(DATA_DIR, dataset_name), download=True,
        )
    elif dataset_name == "gonuclear":
        # TODO: Container format and get 2d crops.
        ...
    elif dataset_name == "ifnuclei":  # TODO: Double check
        image_paths, label_paths = datasets.light_microscopy.ifnuclei.get_ifnuclei_paths(
            os.path.join(DATA_DIR, dataset_name), download=True,
        )
    elif dataset_name == "nis3d":
        # TODO: 3d images need to be converted to 2d.
        ...
    elif dataset_name == "parhyale_regen":
        # TODO: Container data formats and need to be converted to 2d.
        ...
    elif dataset_name == "plantseg_nuclei":
        # TODO: Convert 3d images to 2d crops.
        ...

    # Fluorescence (Cells)
    elif dataset_name == "tissuenet":
        image_paths, label_paths = prepare_data_paths(
            dataset_name=dataset_name, split=split, base_dir=os.path.join(DATA_DIR, dataset_name),
        )
    elif dataset_name == "cellpose":
        ...
    elif dataset_name == "plantseg_root":
        # TODO: Convert 3d images from container format and make splits.
        ...
    elif dataset_name == "covid_if":
        # TODO: Convert from container format
        ...
    elif dataset_name == "hpa":
        # TODO: Choose the three channels as PEFT-SAM paper.
        ...
    elif dataset_name == "cellbindb":
        ...
    elif dataset_name == "mouse_embryo":
        ...
    elif dataset_name == "plantseg_ovules":
        ...
    elif dataset_name == "pnas_arabidopsis":
        ...

    else:
        raise ValueError

    return image_paths, label_paths
