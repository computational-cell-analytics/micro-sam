import os
import time
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Optional, List, Literal

import numpy as np
import pandas as pd
import imageio.v3 as imageio
from skimage.measure import label as connected_components

from nifty.tools import blocking

import torch

from torch_em.data import datasets

from micro_sam import util

from . import run_evaluation
from ..training.training import _filter_warnings
from .inference import run_inference_with_iterative_prompting
from .evaluation import run_evaluation_for_iterative_prompting
from .multi_dimensional_segmentation import segment_slices_from_ground_truth
from ..automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter


LM_2D_DATASETS = [
    # in-domain
    "livecell",  # cell segmentation in PhC (has a TEST-set)
    "deepbacs",  # bacteria segmentation in label-free microscopy (has a TEST-set),
    "tissuenet",  # cell segmentation in tissue microscopy images (has a TEST-set),
    "neurips_cellseg",  # cell segmentation in various (has a TEST-set),
    "cellpose",  # cell segmentation in FM (has 'cyto2' on which we can TEST on),
    "dynamicnuclearnet",  # nuclei segmentation in FM (has a TEST-set)
    "orgasegment",  # organoid segmentation in BF (has a TEST-set)
    "yeaz",  # yeast segmentation in BF (has a TEST-set)

    # out-of-domain
    "arvidsson",  # nuclei segmentation in HCS FM
    "bitdepth_nucseg",  # nuclei segmentation in FM
    "cellbindb",  # cell segmentation in various microscopy
    "covid_if",  # cell segmentation in IF
    "deepseas",  # cell segmentation in PhC,
    "hpa",  # cell segmentation in confocal,
    "ifnuclei",  # nuclei segmentation in IFM
    "lizard",  # nuclei segmentation in H&E histopathology,
    "organoidnet",  # organoid segmentation in BF
    "toiam",  # microbial cell segmentation in PhC
    "vicar",  # cell segmentation in label-free
]

LM_3D_DATASETS = [
    # in-domain
    "plantseg_root",  # cell segmentation in lightsheet (has a TEST-set)

    # out-of-domain
    "plantseg_ovules",  # cell segmentation in confocal
    "gonuclear",  # nuclei segmentation in FM
    "mouse_embryo",  # cell segmentation in lightsheet
    "cellseg3d",  # nuclei segmentation in FM
]

EM_2D_DATASETS = ["mitolab_tem"]

EM_3D_DATASETS = [
    # out-of-domain
    "lucchi",  # mitochondria segmentation in vEM
    "mitolab",  # mitochondria segmentation in various
    "uro_cell",  # mitochondria segmentation (and other organelles) in FIB-SEM
    "sponge_em",  # microvili segmentation (and other organelles) in sponge chamber vEM
    "vnc",  # mitochondria segmentation in drosophila brain TEM
    "nuc_mm_mouse",  # nuclei segmentation in microCT
    "num_mm_zebrafish",  # nuclei segmentation in EM
    "platynereis_cilia",  # cilia segmentation (and other structures) in platynereis larvae vEM
    "asem_mito",  # mitochondria segmentation (and other organelles) in FIB-SEM
]

DATASET_RETURNS_FOLDER = {
    "deepbacs": "*.tif",
    "mitolab_tem": "*.tiff"
}

DATASET_CONTAINER_KEYS = {
    # 2d (LM)
    "tissuenet": ["raw/rgb", "labels/cell"],
    "covid_if": ["raw/serum_IgG/s0", "labels/cells/s0"],
    "dynamicnuclearnet": ["raw", "labels"],
    "hpa": [["raw/protein", "raw/microtubules", "raw/er"], "labels"],
    "lizard": ["image", "labels/segmentation"],

    # 3d (LM)
    "plantseg_root": ["raw", "label"],
    "plantseg_ovules": ["raw", "label"],
    "gonuclear": ["raw/nuclei", "labels/nuclei"],
    "mouse_embryo": ["raw", "label"],
    "cellseg_3d": [None, None],

    # 3d (EM)
    "lucchi": ["raw", "labels"],
    "uro_cell": ["raw", "labels/mito"],
    "mitolab_3d": [None, None],
    "sponge_em": ["volumes/raw", "volumes/labels/instances"],
    "vnc": ["raw", "labels/mitochondria"]
}


def _download_benchmark_datasets(path, dataset_choice):
    """Ensures whether all the datasets have been downloaded or not.

    Args:
        path: The path to directory where the supported datasets will be downloaded
            for benchmarking Segment Anything models.
        dataset_choice: The choice of dataset, expects the lower case name for the dataset.

    Returns:
        List of choice of dataset(s).
    """
    available_datasets = {
        # Light Microscopy datasets

        # 2d datasets: in-domain
        "livecell": lambda: datasets.livecell.get_livecell_data(
            path=os.path.join(path, "livecell"), download=True,
        ),
        "deepbacs": lambda: datasets.deepbacs.get_deepbacs_data(
            path=os.path.join(path, "deepbacs"), bac_type="mixed", download=True,
        ),
        "tissuenet": lambda: datasets.tissuenet.get_tissuenet_data(
            path=os.path.join(path, "tissuenet"), split="test", download=True,
        ),
        "neurips_cellseg": lambda: datasets.neurips_cell_seg.get_neurips_cellseg_data(
            root=os.path.join(path, "neurips_cellseg"), split="test", download=True,
        ),
        "cellpose": lambda: datasets.cellpose.get_cellpose_data(
            path=os.path.join(path, "cellpose"), split="train", choice="cyto2", download=True,
        ),
        "dynamicnuclearnet": lambda: datasets.dynamicnuclearnet.get_dynamicnuclearnet_data(
            path=os.path.join(path, "dynamicnuclearnet"), split="test", download=True,
        ),
        "orgasegment": lambda: datasets.orgasegment.get_orgasegment_data(
            path=os.path.join(path, "orgasegment"), split="eval", download=True,
        ),
        "yeaz": lambda: datasets.yeaz.get_yeaz_data(
            path=os.path.join(path, "yeaz"), choice="bf", download=True,
        ),

        # 2d datasets: out-of-domain
        "arvidsson": lambda: datasets.arvidsson.get_arvidsson_data(
            path=os.path.join(path, "arvidsson"), split="test", download=True,
        ),
        "bitdepth_nucseg": lambda: datasets.bitdepth_nucseg.get_bitdepth_nucseg_data(
            path=os.path.join(path, "bitdepth_nucseg"), download=True,
        ),
        "cellbindb": lambda: datasets.cellbindb.get_cellbindb_data(
            path=os.path.join(path, "cellbindb"), download=True,
        ),
        "covid_if": lambda: datasets.covid_if.get_covid_if_data(
            path=os.path.join(path, "covid_if"), download=True,
        ),
        "deepseas": lambda: datasets.deepseas.get_deepseas_data(
            path=os.path.join(path, "deepseas"), split="test",
        ),
        "hpa": lambda: datasets.hpa.get_hpa_segmentation_data(
            path=os.path.join(path, "hpa"), download=True,
        ),
        "ifnuclei": lambda: datasets.ifnuclei.get_ifnuclei_data(
            path=os.path.join(path, "ifnuclei"), download=True,
        ),
        "lizard": lambda: datasets.lizard.get_lizard_data(
            path=os.path.join(path, "lizard"), split="test", download=True,
        ),
        "organoidnet": lambda: datasets.organoidnet.get_organoidnet_data(
            path=os.path.join(path, "organoidnet"), split="Test", download=True,
        ),
        "toiam": lambda: datasets.toiam.get_toiam_data(
            path=os.path.join(path, "toiam"), download=True,
        ),
        "vicar": lambda: datasets.vicar.get_vicar_data(
            path=os.path.join(path, "vicar"), download=True,
        ),

        # 3d datasets: in-domain
        "plantseg_root": lambda: datasets.plantseg.get_plantseg_data(
            path=os.path.join(path, "plantseg_root"), split="test", download=True, name="root",
        ),

        # 3d datasets: out-of-domain
        "plantseg_ovules": lambda: datasets.plantseg.get_plantseg_data(
            path=os.path.join(path, "plantseg_ovules"), split="test", download=True, name="ovules",
        ),
        "gonuclear": lambda: datasets.gonuclear.get_gonuclear_data(
            path=os.path.join(path, "gonuclear"), download=True,
        ),
        "mouse_embryo": lambda: datasets.mouse_embryo.get_mouse_embryo_data(
            path=os.path.join(path, "mouse_embryo"), download=True,
        ),
        "cellseg_3d": lambda: datasets.cellseg_3d.get_cellseg_3d_data(
            path=os.path.join(path, "cellseg_3d"), download=True,
        ),

        # Electron Microscopy datasets

        # 2d datasets: out-of-domain
        "mitolab_tem": lambda: datasets.cem.get_benchmark_data(
            path=os.path.join(path, "mitolab"), dataset_id=7, download=True
        ),

        # 3d datasets: out-of-domain
        "lucchi": lambda: datasets.lucchi.get_lucchi_data(
            path=os.path.join(path, "lucchi"), split="test", download=True,
        ),
        "mitolab_3d": lambda: [
            datasets.cem.get_benchmark_data(
                path=os.path.join(path, "mitolab"), dataset_id=dataset_id, download=True,
            ) for dataset_id in range(1, 7)
        ],
        "uro_cell": lambda: datasets.uro_cell.get_uro_cell_data(
            path=os.path.join(path, "uro_cell"), download=True,
        ),
        "vnc": lambda: datasets.vnc.get_vnc_data(
            path=os.path.join(path, "vnc"), download=True,
        ),
        "sponge_em": lambda: datasets.sponge_em.get_sponge_em_data(
            path=os.path.join(path, "sponge_em"), download=True,
        ),
        "nuc_mm_mouse": lambda: datasets.nuc_mm.get_nuc_mm_data(
            path=os.path.join(path, "nuc_mm"), sample="mouse", download=True,
        ),
        "nuc_mm_zebrafish": lambda: datasets.nuc_mm.get_nuc_mm_data(
            path=os.path.join(path, "nuc_mm"), sample="zebrafish", download=True,
        ),
        "asem_mito": lambda: datasets.asem.get_asem_data(
            path=os.path.join(path, "asem"), volume_ids=datasets.asem.ORGANELLES["mito"], download=True,
        ),
        "platynereis_cilia": lambda: datasets.platynereis.get_platynereis_data(
            path=os.path.join(path, "platynereis"), name="cilia", download=True,
        ),
    }

    if dataset_choice is None:
        dataset_choice = available_datasets.keys()
    else:
        if not isinstance(dataset_choice, list):
            dataset_choice = [dataset_choice]

    for choice in dataset_choice:
        if choice in available_datasets:
            available_datasets[choice]()
        else:
            raise ValueError(f"'{choice}' is not a supported choice of dataset.")

    return dataset_choice


def _extract_slices_from_dataset(path, dataset_choice, crops_per_input=10):
    """Extracts crops of desired shapes for performing evaluation in both 2d and 3d using `micro-sam`.

    Args:
        path: The path to directory where the supported datasets have be downloaded
            for benchmarking Segment Anything models.
        dataset_choice: The name of the dataset of choice to extract crops.
        crops_per_input: The maximum number of crops to extract per inputs.
        extract_2d: Whether to extract 2d crops from 3d patches.

    Returns:
        Filepath to the folder where extracted images are stored.
        Filepath to the folder where corresponding extracted labels are stored.
        The number of dimensions supported by the input.
    """
    ndim = 2 if dataset_choice in [*LM_2D_DATASETS, *EM_2D_DATASETS] else 3
    tile_shape = (512, 512) if ndim == 2 else (32, 512, 512)

    # For 3d inputs, we extract both 2d and 3d crops.
    extract_2d_crops_from_volumes = (ndim == 3)

    available_datasets = {
        # Light Microscopy datasets

        # 2d: in-domain
        "livecell": lambda: datasets.livecell.get_livecell_paths(path=path, split="test"),
        "deepbacs": lambda: datasets.deepbacs.get_deepbacs_paths(path=path, split="test", bac_type="mixed"),
        "tissuenet": lambda: datasets.tissuenet.get_tissuenet_paths(path=path, split="test"),
        "neurips_cellseg": lambda: datasets.neurips_cell_seg.get_neurips_cellseg_paths(root=path, split="test"),
        "cellpose": lambda: datasets.cellpose.get_cellpose_paths(path=path, split="train", choice="cyto2"),
        "dynamicnuclearnet": lambda: datasets.dynamicnuclearnet.get_dynamicnuclearnet_paths(path=path, split="test"),
        "orgasegment": lambda: datasets.orgasegment.get_orgasegment_paths(path=path, split="eval"),
        "yeaz": lambda: datasets.yeaz.get_yeaz_paths(path=path, choice="bf", split="test"),

        # 2d: out-of-domain
        "arvidsson": lambda: datasets.arvidsson.get_arvidsson_paths(path=path, split="test"),
        "bitdepth_nucseg": lambda: datasets.bitdepth_nucseg.get_bitdepth_nucseg_paths(path=path, magnification="20x"),
        "cellbindb": lambda: datasets.cellbindb.get_cellbindb_paths(
            path=path, data_choice=["10×Genomics_DAPI", "DAPI", "mIF"]
        ),
        "covid_if": lambda: datasets.covid_if.get_covid_if_paths(path=path),
        "deepseas": lambda: datasets.deepseas.get_deepseas_paths(path=path, split="test"),
        "hpa": lambda: datasets.hpa.get_hpa_segmentation_paths(path=path, split="val"),
        "ifnuclei": lambda: datasets.ifnuclei.get_ifnuclei_paths(path=path),
        "lizard": lambda: datasets.lizard.get_lizard_paths(path=path, split="test"),
        "organoidnet": lambda: datasets.organoidnet.get_organoidnet_paths(path=path, split="Test"),
        "toiam": lambda: datasets.toiam.get_toiam_paths(path=path),
        "vicar": lambda: datasets.vicar.get_vicar_paths(path=path),

        # 3d: in-domain
        "plantseg_root": lambda: datasets.plantseg.get_plantseg_paths(path=path, name="root", split="test"),

        # 3d: out-of-domain
        "plantseg_ovules": lambda: datasets.plantseg.get_plantseg_paths(path=path, name="ovules", split="test"),
        "gonuclear": lambda: datasets.gonuclear.get_gonuclear_paths(path=path),
        "mouse_embryo": lambda: datasets.mouse_embryo.get_mouse_embryo_paths(path=path, name="nuclei", split="val"),
        "cellseg_3d": lambda: datasets.cellseg_3d.get_cellseg_3d_paths(path=path),

        # Electron Microscopy datasets

        # 2d: out-of-domain
        "mitolab_tem": lambda: datasets.cem.get_benchmark_paths(
            path=os.path.join(os.path.dirname(path), "mitolab"), dataset_id=7
        )[:2],

        # 3d: out-of-domain"lucchi": lambda: datasets.lucchi.get_lucchi_paths(path=path, split="test"),
        "platynereis_cilia": lambda: datasets.platynereis.get_platynereis_paths(path, sample_ids=None, name="cilia"),
        "uro_cell": lambda: datasets.uro_cell.get_uro_cell_paths(path=path, target="mito"),
        "vnc": lambda: datasets.vnc.get_vnc_mito_paths(path=path),
        "sponge_em": lambda: datasets.sponge_em.get_sponge_em_paths(path=path, sample_ids=None),
        "mitolab_3d": lambda: (
            [
                datasets.cem.get_benchmark_paths(
                    path=os.path.join(os.path.dirname(path), "mitolab"), dataset_id=i
                )[0] for i in range(1, 7)
            ],
            [
                datasets.cem.get_benchmark_paths(
                    path=os.path.join(os.path.dirname(path), "mitolab"), dataset_id=i
                )[1] for i in range(1, 7)
            ]
        ),
        "nuc_mm_mouse": lambda: datasets.nuc_mm.get_nuc_mm_paths(path=path, sample="mouse", split="val"),
        "nuc_mm_zebrafish": lambda: datasets.nuc_mm.get_nuc_mm_paths(path=path, sample="zebrafish", split="val"),
        "asem_mito": lambda: datasets.asem.get_asem_paths(path=path, volume_ids=datasets.asem.ORGANELLES["mito"])
    }

    if (ndim == 2 and dataset_choice not in DATASET_CONTAINER_KEYS) or dataset_choice in ["cellseg_3d", "mitolab_3d"]:
        image_paths, gt_paths = available_datasets[dataset_choice]()

        if dataset_choice in DATASET_RETURNS_FOLDER:
            image_paths = glob(os.path.join(image_paths, DATASET_RETURNS_FOLDER[dataset_choice]))
            gt_paths = glob(os.path.join(gt_paths, DATASET_RETURNS_FOLDER[dataset_choice]))

        image_paths, gt_paths = natsorted(image_paths), natsorted(gt_paths)
        assert len(image_paths) == len(gt_paths)

        paths_set = zip(image_paths, gt_paths)

    else:
        image_paths = available_datasets[dataset_choice]()
        if isinstance(image_paths, str):
            paths_set = [image_paths]
        else:
            paths_set = natsorted(image_paths)

    # Directory where we store the extracted ROIs.
    save_image_dir = [os.path.join(path, f"roi_{ndim}d", "inputs")]
    save_gt_dir = [os.path.join(path, f"roi_{ndim}d", "labels")]
    if extract_2d_crops_from_volumes:
        save_image_dir.append(os.path.join(path, "roi_2d", "inputs"))
        save_gt_dir.append(os.path.join(path, "roi_2d", "labels"))

    _dir_exists = [os.path.exists(idir) and os.path.exists(gdir) for idir, gdir in zip(save_image_dir, save_gt_dir)]
    if all(_dir_exists):
        return ndim

    [os.makedirs(idir, exist_ok=True) for idir in save_image_dir]
    [os.makedirs(gdir, exist_ok=True) for gdir in save_gt_dir]

    # Logic to extract relevant patches for inference
    image_counter = 1
    for per_paths in tqdm(paths_set, desc=f"Extracting {ndim}d patches for {dataset_choice}"):
        if (ndim == 2 and dataset_choice not in DATASET_CONTAINER_KEYS) or dataset_choice in ["cellseg_3d", "mitolab_3d"]:  # noqa
            image_path, gt_path = per_paths
            image, gt = util.load_image_data(image_path), util.load_image_data(gt_path)

        else:
            image_path = per_paths
            gt = util.load_image_data(image_path, DATASET_CONTAINER_KEYS[dataset_choice][1])
            if dataset_choice == "hpa":
                # Get inputs per channel and stack them together to make the desired 3 channel image.
                image = np.stack(
                    [util.load_image_data(image_path, k) for k in DATASET_CONTAINER_KEYS[dataset_choice][0]], axis=0,
                )
                # Resize inputs to desired tile shape, in favor of working with the shape of foreground.
                from torch_em.transform.generic import ResizeLongestSideInputs
                raw_transform = ResizeLongestSideInputs(target_shape=tile_shape, is_rgb=True)
                label_transform = ResizeLongestSideInputs(target_shape=tile_shape, is_label=True)
                image, gt = raw_transform(image).transpose(1, 2, 0), label_transform(gt)

            else:
                image = util.load_image_data(image_path, DATASET_CONTAINER_KEYS[dataset_choice][0])

        if dataset_choice in ["tissuenet", "lizard"]:
            if image.ndim == 3 and image.shape[0] == 3:  # Make channels last for tissuenet RGB-style images.
                image = image.transpose(1, 2, 0)

        # Allow RGBs to stay as it is with channels last
        if image.ndim == 3 and image.shape[-1] == 3:
            skip_smaller_shape = (np.array(image.shape) >= np.array((*tile_shape, 3))).all()
        else:
            skip_smaller_shape = (np.array(image.shape) >= np.array(tile_shape)).all()

        # Ensure ground truth has instance labels.
        gt = connected_components(gt)

        if len(np.unique(gt)) == 1:  # There could be labels which does not have any annotated foreground.
            continue

        # Let's extract and save all the crops.
        # The first round of extraction is always to match the desired input dimensions.
        image_crops, gt_crops = _get_crops_for_input(image, gt, ndim, tile_shape, skip_smaller_shape, crops_per_input)
        image_counter = _save_image_label_crops(
            image_crops, gt_crops, dataset_choice, ndim, image_counter, save_image_dir[0], save_gt_dir[0]
        )

        # The next round of extraction is to get 2d crops from 3d inputs.
        if extract_2d_crops_from_volumes:
            curr_tile_shape = tile_shape[1:]  # We expect 2d tile shape for this stage.

            curr_image_crops, curr_gt_crops = [], []
            for per_z_im, per_z_gt in zip(image, gt):
                curr_skip_smaller_shape = (np.array(per_z_im.shape) >= np.array(curr_tile_shape)).all()

                image_crops, gt_crops = _get_crops_for_input(
                    image=per_z_im, gt=per_z_gt, ndim=2,
                    tile_shape=curr_tile_shape,
                    skip_smaller_shape=curr_skip_smaller_shape,
                    crops_per_input=crops_per_input,
                )
                curr_image_crops.extend(image_crops)
                curr_gt_crops.extend(gt_crops)

            image_counter = _save_image_label_crops(
                curr_image_crops, curr_gt_crops, dataset_choice, 2, image_counter, save_image_dir[1], save_gt_dir[1]
            )

    return ndim


def _get_crops_for_input(image, gt, ndim, tile_shape, skip_smaller_shape, crops_per_input):
    tiling = blocking([0] * ndim, gt.shape, tile_shape)
    n_tiles = tiling.numberOfBlocks
    tiles = [tiling.getBlock(tile_id) for tile_id in range(n_tiles)]
    crop_boxes = [
        tuple(slice(beg, end) for beg, end in zip(tile.begin, tile.end)) for tile in tiles
    ]
    n_ids = [idx for idx in range(len(crop_boxes))]
    n_instances = [len(np.unique(gt[crop])) for crop in crop_boxes]

    # Extract the desired number of patches with higher number of instances.
    image_crops, gt_crops = [], []
    for i, (per_n_instance, per_id) in enumerate(sorted(zip(n_instances, n_ids), reverse=True), start=1):
        crop_box = crop_boxes[per_id]
        crop_image, crop_gt = image[crop_box], gt[crop_box]

        # NOTE: We avoid using the crops which do not match the desired tile shape.
        _rtile_shape = (*tile_shape, 3) if image.ndim == 3 and image.shape[-1] == 3 else tile_shape  # For RGB images.
        if skip_smaller_shape and crop_image.shape != _rtile_shape:
            continue

        # NOTE: There could be a case where some later patches are invalid.
        if per_n_instance == 1:
            break

        image_crops.append(crop_image)
        gt_crops.append(crop_gt)

        # NOTE: If the number of patches extracted have been fulfiled, we stop sampling patches.
        if len(image_crops) > 0 and i >= crops_per_input:
            break

    return image_crops, gt_crops


def _save_image_label_crops(image_crops, gt_crops, dataset_choice, ndim, image_counter, save_image_dir, save_gt_dir):
    for image_crop, gt_crop in tqdm(
        zip(image_crops, gt_crops), total=len(image_crops), desc=f"Saving {ndim}d crops for {dataset_choice}"
    ):
        fname = f"{dataset_choice}_{image_counter:05}.tif"

        if image_crop.ndim == 3 and image_crop.shape[-1] == 3:
            assert image_crop.shape[:2] == gt_crop.shape
        else:
            assert image_crop.shape == gt_crop.shape

        imageio.imwrite(os.path.join(save_image_dir, fname), image_crop, compression="zlib")
        imageio.imwrite(os.path.join(save_gt_dir, fname), gt_crop, compression="zlib")

        image_counter += 1

    return image_counter


def _get_image_label_paths(path, ndim):
    image_paths = natsorted(glob(os.path.join(path, f"roi_{ndim}d", "inputs", "*")))
    gt_paths = natsorted(glob(os.path.join(path, f"roi_{ndim}d", "labels", "*")))
    return image_paths, gt_paths


def _run_automatic_segmentation_per_dataset(
    image_paths: List[Union[os.PathLike, str]],
    gt_paths: List[Union[os.PathLike, str]],
    model_type: str,
    output_folder: Union[os.PathLike, str],
    ndim: Optional[int] = None,
    device: Optional[Union[torch.device, str]] = None,
    checkpoint_path: Optional[Union[os.PathLike, str]] = None,
    run_amg: Optional[bool] = None,
    **auto_seg_kwargs
):
    """Functionality to run automatic segmentation for multiple input files at once.
    It stores the evaluated automatic segmentation results (quantitative).

    Args:
        image_paths: List of filepaths for the input image data.
        gt_paths: List of filepaths for the corresponding label data.
        model_type: The choice of image encoder for the Segment Anything model.
        output_folder: Filepath to the folder where we store all the results.
        ndim: The number of input dimensions.
        device: The torch device.
        checkpoint_path: The filepath where the model checkpoints are stored.
        run_amg: Whether to run automatic segmentation in AMG mode.
        auto_seg_kwargs: Additional arguments for automatic segmentation parameters.
    """
    # First, we check if 'run_amg' is done, whether decoder is available or not.
    # Depending on that, we can set 'run_amg' to the default best automatic segmentation (i.e. AIS > AMG).
    if run_amg is None or (not run_amg):  # The 2nd condition checks if you want AIS and if decoder state exists or not.
        _, state = util.get_sam_model(
            model_type=model_type, checkpoint_path=checkpoint_path, device=device, return_state=True
        )
        run_amg = ("decoder_state" not in state)

    experiment_name = "AMG" if run_amg else "AIS"
    fname = f"{experiment_name.lower()}_{ndim}d"

    result_path = os.path.join(output_folder, "results", f"{fname}.csv")
    if os.path.exists(result_path):
        return

    prediction_dir = os.path.join(output_folder, fname, "inference")
    os.makedirs(prediction_dir, exist_ok=True)

    # Get the predictor (and the additional instance segmentation decoder, if available).
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type, checkpoint=checkpoint_path, device=device, amg=run_amg, is_tiled=False,
    )

    for image_path in tqdm(image_paths, desc=f"Run {experiment_name} in {ndim}d"):
        output_path = os.path.join(prediction_dir, os.path.basename(image_path))
        if os.path.exists(output_path):
            continue

        # Run Automatic Segmentation (AMG and AIS)
        automatic_instance_segmentation(
            predictor=predictor,
            segmenter=segmenter,
            input_path=image_path,
            output_path=output_path,
            ndim=ndim,
            verbose=False,
            **auto_seg_kwargs
        )

    prediction_paths = natsorted(glob(os.path.join(prediction_dir, "*")))
    run_evaluation(gt_paths=gt_paths, prediction_paths=prediction_paths, save_path=result_path)


def _run_interactive_segmentation_per_dataset(
    image_paths: List[Union[os.PathLike, str]],
    gt_paths: List[Union[os.PathLike, str]],
    output_folder: Union[os.PathLike, str],
    model_type: str,
    prompt_choice: Literal["box", "points"],
    device: Optional[Union[torch.device, str]] = None,
    ndim: Optional[int] = None,
    checkpoint_path: Optional[Union[os.PathLike, str]] = None,
    use_masks: bool = False,
):
    """Functionality to run interactive segmentation for multiple input files at once.
    It stores the evaluated interactive segmentation results.

    Args:
        image_paths: List of filepaths for the input image data.
        gt_paths: List of filepaths for the corresponding label data.
        output_folder: Filepath to the folder where we store all the results.
        model_type: The choice of model type for Segment Anything.
        prompt_choice: The choice of initial prompts to begin the interactive segmentation.
        device: The torch device.
        ndim: The number of input dimensions.
        checkpoint_path: The filepath for stored checkpoints.
        use_masks: Whether to use masks for iterative prompting.
    """
    if ndim == 2:
        # Get the Segment Anything predictor.
        predictor = util.get_sam_model(model_type=model_type, device=device, checkpoint_path=checkpoint_path)

        prediction_root = os.path.join(
            output_folder, "interactive_segmentation_2d", f"start_with_{prompt_choice}",
            "iterative_prompting_" + ("with_masks" if use_masks else "without_masks")
        )

        # Run interactive instance segmentation
        # (starting with box and points followed by iterative prompt-based correction)
        run_inference_with_iterative_prompting(
            predictor=predictor,
            image_paths=image_paths,
            gt_paths=gt_paths,
            embedding_dir=None,  # We set this to None to compute embeddings on-the-fly.
            prediction_dir=prediction_root,
            start_with_box_prompt=(prompt_choice == "box"),
            use_masks=use_masks,
            # TODO: add parameter for deform over box prompts (to simulate prompts in practice).
        )

        # Evaluate the interactive instance segmentation.
        run_evaluation_for_iterative_prompting(
            gt_paths=gt_paths,
            prediction_root=prediction_root,
            experiment_folder=output_folder,
            start_with_box_prompt=(prompt_choice == "box"),
            use_masks=use_masks,
        )

    else:
        save_path = os.path.join(output_folder, "results", f"interactive_segmentation_3d_with_{prompt_choice}.csv")
        if os.path.exists(save_path):
            print(
                f"Results for 3d interactive segmentation with '{prompt_choice}' are already stored at '{save_path}'."
            )
            return

        results = []
        for image_path, gt_path in tqdm(
            zip(image_paths, gt_paths), total=len(image_paths),
            desc=f"Run interactive segmentation in 3d with '{prompt_choice}'"
        ):
            prediction_dir = os.path.join(output_folder, "interactive_segmentation_3d", f"{prompt_choice}")
            os.makedirs(prediction_dir, exist_ok=True)

            prediction_path = os.path.join(prediction_dir, os.path.basename(image_path))
            if os.path.exists(prediction_path):
                continue

            per_vol_result = segment_slices_from_ground_truth(
                volume=imageio.imread(image_path),
                ground_truth=imageio.imread(gt_path),
                model_type=model_type,
                checkpoint_path=checkpoint_path,
                save_path=prediction_path,
                device=device,
                interactive_seg_mode=prompt_choice,
                min_size=10,
            )
            results.append(per_vol_result)

        results = pd.concat(results)
        results = results.groupby(results.index).mean()
        results.to_csv(save_path)


def _run_benchmark_evaluation_series(
    image_paths, gt_paths, model_type, output_folder, ndim, device, checkpoint_path, run_amg, evaluation_methods,
):
    seg_kwargs = {
        "image_paths": image_paths,
        "gt_paths": gt_paths,
        "output_folder": output_folder,
        "ndim": ndim,
        "model_type": model_type,
        "device": device,
        "checkpoint_path": checkpoint_path,
    }

    # Perform:
    # a. automatic segmentation (supported in both 2d and 3d, wherever relevant)
    #    The automatic segmentation steps below are configured in a way that AIS has priority (if decoder is found)
    #    Otherwise, it runs for AMG.
    #    Next, we check if the user expects to run AMG as well (after the run for AIS).

    if evaluation_methods != "interactive":  # Avoid auto. seg. evaluation for 'interactive'-only run choice.
        # i. Run automatic segmentation method supported with the SAM model (AMG or AIS).
        _run_automatic_segmentation_per_dataset(run_amg=None, **seg_kwargs)

        # ii. Run automatic mask generation (AMG).
        #     NOTE: This would only run if the user wants to. Else by default, it is set to 'False'.
        _run_automatic_segmentation_per_dataset(run_amg=run_amg, **seg_kwargs)

    if evaluation_methods != "automatic":  # Avoid int. seg. evaluation for 'automatic'-only run choice.
        # b. Run interactive segmentation (supported in both 2d and 3d, wherever relevant)
        _run_interactive_segmentation_per_dataset(prompt_choice="box", **seg_kwargs)
        _run_interactive_segmentation_per_dataset(prompt_choice="box", use_masks=True, **seg_kwargs)
        _run_interactive_segmentation_per_dataset(prompt_choice="points", **seg_kwargs)
        _run_interactive_segmentation_per_dataset(prompt_choice="points", use_masks=True, **seg_kwargs)


def _clear_cached_items(retain, path, output_folder):
    import shutil
    from pathlib import Path

    REMOVE_LIST = ["data", "crops", "automatic", "interactive"]
    if retain is None:
        remove_list = REMOVE_LIST
    else:
        assert isinstance(retain, list)
        remove_list = set(REMOVE_LIST) - set(retain)

    paths = []
    # Stage 1: Remove inputs.
    if "data" in remove_list or "crops" in remove_list:
        all_paths = glob(os.path.join(path, "*"))

        # In case we want to remove both data and crops, we remove the data folder entirely.
        if "data" in remove_list and "crops" in remove_list:
            paths.extend(all_paths)
            return

        # Next, we verify whether the we only remove either of data or crops.
        for curr_path in all_paths:
            if os.path.basename(curr_path).startswith("roi") and "crops" in remove_list:
                paths.append(curr_path)
            elif "data" in remove_list:
                paths.append(curr_path)

    # Stage 2: Remove predictions
    if "automatic" in remove_list:
        paths.extend(glob(os.path.join(output_folder, "amg_*")))
        paths.extend(glob(os.path.join(output_folder, "ais_*")))

    if "interactive" in remove_list:
        paths.extend(glob(os.path.join(output_folder, "interactive_segmentation_*")))

    [shutil.rmtree(_path) if Path(_path).is_dir() else os.remove(_path) for _path in paths]


def run_benchmark_evaluations(
    input_folder: Union[os.PathLike, str],
    dataset_choice: str,
    model_type: str = util._DEFAULT_MODEL,
    output_folder: Optional[Union[str, os.PathLike]] = None,
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    run_amg: bool = False,
    retain: Optional[List[str]] = None,
    evaluation_methods: Literal["all", "automatic", "interactive"] = "all",
    ignore_warnings: bool = False,
):
    """Run evaluation for benchmarking Segment Anything models on microscopy datasets.

    Args:
        input_folder: The path to directory where all inputs will be stored and preprocessed.
        dataset_choice: The dataset choice.
        model_type: The model choice for SAM.
        output_folder: The path to directory where all outputs will be stored.
        checkpoint_path: The checkpoint path
        run_amg: Whether to run automatic segmentation in AMG mode.
        retain: Whether to retain certain parts of the benchmark runs.
            By default, removes everything besides quantitative results.
            There is the choice to retain 'data', 'crops', 'automatic', or 'interactive'.
        evaluation_methods: The choice of evaluation methods.
            By default, runs 'all' evaluation methods (i.e. both 'automatic' or 'interactive').
            Otherwise, specify either 'automatic' / 'interactive' for specific evaluation runs.
        ignore_warnings: Whether to ignore warnings.
    """
    start = time.time()

    with _filter_warnings(ignore_warnings):
        device = util._get_default_device()

        # Ensure if all the datasets have been installed by default.
        dataset_choice = _download_benchmark_datasets(path=input_folder, dataset_choice=dataset_choice)

        for choice in dataset_choice:
            output_folder = os.path.join(output_folder, choice)
            result_dir = os.path.join(output_folder, "results")
            os.makedirs(result_dir, exist_ok=True)

            data_path = os.path.join(input_folder, choice)

            # Extrapolate desired set from the datasets:
            # a. for 2d datasets - 2d patches with the most number of labels present
            #    (in case of volumetric data, choose 2d patches per slice).
            # b. for 3d datasets - 3d regions of interest with the most number of labels present.
            ndim = _extract_slices_from_dataset(path=data_path, dataset_choice=choice, crops_per_input=10)

            # Run inference and evaluation scripts on benchmark datasets.
            image_paths, gt_paths = _get_image_label_paths(path=data_path, ndim=ndim)
            _run_benchmark_evaluation_series(
                image_paths=image_paths,
                gt_paths=gt_paths,
                model_type=model_type,
                output_folder=output_folder,
                ndim=ndim,
                device=device,
                checkpoint_path=checkpoint_path,
                run_amg=run_amg,
                evaluation_methods=evaluation_methods,
            )

            # Run inference and evaluation scripts on '2d' crops for volumetric datasets
            if ndim == 3:
                image_paths, gt_paths = _get_image_label_paths(path=data_path, ndim=2)
                _run_benchmark_evaluation_series(
                    image_paths=image_paths,
                    gt_paths=gt_paths,
                    model_type=model_type,
                    output_folder=output_folder,
                    ndim=2,
                    device=device,
                    checkpoint_path=checkpoint_path,
                    run_amg=run_amg,
                    evaluation_methods=evaluation_methods,
                )

            _clear_cached_items(retain=retain, path=data_path, output_folder=output_folder)

    diff = time.time() - start
    hours, rest = divmod(diff, 3600)
    minutes, seconds = divmod(rest, 60)
    print("Time taken for running benchmarks: ", f"{int(hours)}h {int(minutes)}m {int(seconds)}s")


def main():
    """@private"""
    import argparse

    available_models = list(util.get_model_names())
    available_models = ", ".join(available_models)

    parser = argparse.ArgumentParser(
        description="Run evaluation for benchmarking Segment Anything models on microscopy datasets."
    )
    parser.add_argument(
        "-i", "--input_folder", type=str, required=True,
        help="The path to a directory where the microscopy datasets are and/or will be stored."
    )
    parser.add_argument(
        "-m", "--model_type", type=str, default=util._DEFAULT_MODEL,
        help=f"The segment anything model that will be used, one of '{available_models}'. By default, "
        f"it uses'{util._DEFAULT_MODEL}'."
    )
    parser.add_argument(
        "-c", "--checkpoint_path", type=str, default=None,
        help="The filepath to checkpoint from which the SAM model will be loaded."
    )
    parser.add_argument(
        "-d", "--dataset_choice", type=str, nargs='*', default=None,
        help="The choice(s) of dataset for evaluating SAM models. Multiple datasets can be specified. "
        "By default, it evaluates on all datasets."
    )
    parser.add_argument(
        "-o", "--output_folder", type=str, required=True,
        help="The path where the results for (automatic and interactive) instance segmentation results "
        "will be stored as 'csv' files."
    )
    parser.add_argument(
        "--amg", action="store_true",
        help="Whether to run automatic segmentation in AMG mode (i.e. the default auto-seg approach for SAM)."
    )
    parser.add_argument(
        "--retain", nargs="*", default=None,
        help="By default, the functionality removes all besides quantitative results required for running benchmarks. "
        "In case you would like to retain parts of the benchmark evaluation for visualization / reproducibility, "
        "you should choose one or multiple of 'data', 'crops', 'automatic', 'interactive'. "
        "where they are responsible for either retaining original inputs / extracted crops / "
        "predictions of automatic segmentation / predictions of interactive segmentation, respectively."
    )
    parser.add_argument(
        "--evaluate", type=str, default="all", choices=["all", "automatic", "interactive"],
        help="The choice of methods for benchmarking evaluation for reproducibility. "
        "By default, we run all evaluations with 'all'. If 'automatic' is chosen, it runs automatic segmentation only "
        "/ 'interactive' runs interactive segmentation (starting from box and single point) with iterative prompting."
    )
    args = parser.parse_args()

    run_benchmark_evaluations(
        input_folder=args.input_folder,
        dataset_choice=args.dataset_choice,
        model_type=args.model_type,
        output_folder=args.output_folder,
        checkpoint_path=args.checkpoint_path,
        run_amg=args.amg,
        retain=args.retain,
        evaluation_methods=args.evaluate,
        ignore_warnings=True,
    )


if __name__ == "__main__":
    main()
