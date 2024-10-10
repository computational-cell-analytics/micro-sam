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
from ..automatic_segmentation import automatic_instance_segmentation
from .multi_dimensional_segmentation import segment_slices_from_ground_truth
from ..instance_segmentation import get_amg, get_decoder


LM_2D_DATASETS = [
    "livecell", "deepbacs", "tissuenet", "neurips_cellseg", "dynamicnuclearnet",
    "hpa", "covid_if", "pannuke", "lizard", "orgasegment", "omnipose", "dic_hepg2",
]

LM_3D_DATASETS = [
    "plantseg_root", "plantseg_ovules", "gonuclear", "mouse_embryo", "embegseg", "cellseg3d"
]

EM_2D_DATASETS = ["mitolab_tem"]

EM_3D_DATASETS = [
    "mitoem_rat", "mitoem_human", "platynereis_nuclei", "lucchi", "mitolab", "nuc_mm_mouse",
    "num_mm_zebrafish", "uro_cell", "sponge_em", "platynereis_cilia", "vnc", "asem_mito",
]

DATASET_RETURNS_FOLDER = {
    "deepbacs": "*.tif"
}

DATASET_CONTAINER_KEYS = {
    "lucchi": ["raw", "labels"],
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
        "livecell": lambda: datasets.livecell.get_livecell_data(
            path=os.path.join(path, "livecell"), split="test", download=True,
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
        "plantseg_root": lambda: datasets.plantseg.get_plantseg_data(
            path=os.path.join(path, "plantseg"), download=True, name="root",
        ),
        "plantseg_ovules": lambda: datasets.plantseg.get_plantseg_data(
            path=os.path.join(path, "plantseg"), download=True, name="ovules",
        ),
        "covid_if": lambda: datasets.covid_if.get_covid_if_data(
            path=os.path.join(path, "covid_if"), download=True,
        ),
        "hpa": lambda: datasets.hpa.get_hpa_segmentation_data(
            path=os.path.join(path, "hpa"), download=True,
        ),
        "dynamicnuclearnet": lambda: datasets.dynamicnuclearnet.get_dynamicnuclearnet_data(
            path=os.path.join(path, "dynamicnuclearnet"), split="test", download=True,
        ),
        "pannuke": lambda: datasets.pannuke.get_pannuke_data(
            path=os.path.join(path, "pannuke"), download=True, folds=["fold_1", "fold_2", "fold_3"],
        ),
        "lizard": lambda: datasets.lizard.get_lizard_data(
            path=os.path.join(path, "lizard"), download=True,
        ),
        "orgasegment": lambda: datasets.orgasegment.get_orgasegment_data(
            path=os.path.join(path, "orgasegment"), split="eval", download=True,
        ),
        "omnipose": lambda: datasets.omnipose.get_omnipose_data(
            path=os.path.join(path, "omnipose"), download=True,
        ),
        "gonuclear": lambda: datasets.gonuclear.get_gonuclear_data(
            path=os.path.join(path, "gonuclear"), download=True,
        ),
        "mouse_embryo": lambda: datasets.mouse_embryo.get_mouse_embryo_data(
            path=os.path.join(path, "mouse_embryo"), download=True,
        ),
        "embedseg_data": lambda: [
            datasets.embedseg_data.get_embedseg_data(path=os.path.join(path, "embedseg_data"), download=True, name=name)
            for name in datasets.embedseg_data.URLS.keys()
        ],
        "cellseg_3d": lambda: datasets.cellseg_3d.get_cellseg_3d_data(
            path=os.path.join(path, "cellseg_3d"), download=True,
        ),
        "dic_hepg2": lambda: datasets.dic_hepg2.get_dic_hepg2_data(
            path=os.path.join(path, "dic_hepg2"), download=True,
        ),

        # Electron Microscopy datasets
        "mitoem_rat": lambda: datasets.mitoem.get_mitoem_data(
            path=os.path.join(path, "mitoem"), samples="rat", split="test", download=True,
        ),
        "mitoem_human": lambda: datasets.mitoem.get_mitoem_data(
            path=os.path.join(path, "mitoem"), samples="human", split="test", download=True,
        ),
        "platynereis_nuclei": lambda: datasets.platynereis.get_platy_data(
            path=os.path.join(path, "platynereis"), name="nuclei", download=True,
        ),
        "platynereis_cilia": lambda: datasets.platynereis.get_platy_data(
            path=os.path.join(path, "platynereis"), name="cilia", download=True,
        ),
        "lucchi": lambda: datasets.lucchi.get_lucchi_data(
            path=os.path.join(path, "lucchi"), split="test", download=True,
        ),
        # TODO: split mitolab to all 3d vs TEM (i.e. 2d)
        "mitolab": lambda: [
            datasets.cem.get_benchmark_data(
                path=os.path.join(path, "mitolab"), dataset_id=dataset_id, download=True,
            ) for dataset_id in datasets.cem.BENCHMARK_DATASETS.keys()
        ],
        "nuc_mm_mouse": lambda: datasets.nuc_mm.get_nuc_mm_data(
            path=os.path.join(path, "nuc_mm"), sample="mouse", download=True,
        ),
        "nuc_mm_zebrafish": lambda: datasets.nuc_mm.get_nuc_mm_data(
            path=os.path.join(path, "nuc_mm"), sample="zebrafish", download=True,
        ),
        "uro_cell": lambda: datasets.uro_cell.get_uro_cell_data(
            path=os.path.join(path, "uro_cell"), download=True,
        ),
        "sponge_em": lambda: datasets.sponge_em.get_sponge_em_data(
            path=os.path.join(path, "sponge_em"), download=True,
        ),
        "vnc": lambda: datasets.vnc.get_vnc_data(
            path=os.path.join(path, "vnc"), download=True,
        ),
        "asem_mito": lambda: datasets.asem.get_asem_data(
            path=os.path.join(path, "asem"), volume_ids=datasets.asem.ORGANELLES["mito"], download=True,
        )
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
        "livecell": lambda: datasets.livecell.get_livecell_paths(path=path, split="test"),
        "deepbacs": lambda: datasets.deepbacs.get_deepbacs_paths(path=path, split="test", bac_type="mixed"),
        "plantseg_root": lambda: datasets.plantseg.get_plantseg_paths(path=path, name="root", split="test"),
        "plantseg_ovules": lambda: datasets.plantseg.get_plantseg_paths(path=path, name="ovules", split="test"),

        # Electron Microscopy datasets
        "lucchi": lambda: datasets.lucchi.get_lucchi_paths(path=path, split="test"),
    }

    if ndim == 2:
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

    _dir_exists = [
         os.path.exists(idir) and os.path.exists(gdir) for idir, gdir in zip(save_image_dir, save_gt_dir)
    ]
    if all(_dir_exists):
        return ndim

    [os.makedirs(idir, exist_ok=True) for idir in save_image_dir]
    [os.makedirs(gdir, exist_ok=True) for gdir in save_gt_dir]

    # Logic to extract relevant patches for inference
    image_counter = 1
    for per_paths in tqdm(paths_set, total=len(paths_set), desc=f"Extracting patches for {dataset_choice}"):
        if ndim == 2:
            image_path, gt_path = per_paths
            image, gt = util.load_image_data(image_path), util.load_image_data(gt_path)
        else:
            image_path = per_paths
            image = util.load_image_data(image_path, DATASET_CONTAINER_KEYS[dataset_choice][0])
            gt = util.load_image_data(image_path, DATASET_CONTAINER_KEYS[dataset_choice][1])

        skip_smaller_shape = (np.array(image.shape) >= np.array(tile_shape)).all()

        # Ensure ground truth has instance labels.
        gt = connected_components(gt)

        if len(np.unique(gt)) == 1:  # There could be labels which does not have any annotated foreground.
            continue

        # Let's extract and save all the crops.
        # NOTE: The first round of extraction is always to match the desired input dimensions.
        image_crops, gt_crops = _get_crops_for_input(image, gt, ndim, tile_shape, skip_smaller_shape, crops_per_input)
        _save_image_label_crops(
            image_crops, gt_crops, dataset_choice, ndim, image_counter, save_image_dir[0], save_gt_dir[0]
        )

        # NOTE: The next round of extraction is to get 2d crops from 3d inputs.
        if extract_2d_crops_from_volumes:
            curr_tile_shape = tile_shape[-2:]  # NOTE: We expect 2d tile shape for this stage.

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

            _save_image_label_crops(
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
        if skip_smaller_shape and crop_image.shape != tile_shape:
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
        assert image_crop.shape == gt_crop.shape
        imageio.imwrite(os.path.join(save_image_dir, fname), image_crop, compression="zlib")
        imageio.imwrite(os.path.join(save_gt_dir, fname), gt_crop, compression="zlib")
        image_counter += 1


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
    run_amg: bool = False,
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
    experiment_name = "AMG" if run_amg else "AIS"
    fname = f"{experiment_name.lower()}_{ndim}d"

    result_path = os.path.join(output_folder, "results", f"{fname}.csv")
    prediction_dir = os.path.join(output_folder, fname, "inference")
    if os.path.exists(prediction_dir):
        return
    
    os.makedirs(prediction_dir, exist_ok=True)

    # Get the predictor (and the additional instance segmentation decoder, if available).
    predictor, state = util.get_sam_model(
        model_type=model_type, device=device, checkpoint_path=checkpoint_path, return_state=True,
    )

    segmenter = get_amg(
        predictor=predictor,
        is_tiled=False,
        decoder=get_decoder(
            predictor.model.image_encoder, state["decoder_state"], device
        ) if "decoder_state" in state and not run_amg else None
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
    """
    if ndim == 2:
        # Get the Segment Anything predictor.
        predictor = util.get_sam_model(model_type=model_type, device=device, checkpoint_path=checkpoint_path)

        # Run interactive instance segmentation
        # (starting with box and points followed by iterative prompt-based correction)
        run_inference_with_iterative_prompting(
            predictor=predictor,
            image_paths=image_paths,
            gt_paths=gt_paths,
            embedding_dir=None,  # We set this to None to compute embeddings on-the-fly.
            prediction_dir=os.path.join(output_folder, "interactive_segmentation_2d", f"start_with_{prompt_choice}"),
            start_with_box_prompt=(prompt_choice == "box"),
            # TODO: add parameter for deform over box prompts (to simulate prompts in practice).
        )

        # Evaluate the interactive instance segmentation.
        run_evaluation_for_iterative_prompting(
            gt_paths=gt_paths,
            prediction_root=os.path.join(output_folder, "interactive_segmentation_2d", f"start_with_{prompt_choice}"),
            experiment_folder=output_folder,
            start_with_box_prompt=(prompt_choice == "box"),
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
    image_paths, gt_paths, model_type, output_folder, ndim, device, checkpoint_path, run_amg,
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
    #    Else, it runs for AMG.
    #    Next, we check if the user expects to run AMG as well (after the run for AIS).

    # i. Run automatic segmentation method supported with the SAM model (AMG or AIS).
    _run_automatic_segmentation_per_dataset(run_amg=run_amg, **seg_kwargs)

    # ii. Run automatic mask generation (AMG) (in case the first run is AIS).
    _run_automatic_segmentation_per_dataset(run_amg=run_amg, **seg_kwargs)

    # b. Run interactive segmentation (supported in both 2d and 3d, wherever relevant)
    _run_interactive_segmentation_per_dataset(prompt_choice="box", **seg_kwargs)
    _run_interactive_segmentation_per_dataset(prompt_choice="points", **seg_kwargs)


def run_benchmark_evaluations(
    input_folder: Union[os.PathLike, str],
    dataset_choice: str,
    model_type: str = util._DEFAULT_MODEL,
    output_folder: Optional[Union[str, os.PathLike]] = None,
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    run_amg: bool = False,
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
        ignore_warnings: Whether to ignore warnings.
    """
    start = time.time()

    with _filter_warnings(ignore_warnings):
        device = util._get_default_device()

        # Ensure if all the datasets have been installed by default.
        dataset_choice = _download_benchmark_datasets(path=input_folder, dataset_choice=dataset_choice)

        for choice in dataset_choice:
            output_folder = os.path.join(output_folder, choice)
            os.makedirs(os.path.join(output_folder, "results"), exist_ok=True)

            # Extrapolate desired set from the datasets:
            # a. for 2d datasets - 2d patches with the most number of labels present
            #    (in case of volumetric data, choose 2d patches per slice).
            # b. for 3d datasets - 3d regions of interest with the most number of labels present.
            ndim = _extract_slices_from_dataset(
                path=os.path.join(input_folder, choice), dataset_choice=choice, crops_per_input=10,
            )

            # Run inference and evaluation scripts on benchmark datasets.
            image_paths, gt_paths = _get_image_label_paths(path=os.path.join(input_folder, choice), ndim=ndim)
            _run_benchmark_evaluation_series(
                image_paths, gt_paths, model_type, output_folder, ndim, device, checkpoint_path, run_amg
            )

            # Run inference and evaluation scripts on '2d' crops for volumetric datasets
            if ndim == 3:
                image_paths, gt_paths = _get_image_label_paths(path=os.path.join(input_folder, choice), ndim=2)
                _run_benchmark_evaluation_series(
                    image_paths, gt_paths, model_type, output_folder, 2, device, checkpoint_path, run_amg
                )

    diff = time.time() - start
    hours, rest = divmod(diff, 3600)
    minutes, seconds = divmod(rest, 60)
    print("Time taken for running benchmarks: ", f"{int(hours)}h {int(minutes)}m {seconds:.2f}s")


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
        help="The path to a directory where the microscopy datasets are / will be stored."
    )
    parser.add_argument(
        "-m", "--model_type", type=str, default=util._DEFAULT_MODEL,
        help=f"The segment anything model that will be used, one of {available_models}."
    )
    parser.add_argument(
        "-c", "--checkpoint_path", type=str, default=None,
        help="Checkpoint from which the SAM model will be loaded loaded."
    )
    parser.add_argument(
        "-d", "--dataset_choice", type=str, nargs='*', default=None,
        help="The choice(s) of dataset for evaluating SAM models. Multiple datasets can be specified."
    )
    parser.add_argument(
        "-o", "--output_folder", type=str, required=True,
        help="The path where the results for automatic and interactive instance segmentation will be stored as 'csv'."
    )
    parser.add_argument(
        "--amg", action="store_true",
        help="Whether to run automatic segmentation in AMG mode (i.e. the default auto-seg approach for SAM)."
    )
    args = parser.parse_args()

    run_benchmark_evaluations(
        input_folder=args.input_folder,
        dataset_choice=args.dataset_choice,
        model_type=args.model_type,
        output_folder=args.output_folder,
        checkpoint_path=args.checkpoint_path,
        run_amg=args.amg,
        ignore_warnings=True,
    )


if __name__ == "__main__":
    main()
