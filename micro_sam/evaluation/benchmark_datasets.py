import os
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from typing import Union, Optional, List, Literal

import numpy as np
import imageio.v3 as imageio

from nifty.tools import blocking

import torch

from torch_em.data import datasets

from micro_sam import util

from . import run_evaluation
from .inference import run_inference_with_iterative_prompting
from .evaluation import run_evaluation_for_iterative_prompting
from ..automatic_segmentation import automatic_instance_segmentation


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


def _download_benchmark_datasets(path, dataset_choice):
    """Ensures whether all the datasets have been downloaded or not.

    Args:
        path: The path to directory where the supported datasets will be downloaded
            for benchmarking Segment Anything models.
        dataset_choice: The choice of dataset, expects the lower case name for the dataset.
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


def _extract_slices_from_dataset(path, dataset_choice):
    if dataset_choice in LM_2D_DATASETS or EM_2D_DATASETS:
        ndim, tile_shape = 2, (512, 512)
    else:
        ndim, tile_shape = 3, (32, 512, 512)

    available_datasets = {
        # Light Microscopy datasets
        "livecell": lambda: datasets.livecell._get_livecell_paths(path=path, split="test"),
        "deepbacs": lambda: datasets.deepbacs._get_deepbacs_paths(path=path, split="test", bac_type="mixed"),
    }

    image_paths, gt_paths = available_datasets[dataset_choice]()
    if dataset_choice in DATASET_RETURNS_FOLDER:
        image_paths = glob(os.path.join(image_paths, DATASET_RETURNS_FOLDER[dataset_choice]))
        gt_paths = glob(os.path.join(gt_paths, DATASET_RETURNS_FOLDER[dataset_choice]))

    image_paths, gt_paths = natsorted(image_paths), natsorted(gt_paths)
    assert len(image_paths) == len(gt_paths)

    # Directory where we store the extracted ROIs.
    save_image_dir = os.path.join(path, f"roi_{ndim}d", "inputs")
    save_gt_dir = os.path.join(path, f"roi_{ndim}d", "labels")
    if os.path.exists(save_image_dir) and os.path.exists(save_gt_dir):
        return save_image_dir, save_gt_dir, ndim

    os.makedirs(save_image_dir, exist_ok=True)
    os.makedirs(save_gt_dir, exist_ok=True)

    # Logic to extract relevant patches for inference
    image_counter = 1
    for image_path, gt_path in tqdm(
        zip(image_paths, gt_paths), total=len(image_paths),
        desc=f"Extracting patches for {dataset_choice}"
    ):
        image = imageio.imread(image_path)
        gt = imageio.imread(gt_path)

        if len(np.unique(gt)) == 1:  # There could be labels which does not have any annotated foreground.
            continue

        tiling = blocking([0, 0], gt.shape, tile_shape)
        n_tiles = tiling.numberOfBlocks
        tiles = [tiling.getBlock(tile_id) for tile_id in range(n_tiles)]
        crop_boxes = [[tile.begin[1], tile.begin[0], tile.end[1], tile.end[0]] for tile in tiles]

        n_ids = [idx for idx in range(len(crop_boxes))]
        n_instances = [
            len(np.unique(gt[crop_box[1]: crop_box[3], crop_box[0]: crop_box[2]])) for crop_box in crop_boxes
        ]

        # Extract the desired number of patches with higher number of instances.
        desired_tiles_per_image = 1
        image_crops, gt_crops = [], []
        for i, (per_n_instance, per_id) in enumerate(sorted(zip(n_instances, n_ids)), start=1):
            crop_box = crop_boxes[per_id]
            x0, y0, x1, y1 = crop_box
            crop_image, crop_gt = image[y0: y1, x0: x1], gt[y0: y1, x0: x1]

            # NOTE: There could be a case where some later patches are invalid.
            if per_n_instance == 1:
                break

            image_crops.append(crop_image)
            gt_crops.append(crop_gt)

            # NOTE: If the number of patches extracted have been fulfiled, we stop sampling patches.
            if len(image_crops) > 0 and i >= desired_tiles_per_image:
                break

        # Finally, let's store all the patches
        for image_crop, gt_crop in zip(image_crops, gt_crops):
            fname = f"{dataset_choice}_{image_counter:05}.tif"
            assert image_crop.shape == gt_crop.shape
            imageio.imwrite(os.path.join(save_image_dir, fname), image_crop, compression="zlib")
            imageio.imwrite(os.path.join(save_gt_dir, fname), gt_crop, compression="zlib")
            image_counter += 1

    return save_image_dir, save_gt_dir, ndim


def _get_image_label_paths(image_dir, gt_dir):
    image_paths = natsorted(glob(os.path.join(image_dir, "*")))
    gt_paths = natsorted(glob(os.path.join(gt_dir, "*")))
    return image_paths, gt_paths


def _run_automatic_segmentation_per_dataset(
    image_paths: List[Union[os.PathLike, str]],
    gt_paths: List[Union[os.PathLike, str]],
    model_type: str,
    output_folder: Union[os.PathLike, str],
    device: Optional[Union[torch.device, str]] = None,
    ndim: Optional[int] = None,
    checkpoint_path: Union[os.PathLike, str] = None,
    **auto_seg_kwargs
):
    """Functionality to run automatic segmentation for multiple input files at once.
    It stores the evaluated automatic segmentation results (quantitative).

    Args:
        image_paths:
        gt_paths:
        model_type:
        output_folder,
        ndim:
        checkpoint_path:
        auto_seg_kwargs:
    """
    result_path = os.path.join(output_folder, "results", "automatic_segmentation.csv")
    prediction_dir = os.path.join(output_folder, "automatic_instance_segmentation", "inference")
    os.makedirs(prediction_dir, exist_ok=True)

    for image_path in tqdm(image_paths, desc="Run automatic segmentation"):
        output_path = os.path.join(prediction_dir, os.path.basename(image_path))
        if os.path.exists(output_path):
            continue

        # Run Automatic Instance Segmentation (AIS)
        automatic_instance_segmentation(
            input_path=image_path,
            output_path=output_path,
            model_type=model_type,
            device=device,
            checkpoint_path=checkpoint_path,
            ndim=ndim,
            verbose=False,
            **auto_seg_kwargs
        )

    prediction_paths = natsorted(glob(os.path.join(prediction_dir, "*")))
    res = run_evaluation(gt_paths=gt_paths, prediction_paths=prediction_paths, save_path=result_path)
    print(res)


def _run_interactive_segmentation_per_dataset(
    image_paths: List[Union[os.PathLike, str]],
    gt_paths: List[Union[os.PathLike, str]],
    output_folder: Union[os.PathLike, str],
    model_type: str,
    prompt_choice: Literal["box", "point"],
    device: Optional[Union[torch.device, str]] = None,
    ndim: Optional[int] = None,  # TODO: extend iterative prompting backbone to 3d (cc: SAM)
    checkpoint_path=None,
):
    """Functionality to run interactive segmentation for multiple input files at once.
    It stores the evaluated interactive segmentation results.

    Args:
        image_paths:
        gt_paths:
        model_type:
        output_folder:
        prompt_choice:
        device:
        ndim:
        checkpoint_path:
    """
    if ndim == 3:
        raise NotImplementedError("Integration WIP")

    # Get the Segment Anything predictor.
    predictor = util.get_sam_model(model_type=model_type, device=device, checkpoint_path=checkpoint_path)

    # Path where the interactive segmentation results will be stored.
    prediction_dir = os.path.join(output_folder, "interactive_segmentation", f"start_with_{prompt_choice}")

    # Run interactive instance segmentation
    # (starting with box and points followed by iterative prompt-based correction)
    run_inference_with_iterative_prompting(
        predictor=predictor,
        image_paths=image_paths,
        gt_paths=gt_paths,
        prediction_dir=prediction_dir,
        start_with_box_prompt=(prompt_choice == "box"),
        # TODO: add parameter for deform over box prompts (to simulate prompts in practice).
    )

    # Evaluate the interactive instance segmentation.
    run_evaluation_for_iterative_prompting(
        gt_paths=gt_paths,
        prediction_root=prediction_dir,
        experiment_folder=output_folder,
        start_with_box_prompt=(prompt_choice == "box"),
    )


def run_benchmark_evaluations(
    input_folder: Union[os.PathLike, str],
    dataset_choice: str,
    model_type: str = util._DEFAULT_MODEL,
    output_folder: Optional[Union[str, os.PathLike]] = None,
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
):
    """Run evaluation for benchmarking Segment Anything models on microscopy datasets.

    Args:
        input_folder: The path to directory where all inputs will be stored and preprocessed.
        dataset_choice: The dataset choice.
        model_type: The model choice for SAM.
        output_folder: The path to directory where all outputs will be stored.
        checkpoint_path: The checkpoint path
    """
    device = util._get_default_device()

    # Ensure if all the datasets have been installed by default.
    dataset_choice = _download_benchmark_datasets(path=input_folder, dataset_choice=dataset_choice)

    for choice in dataset_choice:
        output_folder = os.path.join(output_folder, choice)

        # Extrapolate desired set from the datasets:
        # a. for 2d datasets - 2d patches with the most number of labels present
        #    (in case of volumetric data, choose 2d patches per slice).
        # b. for 3d datasets - 3d regions of interest with the most number of labels present.
        image_dir, gt_dir, ndim = _extract_slices_from_dataset(
            path=os.path.join(input_folder, choice), dataset_choice=choice
        )
        image_paths, gt_paths = _get_image_label_paths(image_dir=image_dir, gt_dir=gt_dir)

        os.makedirs(os.path.join(output_folder, "results"), exist_ok=True)
        seg_kwargs = {
            "image_paths": image_paths,
            "gt_paths": gt_paths,
            "model_type": model_type,
            "output_folder": output_folder,
            "ndim": ndim,
            "device": device,
            "checkpoint_path": checkpoint_path
        }
        # Perform a. automatic segmentation (in both 2d and 3d, wherever relevant)
        _run_automatic_segmentation_per_dataset(**seg_kwargs)

        # b. interactive segmentation (in both 2d and 3d, wherever relevant)
        _run_interactive_segmentation_per_dataset(prompt_choice="box", **seg_kwargs)
        _run_interactive_segmentation_per_dataset(prompt_choice="point", **seg_kwargs)


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
    args = parser.parse_args()

    run_benchmark_evaluations(
        input_folder=args.input_folder,
        dataset_choice=args.dataset_choice,
        model_type=args.model_type,
        output_folder=args.output_folder,
        checkpoint_path=args.checkpoint_path,
    )


if __name__ == "__main__":
    main()
