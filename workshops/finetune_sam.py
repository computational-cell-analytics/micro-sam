"""Finetuning Segment Anything using µsam.

This python script shows how to use Segment Anything for Microscopy to fine-tune a Segment Anything Model (SAM)
on an open-source data with multiple channels.

We use confocal microscopy images from the HPA Kaggle Challenge for protein identification
(from Ouyang et al. - https://doi.org/10.1038/s41592-019-0658-6) in this script for the cell segmentation task.
The functionalities shown here should work for your (microscopy) images too.
"""


import os
from glob import glob
from pathlib import Path
from natsort import natsorted
from typing import Union, Tuple

import h5py
import numpy as np
import imageio.v3 as imageio
from matplotlib import pyplot as plt
from skimage.measure import label as connected_components

import torch

from torch_em.util.debug import check_loader
from torch_em.util.util import get_random_colors
from torch_em.data.datasets.light_microscopy.hpa import get_hpa_segmentation_paths

import micro_sam.training as sam_training
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation


def download_dataset(path: Union[os.PathLike, str]) -> Tuple[str, str]:
    """Download the HPA dataset.

    This functionality downloads the images, assorts the input data (thanks to `torch-em`)
    and stores the images and corresponding labels as `tif` files.

    Args:
        path: Filepath to the directory where the data will be stored.

    Returns:
        Filepath to the folder for the image data.
        Filepath to the folder for the label data.
    """
    # Download the data into a directory
    volume_paths = get_hpa_segmentation_paths(path=os.path.join(path, "hpa"), split="train", download=True)

    breakpoint()

    # Store inputs as tif files
    image_dir = os.path.join(path, "hpa", "preprocessed", "images")
    label_dir = os.path.join(path, "hpa", "preprocessed", "labels")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for volume_path in volume_paths:
        fname = Path(volume_path).stem

        with h5py.File(volume_path, "r") as f:
            # Get the channel-wise inputs
            image = np.stack([f["raw/microtubule"], f["raw/protein"], f["raw/nuclei"], f["raw/er"]], axis=-1)
            labels = f["labels"]

        image_path = os.path.join(image_dir, f"{fname}.tif")
        label_path = os.path.join(label_dir, f"{fname}.tif")

        imageio.imwrite(image_path, image, compression="zlib")
        imageio.imwrite(label_path, labels, compression="zlib")

    print(f"The inputs have been preprocessed and stored at: '{os.path.join(path, 'hpa', 'preprocessed')}'")

    return image_dir, label_dir


def verify_inputs(image_dir: Union[os.PathLike, str], label_dir: Union[os.PathLike, str]):
    """Verify the downloaded inputs.
    """
    image_paths = natsorted(glob(os.path.join(image_dir, "*.tif")))
    label_paths = natsorted(glob(os.path.join(label_dir, "*.tif")))

    for image_path, label_path in zip(image_paths, label_paths):
        image = imageio.imread(image_path)
        labels = imageio.imread(label_path)

        # The images should be of shape: H, W, 4 -> where, 4 is the number of channels.
        print(f"Shape of inputs: '{image.shape}'")
        # The labels should be of shape: H, W
        print(f"Shape of corresponding labels: '{labels.shape}'")

        break  # comment this line out in case you would like to verify the shapes for all inputs.


def preprocess_inputs(image_dir: Union[os.PathLike, str]):
    """Preprocess the input images.
    """
    image_paths = natsorted(glob(os.path.join(image_dir, "*.tif")))

    # We remove the 'er' channel, i.e. the last channel.
    for image_path in zip(image_paths):
        image = imageio.imread(image_path)
        image = image[..., :-1]
        imageio.imwrite(image_path, image)


def visualize_inputs(image_dir: Union[os.PathLike, str], label_dir: Union[os.PathLike, str]):
    ...


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, default="./data")
    args = parser.parse_args()

    # Step 1: Download the dataset.
    image_dir, label_dir = download_dataset(path=args.input_path)

    breakpoint()

    # Step 2: Verify the spatial shape of inputs.
    verify_inputs(image_dir=image_dir, label_dir=label_dir)

    # Step 3: Preprocess input images.
    preprocess_inputs(image_dir=image_dir)

    # Step 4: Visualize the images and corresponding labels.
    visualize_inputs(image_dir=image_dir, label_dir=label_dir)


if __name__ == "__main__":
    main()
