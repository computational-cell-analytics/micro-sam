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
from torch.utils.data import DataLoader

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
    volume_paths = get_hpa_segmentation_paths(path=os.path.join(path, "hpa"), split="test", download=True)

    # Store inputs as tif files
    image_dir = os.path.join(path, "hpa", "preprocessed", "images")
    label_dir = os.path.join(path, "hpa", "preprocessed", "labels")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    for volume_path in volume_paths:
        fname = Path(volume_path).stem

        with h5py.File(volume_path, "r") as f:
            # Get the channel-wise inputs
            image = np.stack(
                [f["raw/microtubules"][:], f["raw/protein"][:], f["raw/nuclei"][:], f["raw/er"][:]], axis=-1
            )
            # labels = f["labels"][:]

        image_path = os.path.join(image_dir, f"{fname}.tif")
        # label_path = os.path.join(label_dir, f"{fname}.tif")

        imageio.imwrite(image_path, image, compression="zlib")
        # imageio.imwrite(label_path, labels, compression="zlib")

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
    """
    """
    image_paths = natsorted(glob(os.path.join(image_dir, "*.tif")))
    label_paths = natsorted(glob(os.path.join(label_dir, "*.tif")))

    for image_path, label_path in zip(image_paths, label_paths):
        image = imageio.imread(image_path)
        labels = imageio.imread(label_path)

        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].imshow(image, cmap="gray")
        ax[0].set_title("Input Image")
        ax[0].axis("off")

        labels = connected_components(labels)
        ax[1].imshow(labels, cmap=get_random_colors(labels), interpolation="nearest")
        ax[1].set_title("Ground Truth Instances")
        ax[1].axis("off")

        plt.show()
        plt.close()

        break  # comment this out in case you want to visualize all the images


def get_dataloaders(
    image_dir: Union[os.PathLike, str],
    label_dir: Union[os.PathLike, str],
    view: bool,
    train_instance_segmentation: bool,
) -> Tuple[DataLoader, DataLoader]:
    """
    """
    # Get filepaths to the image and corresponding label data.
    image_paths = natsorted(glob(os.path.join(image_dir, "*.tif")))
    label_paths = natsorted(glob(os.path.join(label_dir, "*.tif")))

    # Load images from tif stacks by setting `raw_key` and `label_key` to None.
    raw_key, label_key = None, None

    # Split the image and corresponding labels to establish train-test split.
    # Here, we select the first 2000 images for the train split and the other frames for the val split.
    train_image_paths, val_image_paths = image_paths[:2000], image_paths[2000:]
    train_label_paths, val_label_paths = label_paths[:2000], label_paths[2000:]

    batch_size = 1  # the training batch size
    patch_shape = (512, 512)  # the size of patches for training

    train_loader = sam_training.default_sam_loader(
        raw_paths=train_image_paths,
        raw_key=raw_key,
        label_paths=train_label_paths,
        label_key=label_key,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        with_channels=True,
        with_segmentation_decoder=train_instance_segmentation,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = sam_training.default_sam_loader(
        raw_paths=val_image_paths,
        raw_key=raw_key,
        label_paths=val_label_paths,
        label_key=label_key,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        with_channels=True,
        with_segmentation_decoder=train_instance_segmentation,
        batch_size=batch_size,
        shuffle=True,
    )

    if view:
        check_loader(train_loader, 4, plt=True)

    return train_loader, val_loader


def run_finetuning(
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_root: Union[os.PathLike, str],
    train_instance_segmentation: bool
) -> str:
    """
    """
    # All hyperparameters for training.
    n_objects_per_batch = 5  # the number of objects per batch that will be sampled
    device = "cuda" if torch.cuda.is_available() else "cpu"  # the device/GPU used for training
    n_epochs = 10  # how long we train (in epochs)

    # The model_type determines which base model is used to initialize the weights that are finetuned.
    # We use vit_b here because it can be trained faster. Note that vit_h usually yields higher quality results.
    model_type = "vit_b"

    # The name of the checkpoint. The checkpoints will be stored in './checkpoints/<checkpoint_name>'
    checkpoint_name = "sam_hpa"

    # Run training
    sam_training.train_sam(
        name=checkpoint_name,
        save_root=os.path.join(save_root, "models"),
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        n_objects_per_batch=n_objects_per_batch,
        with_segmentation_decoder=train_instance_segmentation,
        device=device,
    )

    # Let's spot our best checkpoint and download it to get started with the annotation tool
    best_checkpoint = os.path.join(save_root, "models", "checkpoints", checkpoint_name, "best.pt")

    return best_checkpoint


def run_automatic_instance_segmentation(
    model_type: str,
    checkpoint: Union[os.PathLike, str],
    device: Union[torch.device, str],
    test_image_dir: Union[os.PathLike, str],
):
    """
    """
    assert os.path.exists(checkpoint), "Please train the model first to run inference on the finetuned model."

    # Get the 'predictor' and 'segmenter' to perform automatic instance segmentation.
    predictor, segmenter = get_predictor_and_segmenter(model_type=model_type, checkpoint=checkpoint, device=device)

    # Let's check the last 10 images. Feel free to comment out the line below to run inference on all images.
    image_paths = image_paths[-10:]

    for image_path in image_paths:
        image = imageio.imread(image_path)

        # Predicted instances
        prediction = run_automatic_instance_segmentation(
            image=image, checkpoint_path=checkpoint, model_type=model_type, device=device
        )

        # Visualize the predictions
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))

        ax[0].imshow(image, cmap="gray")
        ax[0].axis("off")
        ax[0].set_title("Input Image")

        ax[1].imshow(prediction, cmap=get_random_colors(prediction), interpolation="nearest")
        ax[1].axis("off")
        ax[1].set_title("Predictions (AIS)")

        plt.show()
        plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, default="./data")
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()

    # Step 1: Download the dataset.
    image_dir, label_dir = download_dataset(path=args.input_path)

    # Step 2: Verify the spatial shape of inputs.
    verify_inputs(image_dir=image_dir, label_dir=label_dir)

    # Step 3: Preprocess input images.
    preprocess_inputs(image_dir=image_dir)

    # Step 4: Visualize the images and corresponding labels.
    visualize_inputs(image_dir=image_dir, label_dir=label_dir)

    # Step 5: Get the dataloaders.
    # Train an additional convolutional decoder for end-to-end automatic instance segmentation
    train_instance_segmentation = True

    train_loader, val_loader = get_dataloaders(
        image_dir=image_dir,
        label_dir=label_dir,
        view=args.view,
        train_instance_segmentation=train_instance_segmentation,
    )

    # Step 6: Run the finetuning for Segment Anything Model.
    checkpoint_path = run_finetuning(
        train_loader=train_loader,
        val_loader=val_loader,
        save_root=args.save_root,
        train_instance_segmentation=train_instance_segmentation,
    )

    # Step 7: Run automatic instance segmentation using the finetuned model.
    run_automatic_instance_segmentation()


if __name__ == "__main__":
    main()
