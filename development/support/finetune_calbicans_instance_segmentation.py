"""Finetune only the instance segmentation decoder (UNETR) of a Segment Anything model.

This script is tailored to the data from issue #1214 (computational-cell-analytics/micro-sam):
co-cultured mammalian cells (fluorescence: cytoplasm + DAPI) and hyphal cells (DIC). It trains
only the UNETR decoder for automatic instance segmentation (AIS), without updating the interactive
segmentation components (prompt encoder and mask decoder).

The important difference to 'examples/finetuning/finetune_hela_instance_segmentation.py' is the
'normalize_to_8bit' raw transform. The reported NaN losses under FP16 mixed precision were caused
by un-normalized inputs: the raw images are uint16 with intensities in the tens of thousands, while
SAM's image encoder normalizes internally with statistics calibrated for an 8-bit [0, 255] range.
Feeding the native range produces activations that overflow FP16 (max ~65504) -> NaNs, while FP32
survives. Normalizing each channel to [0, 255] here keeps training numerically stable.
"""

import os
from glob import glob

import numpy as np
import pooch

from torch_em.transform.raw import normalize_percentile

import micro_sam.training as sam_training
from micro_sam.util import get_device


DATA_URL = "https://github.com/user-attachments/files/28712031/data.zip"
DATA_HASH = "f3d63143f2fd16c99d09498620ce5a0e0b4fa1bb4da1d8ee18b43cac226b6d4e"


def download_data(save_directory):
    """Download and extract the example data from issue #1214.

    The archive contains '<name>.tif' images and matching '<name>_label.tif' label files.

    Args:
        save_directory: Folder to download and extract the data into.

    Returns:
        The folder containing the extracted '.tif' files.
    """
    os.makedirs(save_directory, exist_ok=True)
    fname = "data.zip"
    pooch.retrieve(
        url=DATA_URL, known_hash=DATA_HASH, fname=fname, path=save_directory,
        progressbar=True, processor=pooch.Unzip(),
    )
    data_dir = os.path.join(save_directory, f"{fname}.unzip")
    print("Example data directory is:", os.path.abspath(data_dir))
    return data_dir


def normalize_to_8bit(raw):
    """Normalize raw inputs of arbitrary intensity range (e.g. uint16) to the [0, 255] range.

    This is the key fix for the NaN losses under FP16 mixed precision (see issue #1214):
    SAM's image encoder expects inputs in an 8-bit range, and un-normalized uint16 intensities
    overflow FP16 in the forward pass.

    Args:
        raw: The raw image, either single channel '(H, W)' or multi-channel '(C, H, W)'.

    Returns:
        The normalized image in the [0, 255] range, with 3 channels for multi-channel inputs.
    """
    raw = raw.astype("float32")
    if raw.ndim == 2:  # Single channel image (e.g. DIC). Broadcast to 3 channels internally by SAM.
        return np.clip(normalize_percentile(raw), 0, 1) * 255

    # Multi-channel image (e.g. cytoplasm + DAPI). Normalize each channel independently.
    raw = np.stack([np.clip(normalize_percentile(channel), 0, 1) * 255 for channel in raw])

    # SAM requires exactly 3 channels. For 2-channel data, pad a zero channel as the third channel
    # (cytoplasm -> channel 1, DAPI -> channel 2, zeros -> channel 3), matching the training protocol.
    if raw.shape[0] == 2:
        raw = np.concatenate([raw, np.zeros_like(raw[:1])], axis=0)

    return raw


def get_dataloader(image_paths, label_paths, patch_shape, batch_size, with_channels, is_train):
    """Return a data loader for training the instance segmentation decoder.

    The loader returns 'x, y' tensors, where 'x' is the (normalized) image data and 'y' are the
    distance-based targets for the UNETR decoder. The labels must be in instance segmentation
    format, i.e. one consecutive ID per object and 0 for background.

    Args:
        image_paths: List of paths to the image files.
        label_paths: List of paths to the corresponding label files.
        patch_shape: The shape of patches for training.
        batch_size: The training batch size.
        with_channels: Whether the input images have an explicit channel axis (multi-channel data).
        is_train: Whether this loader is used for training or validation.

    Returns:
        The data loader.
    """
    # 'train_instance_segmentation_only=True' configures the label transform to produce only the
    # 3 distance-related channels (normalized distances, boundary distances, foreground probabilities),
    # which is the correct target format for 'train_instance_segmentation'.
    loader = sam_training.default_sam_loader(
        raw_paths=image_paths, raw_key=None,
        label_paths=label_paths, label_key=None,
        patch_shape=patch_shape, batch_size=batch_size,
        with_segmentation_decoder=True,
        train_instance_segmentation_only=True,
        with_channels=with_channels,
        raw_transform=normalize_to_8bit,
        is_train=is_train,
        num_workers=8, shuffle=True,
    )
    return loader


def get_loaders(data_root, patch_shape, batch_size, with_channels):
    """Build the train and validation loaders from a data directory.

    Expects an 'images' and a 'labels' subdirectory with matching '.tif' files. The last file is
    used for validation and the rest for training. Adapt this to your own data layout as needed.

    Args:
        data_root: The directory containing '<name>.tif' images and matching '<name>_label.tif' labels.
        patch_shape: The shape of patches for training.
        batch_size: The training batch size.
        with_channels: Whether the input images have an explicit channel axis.

    Returns:
        The train and validation loaders.
    """
    image_paths = sorted(p for p in glob(os.path.join(data_root, "*.tif")) if not p.endswith("_label.tif"))
    label_paths = [p.replace(".tif", "_label.tif") for p in image_paths]
    assert len(image_paths) > 0, f"No images found in '{data_root}'."
    for label_path in label_paths:
        assert os.path.exists(label_path), f"Missing label file '{label_path}'."

    # Use the last sample for validation and the rest for training. If only a single sample is
    # available (e.g. the downloaded example), reuse it for both splits.
    if len(image_paths) == 1:
        train_image_paths, val_image_paths = image_paths, image_paths
        train_label_paths, val_label_paths = label_paths, label_paths
    else:
        train_image_paths, val_image_paths = image_paths[:-1], image_paths[-1:]
        train_label_paths, val_label_paths = label_paths[:-1], label_paths[-1:]

    print(f"Train images ({len(train_image_paths)}), val images ({len(val_image_paths)})")

    train_loader = get_dataloader(
        train_image_paths, train_label_paths, patch_shape, batch_size, with_channels, is_train=True
    )
    val_loader = get_dataloader(
        val_image_paths, val_label_paths, patch_shape, batch_size, with_channels, is_train=False
    )
    return train_loader, val_loader


def run_training(checkpoint_name, model_type, train_loader, val_loader, n_epochs, save_root):
    """Run training of only the UNETR instance segmentation decoder.

    This trains the SAM image encoder together with the UNETR decoder, but does not train the
    prompt encoder or mask decoder.
    """
    sam_training.train_instance_segmentation(
        name=checkpoint_name,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        device=get_device(),
        save_root=save_root,
    )


def export_model(checkpoint_name, model_type, save_root, export_path):
    """Export the trained model for automatic instance segmentation (AIS).

    Note: the exported model is only suitable for automatic segmentation, not for interactive
    segmentation with prompts.
    """
    checkpoint_path = os.path.join(save_root or ".", "checkpoints", checkpoint_name, "best.pt")
    sam_training.export_instance_segmentation_model(
        trained_model_path=checkpoint_path, output_path=export_path, model_type=model_type,
    )
    print(f"The trained instance segmentation model is saved at '{export_path}'.")


def main():
    """Finetune the UNETR instance segmentation decoder of a Segment Anything model on custom data."""
    # The base model used to initialize the weights. 'vit_b_lm' is a good default for light microscopy.
    model_type = "vit_b_lm"

    # Download the example data from issue #1214. To train on your own data, replace this with the
    # path to a directory containing '<name>.tif' images and matching '<name>_label.tif' labels.
    data_root = download_data("./calbicans_data")

    # Where checkpoints and logs are stored (under '<save_root>/checkpoints/<checkpoint_name>').
    save_root = "./calbicans_instance_segmentation"
    checkpoint_name = "sam_calbicans_instance_segmentation"
    export_path = "./finetuned_calbicans_instance_segmentation_model.pth"

    n_epochs = 100
    batch_size = 1
    patch_shape = (512, 512)

    # Set to True for multi-channel data (e.g. cytoplasm + DAPI fluorescence). For single-channel
    # data (e.g. DIC), keep this False; the single channel is broadcast to 3 channels internally.
    with_channels = False

    train_loader, val_loader = get_loaders(data_root, patch_shape, batch_size, with_channels)
    run_training(checkpoint_name, model_type, train_loader, val_loader, n_epochs, save_root)
    export_model(checkpoint_name, model_type, save_root, export_path)


if __name__ == "__main__":
    main()
