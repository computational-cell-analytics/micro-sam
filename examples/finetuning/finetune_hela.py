import os

import numpy as np

import torch
import torch_em
from torch_em.data.sampler import MinInstanceSampler

from micro_sam.util import get_device
from micro_sam.v2.training import train_sam2
from micro_sam.v2.normalization import normalize_raw
from micro_sam.v2.datasets.wrapper import UniDataWrapper
from micro_sam.sample_data import fetch_tracking_example_data, fetch_tracking_segmentation_data


DATA_FOLDER = "data"


def get_dataloader(split, patch_shape, batch_size):
    """Return the train or val data loader for finetuning micro-sam2.

    The data loader must be a torch data loader that returns `x, y` tensors, where `x` is the image data
    and `y` are the labels. The labels have to be in a label mask instance segmentation format,
    i.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID.
    Important: the ID 0 is reserved for background.

    Here, we use `torch_em.default_segmentation_dataset` to load the example HeLa data.
    You can adapt this for your own data (see comments below) or write a suitable torch dataset yourself.
    """
    assert split in ("train", "val")
    os.makedirs(DATA_FOLDER, exist_ok=True)

    # This will download the image and segmentation data for training.
    image_dir = fetch_tracking_example_data(DATA_FOLDER)
    segmentation_dir = fetch_tracking_segmentation_data(DATA_FOLDER)

    # We load the images and labels from the two folders with tif images that were downloaded above,
    # by setting `raw_key` and `label_key` to `*.tif`. This loads all tif images in a folder as one stack.
    # If you have tif stacks with multiple slices instead of a folder of images, set both keys to None.
    raw_key, label_key = "*.tif", "*.tif"

    # The 'rois' argument selects a part of the data.
    # Here, we select the first 70 frames for the train split and the other frames for the val split.
    roi = np.s_[:70, :, :] if split == "train" else np.s_[70:, :, :]

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_dir, raw_key=raw_key,
        label_paths=segmentation_dir, label_key=label_key,
        patch_shape=patch_shape, ndim=2, rois=roi,
        # Normalize the images to [0, 1]. 'UniDataWrapper' then copies a grayscale image to three channels.
        # Use a function from an importable module here: the checkpoint stores a reference to it.
        raw_transform=normalize_raw,
        sampler=MinInstanceSampler(min_num_instances=3),
        label_dtype=torch.int64,
    )
    # The micro-sam2 trainers expect every sample as (channels, z, y, x). A 2d image has a single z slice.
    dataset = UniDataWrapper(dataset, source_ndim=2)
    return torch_em.get_data_loader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=8)


def main():
    """Finetune a micro-sam2 model for interactive segmentation.

    This example uses image data and segmentations from the cell tracking challenge,
    but can easily be adapted for other data (including data you have annotated with micro_sam beforehand).
    To also finetune the decoder for automatic segmentation, see 'finetune_hela_instance_segmentation.py'.
    """
    # The model that is finetuned. We start from the micro-sam2 model for cells.
    # You can also start from a base SAM2 model, e.g. 'hvit_t', or use a larger model, e.g. 'hvit_b_cells'.
    model_type = "hvit_t_cells"

    # The name of the checkpoint. The checkpoints will be stored in './checkpoints/<checkpoint_name>'.
    checkpoint_name = "sam2_hela"

    batch_size = 1  # The training batch size.
    patch_shape = (1, 512, 512)  # The size of the patches for training.

    train_loader = get_dataloader("train", patch_shape, batch_size)
    val_loader = get_dataloader("val", patch_shape, batch_size)

    train_sam2(
        name=checkpoint_name,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=50,
        max_num_objects=25,  # The number of objects per image that are sampled per training step.
        device=get_device(),
    )

    # The best checkpoint can be used directly in the rest of micro_sam, for example in the annotator
    # (see 'annotator_with_finetuned_model.py'). The model size is read from the checkpoint.
    print("The finetuned model is saved at", os.path.join("checkpoints", checkpoint_name, "best.pt"))


if __name__ == "__main__":
    main()
