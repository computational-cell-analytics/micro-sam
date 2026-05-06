import os

import micro_sam.training as sam_training
from micro_sam.util import get_device
from micro_sam.sample_data import fetch_tracking_example_data, fetch_tracking_segmentation_data


DATA_FOLDER = "data"


def get_dataloader(split, patch_shape, batch_size):
    """Return train or val data loader for training the instance segmentation decoder.

    The data loader must be a torch data loader that returns `x, y` tensors,
    where `x` is the image data and `y` are the labels.
    The labels have to be in a label mask instance segmentation format,
    i.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID.
    Important: the ID 0 is reserved for background, and the IDs must be consecutive.

    Here, we use `sam_training.default_sam_loader` for creating a suitable data loader from
    the example hela data. You can either adapt this for your own data (see comments below)
    or write a suitable torch dataloader yourself.
    """
    import numpy as np

    assert split in ("train", "val")
    os.makedirs(DATA_FOLDER, exist_ok=True)

    # This will download the image and segmentation data for training.
    image_dir = fetch_tracking_example_data(DATA_FOLDER)
    segmentation_dir = fetch_tracking_segmentation_data(DATA_FOLDER)

    # Load images from multiple files in folder via pattern (here: all tif files).
    raw_key, label_key = "*.tif", "*.tif"
    # Alternative: if you have tif stacks you can just set raw_key and label_key to None.
    # raw_key, label_key = None, None

    # The 'roi' argument can be used to subselect parts of the data.
    # Here, we use it to select the first 70 frames for the train split and the other frames for the val split.
    if split == "train":
        roi = np.s_[:70, :, :]
    else:
        roi = np.s_[70:, :, :]

    # 'default_sam_loader' creates a dataloader suitable for SAM training.
    # Setting 'train_instance_segmentation_only=True' configures the label transform to produce
    # only the 3 distance-related channels (normalized distances, boundary distances, foreground
    # probabilities), without the instance segmentation channel. This is the correct format
    # for 'train_instance_segmentation'.
    loader = sam_training.default_sam_loader(
        raw_paths=image_dir, raw_key=raw_key,
        label_paths=segmentation_dir, label_key=label_key,
        patch_shape=patch_shape, batch_size=batch_size,
        with_segmentation_decoder=True,
        train_instance_segmentation_only=True,
        is_train=(split == "train"),
        rois=roi,
        num_workers=8, shuffle=True, raw_transform=sam_training.identity,
    )
    return loader


def run_training(checkpoint_name, model_type):
    """Run training of the UNETR instance segmentation decoder."""

    # All hyperparameters for training.
    batch_size = 1  # the training batch size
    patch_shape = (1, 512, 512)  # the size of patches for training
    device = get_device()  # the device used for training

    # Get the dataloaders.
    train_loader = get_dataloader("train", patch_shape, batch_size)
    val_loader = get_dataloader("val", patch_shape, batch_size)

    # Run training of only the UNETR instance segmentation decoder.
    # This trains the SAM image encoder together with the UNETR decoder,
    # but does not train the prompt encoder or mask decoder.
    sam_training.train_instance_segmentation(
        name=checkpoint_name,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=100,
        device=device,
    )


def export_model(checkpoint_name, model_type):
    """Export the trained model.

    The exported model can be used with the micro_sam automatic instance segmentation (AIS) functionality.
    Note: the exported model is only suitable for automatic segmentation,
    not for interactive segmentation with prompts.
    """
    export_path = "./finetuned_hela_instance_segmentation_model.pth"
    checkpoint_path = os.path.join("checkpoints", checkpoint_name, "best.pt")
    sam_training.export_instance_segmentation_model(
        trained_model_path=checkpoint_path,
        output_path=export_path,
        model_type=model_type,
    )


def main():
    """Finetune only the instance segmentation decoder (UNETR) of a Segment Anything model.

    This example uses image data and segmentations from the cell tracking challenge,
    but can easily be adapted for other data (including data you have annotated with micro_sam beforehand).

    Unlike 'finetune_hela.py', this script trains only the UNETR decoder for automatic instance
    segmentation, without updating the interactive segmentation components (prompt encoder and mask decoder).
    Use this when you only need automatic instance segmentation (AIS) and not interactive segmentation.
    """
    # The model_type determines which base model is used to initialize the weights that are finetuned.
    # We use vit_b here because it can be trained faster. Note that vit_h usually yields higher quality results.
    model_type = "vit_b"

    # The name of the checkpoint. The checkpoints will be stored in './checkpoints/<checkpoint_name>'
    checkpoint_name = "sam_hela_instance_segmentation"

    run_training(checkpoint_name, model_type)
    export_model(checkpoint_name, model_type)


if __name__ == "__main__":
    main()
