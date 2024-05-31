import os

import torch
from torch.utils.data import random_split

import torch_em
from torch_em.transform.label import PerObjectDistanceTransform

import micro_sam.training as sam_training


DATA_FOLDER = "data"  # overwrite to provide the path to your data directory


def get_dataloader(patch_shape, batch_size, train_instance_segmentation):
    """Return train or val data loader for finetuning SAM.

    The data loader must be a torch data loader that retuns `x, y` tensors,
    where `x` is the image data and `y` are the labels.
    The labels have to be in a label mask instance segmentation format.
    I.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID.
    Important: the ID 0 is reseved for background, and the IDs must be consecutive

    Here, we use `torch_em.default_segmentation_loader` for creating a suitable data loader from
    your custom EM data. You can either adapt this for your own data (see comments below)
    or write a suitable torch dataloader yourself.
    """
    os.makedirs(DATA_FOLDER, exist_ok=True)

    # Provide the image and segmentation data directories for training.
    image_dir = os.path.join(DATA_FOLDER, "images")
    segmentation_dir = os.path.join(DATA_FOLDER, "labels")

    # torch_em.default_segmentation_loader is a convenience function to build a torch dataloader
    # from image data and labels for training segmentation models.
    # It supports image data in various formats. Here, we load image data and labels from the two
    # folders with tif images (both inputs and respective segmentations are tif files), by specifying
    # `raw_key` and `label_key` as `*.tif`. This means all images in the respective folders that end with
    # .tif will be loadded.
    # The function supports many other file formats. For example, if you have tif stacks with multiple slices
    # instead of multiple tif images in a foldder, then you can pass raw_key=label_key=None.

    # Load images from multiple files in folder via pattern (here: all tif files)
    raw_key, label_key = "*.tif", "*.tif"
    # Alternative: if you have tif stacks you can just set raw_key and label_key to None
    # raw_key, label_key= None, None

    if train_instance_segmentation:
        # Computes the distance transform for objects to perform end-to-end automatic instance segmentation.
        label_transform = PerObjectDistanceTransform(
            distances=True,
            boundary_distances=True,
            directed_distances=False,
            foreground=True,
            instances=True,
            min_size=25
        )
    else:
        label_transform = torch_em.transform.label.connected_components

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=image_dir,
        raw_key=raw_key,
        label_paths=segmentation_dir,
        label_key=label_key,
        patch_shape=patch_shape,
        ndim=2,
        is_seg_dataset=True,
        label_transform=label_transform,
        raw_transform=sam_training.identity,
    )

    # Use 10% of the dataset - at least one image - for validation.
    n_val = min(1, int(0.1 * len(dataset)))
    train_dataset, val_dataset = random_split(dataset, lengths=[len(dataset) - n_val, n_val])

    train_loader = torch_em.get_data_loader(dataset=train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    val_loader = torch_em.get_data_loader(dataset=val_dataset, batch_size=batch_size, num_workers=8, shuffle=True)

    return train_loader, val_loader


def run_training(checkpoint_name, model_type, train_instance_segmentation):
    """
    Run the actual model training.
    """
    # All hyperparameters for training.
    batch_size = 1  # the training batch size
    patch_shape = (1, 512, 512)  # the size of patches for training
    n_objects_per_batch = 25  # the number of objects per batch that will be sampled
    device = torch.device("cuda")  # the device/GPU used for training

    # Get the dataloaders.
    train_loader, val_loader = get_dataloader(patch_shape, batch_size, train_instance_segmentation)

    # Run training.
    sam_training.train_sam(
        name=checkpoint_name,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=100,
        n_objects_per_batch=n_objects_per_batch,
        with_segmentation_decoder=train_instance_segmentation,
        device=device,
    )


def export_model(checkpoint_name, model_type):
    """
    Export the trained model.
    """
    from micro_sam.util import export_custom_sam_model

    # export the model after training so that it can be used by the rest of the micro_sam library
    export_path = "./finetuned_chromosomes_model.pth"
    checkpoint_path = os.path.join("checkpoints", checkpoint_name, "best.pt")
    export_custom_sam_model(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        save_path=export_path,
    )


def main():
    """
    Finetune a Segment Anything model.

    This script would use the image data and segmentations for segmenting chromosomes in electron microscopy,
    but can easily be adapted for other data (including data you have annoated with micro_sam beforehand).
    """
    # The model_type determines which base model is used to initialize the weights that are finetuned.
    # We use vit_b here because it can be trained faster. Note that vit_h usually yields higher quality results.

    # NOTE: We use the vit_b_em_organelles generalist model as it performs a bit better
    # compare to the default SAM models for segmenting the chromosomes in EM.
    model_type = "vit_b_em_organelles"

    # The name of the checkpoint. The checkpoints will be stored in './checkpoints/<checkpoint_name>'
    checkpoint_name = "sam_chromosomes"

    # Train an additional convolutional decoder for end-to-end automatic instance segmentation
    train_instance_segmentation = True

    run_training(checkpoint_name, model_type, train_instance_segmentation)


if __name__ == "__main__":
    main()
