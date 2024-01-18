import os
import numpy as np

import torch

import torch_em
from torch_em.model import UNETR
from torch_em.loss import DiceBasedDistanceLoss
from torch_em.transform.label import PerObjectDistanceTransform

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model
from micro_sam.sample_data import fetch_tracking_example_data, fetch_tracking_segmentation_data


DATA_FOLDER = "data"


def get_dataloader(split, patch_shape, batch_size, train_instance_segmentation):
    """Return train or val data loader for finetuning SAM.

    The data loader must be a torch data loader that retuns `x, y` tensors,
    where `x` is the image data and `y` are the labels.
    The labels have to be in a label mask instance segmentation format.
    I.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID.
    Important: the ID 0 is reseved for background, and the IDs must be consecutive

    Here, we use `torch_em.default_segmentation_loader` for creating a suitable data loader from
    the example hela data. You can either adapt this for your own data (see comments below)
    or write a suitable torch dataloader yourself.
    """
    assert split in ("train", "val")
    os.makedirs(DATA_FOLDER, exist_ok=True)

    # This will download the image and segmentation data for training.
    image_dir = fetch_tracking_example_data(DATA_FOLDER)
    segmentation_dir = fetch_tracking_segmentation_data(DATA_FOLDER)

    # torch_em.default_segmentation_loader is a convenience function to build a torch dataloader
    # from image data and labels for training segmentation models.
    # It supports image data in various formats. Here, we load image data and labels from the two
    # folders with tif images that were downloaded by the example data functionality, by specifying
    # `raw_key` and `label_key` as `*.tif`. This means all images in the respective folders that end with
    # .tif will be loadded.
    # The function supports many other file formats. For example, if you have tif stacks with multiple slices
    # instead of multiple tif images in a foldder, then you can pass raw_key=label_key=None.

    # Load images from multiple files in folder via pattern (here: all tif files)
    raw_key, label_key = "*.tif", "*.tif"
    # Alternative: if you have tif stacks you can just set raw_key and label_key to None
    # raw_key, label_key= None, None

    # The 'roi' argument can be used to subselect parts of the data.
    # Here, we use it to select the first 70 frames for the train split and the other frames for the val split.
    if split == "train":
        roi = np.s_[:70, :, :]
    else:
        roi = np.s_[70:, :, :]

    if train_instance_segmentation:
        # Computes the distance transform for objects to perform end-to-end automatic instance segmentation.
        label_transform = PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False,
            foreground=True, instances=True, min_size=25
        )
    else:
        label_transform = torch_em.transform.label.connected_components

    loader = torch_em.default_segmentation_loader(
        raw_paths=image_dir, raw_key=raw_key,
        label_paths=segmentation_dir, label_key=label_key,
        patch_shape=patch_shape, batch_size=batch_size,
        ndim=2, is_seg_dataset=True, rois=roi,
        label_transform=label_transform,
        num_workers=8, shuffle=True, raw_transform=sam_training.identity,
    )
    return loader


def run_training(checkpoint_name, model_type, train_instance_segmentation):
    """Run the actual model training."""

    # All hyperparameters for training.
    batch_size = 1  # the training batch size
    patch_shape = (1, 512, 512)  # the size of patches for training
    n_objects_per_batch = 25  # the number of objects per batch that will be sampled
    device = torch.device("cuda")  # the device/GPU used for training
    n_iterations = 10000  # how long we train (in iterations)

    # Get the dataloaders.
    train_loader = get_dataloader("train", patch_shape, batch_size, train_instance_segmentation)
    val_loader = get_dataloader("val", patch_shape, batch_size, train_instance_segmentation)

    # Get the segment anything model
    model = sam_training.get_trainable_sam_model(model_type=model_type, device=device)

    # This class creates all the training data for a batch (inputs, prompts and labels).
    convert_inputs = sam_training.ConvertToSamInputs(transform=model.transform, box_distortion_factor=0.025)

    # Get the optimizer and the LR scheduler
    if train_instance_segmentation:
        # for instance segmentation, we use the UNETR model configuration.
        unetr = UNETR(
            backbone="sam", encoder=model.sam.image_encoder, out_channels=3, use_sam_stats=True,
            final_activation="Sigmoid", use_skip_connection=False, resize_input=True,
        )
        # let's get the parameters for SAM and the decoder from UNETR
        joint_model_params = [params for params in model.parameters()]  # sam parameters
        for name, params in unetr.named_parameters():  # unetr's decoder parameters
            if not name.startswith("encoder"):
                joint_model_params.append(params)
        unetr.to(device)
        optimizer = torch.optim.Adam(joint_model_params, lr=1e-5)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=10, verbose=True)

    # the trainer which performs training and validation (implemented using "torch_em")
    if train_instance_segmentation:
        instance_seg_loss = DiceBasedDistanceLoss(mask_distances_in_bg=True)
        trainer = sam_training.JointSamTrainer(
            name=checkpoint_name, train_loader=train_loader, val_loader=val_loader, model=model,
            optimizer=optimizer, device=device, lr_scheduler=scheduler, logger=sam_training.JointSamLogger,
            log_image_interval=100, mixed_precision=True, convert_inputs=convert_inputs,
            n_objects_per_batch=n_objects_per_batch, n_sub_iteration=8, compile_model=False, unetr=unetr,
            instance_loss=instance_seg_loss, instance_metric=instance_seg_loss
        )
    else:
        trainer = sam_training.SamTrainer(
            name=checkpoint_name, train_loader=train_loader, val_loader=val_loader, model=model,
            optimizer=optimizer, device=device, lr_scheduler=scheduler, logger=sam_training.SamLogger,
            log_image_interval=100, mixed_precision=True, convert_inputs=convert_inputs,
            n_objects_per_batch=n_objects_per_batch, n_sub_iteration=8, compile_model=False
        )
    trainer.fit(n_iterations)


def export_model(checkpoint_name, model_type):
    """Export the trained model."""
    # export the model after training so that it can be used by the rest of the micro_sam library
    export_path = "./finetuned_hela_model.pth"
    checkpoint_path = os.path.join("checkpoints", checkpoint_name, "best.pt")
    export_custom_sam_model(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        save_path=export_path,
    )


def main():
    """Finetune a Segment Anything model.

    This example uses image data and segmentations from the cell tracking challenge,
    but can easily be adapted for other data (including data you have annoated with micro_sam beforehand).
    """
    # The model_type determines which base model is used to initialize the weights that are finetuned.
    # We use vit_b here because it can be trained faster. Note that vit_h usually yields higher quality results.
    model_type = "vit_b"

    # The name of the checkpoint. The checkpoints will be stored in './checkpoints/<checkpoint_name>'
    checkpoint_name = "sam_hela"

    # Train an additional convolutional decoder for end-to-end automatic instance segmentation
    train_instance_segmentation = False

    run_training(checkpoint_name, model_type, train_instance_segmentation)
    export_model(checkpoint_name, model_type)


if __name__ == "__main__":
    main()
