import os

import numpy as np
import torch
import torch_em

import micro_sam.training as sam_training
from micro_sam.sample_data import fetch_tracking_example_data, fetch_tracking_segmentation_data
from micro_sam.util import export_custom_sam_model

DATA_FOLDER = "data"


def get_dataloader(split, patch_shape, batch_size):
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
    # Here, we use it to select the first 70 frames fro the test split and the other frames for the val split.
    if split == "train":
        roi = np.s_[:70, :, :]
    else:
        roi = np.s_[70:, :, :]

    loader = torch_em.default_segmentation_loader(
        raw_paths=image_dir, raw_key=raw_key,
        label_paths=segmentation_dir, label_key=label_key,
        patch_shape=patch_shape, batch_size=batch_size,
        ndim=2, is_seg_dataset=True, rois=roi,
        label_transform=torch_em.transform.label.connected_components,
    )
    return loader


def main():
    """Finetune a Segment Anything model.

    This example uses image data and segmentations from the cell tracking challenge,
    but can easily be adapted for other data (including data you have annoated with micro_sam beforehand).
    """

    # All hyperparameters for training.
    batch_size = 1  # the training batch size
    patch_shape = (1, 512, 512)  # the size of patches for training
    n_objects_per_batch = 25  # the number of objects per batch that will be sampled
    device = torch.device("cuda")  # the device/GPU used for training
    n_iterations = 10000  # how long we train (in iterations)

    model_type = "vit_b"  # the name of the model which is used to initialize the weights that are finetuned
    # We use vit_b here because it can be trained faster. Note that vit_h usually yields higher quality results.

    # Get the dataloaders.
    train_loader = get_dataloader("train", patch_shape, batch_size)
    val_loader = get_dataloader("val", patch_shape, batch_size)

    # Get the segment anything model, the optimizer and the LR scheduler
    model = sam_training.get_trainable_sam_model(model_type=model_type, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=10, verbose=True)

    # This class creates all the training data for a batch (inputs, prompts and labels).
    convert_inputs = sam_training.ConvertToSamInputs()

    checkpoint_name = "sam_hela"
    # the trainer which performs training and validation (implemented using "torch_em")
    trainer = sam_training.SamTrainer(
        name=checkpoint_name,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        # currently we compute loss batch-wise, else we pass channelwise True
        loss=torch_em.loss.DiceLoss(channelwise=False),
        metric=torch_em.loss.DiceLoss(),
        device=device,
        lr_scheduler=scheduler,
        logger=sam_training.SamLogger,
        log_image_interval=10,
        mixed_precision=True,
        convert_inputs=convert_inputs,
        n_objects_per_batch=n_objects_per_batch,
        n_sub_iteration=8,
        compile_model=False
    )
    trainer.fit(n_iterations)

    # export the model after training so that it can be used by the rest of the micro_sam library
    export_path = "./finetuned_hela_model.pth"
    checkpoint_path = os.path.join("checkpoints", checkpoint_name, "best.pt")
    export_custom_sam_model(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        save_path=export_path,
    )


if __name__ == "__main__":
    main()
