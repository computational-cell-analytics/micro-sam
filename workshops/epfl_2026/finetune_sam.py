"""Finetuning Segment Anything using 'µsam'.

This python script shows how to use µsam to fine-tune (improve) for a specific segmentation task.
Note: You need a GPU to fine-tune a model in reasonable time.

Here, we use confocal microscopy images from the HPA Kaggle Challenge for protein identification
(from Ouyang et al. - https://doi.org/10.1038/s41592-019-0658-6).
This example should work for your (microscopy) images with a few minor adaptations.

Note that finetuning requires annotated images. This means that images and the corresponding segmentation masks
are needed. Check out the examples from the HPA data in 'data/hpa/train/images' and 'data/hpa/train/labels' if
you are unsure about the exact format.
"""

import os
from glob import glob

import imageio.v3 as imageio

from torch_em.util.debug import check_loader

import micro_sam.training as sam_training
from micro_sam.training.util import normalize_to_8bit
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation


def load_images_and_labels(folder, view):
    images, labels = [], []
    image_paths = sorted(glob(os.path.join(folder, "images", "*.tif")))
    label_paths = sorted(glob(os.path.join(folder, "labels", "*.tif")))
    assert len(image_paths) == len(label_paths)

    for i, (image_path, label_path) in enumerate(zip(image_paths, label_paths)):
        image = imageio.imread(image_path)
        label = imageio.imread(label_path)
        # We remove the 'er' channel, i.e. the last channel.
        if image.ndim == 3 and image.shape[-1] == 4:  # Convert 4 channel inputs to 3 channels.
            image = image[..., :-1]

        # Check out the first 5 images if view was set to true.
        if view and i < 5:
            import napari

            v = napari.Viewer()
            v.add_image(image)
            v.add_labels(label)
            napari.run()

        # And we move the channel into the first position, which is expected by pytorch
        image = image.transpose((2, 0, 1))
        images.append(image)
        labels.append(label)

    return images, labels


def get_dataloaders(input_root, view):
    """Create the data loaders for processing and feeding the training data.

    Note: The values here are optimized for the HPA dataset.
    The places where you should change this for your data are marked with a comment: 'CHANGE'.
    """
    # CHANGE: For your own data you have to adapt these folder names.
    train_folder = os.path.join(input_root, "hpa/train")
    val_folder = os.path.join(input_root, "hpa/val")

    train_images, train_labels = load_images_and_labels(train_folder, view=view)
    val_images, val_labels = load_images_and_labels(val_folder, view=view)

    # CHANGE: The batch size determines how many images are used for one batch during training.
    # You can usually leave this at 1, but if you have a powerful GPU may set it to 2.
    batch_size = 1
    # CHANGE: This is the size of one patch used for training. It should be chose such that several cells
    # are visible within a typical patch of this size.
    # We usually recommend to go with 512 x 512.
    patch_shape = (1024, 1024)

    # The data loaders take care of all the data loading and pre-processing.
    train_loader = sam_training.default_sam_loader(
        raw_paths=train_images,
        raw_key=None,
        label_paths=train_labels,
        label_key=None,
        patch_shape=patch_shape,
        with_channels=True,
        with_segmentation_decoder=True,
        batch_size=batch_size,
        shuffle=True,
        raw_transform=normalize_to_8bit,
        n_samples=100,
    )
    val_loader = sam_training.default_sam_loader(
        raw_paths=val_images,
        raw_key=None,
        label_paths=val_labels,
        label_key=None,
        is_seg_dataset=False,
        patch_shape=patch_shape,
        with_channels=True,
        with_segmentation_decoder=True,
        batch_size=batch_size,
        shuffle=True,
        raw_transform=normalize_to_8bit,
        n_samples=10,
    )

    if view:
        # Let's check how our samples look from the dataloader.
        check_loader(train_loader, 4)

    return train_loader, val_loader


def run_finetuning(model_name, train_loader, val_loader, model_type, overwrite, n_epochs):
    """Run finetuning for the Segment Anything model on microscopy images.

    Args:
        train_loader: The PyTorch dataloader used for training.
        val_loader: The PyTorch dataloader used for validation.
        model_type: The choice of Segment Anything model (connotated by the size of image encoder).
        overwrite: Whether to overwrite the already finetuned model checkpoints.
        n_epochs: The maximal number of epochs to train for.

    Returns:
        Filepath where the (best) model checkpoint is stored.
    """
    save_root = os.getcwd()
    best_checkpoint = os.path.join(save_root, "checkpoints", model_name, "best.pt")
    if os.path.exists(best_checkpoint) and not overwrite:
        print(
            "It looks like the training has completed. Pass the argument '--overwrite' to overwrite "
            "the already finetuned model (or train a model with a new name using the argument '--model_name')."
        )
        return best_checkpoint

    # Run training
    sam_training.train_sam_for_configuration(
        name=model_name,
        save_root=save_root,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        with_segmentation_decoder=True,
    )

    return best_checkpoint


def run_instance_segmentation_with_decoder(input_root, model_type, checkpoint):
    """Run automatic instance segmentation with the fine-tuned model.

    Args:
        test_image_paths: List of filepaths for the test image data.
        model_type: The choice of Segment Anything model (connotated by the size of image encoder).
        checkpoint: Filepath to the finetuned model checkpoints.
    """
    assert os.path.exists(checkpoint), "Please train the model first to run inference on the finetuned model."

    # CHANGE: For your own data you have to adapt these folder names.
    folder = os.path.join(input_root, "hpa/test/images")
    image_paths = glob(os.path.join(folder, "*.tif"))
    print(len(image_paths))

    # Get the 'predictor' and 'segmenter' to perform automatic instance segmentation.
    predictor, segmenter = get_predictor_and_segmenter(model_type=model_type, checkpoint=checkpoint, is_tiled=True)

    # Iterate over the training images to run the segmentation and visualize the result in napari.
    for image_path in image_paths:
        image = imageio.imread(image_path)
        image = normalize_to_8bit(image)

        # CHANGE: Update the values for 'tile_shape' and 'halo' so that they match the patch_shape used in training,
        # according to: patch_shape = tile_shape + 2 * halo.
        # For example, if you used patch_shape = (512, 512), you can use tile_shape = (384, 384) and halo = (64, 64)
        prediction = automatic_instance_segmentation(
            predictor=predictor, segmenter=segmenter, input_path=image, ndim=2, tile_shape=(768, 768), halo=(128, 128)
        )

        import napari
        v = napari.Viewer()
        v.add_image(image)
        v.add_labels(prediction)
        napari.run()

        break  # comment this out in case you want to run inference for all images.


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run finetuning for Segment Anything model for microscopy images.")
    parser.add_argument(
        "-n", "--model_name", type=str, default="sam-hpa",
        help="The name of the checkpoint. The checkpoints will be stored in './checkpoints/<model_name>'"

    )
    parser.add_argument(
        "-i", "--input_path", type=str, default="./data",
        help="The filepath to the folder where the image data will be downloaded. "
        "By default, the data will be stored in your current working directory at './data'."
    )
    parser.add_argument(
        "--view", action="store_true",
        help="Whether to visualize samples from the dataloader, instance segmentation outputs, etc."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Whether to overwrite the already finetuned model checkpoints."
    )
    parser.add_argument(
        "--n_epochs", type=int, default=50,
        help="The maximum number of epochs to train for."
    )
    args = parser.parse_args()

    # The model_type determines which base model is used to initialize the weights that are finetuned.
    # We use vit_b_lm here, which is the model we trained for light microscopy segmentation.
    # Other choices are:
    # - vit_b_em_organelles: For organelle segmentation in electron microscopy.
    # - vit_b_histopathology: For nucleus segmentation in histopathology.
    model_type = "vit_b_lm"

    # Get the data loaders and run the training.
    train_loader, val_loader = get_dataloaders(args.input_path, view=args.view)
    checkpoint_path = run_finetuning(
        model_name=args.model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        model_type=model_type,
        overwrite=args.overwrite,
        n_epochs=args.n_epochs,
    )
    assert os.path.exists(checkpoint_path), checkpoint_path

    # Use the fine-tuned model for instance segmentation on test data to verify that it worked.
    # Note: You can also use the fine-tuned model within the micro_sam napari plugin
    # or within other python functions from the micro_sam library.
    run_instance_segmentation_with_decoder(args.input_path, model_type=model_type, checkpoint=checkpoint_path)


if __name__ == "__main__":
    main()
