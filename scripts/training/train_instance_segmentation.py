"""This is an example script for training a model only for automated instance segmentation.
"""
import os

# This function downloads the DSB dataset, which we use as exaple data for this script.
# You can use any other data with images and associated label masks for training,
# for example images and label masks stored in a .tif format.
from torch_em.data.datasets.light_microscopy.dsb import get_dsb_paths

# The required functionality for training.
from micro_sam.training import export_instance_segmentation_model, default_sam_loader, train_instance_segmentation

image_paths, label_paths = get_dsb_paths("./data", source="reduced", split="train", download=True)

# Use 10% of the data for validation.
# val_len = int(0.1 * len(image_paths))
# train_images, val_images = image_paths[:-val_len], image_paths[-val_len:]
# train_labels, val_labels = label_paths[:-val_len], label_paths[-val_len:]
train_images, val_images = image_paths[:10], image_paths[-5:]
train_labels, val_labels = label_paths[:10], label_paths[-5:]

# Run the training. This will train a UNETR with instance segmentation decoder.
# This is equivalent to training the additional instance segmentation decoder
# in the full training logic, WITHOUT training the model for interactive segmentation.

# Adjust the patch shape to match your images!
patch_shape = (256, 256)

# First, we create the training and validation loaders.
train_loader = default_sam_loader(
    raw_paths=train_images, label_paths=train_labels,
    raw_key=None, label_key=None, batch_size=1,
    patch_shape=patch_shape, with_segmentation_decoder=True,
    train_instance_segmentation_only=True, is_train=True,
)
val_loader = default_sam_loader(
    raw_paths=val_images, label_paths=val_labels,
    raw_key=None, label_key=None, batch_size=1,
    patch_shape=patch_shape, with_segmentation_decoder=True,
    train_instance_segmentation_only=True, is_train=False,
)

# Choose the model type to start the training from.
# We recommend 'vit_b_lm' for any light microscopy images.
model_type = "vit_t_lm"

# This is the name for the checkpoint that will be trained.
name = "ais-dsb"

# Then run the training. Check out the docstring of the function for more training options.
train_instance_segmentation(
    name=name,
    model_type=model_type,
    train_loader=train_loader,
    val_loader=val_loader
)

# Finally, we export the trained model to a new format that is compatible with micro_sam:
# This exported model can be used by micro_sam functions, the CLI or in the napari plugin.
# However, it may not work well for interactive segmentation, since it may suffer from 'catastrophic forgetting'
# for this task, because its image encoder was updated without training for interactive segmentation.
checkpoint_path = os.path.join("checkpoints", name, "best.pt")
export_path = "./ais-dsb-model.pt"
export_instance_segmentation_model(checkpoint_path, export_path, model_type)
