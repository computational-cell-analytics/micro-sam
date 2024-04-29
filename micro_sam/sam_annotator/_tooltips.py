"""Tooltips for widgets in the annotator.
"""

tooltips = {
  "embedding": {
    "custom_weights": "Select custom model weights. For example for a model you have finetuned",
    "device": "Select the computational device to use for processing.",
    "embeddings_save_path": "Select path to save or load the computed image embeddings.",
    "halo": "Enter overlap values for computing tiled embeddings. Enter only x-value for quadratic size.\n Only active when tiling is used.",
    "image": "Select the napari image layer.",
    "model": "Select the segment anything model.",
    "prefer_decoder": "Choose if the segmentation decoder is used for automatic segmentation. Only if it is available for the selected model..",
    "run_button": "Compute embeddings or load embeddings if embedding_save_path is specified.",
    "tiling": "Enter tile size for computing tiled embeddings. Enter only x-value for quadratic size or both for non-quadratic.",
  },
  "segmentnd": {
    "box_extension": "Enter factor by which box size is increased when projecting to adjacent slices. Larger factors help if object sizes change between slices.",
    "iou_threshold": "Enter the minimal overlap between objects in adjacent slices to continue segmentation.",
    "motion_smoothing": "Enter the motion smoothing factor. It is used to follow objects which have a directed movement, higher values help for objects that are moving fast.",
    "projection_dropdown": "Choose the projection mode. It determines which prompts are derived from the masks projected to adjacent frames to rerun SAM.",
  },
  "autosegment": {
    # General settings.
    "apply_to_volume": "Choose if automatic segmentation is run for the full volume or only the current slice.",
    "gap_closing": "Enter value for closing gaps across slices for volumetric segmentation. Higher values will reduce artifacts due to missing slices in objects but may lead to wrongly merging objects.",
    "min_extent": "Enter the minimal number of slices for objects in volumetric segmentation. To filter out small segmentation artifacts.",
    "min_object_size": "Enter the minimal object size in pixels. This refers to the size per slice for volumetric segmentation.",
    "run_button": "Run automatic segmentation.",
    "with_background": "Choose if your image has a large background area.",
    # Settings for AIS.
    "boundary_distance_thresh": "Enter the boundary distance threshold.",
    "center_distance_thresh": "Enter the center distance threshold.",
    # Settings for AMG.
    "box_nms_thresh": "Enter the non-maximum suppression threshold.",
    "pred_iou_thresh": "Enter the threshold for filtering objects based on the predicted IOU.",
    "stability_score_thresh": "Enter the threshold for filtering objects based on the stability score.",
  },
  "prompt_menu": {
    "labels": "Choose positive prompts to inlcude regions or negative ones to exclude regions. Toggle between the settings by pressing [t].",
  },
  "annotator_tracking": {
    "track_id": "Select the id of the track you are currently annotating.",
    "track_state": "Select the state of the current annotation. Choose 'division' if the object is dviding in the current frame.",
  },
  "image_series_annotator": {
    "folder": "Select the folder with the images to annotate.",
    "output_folder": "Select the folder for saving the segmentation results.",
    "pattern": "Select a pattern for selecting files. E.g. '*.tif' to only select tif files. By default all files in the input folder are selected.",
    "is_volumetric": "Choose if the data you annotate is volumetric.",
  },
  "training": {
    "checkpoint": "Select a checkpoint (saved model) to resume training from.",
    "device": "Select the computational device to use for processing.",
    "initial_model": "Select the model name used as starting point for training.",
    "label_key": "Define the key that holds to the segmentation labels. Use a pattern, e.g. \"*.tif\" select multiple files or an internal path for hdf5, zarr or similar formats.",
    "label_path": "Specify the path to the segmentaiton labels for training. Can either point to a directory or single file.",
    "label_path_val": "Specify the path to the segmentation labels for validation. Can either point to a directory or single file.",
    "name": "Enter the name of the model that will be trained.",
    "patch": "Select the size of image patches used for training.",
    "raw_key": "Define the key that holds to the image data. Use a pattern, e.g. \"*.tif\" select multiple files or an internal path for hdf5, zarr or similar formats.",
    "raw_path": "Specify the path to the image data for training. Can either point to a directory or single file.",
    "raw_path_val": "Specify the path to the image data for training. Can either point to a directory or single file.",
    "segmentation_decoder": "Choose whether to train with additional segmentation decoder or not.",
    "output_path": "Specify the path where you want to save the trained model after the training process.",
    "n_epochs": "Define the number of training epochs for your model.",
    "configuration": "Specifiy the hardware configuration to use for training.",
  },
}


def get_tooltip(widget_type: str, name: str) -> str:
    """Retrieves the tooltip for a given parameter name within a specific widget type.

    Args:
        widget_type: The type of widget (e.g., "embedding", "segmentation").
        name: The name of the parameter to get the tooltip for.

    Returns:
        The tooltip string, or None if not found.
    """
    if widget_type in tooltips:
        return tooltips[widget_type].get(name)
    else:
        return None  # Handle cases where the widget type is not found
