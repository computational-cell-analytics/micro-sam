"""Tooltips for widgets in the annotator.
"""

tooltips = {
  "embedding": {
    "custom_weights": "Select a path for custom weights.",
    "device": "Select the computational device to use for processing.",
    "embeddings_save_path": "Path to save the computed image embeddings.",
    "halo": """Enter values to activate halo. Only x value for quadratic halo and both for custom.
    Only active when tiling is active.""",
    "image": "Select an image to segment objects.",
    "model": "Select a model to use for segmentation tasks.",
    "prefer_decoder": "Choose if to use segmentation decoder.",
    "run_button": "Compute embeddings or load embeddings if embeddings file path is specified.",
    "settings": "",
    "tiling": "Enter values to activate tiling. Only x-value for quadratic tiling or both values for custom tiling.",
    # ... other tooltips for embedding widgets ...
  },
  "segmentnd": {
    "box_extension": "box_extension",
    "custom_weights": "Path to the file containing pre-trained weights for the segmentation model.",
    "iou_threshold": "iou_threshold",
    "motion_smoothing": "motion_smoothing",
    "projection_dropdown": "projection dropdown",
    "settings": "",
    # ... other tooltips for segmentation widgets ...
  },
  "autosegment": {
    "apply_to_volume": "apply_to_volume",
    "boundary_distance_thresh": "boundary_distance_thresh",
    "box_nms_thresh": "box_nms_thresh",
    "center_distance_thresh": "center_distance_thresh",
    "gap_closing": "gap_closing",
    "min_extent": "min_extent",
    "min_object_size": "min_object_size",
    "pred_iou_thresh": "pred_iou_thresh",
    "run_button": "Run automatic segmentation.",
    "stability_score_thresh": "stability_score_thresh",
    "with_background": "with_background",
    # ... tooltips for autosegment widgets ...
  },
  "segment": {
    "run_button": "run_button",
  },
  "prompt_menu": {
    "labels": "Choose positive prompts to inlcude regions or negative ones to exclude regions.",
  },
  "annotator_tracking": {
    "track_id": "track_id",
    "track_state": "track_state",
  },
  "training": {
    "checkpoint": "checkpoint",
    "device": "device",
    "initial_model": "initial_model",
    "label_key": "label_key",
    "label_path": "label_path",
    "label_path_val": "label_path_val",
    "name": "Name of the new model.",
    "patch": "patch",
    "raw_key": "raw_key",
    "raw_path": "raw_path",
    "raw_path_val": "raw_path_val",
    "segmentation_decoder": "segmentation_decoder",
    "setting": "setting"
  }
}


def get_tooltip(widget_type, name):
    """
    Retrieves the tooltip for a given parameter name within a specific widget type.

    Args:
        widget_type (str): The type of widget (e.g., "embedding", "segmentation").
        name (str): The name of the parameter to get the tooltip for.

    Returns:
        str: The tooltip string, or None if not found.
    """
    if widget_type in tooltips:
        return tooltips[widget_type].get(name)
    else:
        return None  # Handle cases where the widget type is not found