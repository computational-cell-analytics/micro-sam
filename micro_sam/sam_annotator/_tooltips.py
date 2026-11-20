"""Tooltips for widgets in the annotator."""

tooltips = {
    "embedding": {
        "batch_size": "Number of slices or tiles per GPU step. Higher is faster, but uses more GPU memory.",
        "cache_state": "Save the automatic segmentation state to disk for faster runs.",
        "custom_weights": (
            "Your own model weights. For automatic segmentation, use 'best.pt' "
            "or a model file with its '_decoder' file next to it."
        ),
        "device": "The device for the computation. 'auto' uses all GPUs.",
        "embeddings_save_path": "File to save or load the embeddings.",
        "halo": "Tile overlap. Enter only x for a square overlap.",
        "image": "The image layer.",
        "image_ndim": "'2d': read a multi-channel array as one image. '3d': read it as a volume.",
        "model_family": "The model family.",
        "model_family_advanced": "The advanced model family, for example SAM1.",
        "model_size": "The model size. Larger is more accurate, but slower.",
        "advanced_model": "Show the advanced models (SAM1).",
        "automatic_segmentation_mode": "The automatic segmentation mode.",
        "run_button": "Compute the embeddings, or load them from the save path.",
        "tiling": "Tile size. Enter only x for square tiles.",
        "settings": "Model, tiling, batch size, save path and device.",
    },
    "segmentnd": {
        "early_stop_patience": "Stop after this many slices without the object. 0: no stop.",
        "use_full_z_range": "Segment through all slices. Clear to set a slice range.",
        "z_range": "The slices to segment.",
        "settings": "Early stop and slice range for the volume segmentation.",
    },
    "unified_segment": {
        "apply_to_volume": "Segment the current slice or the full volume.",
        "batched": "Segment one object per positive point and per box.",
        "batched_scribble_disabled": "Not available with scribble prompts. Remove the scribbles to use it.",
        "segment_button": "Segment the object from the prompts. Shortcut: S.",
        "clear_button": "Clear the prompts and the current object. Shortcut: Shift + C.",
    },
    "autosegment": {
        # General settings.
        "apply_to_volume": "Segment the current slice or the full volume.",
        "min_object_size": "Minimum object size in pixels (per slice in 3d).",
        "run_button": "Run the automatic segmentation.",
        "tile_z": "Slices per block in 3d. Less uses less memory.",
        "halo_z": "Overlap slices between the 3d blocks.",
        # Settings for the SAM2 automatic segmentation modes.
        "mode": "'apg': default on GPU. 'sparse': default on CPU. 'dense': multicut. All need a decoder.",
        "foreground_threshold": "Threshold for the foreground.",
        "density_threshold": "Threshold for the seeds.",
        "candidate_threshold": "Threshold for the APG prompts.",
        "candidate_threshold_high": "Second APG prompt threshold for volumes.",
        "min_candidate_size": "Minimum APG prompt size in pixels.",
        "score_threshold": "Minimum mask score.",
        "max_overlap": "Maximum overlap between masks.",
        "multimasking": "Predict 3 masks per prompt and keep the best one.",
        "refine_with_box_prompts": "Refine each mask with a box.",
        "box_extension": "Pixels added to each refinement box.",
        "prompt_batch_size": "Prompts per model step.",
        "n_objects_per_pass": "Objects per pass through the volume.",
        "early_stop_patience": "Stop after this many slices without the object. 0: no stop.",
        "beta": "Boundary bias. Higher merges more objects.",
        "n_iter": "Number of flow steps.",
        "dt": "Flow step size.",
        "sigma": "Smoothing of the seed map.",
        "n_threads": "Number of threads.",
        "advanced_settings": "Advanced parameters.",
    },
    "autotrack": {
        "run_button": "Run the automatic tracking.",
        "run_tracking": "Track all frames, or segment only the current frame.",
    },
    "prompt_menu": {
        "labels": "Positive prompts add to the object. Negative prompts remove from it. Shortcut: T.",
    },
    "annotator_tracking": {
        "track_id": "The track that you annotate.",
        "track_state": "Select 'division' if the object divides in this frame.",
        "export_button": "Export the tracks (CTC, GEFF or TrackMate XML).",
    },
    "batch_annotator": {
        "folder": "Folder with the images.",
        "output_folder": "Folder for the results.",
        "continue_annotation": "Start at the first image without a result. Clear to start at the first image.",
        "pattern": "File pattern, for example '*.tif'. Empty: all files.",
        "ndim": "2d or 3d data.",
        "task": "Segmentation, tracking, object classification or pixel classification.",
        "segmentation_folder": "Folder with one segmentation per image. Empty: segment in the tool.",
        "next_image": "Go to the next image. Shortcut: N.",
        "previous_image": "Go to the previous image. Shortcut: Shift + N.",
    },
    "training": {
        "checkpoint": "Checkpoint to continue the training.",
        "device": "The device for the training.",
        "initial_model": "The model to start from.",
        "label_key": "File pattern (for example '*.tif') or the key in a hdf5 or zarr file.",
        "label_path": "File or folder with the training labels.",
        "label_path_val": "File or folder with the validation labels.",
        "name": "Name of the new model.",
        "patch": "Patch size for the training.",
        "raw_key": "File pattern (for example '*.tif') or the key in a hdf5 or zarr file.",
        "raw_path": "File or folder with the training images.",
        "raw_path_val": "File or folder with the validation images.",
        "segmentation_decoder": "Also train a decoder for automatic segmentation.",
        "output_path": "Where to save the model.",
        "n_epochs": "Number of epochs.",
        "configuration": "Your hardware.",
    },
    "commit": {
        "layer": "The layer to commit: the prompt result or the automatic result.",
        "preserve_mode": "Keep committed objects: per object, per pixel, or not at all.",
        "commit_path": "Zarr file to save the committed objects and prompts.",
        "commit_button": "Commit the segmentation. Shortcut: C.",
    },
    "classification": {
        "settings": "Features, seed, and load or export of the classifier.",
        "forward_classifier_state": "Use the annotations of all images. Clear to classify each image alone.",
        "segmentation": "The segmentation with the objects to classify.",
        "train_button": "Train and predict. Shortcut: Shift + T.",
        "clear_button": "Clear the annotations and the prediction. Shortcut: Shift + C.",
        "apply_to_volume": "Predict and clear in the full volume. Clear: current slice only.",
        "use_top_features_pixel": "Reduce the features with PCA.",
        "use_top_features_object": "Reduce the features with PCA.",
        "top_features_pixel": "Number of PCA features (1 to 256).",
        "top_features_object": "Number of PCA features (1 to 257).",
        "use_classical_features_pixel": "Add filter features, as in ilastik. Good for intensity and texture.",
        "use_classical_features_object": "Add size, intensity and shape features.",
        "classical_features_pixel": "The filters to add.",
        "classical_features_object": "The features to add.",
        "use_anyup": "Sharper features at object boundaries.",
        "random_seed": "'fixed': same result each run. 'random': results change a little.",
        "load_path": "Classifier file (.joblib) to load.",
        "load_button": "Load the classifier and predict.",
        "export_dir": "Folder to save the classifier.",
        "export_button": "Save the classifier.",
        "label_names": "Your classes. Select a class to paint with it. Previous / next class: Alt + Left / Right.",
        "label_name_row": "Name of the class (optional).",
        "class_select": "Paint with this class. Shortcut: Alt + Right / Left.",
        "class_color": "Color of the class.",
        "class_remove": "Remove the class and its annotations.",
        "add_class": "Add a class.",
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
