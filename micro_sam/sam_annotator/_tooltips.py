"""Tooltips for widgets in the annotator."""

tooltips = {
    "embedding": {
        "batch_size": (
            "Number of image slices or tiles encoded together per GPU. Larger values can improve throughput. "
            "They can also be slower or run out of GPU memory. The safe default is 1."
        ),
        "cache_state": "Cache the automatic segmentation state to disk for faster (re)runs.",
        "custom_weights": "Select custom model weights, for example from a model that you finetuned.",
        "device": "Select the computational device to use for processing.",
        "embeddings_save_path": "Select path to save or load the computed image embeddings.",
        "halo": "Enter overlap values to compute tiled embeddings. Enter only the x-value for a square size.\n Active only when you use tiling.",  # noqa
        "image": "Select the napari image layer.",
        "model_family": "Select the segment anything 2 model family.",
        "model_family_advanced": "Select the advanced (non-SAM2) model family, for example a SAM1 family. Turn it on via 'Advanced Models' in the embedding settings.",  # noqa
        "model_size": "Select the image encoder size of the segment anything 2 model.",
        "advanced_model": "Switch the model list above to advanced models beyond the default SAM2 models (currently SAM1). Only available for the classification tools.",  # noqa
        "automatic_segmentation_mode": "Select the automatic segmentation mode.",
        "run_button": "Compute embeddings or load embeddings if embedding_save_path is specified.",
        "tiling": "Enter the tile size to compute tiled embeddings. Enter only the x-value for a square size, or both values for a non-square size.",  # noqa
        "settings": "Settings for computing the image embeddings: model family and size, tiling, batch size, the embedding save path and the compute device.",  # noqa
    },
    "segmentnd": {
        "box_extension": "Enter the factor by which the box size grows when it projects to adjacent slices. Larger factors help if object sizes change between slices.",  # noqa
        "iou_threshold": "Enter the minimal overlap between objects in adjacent slices to continue segmentation.",
        "early_stop_patience": "SAM2 volume mode: stop propagation once the object is absent for this many slices in a row. Lower values stop sooner (faster). A value of 0 disables early stopping and propagates through the whole volume.",  # noqa
        "use_full_z_range": "SAM2 volume mode: propagate through all slices along z. Uncheck to restrict propagation to a chosen slice range with the slider below (faster, and a hard guardrail against leaking into neighbouring structures).",  # noqa
        "z_range": "SAM2 volume mode: the inclusive range of slices (along z) that propagation is allowed to cover. Only used when 'Propagate through full volume' is unchecked.",  # noqa
        "motion_smoothing": "Enter the motion smoothing factor. It helps to follow objects that have a directed movement. Higher values help for fast objects.",  # noqa
        "projection_dropdown": "Choose the projection mode. It determines which prompts are derived from the masks projected to adjacent frames to rerun SAM.",  # noqa
        "batched": "Enable to segment multiple objects with separate point prompts. The tool tracks each positive point as a separate object. Only available for SAM2 models.",  # noqa
        "settings": "Settings controlling how a 2d segmentation is propagated through the volume (projection mode, IoU threshold, early stopping and the z-range).",  # noqa
    },
    "unified_segment": {
        "apply_to_volume": "Choose whether to segment only the current slice or frame, or the full volume or all frames.",  # noqa
        "batched": "Enable to segment multiple objects at once: each positive point and each box defines a separate object. Only available for SAM2 models.",  # noqa
        "batched_scribble_disabled": "Batched segmentation is unavailable while scribble prompts are present. Remove all path, polyline and line prompts to re-enable it.",  # noqa
        "segment_button": "Run Segment Anything 2 on the current point/box prompts to segment the object. Shortcut: S.",  # noqa
        "clear_button": "Clear the current prompts and the current-object segmentation (whole volume or current slice per 'Apply to Volume' for 3d data). Shortcut: Shift + C.",  # noqa
        "settings": "Settings for interactive segmentation across slices (projection mode and propagation parameters).",  # noqa
    },
    "autosegment": {
        # General settings.
        "apply_to_volume": "Choose whether to run automatic segmentation on the full volume or only the current slice.",
        "gap_closing": "Enter the value to close gaps across slices for volumetric segmentation. Higher values reduce artifacts from missing slices in objects, but can wrongly merge objects.",  # noqa
        "min_extent": "Enter the minimal number of slices for objects in volumetric segmentation. This filters out small segmentation artifacts.",  # noqa
        "min_object_size": "Enter the minimal object size in pixels. This refers to the size per slice for volumetric segmentation.",  # noqa
        "run_button": "Run automatic segmentation.",
        "with_background": "Choose whether your image has a large background area.",
        "tile_z": "Number of slices per z-block for 3d automatic segmentation. The volume is decoded in z-blocks to bound memory. Set this to the number of slices (or more) to process the whole volume in one block (no z-tiling).",  # noqa
        "halo_z": "Number of overlapping slices between z-blocks for 3d automatic segmentation, used as context and discarded when stitching.",  # noqa
        # Settings for AIS.
        "boundary_distance_thresh": "Enter the boundary distance threshold.",
        "center_distance_thresh": "Enter the center distance threshold.",
        # Settings for AMG.
        "box_nms_thresh": "Enter the non-maximum suppression threshold.",
        "pred_iou_thresh": "Enter the threshold for filtering objects based on the predicted IOU.",
        "stability_score_thresh": "Enter the threshold for filtering objects based on the stability score.",
        "points_per_side": "Number of points sampled along one side of the image for AMG grid prompting.",
        # Settings for the SAM2 dense/sparse modes.
        "mode": (
            "Select the automatic segmentation mode: 'amg' (grid-based, no decoder needed), "
            "'sparse' (flow-based) or 'dense' (multicut-based)."
        ),
        "foreground_threshold": "Enter the threshold for binarizing the foreground probability map.",
        "density_threshold": "Enter the convergence-density threshold used for seed extraction.",
        "beta": "Enter the multicut boundary bias. Higher values favour merging objects.",
        "n_iter": "Enter the number of flow-integration steps.",
        "dt": "Enter the flow-integration step size.",
        "sigma": "Enter the Gaussian sigma for smoothing the convergence-density map.",
        "n_threads": "Enter the number of threads for the post-processing.",
        "settings": "Settings for automatic segmentation: the mode and its thresholds.",
        "advanced_settings": "Advanced automatic-segmentation parameters.",
    },
    "autotrack": {
        "run_button": "Run automatic tracking.",
        "run_tracking": "Choose whether to run tracking for the whole timeseries, or to segment only the current timeframe.",  # noqa
    },
    "prompt_menu": {
        "labels": "Choose positive point/scribble prompts to include regions or negative ones to exclude regions. Toggle between the settings by pressing [t]. In 3d, a scribble belongs to the z-slice where it was drawn and can seed or correct volume propagation.",  # noqa
    },
    "annotator_tracking": {
        "track_id": "Select the id of the track you are currently annotating.",
        "track_state": "Select the state of the current annotation. Choose 'division' if the object divides in the current frame.",  # noqa
        "export_button": "Export the committed tracking result in the chosen format (CTC, GEFF or TrackMate XML).",  # noqa
    },
    "batch_annotator": {
        "folder": "Select the folder with the images to annotate.",
        "output_folder": "Select the folder for saving the segmentation results.",
        "continue_annotation": "Resume at the first image without a saved result in the output folder. Uncheck to restart at the first image and load existing segmentations for review or editing.",  # noqa
        "pattern": "Select a pattern for selecting files. E.g. '*.tif' to only select tif files. By default all files in the input folder are selected.",  # noqa
        "ndim": "The spatial dimensionality of the data.",
        "task": "The annotation task to run over the batch: interactive segmentation, tracking (each file is a timeseries), or object / pixel classification.",  # noqa
        "segmentation_folder": "Object classification only: a folder with one segmentation per image to classify. Leave empty to produce the segmentations in the tool.",  # noqa
    },
    "training": {
        "checkpoint": "Select a checkpoint (saved model) to resume training from.",
        "device": "Select the computational device to use for processing.",
        "initial_model": "Select the model name used as starting point for training.",
        "label_key": 'Define the key that holds the segmentation labels. Use a pattern, e.g. "*.tif", to select multiple files, or an internal path for hdf5, zarr or similar formats.',  # noqa
        "label_path": "Specify the path to the segmentation labels for training. Can either point to a directory or single file.",  # noqa
        "label_path_val": "Specify the path to the segmentation labels for validation. Can either point to a directory or single file.",  # noqa
        "name": "Enter the name of the model that will be trained.",
        "patch": "Select the size of image patches used for training.",
        "raw_key": 'Define the key that holds the image data. Use a pattern, e.g. "*.tif", to select multiple files, or an internal path for hdf5, zarr or similar formats.',  # noqa
        "raw_path": "Specify the path to the image data for training. Can either point to a directory or single file.",
        "raw_path_val": "Specify the path to the image data for validation. Can either point to a directory or single file.",  # noqa
        "segmentation_decoder": "Choose whether to train with additional segmentation decoder or not.",
        "output_path": "Specify the path where you want to save the trained model after the training process.",
        "n_epochs": "Define the number of training epochs for your model.",
        "configuration": "Specify the hardware configuration to use for training.",
    },
    "commit": {
        "layer": "The layer to commit. Either 'current_object' to commit results from prompt-based segmentation or 'auto_segmentation' to commit results from automatic segmentation.",  # noqa
        "preserve_mode": "The mode for preserving already committed objects. Either 'objects' to preserve on a per-object level, 'pixels' to preserve on a per-pixel level, or 'none' to not preserve.",  # noqa
        "commit_path": "The path to a zarr file for saving committed objects, prompts and other segmentation settings.",
        "commit_button": "Commit the current segmentation to the committed-objects layer. Shortcut: C.",
    },
    "classification": {
        "settings": "Optional classifier settings: PCA feature reduction, AnyUp upsampling, the random seed, and loading or exporting a trained classifier.",  # noqa
        "forward_classifier_state": "Carry the classifier across images in the batch. The tool stacks the annotated features from earlier images with the current image. It trains a fresh random forest on the combined set. Then it applies the forest to the next image automatically, even without new annotations. Uncheck to classify each image on its own.",  # noqa
        "segmentation": "Select the segmentation (labels) layer whose objects will be classified.",
        "train_button": "Train the random forest on all current annotations and predict on the image. Shortcut: Shift + T.",  # noqa
        "clear_button": "Clear the annotation scribbles and the prediction (whole volume, or the current slice for 3d data when 'Apply to Volume' is unchecked). Shortcut: C.",  # noqa
        "apply_to_volume": "Apply 'Train and predict' and 'Clear Annotations' to the whole volume. When unchecked, they act only on the current slice (training always uses all annotations). Only relevant for 3d data.",  # noqa
        "use_top_features_pixel": "Reduce the SAM/SAM2 embedding to its most informative channels via PCA before training. When unchecked, all 256 embedding channels are used and no PCA is applied.",  # noqa
        "use_top_features_object": "Reduce the object features to their most informative components via PCA before training. When unchecked, all features are used and no PCA is applied.",  # noqa
        "top_features_pixel": "Number of top PCA components to reduce the embedding to, between 1 and 256 (the SAM/SAM2 image embedding has 256 channels for all model sizes).",  # noqa
        "top_features_object": "Number of top PCA components to reduce the object features to, between 1 and 257 (object area plus the 256 per-channel embedding means).",  # noqa
        "use_anyup": "Use AnyUp to upsample the embedding with the original image as guidance, for sharper features near object boundaries. When unchecked, plain interpolation is used.",  # noqa
        "random_seed": "'fixed' trains with a fixed random seed so the prediction is reproducible across runs. 'random' leaves the random forest unseeded, so results change slightly each time you train.",  # noqa
        "load_path": "Path to a stored classifier (.joblib) to load and apply to the current image.",
        "load_button": "Load the selected classifier and predict on the current image.",
        "export_dir": "Folder where the trained classifier is saved. The file name is generated automatically as <image>_<nclasses>classes_<date>_<time>_<hash>.joblib. Defaults to the current working directory.",  # noqa
        "export_button": "Save the current classifier into the selected folder.",
        "label_names": "Optional names for your classes. Each painted label id gets a row here; type a name to keep track of what each class represents. Names are saved with the exported classifier.",  # noqa
        "label_name_row": "Optional name for this label id.",
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
