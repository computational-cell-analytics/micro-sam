import os
from glob import glob
from typing import List, Optional, Union, Tuple

import numpy as np
import imageio.v3 as imageio

import torch

import napari
from qtpy import QtWidgets
from qtpy.QtCore import QTimer

from ..v1.util import get_model_names
from ..v2.util import DEFAULT_MODEL
from . import _widgets as widgets
from ._series import SeriesAnnotatorTask, run_image_series
from ._tooltips import get_tooltip
from ._state import AnnotatorState
from .annotator import Annotator, detect_ndim
from .util import _sync_embedding_widget

# The tasks the unified image series annotator can run over a series.
TASKS = ["Segmentation", "Tracking", "Object Classification", "Pixel Classification"]


def _get_input_shape(image, ndim):
    if image.ndim == 2:
        image_shape = image.shape
    elif image.ndim == 3:
        if ndim == 3:
            image_shape = image.shape
        else:
            image_shape = image.shape[:-1]
    elif image.ndim == 4:
        image_shape = image.shape[:-1]

    return image_shape


class SegmentationSeriesTask(SeriesAnnotatorTask):
    """Series task for 2d/3d interactive segmentation (the original image series annotator)."""

    empty_item_message = "Nothing is segmented yet. Do you wish to continue to the next image?"

    def __init__(
        self, *, ndim, model_type, embedding_path, tile_shape, halo,
        precompute_amg_state, checkpoint_path, device, prefer_decoder, initial_segmentations=None,
    ):
        self.ndim = ndim
        self.model_type = model_type
        self.embedding_path = embedding_path
        self.tile_shape = tile_shape
        self.halo = halo
        self.precompute_amg_state = precompute_amg_state
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.prefer_decoder = prefer_decoder
        self.initial_segmentations = initial_segmentations
        self.predictor = None
        self.decoder = None

    def result_filename(self, entry, index):
        if self.have_inputs_as_arrays:
            return f"seg_{index:05}.tif"
        return os.path.splitext(os.path.basename(entry))[0] + ".tif"

    def precompute(self, images):
        # Embeddings are computed lazily per item in start/advance (via 'initialize_predictor', which
        # routes SAM1/SAM2 and loads the model once, reused across items). Here we only build the
        # per-item embedding paths; 'None' for every item when no embedding folder is given.
        if self.embedding_path is None:
            return [None] * len(images)
        os.makedirs(self.embedding_path, exist_ok=True)
        if self.have_inputs_as_arrays:
            return [os.path.join(self.embedding_path, f"embedding_{i:05}.zarr") for i in range(len(images))]
        return [
            os.path.join(self.embedding_path, os.path.splitext(os.path.basename(p))[0] + ".zarr") for p in images
        ]

    def _resolve_initial_result(self, entry, index):
        # Load an existing saved result if present, otherwise an initial segmentation if provided.
        save_path = os.path.join(self.output_folder, self.result_filename(entry, index))
        if os.path.exists(save_path):
            return imageio.imread(save_path)
        if self.initial_segmentations is not None:
            initial = self.initial_segmentations[index]
            return initial if isinstance(initial, np.ndarray) else imageio.imread(initial)
        return None

    def _init_predictor(self, viewer, image, embedding_path):
        state = AnnotatorState()
        # Reuse the already-loaded model on later items (and for SAM1, which preloads it in 'precompute');
        # only the first SAM2 item actually loads the model and its decoder.
        if self.predictor is not None:
            kwargs = dict(predictor=self.predictor, decoder=self.decoder, prefer_decoder=False)
        else:
            kwargs = dict(prefer_decoder=self.prefer_decoder)
        state.initialize_predictor(
            image, model_type=self.model_type, save_path=embedding_path, halo=self.halo, tile_shape=self.tile_shape,
            ndim=self.ndim, precompute_amg_state=self.precompute_amg_state, checkpoint_path=self.checkpoint_path,
            device=self.device, skip_load=False, use_cli=True, **kwargs,
        )
        # Capture the loaded model so subsequent items reuse it instead of reloading.
        self.predictor, self.decoder = state.predictor, state.decoder
        state.image_shape = _get_input_shape(image, self.ndim)
        # Establish the scale for this image (matching the segmentation annotator) so the layers do not
        # inherit a stale 'image_scale' of a different dimensionality from a previous image / session.
        state.image_scale = tuple(viewer.layers["image"].scale)

    def start(self, viewer, entry, image, embedding_path, index):
        viewer.add_image(image, name="image")
        self._init_predictor(viewer, image, embedding_path)

        annotator = Annotator(viewer, ndim=self.ndim, reset_state=False)
        annotator._update_image(segmentation_result=self._resolve_initial_result(entry, index))

        state = AnnotatorState()
        viewer.window.add_dock_widget(annotator, name="Segment Anything for Microscopy (Segmentation)")
        _sync_embedding_widget(
            widget=state.widgets["embeddings"],
            model_type=self.model_type if self.checkpoint_path is None else state.predictor.model_type,
            save_path=self.embedding_path, checkpoint_path=self.checkpoint_path,
            device=self.device, tile_shape=self.tile_shape, halo=self.halo,
        )
        return annotator

    def advance(self, viewer, annotator, entry, image, embedding_path, index):
        state = AnnotatorState()
        # Clear the committed segmentation first to avoid laggy removal of the previous result.
        viewer.layers["committed_objects"].data = np.zeros_like(viewer.layers["committed_objects"].data)
        segmentation_result = self._resolve_initial_result(entry, index)
        viewer.layers["image"].data = image
        if state.amg is not None:
            state.amg.clear_state()
        self._init_predictor(viewer, image, embedding_path)
        annotator._update_image(segmentation_result=segmentation_result)

    def has_unsaved_content(self, viewer):
        return viewer.layers["committed_objects"].data.sum() != 0

    def save_item(self, viewer, entry, index):
        save_path = os.path.join(self.output_folder, self.result_filename(entry, index))
        imageio.imwrite(save_path, viewer.layers["committed_objects"].data, compression="zlib")


def image_series_annotator(
    images: Union[List[Union[os.PathLike, str]], List[np.ndarray]],
    output_folder: str,
    *,
    ndim: Optional[int] = None,
    model_type: str = DEFAULT_MODEL,
    embedding_path: Optional[str] = None,
    initial_segmentations: Optional[Union[List[Union[os.PathLike, str]], List[np.ndarray]]] = None,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    viewer: Optional["napari.viewer.Viewer"] = None,
    return_viewer: bool = False,
    precompute_amg_state: bool = False,
    checkpoint_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    prefer_decoder: bool = True,
    skip_segmented: bool = True,
) -> Optional["napari.viewer.Viewer"]:
    """Run the segmentation annotation tool for a series of images (2d or 3d).

    Args:
        images: List of the file paths or list of (set of) slices for the images to be annotated.
        output_folder: The folder where the segmentation results are saved.
        ndim: The number of spatial dimensions (2 or 3). If None, auto-detected from image shape.
        model_type: The Segment Anything model to use. For details on the available models check out
            https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#finetuned-models.
        embedding_path: Filepath where to save the embeddings.
        initial_segmentations: Initial segmentations to be corrected.
            By default no initial segmentations are loaded.
            If given, the initial segmentations will be loaded into 'committed_objects'.
        tile_shape: Shape of tiles for tiled embedding prediction.
            If `None` then the whole image is passed to Segment Anything.
        halo: Shape of the overlap between tiles, which is needed to segment objects on tile borders.
        viewer: The viewer to which the Segment Anything functionality should be added.
            This enables using a pre-initialized viewer.
        return_viewer: Whether to return the napari viewer to further modify it before starting the tool.
            By default, does not return the napari viewer.
        precompute_amg_state: Whether to precompute the state for automatic mask generation.
            This will take more time when precomputing embeddings, but will then make
            automatic mask generation much faster. By default, set to 'False'.
        checkpoint_path: Path to a custom checkpoint from which to load the SAM model.
        prefer_decoder: Whether to use decoder based instance segmentation if
            the model used has an additional decoder for instance segmentation.
            By default, set to 'True'.
        skip_segmented: Whether to skip images that were already segmented.
            If set to False, then segmentations that already exist will be loaded
            and used to populate the 'committed_objects' layer.

    Returns:
        The napari viewer, only returned if `return_viewer=True`.
    """
    if initial_segmentations is not None and len(initial_segmentations) != len(images):
        raise ValueError(
            "You have passed initial segmentations, but the number of images and segmentations is not the same: "
            f"{len(images)} != {len(initial_segmentations)}."
        )

    have_inputs_as_arrays = isinstance(images[0], np.ndarray)

    # Auto-detect the dimensionality from the first image if not given.
    if ndim is None:
        first_image = images[0] if have_inputs_as_arrays else imageio.imread(images[0])
        ndim = detect_ndim(first_image)

    task = SegmentationSeriesTask(
        ndim=ndim, model_type=model_type, embedding_path=embedding_path,
        tile_shape=tile_shape, halo=halo, precompute_amg_state=precompute_amg_state,
        checkpoint_path=checkpoint_path, device=device, prefer_decoder=prefer_decoder,
        initial_segmentations=initial_segmentations,
    )
    return run_image_series(
        images, output_folder, task, have_inputs_as_arrays=have_inputs_as_arrays,
        viewer=viewer, return_viewer=return_viewer, skip_done=skip_segmented,
    )


def image_folder_annotator(
    input_folder: str,
    output_folder: str,
    *,
    ndim: Optional[int] = None,
    pattern: str = "*",
    initial_segmentation_folder: Optional[str] = None,
    initial_segmentation_pattern: str = "*",
    viewer: Optional["napari.viewer.Viewer"] = None,
    return_viewer: bool = False,
    **kwargs
) -> Optional["napari.viewer.Viewer"]:
    """Run the segmentation annotation tool for a series of images (2d or 3d) in a folder.

    Args:
        input_folder: The folder with the images to be annotated.
        output_folder: The folder where the segmentation results are saved.
        ndim: The number of spatial dimensions (2 or 3). If None, auto-detected from image shape.
        pattern: The glob pattern for loading files from `input_folder`.
            By default all files will be loaded.
        initial_segmentation_folder: A folder with initial segmentation results.
            By default no initial segmentations are loaded.
        initial_segmentation_pattern: The glob pattern for loading files from `initial_segmentation_folder`.
        viewer: The viewer to which the Segment Anything functionality should be added.
            This enables using a pre-initialized viewer.
        return_viewer: Whether to return the napari viewer to further modify it before starting the tool.
            By default, does not return the napari viewer.
        kwargs: The keyword arguments for `micro_sam.sam_annotator.image_series_annotator`.

    Returns:
        The napari viewer, only returned if `return_viewer=True`.
    """
    image_files = sorted(glob(os.path.join(input_folder, pattern)))
    if initial_segmentation_folder is None:
        initial_segmentations = None
    else:
        initial_segmentations = sorted(glob(os.path.join(
            initial_segmentation_folder, initial_segmentation_pattern
        )))

    return image_series_annotator(
        image_files, output_folder, ndim=ndim,
        initial_segmentations=initial_segmentations,
        viewer=viewer, return_viewer=return_viewer, **kwargs
    )


def _hide_layout_widgets(item):
    """Recursively hide every widget held by a layout item.

    Used to hide the embedding widget's top image / model row once its model dropdown has been
    relocated next to the launcher's Task dropdown.
    """
    if item is None:
        return
    layout = item.layout()
    if layout is None:
        widget = item.widget()
        if widget is not None:
            widget.hide()
        return
    for i in range(layout.count()):
        _hide_layout_widgets(layout.itemAt(i))


class ImageSeriesAnnotator(widgets._WidgetBase):
    def __init__(self, viewer: napari.Viewer, parent=None):
        super().__init__(parent=parent)
        self._viewer = viewer

        # Create the UI: options + the embedded model / embedding settings.
        self._create_options()

        # Add the run button to trigger the embedding computation.
        self.run_button = QtWidgets.QPushButton("Annotate Images")
        self.run_button.clicked.connect(self.__call__)
        self.layout().addWidget(self.run_button)

        # Pack the menus to the top: the dock's extra vertical space collapses below the button
        # instead of being distributed as fixed gaps between the rows.
        self.layout().addStretch()

    def _create_options(self):
        self.folder = None
        self._folder_textbox, layout = self._add_path_param(
            "folder", self.folder, "directory",
            title="Input Folder", placeholder="Folder with images ...",
            tooltip=get_tooltip("image_series_annotator", "folder")
        )
        self.layout().addLayout(layout)
        self._folder_label = layout.itemAt(0).widget()

        # File pattern qualifying the input folder: which files form the series.
        self.pattern = "*"
        self._pattern_param, layout = self._add_string_param(
            "pattern", self.pattern, tooltip=get_tooltip("image_series_annotator", "pattern")
        )
        self.layout().addLayout(layout)
        self._pattern_label = layout.itemAt(0).widget()

        self.output_folder = None
        _, layout = self._add_path_param(
            "output_folder", self.output_folder, "directory",
            title="Output Folder", placeholder="Folder to save the results ...",
            tooltip=get_tooltip("image_series_annotator", "output_folder")
        )
        self.layout().addLayout(layout)
        self._output_label = layout.itemAt(0).widget()

        # Model dropdown on top, then the Task dropdown below it (stacked). The model dropdown is owned
        # by the embedded embedding widget and relocated into '_model_row' in '_rebuild_embedding_widget'.
        self._model_row = QtWidgets.QHBoxLayout()
        self.layout().addLayout(self._model_row)
        self._model_label = None
        self._relocated_model_dropdown = None

        self.task = "Segmentation"
        self.task_dropdown, task_layout = self._add_choice_param(
            "task", self.task, TASKS, title="Task:", tooltip=get_tooltip("image_series_annotator", "task"),
        )
        # Let the dropdown absorb the row's extra width so the 'Task:' label hugs it (otherwise the
        # label expands and leaves a gap between the text and the dropdown).
        size_policy = getattr(QtWidgets.QSizePolicy, "Policy", QtWidgets.QSizePolicy)
        self.task_dropdown.setSizePolicy(size_policy.Expanding, size_policy.Fixed)
        self.layout().addLayout(task_layout)
        self._task_label = task_layout.itemAt(0).widget()

        # Segmentation folder (object classification only), toggled by the task selector.
        self.segmentation_folder = None
        self.segmentation_pattern = "*"
        self._seg_folder_container = QtWidgets.QWidget()
        seg_layout = QtWidgets.QVBoxLayout()
        seg_layout.setContentsMargins(0, 0, 0, 0)
        _, path_layout = self._add_path_param(
            "segmentation_folder", self.segmentation_folder, "directory",
            title="Segmentation Folder", placeholder="Folder with segmentations (optional) ...",
            tooltip=get_tooltip("image_series_annotator", "segmentation_folder"),
        )
        seg_layout.addLayout(path_layout)
        self._seg_folder_container.setLayout(seg_layout)
        self._seg_folder_container.setVisible(False)
        self.layout().addWidget(self._seg_folder_container)

        # Embedded model / embedding settings, reusing the annotator's embedding widget so the model
        # family/size, image-dimensions and tiling controls (and, for the classifier tasks, the
        # 'Advanced Models' selector) are not duplicated. Swapped to match the selected task.
        self._embedding_container = QtWidgets.QWidget()
        self._embedding_container.setLayout(QtWidgets.QVBoxLayout())
        self._embedding_container.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(self._embedding_container)
        self._embedding_widget = None
        self._rebuild_embedding_widget()

        # Swap the embedding widget + toggle the segmentation folder on task change, and re-judge the
        # default tiling from the first image when the input folder or pattern changes.
        self.task_dropdown.currentTextChanged.connect(self._on_task_changed)
        self._folder_textbox.textChanged.connect(self._update_default_tiling)
        self._pattern_param.textChanged.connect(self._update_default_tiling)

    def _build_embedding_widget(self):
        # The classifier tasks use the classification embedding widget (which adds the 'Advanced
        # Models' selector); tracking uses the SAM2-only timeseries widget; segmentation the default.
        if self.task in ("Object Classification", "Pixel Classification"):
            ew = widgets.ClassificationEmbeddingWidget(ndim_choice=True)
        elif self.task == "Tracking":
            ew = widgets.EmbeddingWidget(sam2_only=True, is_timeseries=True)
        else:
            ew = widgets.EmbeddingWidget(ndim_choice=True)
        # The launcher works on a folder and the harness computes embeddings itself, so the
        # 'Compute Embeddings' button is not needed (the image / model row is hidden in the rebuild,
        # after the model dropdown has been relocated next to the Task dropdown).
        ew.run_button.hide()
        return ew

    def _rebuild_embedding_widget(self, *args):
        # Drop the previous embedding widget and the model row relocated from it.
        if self._model_label is not None:
            self._model_label.setParent(None)
            self._model_label.deleteLater()
            self._model_label = None
        if self._relocated_model_dropdown is not None:
            self._relocated_model_dropdown.setParent(None)
            self._relocated_model_dropdown.deleteLater()
            self._relocated_model_dropdown = None
        if self._embedding_widget is not None:
            self._embedding_widget.setParent(None)
            self._embedding_widget.deleteLater()

        self._embedding_widget = self._build_embedding_widget()
        self._embedding_container.layout().addWidget(self._embedding_widget)

        # Relocate the model-family dropdown into the model row (above Task), then hide the now-empty
        # image / model row at the top of the embedding widget.
        self._model_label = QtWidgets.QLabel("Model:")
        self._relocated_model_dropdown = self._embedding_widget.model_family_dropdown
        # Expanding so the 'Model:' label hugs the dropdown (matching the Task field).
        size_policy = getattr(QtWidgets.QSizePolicy, "Policy", QtWidgets.QSizePolicy)
        self._relocated_model_dropdown.setSizePolicy(size_policy.Expanding, size_policy.Fixed)
        self._model_row.addWidget(self._model_label)
        self._model_row.addWidget(self._relocated_model_dropdown)
        _hide_layout_widgets(self._embedding_widget.layout().itemAt(0))

        # Uniform label widths so the input / output / pattern fields and the model / task dropdowns
        # all start at the same x and span the same width.
        self._align_widths(
            [self._folder_label, self._pattern_label, self._output_label, self._task_label, self._model_label]
        )

        self._update_default_tiling()

    def _on_task_changed(self, *args):
        self.task = self.task_dropdown.currentText()
        self._seg_folder_container.setVisible(self.task == "Object Classification")
        self._rebuild_embedding_widget()

    def _update_default_tiling(self, *args):
        # Judge default tiling from the first image in the series, mirroring the embedding widget's
        # per-image auto-tiling (which keys off a selected layer that the launcher does not have).
        ew = self._embedding_widget
        if ew is None or not self.folder:
            return
        files = sorted(glob(os.path.join(self.folder, self.pattern)))
        if not files:
            return
        try:
            shape = imageio.improps(files[0]).shape
        except Exception:
            return
        # Drop a trailing channel axis (RGB/RGBA) for the in-plane size judgement.
        spatial = shape[:-1] if (len(shape) >= 3 and shape[-1] in (3, 4)) else shape
        ew._apply_default_tiling_for_shape(spatial)

    def _validate_inputs(self):
        missing_data = self.folder is None or len(glob(os.path.join(self.folder, self.pattern))) == 0
        missing_output = self.output_folder is None
        if missing_data or missing_output:
            msg = ""
            if missing_data:
                msg += "The input folder is missing or empty. "
            if missing_output:
                msg += "The output folder is missing."
            return widgets._generate_message("error", msg)

        # For object classification with provided segmentations, the counts must match.
        if self.task == "Object Classification" and self.segmentation_folder:
            n_img = len(glob(os.path.join(self.folder, self.pattern)))
            n_seg = len(glob(os.path.join(self.segmentation_folder, self.segmentation_pattern)))
            if n_img != n_seg:
                return widgets._generate_message(
                    "error", f"The number of images ({n_img}) and segmentations ({n_seg}) does not match."
                )
        return False

    def _embedding_paths_for(self, image_files):
        # Per-item embedding zarr paths under the chosen folder (the classification series functions
        # take an explicit list); 'None' when no embedding folder is set.
        save_path = self._embedding_widget.embeddings_save_path
        if not save_path:
            return None
        os.makedirs(save_path, exist_ok=True)
        return [
            os.path.join(save_path, os.path.splitext(os.path.basename(f))[0] + ".zarr") for f in image_files
        ]

    def __call__(self, skip_validate=False):
        ew = self._embedding_widget
        ew._validate_model_type_and_custom_weights()

        if not skip_validate and self._validate_inputs():
            return

        # Only forward tiling params when tiling is enabled (the tile/halo defaults are nonzero even
        # when tiling is 'no'). 'ndim' comes from the image-dimensions dropdown (None = auto-detect).
        if ew.tiling == "yes":
            tile_shape, halo = widgets._process_tiling_inputs(ew.tile_x, ew.tile_y, ew.halo_x, ew.halo_y)
        else:
            tile_shape, halo = None, None
        ndim = ew._ndim_override()

        common = dict(
            model_type=ew.model_type, tile_shape=tile_shape, halo=halo,
            checkpoint_path=ew.custom_weights, device=ew.device,
            viewer=self._viewer, return_viewer=True,
        )

        if self.task == "Segmentation":
            image_folder_annotator(
                input_folder=self.folder, output_folder=self.output_folder, ndim=ndim,
                pattern=self.pattern, embedding_path=ew.embeddings_save_path, **common,
            )
        else:
            image_files = sorted(glob(os.path.join(self.folder, self.pattern)))
            if self.task == "Tracking":
                from .annotator_tracking import image_series_tracking_annotator
                image_series_tracking_annotator(
                    image_files, self.output_folder, embedding_path=ew.embeddings_save_path, **common,
                )
            else:
                embedding_paths = self._embedding_paths_for(image_files)
                if self.task == "Pixel Classification":
                    from .pixel_classifier import image_series_pixel_classifier
                    image_series_pixel_classifier(
                        image_files, self.output_folder, embedding_paths=embedding_paths, ndim=ndim, **common,
                    )
                else:
                    # Object Classification: load the per-image segmentations if a folder is given.
                    from .object_classifier import image_series_object_classifier
                    seg_files = None
                    if self.segmentation_folder:
                        seg_files = sorted(glob(os.path.join(self.segmentation_folder, self.segmentation_pattern)))
                    image_series_object_classifier(
                        image_files, seg_files, self.output_folder,
                        embedding_paths=embedding_paths, ndim=ndim, **common,
                    )

        # The console has done its job (task + settings are locked in for this session); remove it so
        # the annotator has the screen to itself.
        self._dismiss()

    def _dismiss(self):
        # Remove the console dock after a successful launch. Deferred so it runs after this click
        # handler returns (deleting the widget mid-handler is unsafe); a no-op if the console was not
        # added as a dock (e.g. in tests that construct it directly).
        def _remove():
            try:
                self._viewer.window.remove_dock_widget(self)
            except Exception:
                pass
        QTimer.singleShot(0, _remove)


def main():
    """@private"""
    import argparse

    available_models = list(get_model_names())
    available_models = ", ".join(available_models)

    parser = argparse.ArgumentParser(description="Annotate a series of images from a folder.")
    parser.add_argument(
        "-i", "--input_folder", required=True,
        help="The folder containing the image data. The data can be stored in any common format (tif, jpg, png, ...)."
    )
    parser.add_argument(
        "-o", "--output_folder", required=True,
        help="The folder where the segmentation results will be stored."
    )
    parser.add_argument(
        "--ndim", help="The number of spatial dimensions (2 or 3). If None, auto-detected from image shape."
    )
    parser.add_argument(
        "-p", "--pattern", default="*",
        help="The pattern to select the images to annotator from the input folder. E.g. *.tif to annotate all tifs."
        "By default all files in the folder will be loaded and annotated."
    )
    parser.add_argument(
        "--initial_segmentation_folder",
        help="A folder with initial segmentation results. By default no initial segmentations are loaded."
    )
    parser.add_argument(
        "--initial_segmentation_pattern",
        help="The glob pattern for loading files from `initial_segmentation_folder`."
    )
    parser.add_argument(
        "-e", "--embedding_path",
        help="The filepath for saving/loading the pre-computed image embeddings. "
        "NOTE: It is recommended to pass this argument and store the embeddings, "
        "otherwise they will be recomputed every time (which can take a long time)."
    )
    parser.add_argument(
        "-m", "--model_type", default=DEFAULT_MODEL,
        help=f"The segment anything model that will be used, one of {available_models}."
    )
    parser.add_argument(
        "-c", "--checkpoint", default=None,
        help="Checkpoint from which the SAM model will be loaded."
    )
    parser.add_argument(
        "-d", "--device", default=None,
        help="The device to use for the predictor. Can be one of 'cuda', 'cpu' or 'mps' (only MAC)."
        "By default the most performant available device will be selected."
    )

    parser.add_argument(
        "--tile_shape", nargs="+", type=int, help="The tile shape for using tiled prediction", default=None
    )
    parser.add_argument(
        "--halo", nargs="+", type=int, help="The halo for using tiled prediction", default=None
    )
    parser.add_argument("--precompute_amg_state", action="store_true")
    parser.add_argument("--prefer_decoder", action="store_false")
    parser.add_argument("--skip_segmented", action="store_false")

    args = parser.parse_args()

    image_folder_annotator(
        args.input_folder, args.output_folder, pattern=args.pattern, ndim=args.ndim,
        initial_segmentation_folder=args.initial_segmentation_folder,
        initial_segmentation_pattern=args.initial_segmentation_pattern,
        embedding_path=args.embedding_path, model_type=args.model_type,
        tile_shape=args.tile_shape, halo=args.halo, precompute_amg_state=args.precompute_amg_state,
        checkpoint_path=args.checkpoint, device=args.device,
        prefer_decoder=args.prefer_decoder, skip_segmented=args.skip_segmented
    )
