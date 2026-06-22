import os

from joblib import dump, load
from typing import List, Optional, Tuple, Union

import napari
import numpy as np
import torch

from magicgui.widgets import Widget, Container, FileEdit, FunctionGui, PushButton
from qtpy import QtWidgets

from .. import util
from ..pixel_classification import (
    accumulate_pixel_labels, compute_pixel_features, project_prediction_to_image, train_pixel_classifier,
)
from ._state import AnnotatorState
from . import _widgets as widgets
from .util import _sync_embedding_widget


def _compute_pixel_features_if_needed(viewer):
    # Returns (features, grid_shape) for the current image, computing and caching them if needed.
    state = AnnotatorState()
    if state.pixel_features is None:
        if widgets._validate_embeddings(viewer):
            return None, None
        features, grid_shape = compute_pixel_features(state.image_embeddings, state.image_shape)
        state.pixel_features, state.pixel_grid_shape = features, grid_shape
    return state.pixel_features, state.pixel_grid_shape


def _predict_and_show(viewer, rf, features, grid_shape):
    state = AnnotatorState()
    try:
        pred = rf.predict(features)
    except ValueError:
        return widgets._generate_message(
            "error", "The loaded classifier does not match the current embeddings. Use the same model type."
        )
    viewer.layers["prediction"].data = project_prediction_to_image(pred, grid_shape, state.image_shape)
    state.annotator._refresh_label_widget()


def _run_train_and_predict(viewer):
    # Get the per-pixel features and the annotations.
    state = AnnotatorState()
    state.annotator._require_layers()
    annotations = viewer.layers["annotations"].data

    features, grid_shape = _compute_pixel_features_if_needed(viewer)
    if features is None:
        return None

    previous_features, previous_labels = state.previous_features, state.previous_labels
    labels = accumulate_pixel_labels(annotations, grid_shape)
    if (labels == 0).all() and (previous_labels is None):
        return widgets._generate_message("error", "You have not provided any annotations.")

    # Run RF training and store it in the state.
    state.pixel_rf = train_pixel_classifier(
        features, labels, previous_features=previous_features, previous_labels=previous_labels,
    )

    # Run and set the prediction.
    _predict_and_show(viewer, state.pixel_rf, features, grid_shape)


def _load_rf(viewer, model_path):
    state = AnnotatorState()
    model_path = str(model_path)
    if not model_path or not os.path.exists(model_path):
        return widgets._generate_message("error", "You have to provide a valid path to load the classifier.")
    state.pixel_rf = load(model_path)

    # Predict on the current image if embeddings are available.
    features, grid_shape = _compute_pixel_features_if_needed(viewer)
    if features is None:
        return None
    state.annotator._require_layers()
    _predict_and_show(viewer, state.pixel_rf, features, grid_shape)


def _save_rf(model_path):
    state = AnnotatorState()
    if state.pixel_rf is None:
        return widgets._generate_message("error", "You have not trained or loaded a classifier yet.")
    model_path = str(model_path)
    if not model_path:
        return widgets._generate_message("error", "You have to provide a path to save the classifier.")
    dump(state.pixel_rf, model_path)


def _create_train_and_predict_widget(viewer):
    # Combine training/prediction and classifier load/save into a single area. The model path
    # field follows the same path-selector style as the custom weights in the embedding widget.
    train_button = PushButton(text="Train and predict")
    model_path = FileEdit(label="classifier path:", mode="w", filter="*.joblib")
    load_button = PushButton(text="Load classifier")
    save_button = PushButton(text="Save classifier")

    train_button.clicked.connect(lambda: _run_train_and_predict(viewer))
    load_button.clicked.connect(lambda: _load_rf(viewer, model_path.value))
    save_button.clicked.connect(lambda: _save_rf(model_path.value))

    io_row = Container(layout="horizontal", widgets=[load_button, save_button], labels=False)
    return Container(widgets=[train_button, model_path, io_row], labels=False)


class PixelClassifier(QtWidgets.QScrollArea):

    def _require_layers(self, layer_choices: Optional[List[str]] = None):
        # Check whether the image is initialized already. And use the image shape and scale for the layers.
        state = AnnotatorState()
        shape = self._shape if state.image_shape is None else state.image_shape

        # Add the label layers for the annotations and the prediction.
        dummy_data = np.zeros(shape, dtype="uint32")
        image_scale = state.image_scale

        # Before adding new layers, we always check whether a layer with this name already exists or not.
        if "annotations" not in self._viewer.layers:
            if layer_choices and "annotations" in layer_choices:
                widgets._validation_window_for_missing_layer("annotations")
            annotation_layer = self._viewer.add_labels(data=dummy_data, name="annotations")
            if image_scale is not None:
                self._viewer.layers["annotations"].scale = image_scale
            # Reduce the brush size and set the default mode to "paint" brush mode.
            annotation_layer.brush_size = 3
            annotation_layer.mode = "paint"
            # Start painting with label id 1 (id 0 is the unlabeled background).
            annotation_layer.selected_label = 1

        if "prediction" not in self._viewer.layers:
            if layer_choices and "prediction" in layer_choices:
                widgets._validation_window_for_missing_layer("prediction")
            self._viewer.add_labels(data=dummy_data, name="prediction")
            if image_scale is not None:
                self._viewer.layers["prediction"].scale = image_scale

    def _create_label_widget(self):
        self._label_form = QtWidgets.QFormLayout()
        scroll_area = QtWidgets.QScrollArea()
        inner = QtWidgets.QWidget()
        inner.setLayout(self._label_form)
        scroll_area.setWidget(inner)
        scroll_area.setWidgetResizable(True)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Pixel label names:"))
        layout.addWidget(scroll_area)

        return layout

    def _make_label_role_widget(self, lbl):
        # Build the left-hand widget for a label row: a color swatch matching the annotation
        # layer color for this id, followed by the exact id ("ID <n>"). The id is stored as the
        # object name so it can be read back reliably when removing vanished rows.
        container = QtWidgets.QWidget()
        container.setObjectName(str(int(lbl)))
        row_layout = QtWidgets.QHBoxLayout(container)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)

        swatch = QtWidgets.QLabel()
        swatch.setFixedSize(14, 14)
        color = self._viewer.layers["annotations"].get_color(int(lbl))
        r, g, b = (int(round(255 * c)) for c in color[:3])
        swatch.setStyleSheet(f"background-color: rgb({r}, {g}, {b}); border: 1px solid #888;")

        row_layout.addWidget(swatch)
        row_layout.addWidget(QtWidgets.QLabel(f"ID {int(lbl)}"))
        row_layout.addStretch(1)
        return container

    def _refresh_label_widget(self):
        state = AnnotatorState()

        # Get the current label ids.
        ids = np.unique(self._viewer.layers["annotations"].data)[1:]
        if state.previous_labels is not None:
            ids = np.union1d(ids, np.unique(state.previous_labels))

        # Add new rows.
        for lbl in ids:
            if lbl in self._label_names:
                continue
            line = QtWidgets.QLineEdit(self._label_names.get(lbl, ""))
            self._label_names[lbl] = ""
            self._label_form.addRow(self._make_label_role_widget(lbl), line)
            line.textChanged.connect(lambda txt, lbl=lbl: self._label_names.__setitem__(lbl, txt))

        # Remove rows whose label vanished. 'removeRow' deletes the row's widgets itself.
        for row in reversed(range(self._label_form.rowCount())):
            lbl_id = int(self._label_form.itemAt(row, QtWidgets.QFormLayout.LabelRole).widget().objectName())
            if lbl_id not in ids:
                self._label_form.removeRow(row)
                self._label_names.pop(lbl_id, None)

    def _create_widgets(self):
        # Create the embedding widget and connect all events related to it.
        self._embedding_widget = widgets.ClassificationEmbeddingWidget()
        # Connect events for the image selection box.
        self._viewer.layers.events.inserted.connect(self._embedding_widget.image_selection.reset_choices)
        self._viewer.layers.events.removed.connect(self._embedding_widget.image_selection.reset_choices)
        # Connect the run button with the function to update the image.
        self._embedding_widget.run_button.clicked.connect(self._update_image)

        # Create the widget for training/prediction and classifier load/save (one combined area).
        self._train_and_predict_widget = _create_train_and_predict_widget(self._viewer)

        # Create the widget for displaying the current label state.
        self._label_widget = self._create_label_widget()

        self._widgets = {
            "embeddings": self._embedding_widget,
            "train_and_predict": self._train_and_predict_widget,
            "label_widget": self._label_widget,
        }

    def __init__(self, viewer: "napari.viewer.Viewer") -> None:
        """Create the GUI for the pixel classifier.

        Args:
            viewer: The napari viewer.
        """
        super().__init__()
        self._viewer = viewer
        self._annotator_widget = QtWidgets.QWidget()
        self._annotator_widget.setLayout(QtWidgets.QVBoxLayout())

        # Add the layers for annotations and prediction.
        # Initialize with a dummy shape, which is reset to the correct shape once an image is set.
        self._shape = (256, 256)
        self._require_layers()
        self._ndim = len(self._shape)

        # Create all the widgets and add them to the layout.
        self._label_names = {}  # The names for the pixel labels.
        self._create_widgets()

        for widget_name, widget in self._widgets.items():
            widget_frame = QtWidgets.QGroupBox()
            widget_layout = QtWidgets.QVBoxLayout()
            if isinstance(widget, (Container, FunctionGui, Widget)):
                # This is a magicgui type and we need to get the native qt widget.
                widget_layout.addWidget(widget.native)
            elif isinstance(widget, QtWidgets.QLayout):
                widget_layout.addLayout(widget)
            else:
                # This is a qt type and we add the widget directly.
                widget_layout.addWidget(widget)
            widget_frame.setLayout(widget_layout)
            self._annotator_widget.layout().addWidget(widget_frame)

        # Connect the label layer and the refresh function.
        self._refresh_label_widget()

        # Set the expected annotator class to the state.
        state = AnnotatorState()
        state.annotator = self

        # Add the widgets to the state.
        state.widgets = self._widgets

        # Add the widget to the scroll area.
        self.setWidgetResizable(True)  # Allow widget to resize within scroll area.
        self.setWidget(self._annotator_widget)

    def _update_image(self, segmentation_result=None):
        state = AnnotatorState()

        # Whether embeddings already exist and avoid clearing objects in layers.
        if state.skip_recomputing_embeddings:
            return

        if state.image_shape is None:
            return

        # Update the dimension and image shape if it has changed.
        if state.image_shape != self._shape:
            self._ndim = len(state.image_shape)
            self._shape = state.image_shape

        # The features depend on the image, so they have to be recomputed for a new image.
        state.pixel_features = None
        state.pixel_grid_shape = None

        # Before we reset the layers, we ensure all expected layers exist.
        self._require_layers()

        # Update the image scale.
        scale = state.image_scale

        # Reset all layers.
        self._viewer.layers["annotations"].data = np.zeros(self._shape, dtype="uint32")
        self._viewer.layers["annotations"].scale = scale
        self._viewer.layers["prediction"].data = np.zeros(self._shape, dtype="uint32")
        self._viewer.layers["prediction"].scale = scale


def pixel_classifier(
    image: np.ndarray,
    embedding_path: Optional[Union[str, util.ImageEmbeddings]] = None,
    model_type: str = util._DEFAULT_MODEL,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    return_viewer: bool = False,
    viewer: Optional["napari.viewer.Viewer"] = None,
    checkpoint_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    ndim: Optional[int] = None,
) -> Optional["napari.viewer.Viewer"]:
    """Start the pixel classifier for a given image.

    Args:
        image: The image data.
        embedding_path: Filepath where to save the embeddings
            or the precompted image embeddings computed by `precompute_image_embeddings`.
        model_type: The Segment Anything model to use. For details on the available models check out
            https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#finetuned-models.
        tile_shape: Shape of tiles for tiled embedding prediction.
            If `None` then the whole image is passed to Segment Anything.
        halo: Shape of the overlap between tiles, which is needed to segment objects on tile borders.
        return_viewer: Whether to return the napari viewer to further modify it before starting the tool.
            By default, does not return the napari viewer.
        viewer: The viewer to which the Segment Anything functionality should be added.
            This enables using a pre-initialized viewer.
        checkpoint_path: Path to a custom checkpoint from which to load the SAM model.
        device: The computational device to use for the SAM model.
            By default, automatically chooses the best available device.
        ndim: The dimensionality of the data. If not given will be derived from the data.

    Returns:
        The napari viewer, only returned if `return_viewer=True`.
    """
    if ndim is None:
        ndim = image.ndim - 1 if image.shape[-1] == 3 and image.ndim in (3, 4) else image.ndim

    state = AnnotatorState()
    state.image_shape = image.shape[:ndim]

    state.initialize_predictor(
        image, model_type=model_type, save_path=embedding_path,
        halo=halo, tile_shape=tile_shape, precompute_amg_state=False,
        ndim=ndim, checkpoint_path=checkpoint_path, device=device,
        skip_load=False, use_cli=True,
    )

    if viewer is None:
        viewer = napari.Viewer()

    viewer.add_image(image, name="image")

    annotator = PixelClassifier(viewer)

    # Trigger layer update of the annotator so that layers have the correct shape.
    annotator._update_image()

    # Add the annotator widget to the viewer and sync widgets.
    viewer.window.add_dock_widget(annotator)
    _sync_embedding_widget(
        widget=state.widgets["embeddings"],
        model_type=model_type if checkpoint_path is None else state.predictor.model_type,
        save_path=embedding_path,
        checkpoint_path=checkpoint_path,
        device=device,
        tile_shape=tile_shape,
        halo=halo,
    )

    if return_viewer:
        return viewer

    napari.run()
