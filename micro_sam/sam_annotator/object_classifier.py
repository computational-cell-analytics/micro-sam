import os
from datetime import datetime
from joblib import dump, hash as joblib_hash, load
from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Optional, Tuple, Union

import imageio.v3 as imageio
import napari
import numpy as np
import torch

from magicgui import magicgui
from magicgui.widgets import Widget, Container, FileEdit, FunctionGui, PushButton, create_widget
from napari.utils.notifications import show_info
from qtpy import QtWidgets

from skimage.measure import regionprops_table
from sklearn.ensemble import RandomForestClassifier

from .. import util
from ..object_classification import compute_object_features, project_prediction_to_segmentation
from ._state import AnnotatorState
from . import _widgets as widgets
from .util import _sync_embedding_widget

#
# Utility functionality.
# Some of this could be refactored to general purpose functionality that can also
# be used for inference with the trained classifier.
#


def _accumulate_labels(segmentation, annotations):

    def majority_label(mask, annotation):
        ids, counts = np.unique(annotation[mask], return_counts=True)
        if len(ids) == 1 and ids[0] == 0:
            return 0
        if ids[0] == 0:
            ids, counts = ids[1:], counts[1:]
        return ids[np.argmax(counts)]

    all_features = regionprops_table(
        segmentation, intensity_image=annotations, properties=("label",),
        extra_properties=[majority_label],
    )
    return all_features["majority_label"].astype("int")


def _train_rf(features, labels, previous_features=None, previous_labels=None, **rf_kwargs):
    assert len(features) == len(labels)
    valid = labels != 0
    X, y = features[valid], labels[valid]

    if previous_features is not None:
        assert previous_labels is not None and len(previous_features) == len(previous_labels)
        X = np.concatenate([previous_features, X], axis=0)
        y = np.concatenate([previous_labels, y], axis=0)

    rf = RandomForestClassifier(**rf_kwargs)
    rf.fit(X, y)
    return rf


def _compute_object_features_if_needed(viewer):
    # Returns (features, seg_ids) for the current image+segmentation, computing/caching if needed.
    state = AnnotatorState()
    if state.object_features is None:
        if widgets._validate_embeddings(viewer):
            return None, None
        segmentation = state.segmentation_selection.get_value().data
        seg_ids, features = compute_object_features(state.image_embeddings, segmentation)
        state.seg_ids, state.object_features = seg_ids, features
    return state.object_features, state.seg_ids


def _predict_and_show(viewer, rf, features, seg_ids):
    state = AnnotatorState()
    segmentation = state.segmentation_selection.get_value().data
    try:
        pred = rf.predict(features)
    except ValueError:
        return widgets._generate_message(
            "error", "The loaded classifier does not match the current embeddings. Use the same model type."
        )
    viewer.layers["prediction"].data = project_prediction_to_segmentation(segmentation, pred, seg_ids)
    state.annotator._refresh_label_widget()


def _run_train_and_predict(viewer):
    # Get the object features and the annotations.
    state = AnnotatorState()
    state.annotator._require_layers()
    annotations = viewer.layers["annotations"].data
    segmentation = state.segmentation_selection.get_value().data

    features, seg_ids = _compute_object_features_if_needed(viewer)
    if features is None:
        return None

    previous_features, previous_labels = state.previous_features, state.previous_labels
    labels = _accumulate_labels(segmentation, annotations)
    if (labels == 0).all() and (previous_labels is None):
        return widgets._generate_message("error", "You have not provided any annotations.")

    # Run RF training and store it in the state.
    state.object_rf = _train_rf(
        features, labels, previous_features=previous_features, previous_labels=previous_labels,
        n_estimators=200, max_depth=10, n_jobs=cpu_count(),
    )

    # Run and set the prediction.
    _predict_and_show(viewer, state.object_rf, features, seg_ids)


def _load_rf(viewer, model_path):
    state = AnnotatorState()
    model_path = str(model_path)
    if not model_path or not os.path.exists(model_path):
        return widgets._generate_message("error", "You have to provide a valid path to load the classifier.")
    state.object_rf = load(model_path)

    # Predict on the current image if embeddings are available.
    features, seg_ids = _compute_object_features_if_needed(viewer)
    if features is None:
        return None
    state.annotator._require_layers()
    _predict_and_show(viewer, state.object_rf, features, seg_ids)


def _resolve_export_path(viewer, export_dir, rf):
    # The classifier is saved into the chosen folder (defaulting to the current working directory)
    # with a descriptive auto-generated name: <image>_<nclasses>classes_<date>_<time>_<hash>.joblib.
    state = AnnotatorState()
    name = state.image_name or (viewer.layers["image"].name if "image" in viewer.layers else "image")
    name = os.path.splitext(os.path.basename(str(name)))[0]
    n_classes = len(rf.classes_)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{name}_{n_classes}classes_{stamp}_{joblib_hash(rf)[:8]}.joblib"
    base_dir = str(export_dir).strip() or os.getcwd()
    return os.path.join(base_dir, fname)


def _gather_class_names(rf):
    # Collect the user-provided class names from the label widget, keyed by class id. Only
    # non-empty names for classes the classifier actually knows are included. Returns None if
    # no names were given, so the attribute is only attached when found.
    state = AnnotatorState()
    names = getattr(state.annotator, "_label_names", None) or {}
    class_names = {int(k): v for k, v in names.items() if v and int(k) in rf.classes_}
    return class_names or None


def _save_rf(viewer, export_dir):
    state = AnnotatorState()
    if state.object_rf is None:
        return widgets._generate_message("error", "You have not trained or loaded a classifier yet.")
    out_path = _resolve_export_path(viewer, export_dir, state.object_rf)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Attach the class names (if any) to the classifier so they travel with the saved model.
    class_names = _gather_class_names(state.object_rf)
    if class_names is not None:
        state.object_rf.class_names_ = class_names
    dump(state.object_rf, out_path)
    show_info(f"Exported classifier to {out_path}")


def _create_train_widget(viewer):
    # The 'Train and predict' button is kept at the top level, outside the settings dropdown.
    train_button = PushButton(text="Train and predict [Shift + T]")
    train_button.clicked.connect(lambda: _run_train_and_predict(viewer))

    @viewer.bind_key("Shift-T", overwrite=True)
    def _train_and_predict(event=None):
        _run_train_and_predict(viewer)

    return Container(widgets=[train_button], labels=False)


def _create_classifier_io_widget(viewer):
    # Classifier load/export, with separate paths (load a stored model, retrain, then export
    # the current one elsewhere). The path fields follow the same path-selector style as the
    # custom weights in the embedding widget.
    load_path = FileEdit(label="load classifier path:", mode="r", filter="*.joblib")
    load_path.line_edit.native.setPlaceholderText("/path/to/stored_model.joblib")
    load_path.native.setToolTip(
        "Path to a stored classifier (.joblib) to load and apply to the current image."
    )
    load_button = PushButton(text="Load classifier")
    load_button.native.setToolTip("Load the selected classifier and predict on the current image.")
    load_button.clicked.connect(lambda: _load_rf(viewer, load_path.value))

    # Export saves to the chosen folder (pre-filled with the current working directory) with an
    # auto-generated name: <image>_<nclasses>classes_<date>_<time>_<hash>.joblib.
    export_dir = FileEdit(label="export classifier folder:", mode="d", value=os.getcwd())
    export_dir.native.setToolTip(
        "Folder where the trained classifier is saved. The file name is generated automatically as "
        "<image>_<nclasses>classes_<date>_<time>_<hash>.joblib. Defaults to the current working directory."
    )
    export_button = PushButton(text="Export classifier")
    export_button.native.setToolTip("Save the current classifier into the selected folder.")
    export_button.clicked.connect(lambda: _save_rf(viewer, export_dir.value))

    return Container(widgets=[load_path, load_button, export_dir, export_button], labels=False)

#
# Object classifier implementation.
#


# TODO add a gui element that shows the current label ids, how many objects are labeled, and that
# enables naming them so that the user can keep track of what has been labeled
class ObjectClassifier(QtWidgets.QScrollArea):

    def _require_layers(self, layer_choices: Optional[List[str]] = None):
        # Check whether the image is initialized already. And use the image shape and scale for the layers.
        state = AnnotatorState()
        shape = self._shape if state.image_shape is None else state.image_shape

        # Add the label layers for the current object, the automatic segmentation and the committed segmentation.
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
            # Start painting with label id 1 (id 0 is the unlabeled background). napari already
            # defaults 'selected_label' to 1, so assigning 1 fires no change-event and the layer
            # controls spinbox stays at its displayed 0. Toggle to force the event so it shows 1.
            annotation_layer.selected_label = 2
            annotation_layer.selected_label = 1

        if "prediction" not in self._viewer.layers:
            if layer_choices and "prediction" in layer_choices:
                widgets._validation_window_for_missing_layer("prediction")
            self._viewer.add_labels(data=dummy_data, name="prediction")
            if image_scale is not None:
                self._viewer.layers["prediction"].scale = image_scale

        # Move 'annotations' to the top of the layer stack so scribbles are always visible above
        # the segmentation and prediction, and make it the active layer so the controls (incl. the
        # label id) show it and the user can paint right away rather than on another layer.
        self._viewer.layers.move(self._viewer.layers.index("annotations"), len(self._viewer.layers))
        self._viewer.layers.selection.active = self._viewer.layers["annotations"]

    def _create_segmentation_layer_section(self):
        segmentation_selection = QtWidgets.QVBoxLayout()
        segmentation_layer_widget = QtWidgets.QLabel("Segmentation:")
        segmentation_selection.addWidget(segmentation_layer_widget)
        self.segmentation_selection = create_widget(annotation=napari.layers.Labels)
        state = AnnotatorState()
        state.segmentation_selection = self.segmentation_selection
        segmentation_selection.addWidget(self.segmentation_selection.native)
        return segmentation_selection

    def _create_label_widget(self):
        self._label_form = QtWidgets.QFormLayout()
        scroll_area = QtWidgets.QScrollArea()
        inner = QtWidgets.QWidget()
        inner.setLayout(self._label_form)
        scroll_area.setWidget(inner)
        scroll_area.setWidgetResizable(True)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Object label names:"))
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

        # One section: the "Classification Settings" dropdown (segmentation selection and classifier
        # load/export) on top, with the 'Train and predict' button below it.
        self._train_and_predict_widget = _create_train_widget(self._viewer)
        self._seg_selection_widget = self._create_segmentation_layer_section()
        self._classifier_io_widget = _create_classifier_io_widget(self._viewer)

        settings = QtWidgets.QWidget()
        settings.setLayout(QtWidgets.QVBoxLayout())
        seg_container = QtWidgets.QWidget()
        seg_container.setLayout(self._seg_selection_widget)
        settings.layout().addWidget(seg_container)
        settings.layout().addWidget(self._classifier_io_widget.native)
        collapsible = widgets._make_collapsible(settings, title="Classification Settings")

        classification_section = QtWidgets.QWidget()
        classification_section.setLayout(QtWidgets.QVBoxLayout())
        classification_section.layout().addWidget(collapsible)
        classification_section.layout().addWidget(self._train_and_predict_widget.native)

        # A separate section: the object label names.
        self._label_widget = self._create_label_widget()

        self._widgets = {
            "embeddings": self._embedding_widget,
            "classification": classification_section,
            "label_widget": self._label_widget,
        }

    def __init__(self, viewer: "napari.viewer.Viewer") -> None:
        """Create the GUI for the object classifier.

        Args:
            viewer: The napari viewer.
        """
        super().__init__()
        self._viewer = viewer
        self._annotator_widget = QtWidgets.QWidget()
        self._annotator_widget.setLayout(QtWidgets.QVBoxLayout())

        # Add the layers for prompts and segmented obejcts.
        # Initialize with a dummy shape, which is reset to the correct shape once an image is set.
        self._shape = (256, 256)
        self._require_layers()
        self._ndim = len(self._shape)

        # Create all the widgets and add them to the layout.
        self._label_names = {}  # The names for the object labels.
        self._create_widgets()

        # We could refactor this.
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

        # Before we reset the layers, we ensure all expected layers exist.
        self._require_layers()

        # Update the image scale.
        scale = state.image_scale

        # Reset all layers.
        self._viewer.layers["annotations"].data = np.zeros(self._shape, dtype="uint32")
        self._viewer.layers["annotations"].scale = scale
        self._viewer.layers["prediction"].data = np.zeros(self._shape, dtype="uint32")
        self._viewer.layers["prediction"].scale = scale


def object_classifier(
    image: np.ndarray,
    segmentation: np.ndarray,
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
    """Start the object classifier for a given image and segmentation.

    Args:
        image: The image data.
        segmentation: The segmentation data.
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
    viewer.add_labels(segmentation, name="segmentation")

    annotator = ObjectClassifier(viewer)

    # Trigger layer update of the annotator so that layers have the correct shape.
    # And initialize the 'committed_objects' with the segmentation result if it was given.
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


def image_series_object_classifier(
    images: List[np.ndarray],
    segmentations: List[np.ndarray],
    output_folder: str,
    embedding_paths: Optional[List[Union[str, util.ImageEmbeddings]]] = None,
    model_type: str = util._DEFAULT_MODEL,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    checkpoint_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    ndim: Optional[int] = None,
) -> None:
    """Start the object classifier for a list of images and segmentations.

    This function will save the all features and labels for annotated objects,
    to enable training a random forest on multiple images.

    Args:
        images: The input images.
        segmentations: The input segmentations.
        output_folder: The folder where segmentation results, trained random forest
            and the features, labels aggregated during training will be saved.
        embedding_paths: Filepaths where to save the embeddings
            or the precompted image embeddings computed by `precompute_image_embeddings`.
        model_type: The Segment Anything model to use. For details on the available models check out
            https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#finetuned-models.
        tile_shape: Shape of tiles for tiled embedding prediction.
            If `None` then the whole image is passed to Segment Anything.
        halo: Shape of the overlap between tiles, which is needed to segment objects on tile borders.
        checkpoint_path: Path to a custom checkpoint from which to load the SAM model.
        device: The computational device to use for the SAM model.
            By default, automatically chooses the best available device.
        ndim: The dimensionality of the data. If not given will be derived from the data.
    """
    # TODO precompute the embeddings if not computed, can re-use 'precompute' from image series annotator.
    # TODO support file paths as inputs
    # TODO option to skip segmented
    if len(images) != len(segmentations):
        raise ValueError(
            f"Expect the same number of images and segmentations, got {len(images)}, {len(segmentations)}."
        )

    end_msg = "You have annotated the last image. Do you wish to close napari?"

    # Initialize the object classifier on the fist image / segmentation.
    viewer = object_classifier(
        image=images[0], segmentation=segmentations[0],
        embedding_path=None if embedding_paths is None else embedding_paths[0],
        model_type=model_type, tile_shape=tile_shape, halo=halo,
        return_viewer=True, checkpoint_path=checkpoint_path,
        device=device, ndim=ndim,
    )

    os.makedirs(output_folder, exist_ok=True)
    next_image_id = 0

    def _save_prediction(image, pred, image_id):
        fname = f"{Path(image).stem}_prediction.tif" if isinstance(image, str) else f"prediction_{image_id}.tif"
        save_path = os.path.join(output_folder, fname)
        imageio.imwrite(save_path, pred, compression="zlib")

    # TODO handle cases where rf for the image was not trained, raise a message, enable contnuing
    # Add functionality for going to the next image.
    @magicgui(call_button="Next Image [N]")
    def next_image(*args):
        nonlocal next_image_id

        # Get the state and the current segmentation (note that next image id has not yet been increased)
        state = AnnotatorState()
        segmentation = segmentations[next_image_id]

        # Keep track of the previous features and labels.
        labels = _accumulate_labels(segmentation, viewer.layers["annotations"].data)
        valid = labels != 0
        if valid.sum() > 0:
            features, labels = state.object_features[valid], labels[valid]
            if state.previous_features is None:
                state.previous_features, state.previous_labels = features, labels
            else:
                state.previous_features = np.concatenate([state.previous_features, features], axis=0)
                state.previous_labels = np.concatenate([state.previous_labels, labels], axis=0)
            # Save the accumulated features and labels.
            np.save(os.path.join(output_folder, "features.npy"), state.previous_features)
            np.save(os.path.join(output_folder, "labels.npy"), state.previous_labels)

        # Save the current prediction and RF.
        _save_prediction(images[next_image_id], viewer.layers["prediction"].data, next_image_id)
        dump(state.object_rf, os.path.join(output_folder, "rf.joblib"))

        # Go to the next image.
        next_image_id += 1

        # Check if we are done.
        if next_image_id == len(images):
            # Inform the user via dialog.
            abort = widgets._generate_message("info", end_msg)
            if not abort:
                viewer.close()
            return

        # Get the next image, segmentation and embedding_path.
        image = images[next_image_id]
        segmentation = segmentations[next_image_id]
        embedding_path = None if embedding_paths is None else embedding_paths[next_image_id]

        # Set the new image in the viewer, state and annotator.
        viewer.layers["image"].data = image
        viewer.layers["segmentation"].data = segmentation

        state.initialize_predictor(
            image, model_type=model_type, ndim=ndim,
            save_path=embedding_path,
            tile_shape=tile_shape, halo=halo,
            predictor=state.predictor, device=device,
        )
        state.image_shape = image.shape if image.ndim == ndim else image.shape[:-1]
        state.annotator._update_image()

        # Clear the object features and seg-ids from the state.
        state.object_features = None
        state.seg_ids = None

    viewer.window.add_dock_widget(next_image)

    @viewer.bind_key("n", overwrite=True)
    def _next_image(viewer):
        next_image(viewer)

    napari.run()


# TODO: folder annotator
# TODO: main function
