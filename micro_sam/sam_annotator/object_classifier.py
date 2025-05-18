import os
from joblib import dump
from pathlib import Path
from typing import List, Optional, Tuple, Union

import imageio.v3 as imageio
import napari
import numpy as np
import torch

from magicgui import magic_factory, magicgui
from magicgui.widgets import Widget, Container, FunctionGui, create_widget
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


# TODO do we add a shortcut?
@magic_factory(call_button="Train and predict")
def _train_and_predict_rf_widget(viewer: "napari.viewer.Viewer") -> None:
    # Get the object features and the annotations.
    state = AnnotatorState()
    state.annotator._require_layers()
    annotations = viewer.layers["annotations"].data
    segmentation = state.segmentation_selection.get_value().data

    if state.object_features is None:
        if widgets._validate_embeddings(viewer):
            return None
        image_embeddings = state.image_embeddings
        seg_ids, features = compute_object_features(image_embeddings, segmentation)
        state.seg_ids = seg_ids
        state.object_features = features
    else:
        features, seg_ids = state.object_features, state.seg_ids

    previous_features, previous_labels = state.previous_features, state.previous_labels
    labels = _accumulate_labels(segmentation, annotations)
    if (labels == 0).all() and (previous_labels is None):
        return widgets._generate_message("error", "You have not provided any annotations.")

    # Run RF training and store it in the state.
    # TODO should we over-ride any defaults here?
    rf = _train_rf(features, labels, previous_features=previous_features, previous_labels=previous_labels)
    state.object_rf = rf

    # Run and set the prediction.
    pred = rf.predict(features)
    prediction_data = project_prediction_to_segmentation(segmentation, pred, seg_ids)
    viewer.layers["prediction"].data = prediction_data


@magic_factory(call_button="Export Classifier")
def _create_export_rf_widget(export_path: Optional[Path] = None) -> None:
    state = AnnotatorState()
    rf = state.object_rf
    if rf is None:
        return widgets._generate_message("error", "You have not run training yet.")
    if export_path is None or export_path == "":
        return widgets._generate_message("error", "You have to provide an export path.")
    # Do we add an extension? .joblib?
    dump(rf, export_path)
    # TODO show an info method about the export

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

        if "prediction" not in self._viewer.layers:
            if layer_choices and "prediction" in layer_choices:
                widgets._validation_window_for_missing_layer("prediction")
            self._viewer.add_labels(data=dummy_data, name="prediction")
            if image_scale is not None:
                self._viewer.layers["prediction"].scale = image_scale

    def _create_segmentation_layer_section(self):
        segmentation_selection = QtWidgets.QVBoxLayout()
        segmentation_layer_widget = QtWidgets.QLabel("Segmentation:")
        segmentation_selection.addWidget(segmentation_layer_widget)
        self.segmentation_selection = create_widget(annotation=napari.layers.Labels)
        state = AnnotatorState()
        state.segmentation_selection = self.segmentation_selection
        segmentation_selection.addWidget(self.segmentation_selection.native)
        return segmentation_selection

    def _create_widgets(self):
        # Create the embedding widget and connect all events related to it.
        self._embedding_widget = widgets.EmbeddingWidget()
        # Connect events for the image selection box.
        self._viewer.layers.events.inserted.connect(self._embedding_widget.image_selection.reset_choices)
        self._viewer.layers.events.removed.connect(self._embedding_widget.image_selection.reset_choices)
        # Connect the run button with the function to update the image.
        self._embedding_widget.run_button.clicked.connect(self._update_image)

        # Create the widget for training and prediction of the classifier.
        self._train_and_predict_widget = _train_and_predict_rf_widget()

        # Create the widget for segmentation selection.
        self._seg_selection_widget = self._create_segmentation_layer_section()

        # Cretate the widget for exporting the RF.
        self._export_rf_widget = _create_export_rf_widget()

        self._widgets = {
            "embeddings": self._embedding_widget,
            "segmentation_selection": self._seg_selection_widget,
            "train_and_predict": self._train_and_predict_widget,
            "export_rf": self._export_rf_widget,
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
        self._create_widgets()
        # Could refactor this.
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
