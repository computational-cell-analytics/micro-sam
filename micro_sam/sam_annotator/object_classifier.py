from typing import List, Optional, Tuple, Union

import napari
import numpy as np
import pandas as pd
import torch

from magicgui import magic_factory
from magicgui.widgets import Widget, Container, FunctionGui, create_widget
from qtpy import QtWidgets

from nifty.tools import takeDict
from skimage.transform import resize
from skimage.measure import regionprops_table
from sklearn.ensemble import RandomForestClassifier

from .. import util
from ._state import AnnotatorState
from . import _widgets as widgets
from .util import _sync_embedding_widget

#
# Utility functionality.
# Some of this could be refactored to general purpose functionality that can also
# be used for inference with the trained classifier.
#


def _compute_object_features(image_embeddings, segmentation):
    assert segmentation.ndim == 2, "3D support is not yet implemented"
    embeddings = image_embeddings["features"].squeeze()
    # Put channel last.
    embeddings = embeddings.transpose(1, 2, 0)

    # TODO proper rescaling !!!
    # TODO is this direction of rescaling ok?
    # Probably not, because we loose small segmented objects.
    segmentation_rescaled = resize(
        segmentation, embeddings.shape[:2], order=0, anti_aliasing=False, preserve_range=True
    ).astype(segmentation.dtype)

    # TODO which features do we use?
    all_features = regionprops_table(
        segmentation_rescaled, intensity_image=embeddings, properties=("label", "area", "mean_intensity"),
    )
    seg_ids = all_features["label"]
    features = pd.DataFrame(all_features)[
        ["area"] + [f"mean_intensity-{i}" for i in range(embeddings.shape[-1])]
    ].values

    return seg_ids, features


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


def _train_rf(features, labels):
    assert len(features) == len(labels)
    valid = labels != 0
    X, y = features[valid], labels[valid]
    # TODO other settings than default?
    rf = RandomForestClassifier()
    rf.fit(X, y)
    return rf


def _predict_rf(rf, features, seg_ids):
    pred = rf.predict(features)
    assert len(pred) == len(seg_ids)
    prediction = {seg_id: class_pred for seg_id, class_pred in zip(seg_ids, pred)}
    prediction[0] = 0
    return prediction


def _project_prediction(segmentation, object_prediction):
    return takeDict(object_prediction, segmentation)


# TODO do we add a shortcut?
@magic_factory(call_button="Train and predict")
def _train_and_predict_rf_widget(viewer: "napari.viewer.Viewer") -> None:
    # Get the object features and the annotations.
    state = AnnotatorState()
    annotations = viewer.layers["annotations"].data
    segmentation = state.segmentation_selection.get_value().data

    if state.object_features is None:
        # TODO check here that the image embeddings are computed!
        image_embeddings = state.image_embeddings
        seg_ids, features = _compute_object_features(image_embeddings, segmentation)
        state.seg_ids = seg_ids
        state.object_features = features
    else:
        features, seg_ids = state.object_features, state.seg_ids

    labels = _accumulate_labels(segmentation, annotations)
    # TODO: raise a warining and return as this is not valid
    if (labels == 0).all():
        pass

    # Run RF training and store it in the state.
    rf = _train_rf(features, labels)
    state.object_rf = rf

    # Run and set the prediction.
    object_prediction = _predict_rf(rf, features, seg_ids)
    prediction_data = _project_prediction(segmentation, object_prediction)
    viewer.layers["prediction"].data = prediction_data


#
# Object classifier implementation.
#

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
            self._viewer.add_labels(data=dummy_data, name="annotations")
            if image_scale is not None:
                self.layers["annotations"].scale = image_scale

        if "prediction" not in self._viewer.layers:
            if layer_choices and "prediction" in layer_choices:
                widgets._validation_window_for_missing_layer("prediction")
            self._viewer.add_labels(data=dummy_data, name="prediction")
            if image_scale is not None:
                self.layers["prediction"].scale = image_scale

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

        # TODO widget for exporting the RF.

        self._widgets = {
            "embeddings": self._embedding_widget,
            "segmentation_selection": self._seg_selection_widget,
            "train_and_predict": self._train_and_predict_widget,
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

        # TODO this needs to be more dynamic
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
            elif widget_name == "segmentation_selection":  # This is a hack, we should check the type instead.
                widget_layout.addLayout(widget)
            else:
                # This is a qt type and we add the widget directly.
                widget_layout.addWidget(widget)
            widget_frame.setLayout(widget_layout)
            self._annotator_widget.layout().addWidget(widget_frame)

        # Add the widgets to the state.
        AnnotatorState().widgets = self._widgets

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

        # Update the image shape if it has changed.
        if state.image_shape != self._shape:
            if len(state.image_shape) != self._ndim:
                raise RuntimeError(
                    f"The dim of the annotator {self._ndim} does not match the image data of shape {state.image_shape}."
                )
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

    Returns:
        The napari viewer, only returned if `return_viewer=True`.
    """
    # TODO handle multi-dim / multi-channel data.
    state = AnnotatorState()
    state.image_shape = image.shape
    ndim = 2

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


# TODO: main function
