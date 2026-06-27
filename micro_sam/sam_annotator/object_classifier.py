import os
from joblib import dump
from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Optional, Tuple, Union

import imageio.v3 as imageio
import napari
import numpy as np
import torch

from magicgui import magicgui
from magicgui.widgets import ComboBox
from qtpy import QtWidgets

from skimage.measure import regionprops_table
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .. import util
from ..v2.util import DEFAULT_MODEL
from ..object_classification import compute_object_features, project_prediction_to_segmentation
from ._annotator import _ClassifierBase
from ._state import AnnotatorState
from ._tooltips import get_tooltip
from . import _widgets as widgets
from .util import _sync_embedding_widget

# Object features are the object area plus the per-channel mean of the 256-channel SAM/SAM2 image
# embedding, i.e. 257 features. PCA can reduce to at most this many components.
OBJECT_FEATURES = 257
INTERNAL_LABEL_LAYER_NAMES = {"annotations", "prediction"}


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


def _train_rf(
    features, labels, previous_features=None, previous_labels=None, n_components=None, random_state=0, **rf_kwargs
):
    assert len(features) == len(labels)
    valid = labels != 0
    X, y = features[valid], labels[valid]

    if previous_features is not None:
        assert previous_labels is not None and len(previous_features) == len(previous_labels)
        X = np.concatenate([previous_features, X], axis=0)
        y = np.concatenate([previous_labels, y], axis=0)

    rf = RandomForestClassifier(random_state=random_state, **rf_kwargs)

    # Optionally reduce the features to the top-n PCA components. n_components is clamped to the
    # number of features and samples; if it covers all features we skip PCA and use the plain RF.
    # Object features mix area (large magnitude) with embedding means (small), so we standardize
    # them before PCA to keep area from dominating the components.
    n_features = X.shape[1]
    k = min(int(n_components), n_features, len(X)) if n_components else 0
    if 0 < k < n_features:
        model = Pipeline(
            [("scaler", StandardScaler()), ("pca", PCA(n_components=k, random_state=random_state)), ("rf", rf)]
        )
    else:
        model = rf

    model.fit(X, y)
    return model


# TODO add a gui element that shows the current label ids, how many objects are labeled, and that
# enables naming them so that the user can keep track of what has been labeled
class ObjectClassifier(_ClassifierBase):
    """GUI for the object classifier: trains a random forest on per-object SAM/SAM2 embedding features."""

    rf_attr = "object_rf"
    features_attr = "object_features"
    aux_attr = "seg_ids"
    label_widget_title = "Object label names:"
    max_components = OBJECT_FEATURES
    tool_key = "object"

    def _get_selected_segmentation_layer(self):
        state = AnnotatorState()
        segmentation_layer = None if state.segmentation_selection is None else state.segmentation_selection.get_value()
        if segmentation_layer is None:
            widgets._generate_message("error", "You have to select a segmentation labels layer.")
            return None
        return segmentation_layer

    def _compute_features(self):
        # Returns (features, seg_ids) for the current image+segmentation, computing/caching if needed.
        state = AnnotatorState()
        if state.object_features is None:
            if widgets._validate_embeddings(self._viewer):
                return None, None
            segmentation_layer = self._get_selected_segmentation_layer()
            if segmentation_layer is None:
                return None, None
            image, upsampler, ok = self._resolve_anyup()
            if not ok:
                return None, None
            seg_ids, features = compute_object_features(
                state.image_embeddings, segmentation_layer.data, image=image, upsampler=upsampler,
            )
            state.seg_ids, state.object_features = seg_ids, features
        return state.object_features, state.seg_ids

    def _compute_training_labels(self, aux):
        segmentation_layer = self._get_selected_segmentation_layer()
        if segmentation_layer is None:
            return None
        return _accumulate_labels(segmentation_layer.data, self._viewer.layers["annotations"].data)

    def _train(self, features, labels, previous_features, previous_labels, n_components, random_state):
        return _train_rf(
            features, labels, previous_features=previous_features, previous_labels=previous_labels,
            n_estimators=200, max_depth=10, n_jobs=cpu_count(), n_components=n_components, random_state=random_state,
        )

    def _project_prediction(self, prediction, aux):
        segmentation_layer = self._get_selected_segmentation_layer()
        if segmentation_layer is None:
            return None
        return project_prediction_to_segmentation(segmentation_layer.data, prediction, aux)

    #
    # The segmentation-layer selector (object classifier only).
    #

    def _is_segmentation_layer(self, layer):
        return isinstance(layer, napari.layers.Labels) and layer.name.lower() not in INTERNAL_LABEL_LAYER_NAMES

    def _find_default_segmentation_layer(self):
        candidates = [layer for layer in self._viewer.layers if self._is_segmentation_layer(layer)]
        if not candidates:
            return None

        for layer in candidates:
            if layer.name == "segmentation":
                return layer

        for layer in candidates:
            if "seg" in layer.name.lower():
                return layer

        return candidates[0]

    def _select_default_segmentation_layer(self):
        default_layer = self._find_default_segmentation_layer()
        if default_layer is not None:
            self.segmentation_selection.value = default_layer

    def _segmentation_layer_choices(self):
        return [(layer.name, layer) for layer in self._viewer.layers if self._is_segmentation_layer(layer)]

    def _reset_segmentation_layer_choices(self, *args):
        previous_selection = self.segmentation_selection.value
        self.segmentation_selection.reset_choices()
        choices = self.segmentation_selection.choices
        if any(layer is previous_selection for layer in choices):
            self.segmentation_selection.value = previous_selection
        else:
            self._select_default_segmentation_layer()
        if self.segmentation_selection.value is not previous_selection:
            self._invalidate_features()

    def _create_segmentation_layer_section(self):
        segmentation_selection = QtWidgets.QVBoxLayout()
        seg_label = QtWidgets.QLabel("Segmentation:")
        seg_label.setToolTip(get_tooltip("classification", "segmentation"))
        segmentation_selection.addWidget(seg_label)
        self.segmentation_selection = ComboBox(choices=lambda _: self._segmentation_layer_choices())
        self.segmentation_selection.native.setToolTip(get_tooltip("classification", "segmentation"))
        self._select_default_segmentation_layer()
        self.segmentation_selection.changed.connect(self._invalidate_features)
        AnnotatorState().segmentation_selection = self.segmentation_selection
        segmentation_selection.addWidget(self.segmentation_selection.native)
        # Keep the choices in sync as layers are added, removed or reordered.
        self._viewer.layers.events.inserted.connect(self._reset_segmentation_layer_choices)
        self._viewer.layers.events.removed.connect(self._reset_segmentation_layer_choices)
        self._viewer.layers.events.reordered.connect(self._reset_segmentation_layer_choices)
        return segmentation_selection

    def _extra_classification_sections(self):
        return [self._create_segmentation_layer_section()]


def object_classifier(
    image: np.ndarray,
    segmentation: np.ndarray,
    embedding_path: Optional[Union[str, util.ImageEmbeddings]] = None,
    model_type: str = DEFAULT_MODEL,
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
    state.ndim = ndim

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
    annotator._update_image()

    # Add the annotator widget to the viewer and sync widgets.
    viewer.window.add_dock_widget(annotator, name="(Object Classifier) Segment Anything for Microscopy")
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
    model_type: str = DEFAULT_MODEL,
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

        # Save the current prediction and RF (with its specs, matching the single-image export).
        _save_prediction(images[next_image_id], viewer.layers["prediction"].data, next_image_id)
        dump(
            {"rf": state.object_rf, "model_spec": state.annotator._classifier_spec(state.object_rf)},
            os.path.join(output_folder, "rf.joblib"),
        )

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
        state.ndim = ndim
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
