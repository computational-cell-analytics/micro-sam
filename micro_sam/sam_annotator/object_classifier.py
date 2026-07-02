from multiprocessing import cpu_count
from typing import List, Optional, Tuple, Union

import imageio.v3 as imageio
import napari
import numpy as np
import torch

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
from ._batch import run_batch
from ._batch_classification import ClassificationBatchTask
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
    supports_apply_to_volume = False  # object classification always runs over the full image/volume

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
    viewer.window.add_dock_widget(annotator, name="Segment Anything for Microscopy (Object Classification)")
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


class ObjectClassificationBatchTask(ClassificationBatchTask):
    """Batch task for the object classifier: per-item segmentation layer + projected prediction."""

    dock_name = "Segment Anything for Microscopy (Batch Object Classification)"
    classifier_class = ObjectClassifier
    features_attr = "object_features"
    aux_attr = "seg_ids"
    rf_attr = "object_rf"

    def __init__(self, *, segmentations, **kwargs):
        super().__init__(**kwargs)
        self.segmentations = segmentations

    def _set_layers(self, viewer, index):
        # Load (or carry over) the per-item segmentation. When none is provided, start from an empty
        # segmentation the user can fill in-tool (the 'produce' path).
        seg = None if self.segmentations is None else self.segmentations[index]
        if seg is not None and not isinstance(seg, np.ndarray):
            seg = imageio.imread(seg)
        if "segmentation" in viewer.layers:
            if seg is not None:
                viewer.layers["segmentation"].data = seg
        else:
            if seg is None:
                seg = np.zeros(tuple(AnnotatorState().image_shape), dtype="uint32")
            viewer.add_labels(seg, name="segmentation")


def batch_object_classifier(
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
    viewer: Optional["napari.viewer.Viewer"] = None,
    return_viewer: bool = False,
    skip_done: bool = False,
) -> Optional["napari.viewer.Viewer"]:
    """Start the object classifier for a list of images and segmentations.

    This function saves the features and labels for annotated objects across the batch, so a random
    forest can be trained on multiple images, plus the per-image prediction and the trained classifier.

    Args:
        images: The input images.
        segmentations: The input segmentations, one per image.
        output_folder: The folder where predictions, the trained random forest and the accumulated
            features/labels are saved.
        embedding_paths: Filepaths where to save/load the embeddings, one per image.
        model_type: The Segment Anything model to use. For details on the available models check out
            https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#finetuned-models.
        tile_shape: Shape of tiles for tiled embedding prediction.
            If `None` then the whole image is passed to Segment Anything.
        halo: Shape of the overlap between tiles, which is needed to segment objects on tile borders.
        checkpoint_path: Path to a custom checkpoint from which to load the SAM model.
        device: The computational device to use for the SAM model.
            By default, automatically chooses the best available device.
        ndim: The dimensionality of the data. If not given will be derived from the data.
        viewer: The viewer to which the functionality should be added.
        return_viewer: Whether to return the napari viewer instead of starting the event loop.
        skip_done: Whether to skip images whose prediction already exists in `output_folder`.

    Returns:
        The napari viewer, only returned if `return_viewer=True`.
    """
    if segmentations is not None and len(images) != len(segmentations):
        raise ValueError(
            f"Expect the same number of images and segmentations, got {len(images)}, {len(segmentations)}."
        )

    have_inputs_as_arrays = isinstance(images[0], np.ndarray)
    if ndim is None:
        first = images[0] if have_inputs_as_arrays else imageio.imread(images[0])
        ndim = first.ndim - 1 if first.shape[-1] == 3 and first.ndim in (3, 4) else first.ndim

    task = ObjectClassificationBatchTask(
        segmentations=segmentations, ndim=ndim, model_type=model_type, embedding_paths=embedding_paths,
        tile_shape=tile_shape, halo=halo, checkpoint_path=checkpoint_path, device=device,
    )
    return run_batch(
        images, output_folder, task, have_inputs_as_arrays=have_inputs_as_arrays,
        viewer=viewer, return_viewer=return_viewer, skip_done=skip_done,
    )
