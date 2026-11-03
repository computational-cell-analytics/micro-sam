"""Shared batch-annotation task for the object and pixel classifiers.

Both classifiers share the per-item batch flow: (re)initialize the predictor, build the classifier
widget, store the per-item annotations and labeled features when leaving an item, restore them when
going back to it, and save the prediction plus the trained classifier. Subclasses bind the concrete
classifier widget class, the cached state-attribute names and any extra per-item layers (the object
classifier adds a segmentation layer).
"""

import os
from joblib import dump

import numpy as np
import imageio.v3 as imageio

from magicgui.widgets import CheckBox

from ._state import AnnotatorState
from ._tooltips import get_tooltip
from ._titles import PLUGIN_NAME
from ._batch import BatchAnnotatorTask
from ..v2.util import resolve_default_tiling
from .util import _sync_embedding_widget, _sync_tiling_widget


class ClassificationBatchTask(BatchAnnotatorTask):
    """Batch task base for the classifiers. Subclasses bind the concrete widget and state attributes."""

    # Bound by subclasses.
    classifier_class = None  # ObjectClassifier | PixelClassifier
    dock_name = PLUGIN_NAME
    features_attr = None  # "object_features" | "pixel_features"
    aux_attr = None  # "seg_ids" | "pixel_grid_shape"
    rf_attr = None  # "object_rf" | "pixel_rf"
    supports_previous = True

    def __init__(
        self, *, ndim, model_type, embedding_paths=None, tile_shape=None, halo=None,
        checkpoint_path=None, device=None, batch_size=1,
    ):
        self.ndim = ndim
        self.model_type = model_type
        self.embedding_paths = embedding_paths
        self.tile_shape = tile_shape
        self.halo = halo
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.batch_size = batch_size

    def _set_layers(self, viewer, index):
        """Hook: add or update task-specific layers (e.g. the segmentation layer) for this item."""

    def _item_filename(self, entry, index, suffix):
        if self.have_inputs_as_arrays:
            return f"{suffix}_{index:05}.tif"
        return os.path.splitext(os.path.basename(entry))[0] + f"_{suffix}.tif"

    def result_filename(self, entry, index):
        return self._item_filename(entry, index, "prediction")

    def precompute(self, images):
        # Start the batch with an empty training set and no cached features/classifier, so a new
        # session does not inherit the state of a previous one (the state is a singleton).
        state = AnnotatorState()
        state.batch_training_data, state.batch_index = {}, None
        self.annotations = {}
        setattr(state, self.features_attr, None)
        setattr(state, self.aux_attr, None)
        setattr(state, self.rf_attr, None)
        if self.embedding_paths is None:
            return [None] * len(images)
        return list(self.embedding_paths)

    def _init_predictor(self, viewer, image, embedding_path, reuse):
        state = AnnotatorState()
        # Reuse the already-loaded model on subsequent items so it is not re-downloaded/rebuilt.
        kwargs = {"predictor": state.predictor} if (reuse and state.predictor is not None) else {}
        state.image_shape = image.shape if image.ndim == self.ndim else image.shape[:-1]
        state.ndim = self.ndim
        # Resolve the tiling per image, since the shapes can differ across the batch.
        tile_shape, halo = resolve_default_tiling(state.image_shape, self.tile_shape, self.halo, embedding_path)
        state.initialize_predictor(
            image, model_type=self.model_type, save_path=embedding_path, halo=halo,
            tile_shape=tile_shape, precompute_autoseg_state=False, ndim=self.ndim,
            checkpoint_path=self.checkpoint_path, device=self.device, batch_size=self.batch_size,
            use_cli=True, **kwargs,
        )
        return tile_shape, halo

    def _restore_annotations(self, viewer, entry, index):
        # Restore the annotations of a revisited item, from this session or from an earlier one.
        annotations = self.annotations.get(index)
        path = os.path.join(self.output_folder, self._item_filename(entry, index, "annotations"))
        if annotations is None and os.path.exists(path):
            annotations = imageio.imread(path)
        layer = viewer.layers["annotations"]
        if annotations is None or annotations.shape != layer.data.shape:
            return False
        layer.data = annotations.copy()
        return bool(annotations.any())

    def start(self, viewer, entry, image, embedding_path, index):
        AnnotatorState().batch_index = index
        tile_shape, halo = self._init_predictor(viewer, image, embedding_path, reuse=False)
        viewer.add_image(image, name="image")
        self._set_layers(viewer, index)

        annotator = self.classifier_class(viewer)
        annotator._update_image()

        state = AnnotatorState()
        viewer.window.add_dock_widget(annotator, name=self.dock_name)
        _sync_embedding_widget(
            widget=state.widgets["embeddings"],
            model_type=self.model_type if self.checkpoint_path is None else state.predictor.model_type,
            save_path=embedding_path, checkpoint_path=self.checkpoint_path,
            device=self.device, tile_shape=tile_shape, halo=halo,
        )
        if self._restore_annotations(viewer, entry, index):
            annotator._run_train_and_predict()
        return annotator

    def nav_extra_widgets(self):
        # A checkbox next to the navigation buttons (classification tasks only), on by default, to carry
        # the classifier state forward across the batch. Tracked in the state so it is reachable in tests.
        self._forward_state = CheckBox(value=True, text="Keep Classifier")
        self._forward_state.native.setToolTip(get_tooltip("classification", "forward_classifier_state"))
        AnnotatorState().widgets["batch_forward_state"] = self._forward_state
        return [self._forward_state]

    def _forward_state_enabled(self):
        widget = getattr(self, "_forward_state", None)
        return True if widget is None else bool(widget.value)

    def advance(self, viewer, annotator, entry, image, embedding_path, index):
        state = AnnotatorState()
        state.batch_index = index
        viewer.layers["image"].data = image
        self._set_layers(viewer, index)
        tile_shape, halo = self._init_predictor(viewer, image, embedding_path, reuse=True)
        _sync_tiling_widget(AnnotatorState().widgets["embeddings"], tile_shape, halo)
        annotator._update_image()
        # Drop the cached features/aux of the previous image.
        setattr(state, self.features_attr, None)
        setattr(state, self.aux_attr, None)
        if not self._forward_state_enabled():
            # Independent per image: drop the training set and the classifier of the other images.
            state.batch_training_data = {}
            setattr(state, self.rf_attr, None)

        # Retrain a fresh random forest on the annotations of this image (if revisited) and, when the
        # classifier is carried forward, of all other images. Then predict right away.
        has_annotations = self._restore_annotations(viewer, entry, index)
        if has_annotations or (self._forward_state_enabled() and self._has_other_training_data(index)):
            annotator._run_train_and_predict()

    def _has_other_training_data(self, index):
        return any(idx != index for idx in AnnotatorState().batch_training_data)

    def has_unsaved_content(self, viewer):
        # Classification always has a prediction to save, so it never prompts on advance.
        return True

    def on_leave_item(self, viewer, entry, index):
        state = AnnotatorState()
        annotator = state.annotator
        annotations = viewer.layers["annotations"].data
        self.annotations[index] = annotations.copy()
        path = os.path.join(self.output_folder, self._item_filename(entry, index, "annotations"))
        imageio.imwrite(path, annotations, compression="zlib")

        # Retrain if the annotations changed since the last run, so the saved classifier and prediction
        # include all annotations.
        forward = self._forward_state_enabled()
        has_data = annotations.any() or (forward and self._has_other_training_data(index))
        if has_data and annotator.is_prediction_stale():
            annotator._run_train_and_predict()

        if forward:
            # Replace this image's entry in the training set, which is forwarded to the other images.
            annotator.store_batch_training_data(index)
            features, labels = annotator.batch_training_set()
            if features is not None:
                np.save(os.path.join(self.output_folder, "features.npy"), features)
                np.save(os.path.join(self.output_folder, "labels.npy"), labels)

    def on_close(self, viewer, entry, index):
        # Save the annotations and the classifier, but not the prediction, so the item is not marked as
        # done and a resumed session starts on it with its annotations restored. A programmatic
        # 'viewer.close()' removes the layers before closing the window, so there is nothing to save then.
        if "annotations" not in viewer.layers:
            return
        self.on_leave_item(viewer, entry, index)
        self._save_rf()

    def save_item(self, viewer, entry, index):
        save_path = os.path.join(self.output_folder, self.result_filename(entry, index))
        imageio.imwrite(save_path, viewer.layers["prediction"].data, compression="zlib")
        self._save_rf()

    def _save_rf(self):
        state = AnnotatorState()
        rf = getattr(state, self.rf_attr)
        if rf is not None:
            dump(
                {"rf": rf, "model_spec": state.annotator._classifier_spec(rf)},
                os.path.join(self.output_folder, "rf.joblib"),
            )
