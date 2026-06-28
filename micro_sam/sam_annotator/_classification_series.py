"""Shared image-series task for the object and pixel classifiers.

Both classifiers share the per-item series flow: (re)initialize the predictor, build the classifier
widget, accumulate the per-item labeled features into the running training set when leaving an item,
and save the prediction plus the trained classifier. Subclasses bind the concrete classifier widget
class, the cached state-attribute names and any extra per-item layers (the object classifier adds a
segmentation layer).
"""

import os

import numpy as np
import imageio.v3 as imageio
from joblib import dump

from ._series import SeriesAnnotatorTask
from ._state import AnnotatorState
from .util import _sync_embedding_widget


class ClassificationSeriesTask(SeriesAnnotatorTask):
    """Series task base for the classifiers; subclasses bind the concrete widget and state attrs."""

    # Classification accumulates labeled features forward across the series; revisiting an item would
    # double-count its features, so backward navigation is disabled.
    supports_previous = False

    # Bound by subclasses.
    classifier_class = None  # ObjectClassifier | PixelClassifier
    dock_name = "Segment Anything for Microscopy"
    features_attr = None  # "object_features" | "pixel_features"
    aux_attr = None  # "seg_ids" | "pixel_grid_shape"
    rf_attr = None  # "object_rf" | "pixel_rf"

    def __init__(
        self, *, ndim, model_type, embedding_paths=None, tile_shape=None, halo=None,
        checkpoint_path=None, device=None,
    ):
        self.ndim = ndim
        self.model_type = model_type
        self.embedding_paths = embedding_paths
        self.tile_shape = tile_shape
        self.halo = halo
        self.checkpoint_path = checkpoint_path
        self.device = device

    def _set_layers(self, viewer, index):
        """Hook: add or update task-specific layers (e.g. the segmentation layer) for this item."""

    def result_filename(self, entry, index):
        if self.have_inputs_as_arrays:
            return f"prediction_{index:05}.tif"
        return os.path.splitext(os.path.basename(entry))[0] + "_prediction.tif"

    def precompute(self, images):
        # Start the series with a fresh running training set and no cached features/classifier, so a
        # new session does not inherit accumulated state from a previous one (the state is a singleton).
        state = AnnotatorState()
        state.previous_features, state.previous_labels = None, None
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
        state.initialize_predictor(
            image, model_type=self.model_type, save_path=embedding_path, halo=self.halo,
            tile_shape=self.tile_shape, precompute_amg_state=False, ndim=self.ndim,
            checkpoint_path=self.checkpoint_path, device=self.device, skip_load=False, use_cli=True, **kwargs,
        )
        state.image_shape = image.shape if image.ndim == self.ndim else image.shape[:-1]
        state.ndim = self.ndim

    def start(self, viewer, entry, image, embedding_path, index):
        self._init_predictor(viewer, image, embedding_path, reuse=False)
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
            device=self.device, tile_shape=self.tile_shape, halo=self.halo,
        )
        return annotator

    def advance(self, viewer, annotator, entry, image, embedding_path, index):
        viewer.layers["image"].data = image
        self._set_layers(viewer, index)
        self._init_predictor(viewer, image, embedding_path, reuse=True)
        annotator._update_image()
        # Drop the cached features/aux of the previous image.
        state = AnnotatorState()
        setattr(state, self.features_attr, None)
        setattr(state, self.aux_attr, None)

    def has_unsaved_content(self, viewer):
        # Classification always has a prediction to save, so it never prompts on advance.
        return True

    def on_leave_item(self, viewer, entry, index):
        state = AnnotatorState()
        state.annotator.accumulate_series_features()
        if state.previous_features is not None:
            np.save(os.path.join(self.output_folder, "features.npy"), state.previous_features)
            np.save(os.path.join(self.output_folder, "labels.npy"), state.previous_labels)

    def save_item(self, viewer, entry, index):
        state = AnnotatorState()
        save_path = os.path.join(self.output_folder, self.result_filename(entry, index))
        imageio.imwrite(save_path, viewer.layers["prediction"].data, compression="zlib")
        rf = getattr(state, self.rf_attr)
        if rf is not None:
            dump(
                {"rf": rf, "model_spec": state.annotator._classifier_spec(rf)},
                os.path.join(self.output_folder, "rf.joblib"),
            )
