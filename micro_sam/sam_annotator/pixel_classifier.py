from typing import List, Optional, Tuple, Union

import napari
import numpy as np
import imageio.v3 as imageio
import torch

from .. import util
from ..v2.util import DEFAULT_MODEL
from ..pixel_classification import (
    accumulate_pixel_labels, compute_pixel_features, project_prediction_to_image, train_pixel_classifier,
)
from ._annotator import _ClassifierBase
from ._batch import run_batch
from ._batch_classification import ClassificationBatchTask
from ._state import AnnotatorState
from . import _widgets as widgets
from .util import _sync_embedding_widget

# The SAM and SAM2 image encoders project every model size down to a fixed 256-channel image
# embedding (prompt_embed_dim), so the per-pixel feature dimension is always 256.
EMBEDDING_CHANNELS = 256


class PixelClassifier(_ClassifierBase):
    """GUI for the pixel classifier: trains a random forest on per-pixel SAM/SAM2 embedding features."""

    rf_attr = "pixel_rf"
    features_attr = "pixel_features"
    aux_attr = "pixel_grid_shape"
    label_widget_title = "Pixel label names:"
    max_components = EMBEDDING_CHANNELS
    tool_key = "pixel"

    def _compute_features(self):
        # Returns (features, grid_shape) for the current image, computing and caching them if needed.
        state = AnnotatorState()
        if state.pixel_features is None:
            if widgets._validate_embeddings(self._viewer):
                return None, None
            image, upsampler, ok = self._resolve_anyup()
            if not ok:
                return None, None
            features, grid_shape = compute_pixel_features(
                state.image_embeddings, state.image_shape, image=image, upsampler=upsampler,
            )
            state.pixel_features, state.pixel_grid_shape = features, grid_shape
        return state.pixel_features, state.pixel_grid_shape

    def _compute_training_labels(self, aux):
        return accumulate_pixel_labels(self._viewer.layers["annotations"].data, aux)

    def _train(self, features, labels, previous_features, previous_labels, n_components, random_state):
        return train_pixel_classifier(
            features, labels, previous_features=previous_features, previous_labels=previous_labels,
            n_components=n_components, random_state=random_state,
        )

    def _project_prediction(self, prediction, aux):
        return project_prediction_to_image(prediction, aux, AnnotatorState().image_shape)


def pixel_classifier(
    image: np.ndarray,
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
    state.ndim = ndim

    state.initialize_predictor(
        image, model_type=model_type, save_path=embedding_path,
        halo=halo, tile_shape=tile_shape, precompute_autoseg_state=False,
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
    viewer.window.add_dock_widget(annotator, name="Segment Anything for Microscopy (Pixel Classification)")
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


class PixelClassificationBatchTask(ClassificationBatchTask):
    """Batch task for the pixel classifier."""

    dock_name = "Segment Anything for Microscopy (Batch Pixel Classification)"
    classifier_class = PixelClassifier
    features_attr = "pixel_features"
    aux_attr = "pixel_grid_shape"
    rf_attr = "pixel_rf"


def batch_pixel_classifier(
    images: List[np.ndarray],
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
    batch_size: int = 1,
) -> Optional["napari.viewer.Viewer"]:
    """Start the pixel classifier for a list of images.

    This function saves the per-pixel features and labels across the batch, so a random forest can be
    trained on multiple images, plus the per-image prediction and the trained classifier.

    Args:
        images: The input images.
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
        batch_size: The number of tiles / slices per model call when computing the embeddings.
            Only has an effect on a GPU. By default a single tile / slice is used.
        skip_done: Whether to skip images whose prediction already exists in `output_folder`.

    Returns:
        The napari viewer, only returned if `return_viewer=True`.
    """
    have_inputs_as_arrays = isinstance(images[0], np.ndarray)
    if ndim is None:
        first = images[0] if have_inputs_as_arrays else imageio.imread(images[0])
        ndim = first.ndim - 1 if first.shape[-1] == 3 and first.ndim in (3, 4) else first.ndim

    task = PixelClassificationBatchTask(
        ndim=ndim, model_type=model_type, embedding_paths=embedding_paths,
        tile_shape=tile_shape, halo=halo, checkpoint_path=checkpoint_path, device=device,
        batch_size=batch_size,
    )
    return run_batch(
        images, output_folder, task, have_inputs_as_arrays=have_inputs_as_arrays,
        viewer=viewer, return_viewer=return_viewer, skip_done=skip_done,
    )
