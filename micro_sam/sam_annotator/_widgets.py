from enum import Enum
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from magicgui import magic_factory, widgets
from napari.qt.threading import thread_worker
import zarr
from zarr.errors import PathNotFoundError

from micro_sam.sam_annotator._state import AnnotatorState
from micro_sam.util import (
    ImageEmbeddings,
    get_sam_model,
    precompute_image_embeddings,
    _MODEL_URLS,
    _DEFAULT_MODEL,
    _available_devices,
    _compute_data_signature,
)

if TYPE_CHECKING:
    import napari

Model = Enum("Model", _MODEL_URLS)
available_devices_list = ["auto"] + _available_devices()


@magic_factory(
    pbar={'visible': False, 'max': 0, 'value': 0, 'label': 'working...'},
    call_button="Compute image embeddings",
    device = {"choices": available_devices_list},
    save_path={"mode": "d"},  # choose a directory
    )
def embedding_widget(
    pbar: widgets.ProgressBar,
    image: "napari.layers.Image",
    model: Model = Model.__getitem__(_DEFAULT_MODEL),
    device = "auto",
    save_path: Optional[Path] = None,  # where embeddings for this image are cached (optional)
    optional_custom_weights: Optional[Path] = None,  # A filepath or URL to custom model weights.
) -> ImageEmbeddings:
    """Image embedding widget."""
    state = AnnotatorState()
    state.reset_state()
    # Get image dimensions
    ndim = image.data.ndim
    if image.rgb:
        ndim -= 1

    @thread_worker(connect={'started': pbar.show, 'finished': pbar.hide})
    def _compute_image_embedding(state, image_data, save_path, ndim=None,
                                 device="auto", model=Model.__getitem__(_DEFAULT_MODEL),
                                 optional_custom_weights=None):
        # Make sure save directory exists and is an empty directory
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            if not save_path.is_dir():
                raise NotADirectoryError(
                    f"The user selected 'save_path' is not a direcotry: {save_path}"
                )
            if len(os.listdir(save_path)) > 0:
                try:
                    zarr.open(save_path, "r")
                except PathNotFoundError:
                    raise RuntimeError(
                        "The user selected 'save_path' is not a zarr array "
                        f"or empty directory: {save_path}"
                    )
        # Initialize the model
        state.predictor  = get_sam_model(device=device, model_type=model.name,
                                         checkpoint_path=optional_custom_weights)
        # Compute the image embeddings
        state.image_embeddings = precompute_image_embeddings(
            predictor = state.predictor,
            input_ = image_data,
            save_path = str(save_path),
            ndim=ndim,
        )
        data_signature = _compute_data_signature(image_data)
        state.data_signature = data_signature
        state.image_shape = image_data.shape
        return state  # returns napari._qt.qthreading.FunctionWorker

    return _compute_image_embedding(state, image.data, save_path, ndim=ndim, device=device, model=model, optional_custom_weights=optional_custom_weights)
