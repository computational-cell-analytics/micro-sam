from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from magicgui import magic_factory
from magicgui.tqdm import tqdm
from napari.qt.threading import thread_worker

from micro_sam.util import (
    ImageEmbeddings,
    get_sam_model,
    precompute_image_embeddings,
    _MODEL_URLS,
    _DEFAULT_MODEL,
    _available_devices,
)

if TYPE_CHECKING:
    import napari

Model = Enum("Model", _MODEL_URLS)
available_devices_list = ["auto"] + _available_devices()


@magic_factory(call_button="Compute image embeddings",
               device = {"choices": available_devices_list},
               save_path={"mode": "d"},  # choose a directory
               )
def embedding_widget(
    image: "napari.layers.Image",
    model: Model = Model.__getitem__(_DEFAULT_MODEL),
    device = "auto",
    save_path: Optional[Path] = None,  # where embeddings for this image are cached (optional)
    optional_custom_weights: Optional[Path] = None,  # A filepath or URL to custom model weights.
) -> ImageEmbeddings:
    """Image embedding widget."""
    # for access to the predictor and the image embeddings in the widgets
    global PREDICTOR
    # Initialize the model
    PREDICTOR = get_sam_model(device=device, model_type=model.name,
                              checkpoint_path=optional_custom_weights)

    ndim = image.data.ndim  # Get image dimensions

    # Compute the embeddings for the image data
    with tqdm() as pbar:
        @thread_worker(connect={"finished": lambda: pbar._get_progressbar().hide()})
        def _compute_image_embedding(PREDICTOR, image_data, save_path, ndim=None):
            if save_path is not None:
                save_path = str(save_path)
            global IMAGE_EMBEDDINGS
            IMAGE_EMBEDDINGS = precompute_image_embeddings(
                predictor = PREDICTOR,
                input_ = image_data,
                save_path = save_path,
                ndim=ndim,
            )

        worker = _compute_image_embedding(PREDICTOR, image.data, save_path, ndim=ndim)

    return worker
