from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from magicgui import magic_factory
from magicgui.tqdm import tqdm
from superqt.utils import thread_worker

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
               device = {"choices": available_devices_list})
def embedding_widget(
    image: "napari.layers.Image",
    model: Model = Model.__getitem__(_DEFAULT_MODEL),
    device = "auto",
    save_path: Optional[Path] = None,  # where embeddings for this image are cached (optional)
    custom_model: Optional[str] = None,  # A filepath or URL to custom model weights.
) -> ImageEmbeddings:
    """Image embedding widget."""
    # for access to the predictor and the image embeddings in the widgets
    global PREDICTOR, IMAGE_EMBEDDINGS
    # Initialize the model
    PREDICTOR = get_sam_model(model_type=model.name)

    # Get image dimensions
    if not image.rgb:
        ndim = image.data.ndim
    else:
        # assumes RGB channels are the last dimension
        ndim = image.data.ndim

    # Compute the embeddings for the image data
    with tqdm() as pbar:
        @thread_worker(connect={"finished": lambda: pbar.progressbar.hide()})
        def _compute_image_embedding(PREDICTOR, image, save_path, ndim=None):
            IMAGE_EMBEDDINGS = precompute_image_embeddings(
                predictor = PREDICTOR,
                input_ = image,
                save_path = str(save_path),
                ndim=ndim,
            )

        _compute_image_embedding(PREDICTOR, image.data, save_path, ndim=ndim)
