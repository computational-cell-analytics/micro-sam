from enum import Enum
from pathlib import Path

from magicgui import magicgui
from napari.types import ImageData
from superqt.utils import thread_worker

from magicgui import magicgui
from magicgui.tqdm import tqdm

from micro_sam.util import (
    ImageEmbeddings,
    get_sam_model,
    precompute_image_embeddings,
    _MODEL_URLS,
    _DEFAULT_MODEL,
)


Model = Enum("Model", _MODEL_URLS)


@magicgui(call_button="Compute image embedding")
def embedding_widget(
        image: ImageData,
        model: Model = Model.__getitem__(_DEFAULT_MODEL),
        save_path: Path | str | None = None,
    ) -> ImageEmbeddings:
    """Image embedding widget."""
    PREDICTOR = get_sam_model(model_type=model.name)
    with tqdm() as pbar:
        @thread_worker(connect={"finished": lambda: pbar.progressbar.hide()})
        def _compute_image_embedding(image, model, save_path):
            image_embeddings = precompute_image_embeddings(
                predictor = PREDICTOR,
                input_ = image,
                save_path = str(save_path),
            )
            return image_embeddings

        print("Computing image embedding...")
        image_embeddings = _compute_image_embedding(image, model, save_path)
        print("Finished image embedding computation.")

    return image_embeddings
