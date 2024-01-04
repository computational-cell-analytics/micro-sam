import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Literal

from magicgui import magic_factory, widgets
from napari.qt.threading import thread_worker
import zarr
from zarr.errors import PathNotFoundError

from micro_sam.sam_annotator._state import AnnotatorState
from micro_sam.util import (
    ImageEmbeddings,
    get_sam_model,
    precompute_image_embeddings,
    models,
    _DEFAULT_MODEL,
    _available_devices,
    get_cache_directory,
)
from . import util as vutil

if TYPE_CHECKING:
    import napari


@magic_factory(
    pbar={'visible': False, 'max': 0, 'value': 0, 'label': 'working...'},
    call_button="Compute image embeddings",
    save_path={"mode": "d"},  # choose a directory
)
def embedding_widget(
    pbar: widgets.ProgressBar,
    image: "napari.layers.Image",
    model: Literal[tuple(models().urls.keys())] = _DEFAULT_MODEL,
    device: Literal[tuple(["auto"] + _available_devices())] = "auto",
    save_path: Optional[Path] = None,  # where embeddings for this image are cached (optional)
    optional_custom_weights: Optional[Path] = None,  # A filepath or URL to custom model weights.
) -> ImageEmbeddings:
    """Image embedding widget."""
    state = AnnotatorState()
    state.reset_state()
    # Get image dimensions
    if image.rgb:
        ndim = image.data.ndim - 1
        state.image_shape = image.data.shape[:-1]
    else:
        ndim = image.data.ndim
        state.image_shape = image.data.shape

    @thread_worker(connect={'started': pbar.show, 'finished': pbar.hide})
    def _compute_image_embedding(state, image_data, save_path, ndim=None,
                                 device="auto", model=_DEFAULT_MODEL,
                                 optional_custom_weights=None,
                                 ):
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
        state.predictor = get_sam_model(device=device, model_type=model, checkpoint_path=optional_custom_weights)
        # Compute the image embeddings
        state.image_embeddings = precompute_image_embeddings(
            predictor=state.predictor,
            input_=image_data,
            save_path=save_path,
            ndim=ndim,
        )
        return state  # returns napari._qt.qthreading.FunctionWorker

    return _compute_image_embedding(
        state, image.data, save_path, ndim=ndim, device=device, model=model,
        optional_custom_weights=optional_custom_weights
    )


@magic_factory(
    call_button="Update settings",
    cache_directory={"mode": "d"},  # choose a directory
)
def settings_widget(
    cache_directory: Optional[Path] = get_cache_directory(),
):
    """Update micro-sam settings."""
    os.environ["MICROSAM_CACHEDIR"] = str(cache_directory)
    print(f"micro-sam cache directory set to: {cache_directory}")


# TODO: don't pass the viewer but just the appropriate layers
# TODO: fail more gracefully if image embeddings have not been initialized
@magic_factory(call_button="Segment Object [S]")
def segment_widget(v: "napari.viewer.Viewer", box_extension: float = 0.1) -> None:
    shape = v.layers["current_object"].data.shape

    # get the current box and point prompts
    boxes, masks = vutil.shape_layer_to_prompts(v.layers["prompts"], shape)
    points, labels = vutil.point_layer_to_prompts(v.layers["point_prompts"], with_stop_annotation=False)

    predictor = AnnotatorState().predictor
    image_embeddings = AnnotatorState().image_embeddings
    if image_embeddings["original_size"] is None:  # tiled prediction
        seg = vutil.prompt_segmentation(
            predictor, points, labels, boxes, masks, shape, image_embeddings=image_embeddings,
            multiple_box_prompts=True, box_extension=box_extension,
        )
    else:  # normal prediction and we have set the precomputed embeddings already
        seg = vutil.prompt_segmentation(
            predictor, points, labels, boxes, masks, shape, multiple_box_prompts=True, box_extension=box_extension,
        )

    # no prompts were given or prompts were invalid, skip segmentation
    if seg is None:
        print("You either haven't provided any prompts or invalid prompts. The segmentation will be skipped.")
        return

    v.layers["current_object"].data = seg
    v.layers["current_object"].refresh()
