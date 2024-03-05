import os
import pickle
import warnings
from glob import glob
from pathlib import Path
from typing import Optional, Tuple

import napari
import numpy as np

from segment_anything import SamPredictor

from ._annotator import _AnnotatorBase
from ._state import AnnotatorState
from ._widgets import segment_slice, segment_object, amg_3d
from .util import _initialize_parser
from .. import util


def _load_amg_state(embedding_path):
    if embedding_path is None or not os.path.exists(embedding_path):
        return {"cache_folder": None}

    cache_folder = os.path.join(embedding_path, "amg_state")
    os.makedirs(cache_folder, exist_ok=True)
    amg_state = {"cache_folder": cache_folder}

    state_paths = glob(os.path.join(cache_folder, "*.pkl"))
    for path in state_paths:
        with open(path, "rb") as f:
            state = pickle.load(f)
        i = int(Path(path).stem.split("-")[-1])
        amg_state[i] = state
    return amg_state


class Annotator3d(_AnnotatorBase):
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        segmentation_result: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(
            viewer=viewer,
            ndim=3,
            segment_widget=segment_slice,
            segment_nd_widget=segment_object,
            autosegment_widget=amg_3d,
            segmentation_result=segmentation_result,
        )

    def _update_image(self):
        super()._update_image()
        # Load the amg state from the embedding path.
        state = AnnotatorState()
        state.amg_state = _load_amg_state(state.embedding_path)


def annotator_3d(
    image: np.ndarray,
    embedding_path: Optional[str] = None,
    segmentation_result: Optional[np.ndarray] = None,
    model_type: str = util._DEFAULT_MODEL,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    return_viewer: bool = False,
    viewer: Optional["napari.viewer.Viewer"] = None,
    predictor: Optional["SamPredictor"] = None,
) -> Optional["napari.viewer.Viewer"]:
    """Start the 3d annotation tool for a given image volume.

    Args:
        image: The volumetric image data.
        embedding_path: Filepath for saving the precomputed embeddings.
        segmentation_result: An initial segmentation to load.
            This can be used to correct segmentations with Segment Anything or to save and load progress.
            The segmentation will be loaded as the 'committed_objects' layer.
        model_type: The Segment Anything model to use. For details on the available models check out
            https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#finetuned-models.
        tile_shape: Shape of tiles for tiled embedding prediction.
            If `None` then the whole image is passed to Segment Anything.
        halo: Shape of the overlap between tiles, which is needed to segment objects on tile boarders.
        return_viewer: Whether to return the napari viewer to further modify it before starting the tool.
        viewer: The viewer to which the SegmentAnything functionality should be added.
            This enables using a pre-initialized viewer.
        predictor: The Segment Anything model. Passing this enables using fully custom models.
            If you pass `predictor` then `model_type` will be ignored.

    Returns:
        The napari viewer, only returned if `return_viewer=True`.
    """

    # Initialize the predictor state.
    state = AnnotatorState()
    state.initialize_predictor(
        image, model_type=model_type, save_path=embedding_path, predictor=predictor,
        halo=halo, tile_shape=tile_shape
    )
    state.image_shape = image.shape[:-1] if image.ndim == 4 else image.shape

    if viewer is None:
        viewer = napari.Viewer()

    viewer.add_image(image, name="image")
    annotator = Annotator3d(viewer, segmentation_result=segmentation_result)

    # Trigger layer update of the annotator so that layers have the correct shape.
    annotator._update_image()

    # Add the annotator widget to the viewer.
    viewer.window.add_dock_widget(annotator)

    if return_viewer:
        return viewer

    napari.run()


def main():
    """@private"""
    parser = _initialize_parser(description="Run interactive segmentation for an image volume.")
    args = parser.parse_args()
    image = util.load_image_data(args.input, key=args.key)

    if args.segmentation_result is None:
        segmentation_result = None
    else:
        segmentation_result = util.load_image_data(args.segmentation_result, key=args.segmentation_key)

    if args.embedding_path is None:
        warnings.warn("You have not passed an embedding_path. Restarting the annotator may take a long time.")

    annotator_3d(
        image, embedding_path=args.embedding_path,
        segmentation_result=segmentation_result,
        model_type=args.model_type, tile_shape=args.tile_shape, halo=args.halo,
    )
