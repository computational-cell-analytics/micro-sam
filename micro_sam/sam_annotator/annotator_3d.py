from typing import Optional, Tuple, Union

import napari
import numpy as np
import torch

from ._annotator import _AnnotatorBase
from ._state import AnnotatorState
from . import _widgets as widgets
from .util import _initialize_parser, _sync_embedding_widget, _load_amg_state, _load_is_state
from .. import util


class Annotator3d(_AnnotatorBase):
    def _get_widgets(self):
        autosegment = widgets.AutoSegmentWidget(self._viewer, with_decoder=self._with_decoder, volumetric=True)
        segment_nd = widgets.SegmentNDWidget(self._viewer, tracking=False)
        return {
            "segment": widgets.segment_slice(),
            "segment_nd": segment_nd,
            "autosegment": autosegment,
            "commit": widgets.commit(),
            "clear": widgets.clear_volume(),
        }

    def __init__(self, viewer: "napari.viewer.Viewer") -> None:
        self._with_decoder = AnnotatorState().decoder is not None
        super().__init__(viewer=viewer, ndim=3)

    def _update_image(self, segmentation_result=None):
        super()._update_image(segmentation_result=segmentation_result)
        # Load the amg state from the embedding path.
        state = AnnotatorState()
        if self._with_decoder:
            state.amg_state = _load_is_state(state.embedding_path)
        else:
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
    precompute_amg_state: bool = False,
    checkpoint_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    prefer_decoder: bool = True,
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
        precompute_amg_state: Whether to precompute the state for automatic mask generation.
            This will take more time when precomputing embeddings, but will then make
            automatic mask generation much faster.
        checkpoint_path: Path to a custom checkpoint from which to load the SAM model.
        device: The computational device to use for the SAM model.
        prefer_decoder: Whether to use decoder based instance segmentation if
            the model used has an additional decoder for instance segmentation.

    Returns:
        The napari viewer, only returned if `return_viewer=True`.
    """

    # Initialize the predictor state.
    state = AnnotatorState()
    state.image_shape = image.shape[:-1] if image.ndim == 4 else image.shape
    state.initialize_predictor(
        image, model_type=model_type, save_path=embedding_path,
        halo=halo, tile_shape=tile_shape, ndim=3, precompute_amg_state=precompute_amg_state,
        checkpoint_path=checkpoint_path, device=device, prefer_decoder=prefer_decoder,
    )

    if viewer is None:
        viewer = napari.Viewer()

    viewer.add_image(image, name="image")
    annotator = Annotator3d(viewer)

    # Trigger layer update of the annotator so that layers have the correct shape.
    # And initialize the 'committed_objects' with the segmentation result if it was given.
    annotator._update_image(segmentation_result=segmentation_result)

    # Add the annotator widget to the viewer and sync widgets.
    viewer.window.add_dock_widget(annotator)
    _sync_embedding_widget(
        state.widgets["embeddings"], model_type,
        save_path=embedding_path, checkpoint_path=checkpoint_path,
        device=device, tile_shape=tile_shape, halo=halo
    )

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

    annotator_3d(
        image, embedding_path=args.embedding_path,
        segmentation_result=segmentation_result,
        model_type=args.model_type, tile_shape=args.tile_shape, halo=args.halo,
        checkpoint_path=args.checkpoint, device=args.device,
        precompute_amg_state=args.precompute_amg_state, prefer_decoder=args.prefer_decoder,
    )
