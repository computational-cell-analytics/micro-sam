from typing import Optional, Tuple

import napari
import numpy as np

from segment_anything import SamPredictor

from ._annotator import _AnnotatorBase
from ._state import AnnotatorState
from ._widgets import segment_widget
from .. import util


class Annotator2d(_AnnotatorBase):
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        segmentation_result: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(
            viewer=viewer,
            ndim=2,
            segment_widget=segment_widget,
            segmentation_result=segmentation_result
        )

    # We need to make sure that the pre-computed embeddings are being
    # correctly set in the case of the 2d annotator.
    # (This is not necessary for the other annotators.)
    def _update_image(self):
        super()._update_image()
        state = AnnotatorState()
        util.set_precomputed(state.predictor, state.image_embeddings)


def annotator_2d(
    image: np.ndarray,
    embedding_path: Optional[str] = None,
    segmentation_result: Optional[np.ndarray] = None,
    model_type: str = util._DEFAULT_MODEL,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    return_viewer: bool = False,
    viewer: Optional["napari.viewer.Viewer"] = None,
    predictor: Optional["SamPredictor"] = None,
    precompute_amg_state: bool = False,
) -> Optional["napari.viewer.Viewer"]:
    """TODO update description."""

    state = AnnotatorState()
    state.initialize_predictor(
        image, model_type=model_type, save_path=embedding_path,
        halo=halo, tile_shape=tile_shape, precompute_amg_state=precompute_amg_state,
    )
    state.image_shape = image.shape[:-1] if image.ndim == 3 else image.shape

    if viewer is None:
        viewer = napari.Viewer()

    viewer.add_image(image, name="image")
    annotator = Annotator2d(viewer, segmentation_result=segmentation_result)

    # Trigger layer update of the annotator so that layers have the correct shape.
    annotator._update_image()

    # Add the annotator widget to the viewer.
    viewer.window.add_dock_widget(annotator)

    if return_viewer:
        return viewer

    napari.run()
