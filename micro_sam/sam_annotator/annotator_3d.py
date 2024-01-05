from typing import Optional, Tuple

import napari
import numpy as np

from segment_anything import SamPredictor

from ._annotator import _AnnotatorBase
from ._state import AnnotatorState
from ._widgets import segment_slice_widget
from .. import util


# # TODO: I don't really understand the reason behind this,
# # we need napari anyways, why don't we just import it?
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     import napari

class Annotator3d(_AnnotatorBase):
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        segmentation_result: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(
            viewer=viewer,
            ndim=3,
            segment_widget=segment_slice_widget,
            segmentation_result=segmentation_result
        )
        # TODO do extra stuff for 3d annotator


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
    # precompute_amg_state: bool = False,
) -> Optional["napari.viewer.Viewer"]:
    """TODO update description."""
    if viewer is None:
        viewer = napari.Viewer()

    viewer.add_image(image, name="image")
    annotator = Annotator3d(viewer, segmentation_result=segmentation_result)

    # TODO AMG State
    # Initialize the predictor state.
    state = AnnotatorState()
    if predictor is None:
        state.predictor = util.get_sam_model(model_type=model_type)
    state.image_embeddings = util.precompute_image_embeddings(
        predictor=state.predictor,
        input_=image,
        save_path=embedding_path,
        tile_shape=tile_shape,
        halo=halo,
        ndim=3,
    )
    state.image_shape = image.shape[:-1] if image.ndim == 4 else image.shape

    # Trigger layer update of the annotator so that layers have the correct shape.
    annotator._update_image()

    # Add the annotator widget to the viewer.
    viewer.window.add_dock_widget(annotator)

    if return_viewer:
        return viewer

    napari.run()
