import numpy as np

from magicgui.widgets import Container

from ._widgets import embedding_widget, segment_widget
from ._state import AnnotatorState
from . import util as vutil

# TODO: I don't really understand the reason behind this,
# we need napari anyways, why don't we just import it?
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import napari


class _AnnotatorBase(Container):
    """Base class for micro_sam annotation plugins.

    Implements all common base functionality.
    """

    def _create_common_layers(self):
        labels = ["positive", "negative"]
        prompts = self._viewer.add_points(
            data=[[0.0, 0.0], [0.0, 0.0]],  # FIXME workaround
            name="point_prompts",
            properties={"label": labels},
            edge_color="label",
            edge_color_cycle=vutil.LABEL_COLOR_CYCLE,
            symbol="o",
            face_color="transparent",
            edge_width=0.5,
            size=12,
            ndim=2,
        )
        prompts.edge_color_mode = "cycle"
        self._viewer.add_shapes(
            face_color="transparent", edge_color="green", edge_width=4, name="prompts"
        )

        self._viewer.add_labels(data=np.zeros(self._shape, dtype="uint32"), name="current_object")
        return prompts, labels

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        #
        # Adding the layers for prompts and segmentations.
        #

        # We initialize all layers with a dummy shape.
        # This is updated whenever the image embeddings are recalculated,
        # so tht the layers keep matching the actual image shape
        self._shape = (256, 256)
        prompts, labels = self._create_common_layers()

        #
        # Add widgets
        #

        self._embedding_widget = embedding_widget()
        # Connect the call button of the embedding widget with a function
        # that updates all relevant layers when the image changes.
        self._embedding_widget.call_button.changed.connect(self._update_image)

        # TODO: this should probably go into widgets
        self._prompt_widget = vutil.create_prompt_menu(prompts, labels)
        self._segment_widget = segment_widget()

        # Add the widgets to the container.
        self.extend(
            [
                self._embedding_widget,
                self._prompt_widget,
                self._segment_widget,
            ]
        )

        #
        # Clean up
        #
        vutil.clear_annotations(self._viewer, clear_segmentations=False)

    def _update_image(self):
        state = AnnotatorState()

        # NOTE: maybe this is not quite necessary and we just need to set the new shape and then call clear
        # Update layers.
        if state.image_shape != self._shape:
            self._shape = state.image_shape
            self._viewer.layers["current_object"].data = np.zeros(self._shape, dtype="uint32")

        # TODO we also want to clear stuff
        # vutil.clear_annotations(self._viewer, clear_segmentations=True)
