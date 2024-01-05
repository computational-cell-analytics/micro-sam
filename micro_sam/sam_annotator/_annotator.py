import numpy as np

from magicgui.widgets import Container

from . import _widgets as widgets
from . import util as vutil
from ._state import AnnotatorState

# TODO: I don't really understand the reason behind this,
# we need napari anyways, why don't we just import it?
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    import napari


class _AnnotatorBase(Container):
    """Base class for micro_sam annotation plugins.

    Implements all common base functionality.
    """

    def _create_common_layers(self, segmentation_result):
        # Add the point layer for point prompts.
        self._point_labels = ["positive", "negative"]
        # We need to add dummy data to initialize the point properties correctly.
        # The two points will be cleared at the end.
        dummy_data = 2 * [[0.0] * self._ndim]
        self._point_prompt_layer = self._viewer.add_points(
            data=dummy_data,
            name="point_prompts",
            properties={"label": self._point_labels},
            edge_color="label",
            edge_color_cycle=vutil.LABEL_COLOR_CYCLE,
            symbol="o",
            face_color="transparent",
            edge_width=0.5,
            size=12,
            ndim=self._ndim,
        )
        self._point_prompt_layer.edge_color_mode = "cycle"

        # Add the shape layer for box and other shape prompts.
        self._viewer.add_shapes(
            face_color="transparent", edge_color="green", edge_width=4, name="prompts"
        )

        # Add the label layers for the current object, the automatic segmentation and the committed segmentation.
        dummy_data = np.zeros(self._shape, dtype="uint32")
        self._viewer.add_labels(data=dummy_data, name="current_object")
        self._viewer.add_labels(data=dummy_data, name="auto_segmentation")
        self._viewer.add_labels(
            data=dummy_data if segmentation_result is None else segmentation_result, name="committed_objects"
        )
        # Randomize colors so it is easy to see when object committed.
        self._viewer.layers["committed_objects"].new_colormap()

    def _create_common_widgets(self, segment_widget):
        self._embedding_widget = widgets.embedding_widget()
        # Connect the call button of the embedding widget with a function
        # that updates all relevant layers when the image changes.
        self._embedding_widget.call_button.changed.connect(self._update_image)

        self._prompt_widget = widgets.create_prompt_menu(self._point_prompt_layer, self._point_labels)
        self._segment_widget = segment_widget()
        self._commit_segmentation_widget = widgets.commit_segmentation_widget()

        # TODO autosegment widget

        # Add the widgets to the container.
        self.extend(
            [
                self._embedding_widget,
                self._prompt_widget,
                self._segment_widget,
                self._commit_segmentation_widget,
            ]
        )

    def _create_common_keybindings(self):
        @self._viewer.bind_key("s")
        def _segmet(viewer):
            self._segment_widget(viewer)

        @self._viewer.bind_key("c")
        def _commit(viewer):
            self._commit_segmentation_widget(viewer)

        @self._viewer.bind_key("t")
        def _toggle_label(event=None):
            vutil.toggle_label(self._point_prompt_layer)

        @self._viewer.bind_key("Shift-C")
        def _clear_annotations(viewer):
            vutil.clear_annotations(viewer)

    # TODO more clever way to integrate segmentation result so
    # that we can also choose an active layer?
    # We could allow also passing a label layer and then use it.
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        ndim: int,
        segment_widget,  # TODO what is the type annotation?
        segmentation_result: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        self._viewer = viewer

        # Add the layers for prompts and segmented obejcts.
        # We initialize these with a dummy shape, which is reset to the
        # correct shape once an image is set.
        self._ndim = ndim
        self._shape = (256, 256) if ndim == 2 else (16, 256, 256)
        self._create_common_layers(segmentation_result)

        # Add the widgets in common between all annotators.
        self._create_common_widgets(segment_widget)

        # Add the key bindings in common between all annotators.
        self._create_common_keybindings()

        # Clear the annotations to remove dummy prompts necessary for init.
        vutil.clear_annotations(self._viewer, clear_segmentations=False)

    def _update_image(self):
        state = AnnotatorState()

        if state.image_shape != self._shape:
            if len(state.image_shape) != self._ndim:
                # TODO good error message that makes clear that dims don't match
                raise RuntimeError("")
            self._shape = state.image_shape

            self._viewer.layers["current_object"].data = np.zeros(self._shape, dtype="uint32")
            self._viewer.layers["committed_objects"].data = np.zeros(self._shape, dtype="uint32")
            self._viewer.layers["auto_segmentation"].data = np.zeros(self._shape, dtype="uint32")

        vutil.clear_annotations(self._viewer, clear_segmentations=True)
