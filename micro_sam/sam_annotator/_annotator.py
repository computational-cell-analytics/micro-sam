import numpy as np

from magicgui.widgets import Container, Widget

from . import _widgets as widgets
from . import util as vutil
from ._state import AnnotatorState

from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    import napari


class _AnnotatorBase(Container):
    """Base class for micro_sam annotation plugins.

    Implements the logic for the 2d, 3d and tracking annotator.
    The annotators differ in their data dimensionality and the widgets.
    """

    def _create_layers(self, segmentation_result):
        # Add the point layer for point prompts.
        self._point_labels = ["positive", "negative"]
        self._point_prompt_layer = self._viewer.add_points(
            name="point_prompts",
            property_choices={"label": self._point_labels},
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
            face_color="transparent", edge_color="green", edge_width=4, name="prompts", ndim=self._ndim,
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

    def _create_widgets(self, segment_widget, segment_nd_widget, autosegment_widget, commit_widget, clear_widget):
        self._embedding_widget = widgets.embedding()
        # Connect the call button of the embedding widget with a function
        # that updates all relevant layers when the image changes.
        self._embedding_widget.call_button.changed.connect(self._update_image)

        self._prompt_widget = widgets.create_prompt_menu(self._point_prompt_layer, self._point_labels)
        self._segment_widget = segment_widget()
        widget_list = [self._embedding_widget, self._prompt_widget, self._segment_widget]

        if segment_nd_widget is not None:
            self._segment_nd_widget = segment_nd_widget()
            widget_list.append(self._segment_nd_widget)

        if autosegment_widget is not None:
            self._autosegment_widget = autosegment_widget()
            widget_list.append(self._autosegment_widget)

        self._commit_widget = commit_widget()
        self._clear_widget = clear_widget()
        widget_list.extend([self._commit_widget, self._clear_widget])

        # Add the widgets to the container.
        self.extend(widget_list)

    def _create_keybindings(self):
        @self._viewer.bind_key("s", overwrite=True)
        def _segment(viewer):
            self._segment_widget(viewer)

        @self._viewer.bind_key("c", overwrite=True)
        def _commit(viewer):
            self._commit_widget(viewer)

        @self._viewer.bind_key("t", overwrite=True)
        def _toggle_label(event=None):
            vutil.toggle_label(self._point_prompt_layer)

        @self._viewer.bind_key("Shift-C", overwrite=True)
        def _clear_annotations(viewer):
            self._clear_widget(viewer)

        if hasattr(self, "_segment_nd_widget"):
            @self._viewer.bind_key("Shift-S", overwrite=True)
            def _seg_nd(viewer):
                self._segment_nd_widget(viewer)

    # TODO
    # We could implement a better way of initializing the segmentation result,
    # so that instead of just passing a numpy array an existing layer from the napari
    # viewer can be chosen.
    # See https://github.com/computational-cell-analytics/micro-sam/issues/335
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        ndim: int,
        segment_widget: Widget,
        segment_nd_widget: Optional[Widget] = None,
        autosegment_widget: Optional[Widget] = None,
        commit_widget: Widget = widgets.commit,
        clear_widget: Widget = widgets.clear,
        segmentation_result: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            viewer:
            ndim:
            segment_widget:
            segment_nd_widget:
            autosegment_widget:
            commit_widget:
            clear_widget:
            segmentation_result:
        """
        super().__init__()
        self._viewer = viewer

        # Add the layers for prompts and segmented obejcts.
        # We initialize these with a dummy shape, which is reset to the
        # correct shape once an image is set.
        self._ndim = ndim
        self._shape = (256, 256) if ndim == 2 else (16, 256, 256)
        self._create_layers(segmentation_result)

        # Add the widgets in common between all annotators.
        self._create_widgets(
            segment_widget, segment_nd_widget, autosegment_widget, commit_widget, clear_widget,
        )

        # Add the key bindings in common between all annotators.
        self._create_keybindings()

    def _update_image(self):
        state = AnnotatorState()

        # Update the image shape if it has changed.
        if state.image_shape != self._shape:
            if len(state.image_shape) != self._ndim:
                raise RuntimeError(
                    f"The dim of the annotator {self._ndim} does not match the image data of shape {state.image_shape}."
                )
            self._shape = state.image_shape

        # Reset all layers.
        self._viewer.layers["current_object"].data = np.zeros(self._shape, dtype="uint32")
        self._viewer.layers["committed_objects"].data = np.zeros(self._shape, dtype="uint32")
        self._viewer.layers["auto_segmentation"].data = np.zeros(self._shape, dtype="uint32")

        vutil.clear_annotations(self._viewer, clear_segmentations=False)
