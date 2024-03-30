import napari
import numpy as np

from magicgui.widgets import Widget, Container, FunctionGui
from qtpy import QtWidgets

from . import _widgets as widgets
from . import util as vutil
from ._state import AnnotatorState


class _AnnotatorBase(QtWidgets.QWidget):
    """Base class for micro_sam annotation plugins.

    Implements the logic for the 2d, 3d and tracking annotator.
    The annotators differ in their data dimensionality and the widgets.
    """
    def _create_layers(self):
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
        self._viewer.add_labels(data=dummy_data, name="committed_objects")
        # Randomize colors so it is easy to see when object committed.
        self._viewer.layers["committed_objects"].new_colormap()

    # Child classes have to implement this function and create a dictionary with the widgets.
    def _get_widgets(self):
        raise NotImplementedError("The child classes of _AnnotatorBase have to implement _get_widgets.")

    def _create_widgets(self):
        # Create the embedding widget and connect all events related to it.
        self._embedding_widget = widgets.EmbeddingWidget()
        # Connect events for the image selection box.
        self._viewer.layers.events.inserted.connect(self._embedding_widget.image_selection.reset_choices)
        self._viewer.layers.events.removed.connect(self._embedding_widget.image_selection.reset_choices)
        # Connect the run button with the fundtion to update the image.
        self._embedding_widget.run_button.clicked.connect(self._update_image)

        # Create the prompt widget. (The same for all plugins.)
        self._prompt_widget = widgets.create_prompt_menu(self._point_prompt_layer, self._point_labels)

        # Create the dictionray for the widgets and get the widgets of the child plugin.
        self._widgets = {
            "embeddings": self._embedding_widget,
            "prompts": self._prompt_widget,
        }
        self._widgets.update(self._get_widgets())

    def _create_keybindings(self):
        @self._viewer.bind_key("s", overwrite=True)
        def _segment(viewer):
            self._widgets["segment"](viewer)

        @self._viewer.bind_key("c", overwrite=True)
        def _commit(viewer):
            self._widgets["commit"](viewer)

        @self._viewer.bind_key("t", overwrite=True)
        def _toggle_label(event=None):
            vutil.toggle_label(self._point_prompt_layer)

        @self._viewer.bind_key("Shift-C", overwrite=True)
        def _clear_annotations(viewer):
            self._widgets["clear"](viewer)

        if "segment_nd" in self._widgets:
            @self._viewer.bind_key("Shift-S", overwrite=True)
            def _seg_nd(viewer):
                self._widgets["segment_nd"](viewer)

    # TODO
    # We could implement a better way of initializing the segmentation result,
    # so that instead of just passing a numpy array an existing layer from the napari
    # viewer can be chosen.
    # See https://github.com/computational-cell-analytics/micro-sam/issues/335
    def __init__(self, viewer: "napari.viewer.Viewer", ndim: int) -> None:
        """Create the annotator GUI.

        Args:
            viewer: The napari viewer.
            ndim: The number of spatial dimension of the image data (2 or 3).
        """
        super().__init__()
        self._viewer = viewer

        # Add the layers for prompts and segmented obejcts.
        # Initialize with a dummy shape, which is reset to the correct shape once an image is set.
        self._ndim = ndim
        self._shape = (256, 256) if ndim == 2 else (16, 256, 256)
        self._create_layers()

        # Create all the widgets and add them to the layout.
        self._create_widgets()
        layout = QtWidgets.QVBoxLayout()
        for widget in self._widgets.values():
            # Add the widget to the layout.
            if isinstance(widget, (Container, FunctionGui, Widget)):
                # This is a magicgui type and we need to get the native qt widget.
                layout.addWidget(widget.native)
            else:
                # This is a qt type and we add the widget directly.
                layout.addWidget(widget)
        self.setLayout(layout)

        # Add the widgets to the state.
        AnnotatorState().widgets = self._widgets

        # Add the key bindings in common between all annotators.
        self._create_keybindings()

    def _update_image(self, segmentation_result=None):
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
        self._viewer.layers["auto_segmentation"].data = np.zeros(self._shape, dtype="uint32")
        if segmentation_result is None or segmentation_result is False:
            self._viewer.layers["committed_objects"].data = np.zeros(self._shape, dtype="uint32")
        else:
            assert segmentation_result.shape == self._shape
            self._viewer.layers["committed_objects"].data = segmentation_result

        vutil.clear_annotations(self._viewer, clear_segmentations=False)
