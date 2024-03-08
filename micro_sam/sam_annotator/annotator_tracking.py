import warnings
from typing import Optional, Tuple

import napari
import numpy as np

from magicgui.widgets import ComboBox, Container
from segment_anything import SamPredictor

from ._annotator import _AnnotatorBase
from ._state import AnnotatorState
from . import util as vutil
from . import _widgets as widgets
from .. import util

# Cyan (track) and Magenta (division)
STATE_COLOR_CYCLE = ["#00FFFF", "#FF00FF", ]
"""@private"""


# This solution is a bit hacky, so I won't move it to _widgets.py yet.
def create_tracking_menu(points_layer, box_layer, states, track_ids):
    """@private"""
    state = AnnotatorState()

    state_menu = ComboBox(label="track_state", choices=states)
    track_id_menu = ComboBox(label="track_id", choices=list(map(str, track_ids)))
    tracking_widget = Container(widgets=[state_menu, track_id_menu])

    def update_state(event):
        new_state = str(points_layer.current_properties["state"][0])
        if new_state != state_menu.value:
            state_menu.value = new_state

    def update_track_id(event):
        new_id = str(points_layer.current_properties["track_id"][0])
        if new_id != track_id_menu.value:
            track_id_menu.value = new_id
            state.current_track_id = int(new_id)

    # def update_state_boxes(event):
    #     new_state = str(box_layer.current_properties["state"][0])
    #     if new_state != state_menu.value:
    #         state_menu.value = new_state

    def update_track_id_boxes(event):
        new_id = str(box_layer.current_properties["track_id"][0])
        if new_id != track_id_menu.value:
            track_id_menu.value = new_id
            state.current_track_id = int(new_id)

    points_layer.events.current_properties.connect(update_state)
    points_layer.events.current_properties.connect(update_track_id)
    # box_layer.events.current_properties.connect(update_state_boxes)
    box_layer.events.current_properties.connect(update_track_id_boxes)

    def state_changed(new_state):
        current_properties = points_layer.current_properties
        current_properties["state"] = np.array([new_state])
        points_layer.current_properties = current_properties
        points_layer.refresh_colors()

    def track_id_changed(new_track_id):
        current_properties = points_layer.current_properties
        current_properties["track_id"] = np.array([new_track_id])
        points_layer.current_properties = current_properties
        state.current_track_id = int(new_track_id)

    # def state_changed_boxes(new_state):
    #     current_properties = box_layer.current_properties
    #     current_properties["state"] = np.array([new_state])
    #     box_layer.current_properties = current_properties
    #     box_layer.refresh_colors()

    def track_id_changed_boxes(new_track_id):
        current_properties = box_layer.current_properties
        current_properties["track_id"] = np.array([new_track_id])
        box_layer.current_properties = current_properties
        state.current_track_id = int(new_track_id)

    state_menu.changed.connect(state_changed)
    track_id_menu.changed.connect(track_id_changed)
    # state_menu.changed.connect(state_changed_boxes)
    track_id_menu.changed.connect(track_id_changed_boxes)

    state_menu.set_choice("track")
    return tracking_widget


class AnnotatorTracking(_AnnotatorBase):

    # The tracking annotator needs different settings for the prompt layers
    # to support the additional tracking state.
    # That's why we over-ride this function.
    def _create_layers(self, segmentation_result):
        self._point_labels = ["positive", "negative"]
        self._track_state_labels = ["track", "division"]

        self._point_prompt_layer = self._viewer.add_points(
            name="point_prompts",
            property_choices={
                "label": self._point_labels,
                "state": self._track_state_labels,
                "track_id": ["1"],  # we use string to avoid pandas warning
            },
            edge_color="label",
            edge_color_cycle=vutil.LABEL_COLOR_CYCLE,
            symbol="o",
            face_color="state",
            face_color_cycle=STATE_COLOR_CYCLE,
            edge_width=0.4,
            size=12,
            ndim=self._ndim,
        )
        self._point_prompt_layer.edge_color_mode = "cycle"
        self._point_prompt_layer.face_color_mode = "cycle"

        # Using the box layer to set divisions currently doesn't work.
        # That's why some of the code below is commented out.
        self._box_prompt_layer = self._viewer.add_shapes(
            shape_type="rectangle",
            edge_width=4,
            ndim=self._ndim,
            face_color="transparent",
            name="prompts",
            edge_color="green",
            property_choices={"track_id": ["1"]},
            # property_choces={"track_id": ["1"], "state": self._track_state_labels},
            # edge_color_cycle=STATE_COLOR_CYCLE,
        )
        # self._box_prompt_layer.edge_color_mode = "cycle"

        # Add the label layers for the current object, the automatic segmentation and the committed segmentation.
        dummy_data = np.zeros(self._shape, dtype="uint32")
        self._viewer.add_labels(data=dummy_data, name="current_object")
        self._viewer.add_labels(data=dummy_data, name="auto_segmentation")
        self._viewer.add_labels(
            data=dummy_data if segmentation_result is None else segmentation_result, name="committed_objects"
        )
        # Randomize colors so it is easy to see when object committed.
        self._viewer.layers["committed_objects"].new_colormap()

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        # segmentation_result: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(
            viewer=viewer,
            ndim=3,
            segment_widget=widgets.segment_frame,
            segment_nd_widget=widgets.track_object,
            commit_widget=widgets.commit_track,
            clear_widget=widgets.clear_track,
            # segmentation_result=segmentation_result,
        )

        # Initialize the state for tracking.
        state = AnnotatorState()
        self._init_track_state(state)

        # Create the tracking state menu.
        self._tracking_widget = create_tracking_menu(
            self._point_prompt_layer, self._box_prompt_layer,
            states=self._track_state_labels, track_ids=list(state.lineage.keys()),
        )
        self._save_lineage_widget = widgets.save_lineage()
        # Add the two widgets to the docked widgets.
        self.extend([self._tracking_widget, self._save_lineage_widget])

        # Add the tracking widget to the state so that it can be accessed from within widgets
        # in order to update it when the tracking state changes.
        # NOTE: it would be more elegant to do this by emmitting and connecting events,
        # but I don't know how to create custom events.
        state.tracking_widget = self._tracking_widget

        # Go to t=0.
        self._viewer.dims.current_step = (0, 0, 0) + tuple(sh // 2 for sh in self._shape[1:])

    def _init_track_state(self, state):
        state.current_track_id = 1
        state.lineage = {1: []}
        state.committed_lineages = []

    def _update_image(self):
        super()._update_image()
        # Reset the state for tracking.
        state = AnnotatorState()
        self._init_track_state(state)


def annotator_tracking(
    image: np.ndarray,
    embedding_path: Optional[str] = None,
    # tracking_result: Optional[str] = None,
    model_type: str = util._DEFAULT_MODEL,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    return_viewer: bool = False,
    viewer: Optional["napari.viewer.Viewer"] = None,
    predictor: Optional[SamPredictor] = None,
) -> Optional["napari.viewer.Viewer"]:
    """Start the tracking annotation tool fora given timeseries.

    Args:
        raw: The image data.
        embedding_path: Filepath for saving the precomputed embeddings.
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
        image, model_type=model_type, save_path=embedding_path,
        halo=halo, tile_shape=tile_shape, predictor=predictor,
        ndim=3,
    )
    state.image_shape = image.shape[:-1] if image.ndim == 4 else image.shape

    if viewer is None:
        viewer = napari.Viewer()

    viewer.add_image(image, name="image")
    annotator = AnnotatorTracking(viewer)

    # Trigger layer update of the annotator so that layers have the correct shape.
    annotator._update_image()

    # Add the annotator widget to the viewer.
    viewer.window.add_dock_widget(annotator)

    if return_viewer:
        return viewer

    napari.run()


def main():
    """@private"""
    parser = vutil._initialize_parser(
        description="Run interactive segmentation for an image volume.",
        with_segmentation_result=False,
    )

    # Tracking result is not yet supported, we need to also deserialize the lineage.
    # parser.add_argument(
    #     "-t", "--tracking_result",
    #     help="Optional filepath to a precomputed tracking result. If passed this will be used to initialize the "
    #     "'committed_tracks' layer. This can be useful if you want to correct an existing tracking result or if you "
    #     "have saved intermediate results from the annotator and want to continue. "
    #     "Supports the same file formats as 'input'."
    # )
    # parser.add_argument(
    #     "-tk", "--tracking_key",
    #     help="The key for opening the tracking result. Same rules as for 'key' apply."
    # )

    args = parser.parse_args()
    image = util.load_image_data(args.input, key=args.key)

    if args.embedding_path is None:
        warnings.warn("You have not passed an embedding_path. Restarting the annotator may take a long time.")

    annotator_tracking(
        image, embedding_path=args.embedding_path, model_type=args.model_type,
        tile_shape=args.tile_shape, halo=args.halo,
    )
