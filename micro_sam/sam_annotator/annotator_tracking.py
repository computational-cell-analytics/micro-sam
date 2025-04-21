from typing import Optional, Tuple, Union, List

import napari
import numpy as np

import torch

from magicgui.widgets import ComboBox, Container

from .. import util
from . import util as vutil
from . import _widgets as widgets
from ._tooltips import get_tooltip
from ._state import AnnotatorState
from ._annotator import _AnnotatorBase


# Cyan (track) and Magenta (division)
STATE_COLOR_CYCLE = ["#00FFFF", "#FF00FF", ]
"""@private"""


# This solution is a bit hacky, so I won't move it to _widgets.py yet.
def create_tracking_menu(points_layer, box_layer, states, track_ids, tracking_widget=None):
    """@private"""
    state = AnnotatorState()

    def _get_widget_menu(container, label):
        for w in container:
            if isinstance(w, ComboBox) and w.label == label:
                return w
        raise ValueError(f"ComboBox with label '{label}' not found.")

    if tracking_widget is None:
        state_menu = ComboBox(
            label="track_state", choices=states, tooltip=get_tooltip("annotator_tracking", "track_state")
        )
        track_id_menu = ComboBox(
            label="track_id", choices=list(map(str, track_ids)), tooltip=get_tooltip("annotator_tracking", "track_id")
        )
        tracking_widget = Container(widgets=[state_menu, track_id_menu])
    else:
        state_menu = _get_widget_menu(tracking_widget, "track_state")
        track_id_menu = _get_widget_menu(tracking_widget, "track_id")

    def update_state(event):
        if "state" in points_layer.current_properties:
            new_state = str(points_layer.current_properties["state"][0])
            if new_state != state_menu.value:
                state_menu.value = new_state

    def update_track_id(event):
        if "track_id" in points_layer.current_properties:
            new_id = str(points_layer.current_properties["track_id"][0])
            if new_id != track_id_menu.value:
                track_id_menu.value = new_id
                state.current_track_id = int(new_id)

    # def update_state_boxes(event):
    #     new_state = str(box_layer.current_properties["state"][0])
    #     if new_state != state_menu.value:
    #         state_menu.value = new_state

    def update_track_id_boxes(event):
        if "track_id" in box_layer.current_properties:
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
        # Note: this fails with a key error after committing a lineage with multiple tracks.
        # I think this does not cause any further errors, so we just skip this.
        try:
            points_layer.current_properties = current_properties
        except KeyError:
            pass
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
    def _require_layers(self, layer_choices: Optional[List[str]] = None):

        # Check whether the image is initialized already. And use the image shape and scale for the layers.
        state = AnnotatorState()
        shape = self._shape if state.image_shape is None else state.image_shape

        # Add the label layers for the current object, the automatic segmentation and the committed segmentation.
        dummy_data = np.zeros(shape, dtype="uint32")
        image_scale = state.image_scale

        # Before adding new layers, we always check whether a layer with this name already exists or not.
        if "current_object" not in self._viewer.layers:
            if layer_choices and "current_object" in layer_choices:  # Check at 'commit' call button.
                widgets._validation_window_for_missing_layer("current_object")
            self._viewer.add_labels(data=dummy_data, name="current_object")
            if image_scale is not None:
                self.layers["current_objects"].scale = image_scale

        if "auto_segmentation" not in self._viewer.layers:
            if layer_choices and "auto_segmentation" in layer_choices:  # Check at 'commit' call button.
                widgets._validation_window_for_missing_layer("auto_segmentation")
            self._viewer.add_labels(data=dummy_data, name="auto_segmentation")
            if image_scale is not None:
                self.layers["auto_segmentation"].scale = image_scale

        if "committed_objects" not in self._viewer.layers:
            if layer_choices and "committed_objects" in layer_choices:  # Check at 'commit' call button.
                widgets._validation_window_for_missing_layer("committed_objects")
            self._viewer.add_labels(data=dummy_data, name="committed_objects")
            # Randomize colors so it is easy to see when object committed.
            self._viewer.layers["committed_objects"].new_colormap()
            if image_scale is not None:
                self.layers["committed_objects"].scale = image_scale

        # Add the point prompts layer.
        self._point_labels = ["positive", "negative"]
        self._track_state_labels = ["track", "division"]
        _point_prompt_property_choices = {
            "label": self._point_labels,
            "state": self._track_state_labels,
            "track_id": ["1"],  # we use string to avoid pandas warning
        }

        point_layer_mismatch = True
        if "point_prompts" in self._viewer.layers:
            # Check whether the 'property_choices' match or not.
            curr_property_choices = self._viewer.layers["point_prompts"].property_choices
            point_layer_mismatch = set(curr_property_choices.keys()) != set(_point_prompt_property_choices.keys())

        if point_layer_mismatch and "point_prompts" not in self._viewer.layers:
            self._point_prompt_layer = self._viewer.add_points(
                name="point_prompts",
                property_choices=_point_prompt_property_choices,
                border_color="label",
                border_color_cycle=vutil.LABEL_COLOR_CYCLE,
                symbol="o",
                face_color="state",
                face_color_cycle=STATE_COLOR_CYCLE,
                border_width=0.4,
                size=12,
                ndim=self._ndim,
            )
            self._point_prompt_layer.border_color_mode = "cycle"
            self._point_prompt_layer.face_color_mode = "cycle"
            _new_point_layer = True
        else:
            self._point_prompt_layer = self._viewer.layers["point_prompts"]
            _new_point_layer = False

        # Add the point prompts layer.
        _box_prompt_property_choices = {"track_id": ["1"]}

        box_layer_mismatch = True
        if "prompts" in self._viewer.layers:
            # Check whether the 'property_choices' match or not.
            curr_property_choices = self._viewer.layers["prompts"].property_choices
            box_layer_mismatch = set(curr_property_choices.keys()) != set(_box_prompt_property_choices.keys())

        if box_layer_mismatch and "prompts" not in self._viewer.layers:
            # Using the box layer to set divisions currently doesn't work.
            # That's why some of the code below is commented out.
            self._box_prompt_layer = self._viewer.add_shapes(
                shape_type="rectangle",
                edge_width=4,
                ndim=self._ndim,
                face_color="transparent",
                name="prompts",
                edge_color="green",
                property_choices=_box_prompt_property_choices,
                # property_choices={"track_id": ["1"], "state": self._track_state_labels},
                # edge_color_cycle=STATE_COLOR_CYCLE,
            )
            # self._box_prompt_layer.edge_color_mode = "cycle"
            _new_box_layer = True
        else:
            self._box_prompt_layer = self._viewer.layers["prompts"]
            _new_box_layer = False

        # Trigger a new connection for the tracking state menu only when a new layer is (re)created.
        if _new_point_layer or _new_box_layer:
            self._tracking_widget = create_tracking_menu(
                points_layer=self._point_prompt_layer,
                box_layer=self._box_prompt_layer,
                states=self._track_state_labels,
                track_ids=list(state.lineage.keys()),
                tracking_widget=state.widgets.get("tracking"),
            )
            state.widgets["tracking"] = self._tracking_widget

    def _get_widgets(self):
        state = AnnotatorState()
        self._require_layers()

        # Create the tracking state menu.
        # NOTE: Check whether it exists already from `_require_layers` or needs to be created.
        if state.widgets.get("tracking") is None:
            self._tracking_widget = create_tracking_menu(
                self._point_prompt_layer, self._box_prompt_layer,
                states=self._track_state_labels, track_ids=list(state.lineage.keys()),
            )
        else:
            self._tracking_widget = state.widgets.get("tracking")

        segment_nd = widgets.SegmentNDWidget(self._viewer, tracking=True)
        autotrack = widgets.AutoTrackWidget(self._viewer, with_decoder=self._with_decoder, volumetric=True)
        return {
            "tracking": self._tracking_widget,
            "segment": widgets.segment_frame(),
            "segment_nd": segment_nd,
            "autosegment": autotrack,
            "commit": widgets.commit_track(),
            "clear": widgets.clear_track(),
        }

    def __init__(self, viewer: "napari.viewer.Viewer", reset_state: bool = True) -> None:
        # Initialize the state for tracking.
        self._init_track_state()
        self._with_decoder = AnnotatorState().decoder is not None
        super().__init__(viewer=viewer, ndim=3)
        # Go to t=0.
        self._viewer.dims.current_step = (0, 0, 0) + tuple(sh // 2 for sh in self._shape[1:])

        # Set the expected annotator class to the state.
        state = AnnotatorState()

        # Reset the state.
        if reset_state:
            state.reset_state()

        state.annotator = self

    def _init_track_state(self):
        state = AnnotatorState()
        state.current_track_id = 1
        state.lineage = {1: []}
        state.committed_lineages = []

    def _update_image(self):
        super()._update_image()
        self._init_track_state()
        state = AnnotatorState()
        if self._with_decoder:
            state.amg_state = vutil._load_is_state(state.embedding_path)
        else:
            state.amg_state = vutil._load_amg_state(state.embedding_path)


def annotator_tracking(
    image: np.ndarray,
    embedding_path: Optional[str] = None,
    # tracking_result: Optional[str] = None,
    model_type: str = util._DEFAULT_MODEL,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    return_viewer: bool = False,
    viewer: Optional["napari.viewer.Viewer"] = None,
    precompute_amg_state: bool = False,
    checkpoint_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Optional["napari.viewer.Viewer"]:
    """Start the tracking annotation tool fora given timeseries.

    Args:
        image: The image data.
        embedding_path: Filepath for saving the precomputed embeddings.
        model_type: The Segment Anything model to use. For details on the available models check out
            https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#finetuned-models.
        tile_shape: Shape of tiles for tiled embedding prediction.
            If `None` then the whole image is passed to Segment Anything.
        halo: Shape of the overlap between tiles, which is needed to segment objects on tile boarders.
        return_viewer: Whether to return the napari viewer to further modify it before starting the tool.
        viewer: The viewer to which the Segment Anything functionality should be added.
            This enables using a pre-initialized viewer.
        precompute_amg_state: Whether to precompute the state for automatic mask generation.
            This will take more time when precomputing embeddings, but will then make
            automatic mask generation much faster.
        checkpoint_path: Path to a custom checkpoint from which to load the SAM model.
        device: The computational device to use for the SAM model.

    Returns:
        The napari viewer, only returned if `return_viewer=True`.
    """

    # Initialize the predictor state.
    state = AnnotatorState()
    state.initialize_predictor(
        image, model_type=model_type, save_path=embedding_path,
        halo=halo, tile_shape=tile_shape, prefer_decoder=True,
        ndim=3, checkpoint_path=checkpoint_path, device=device,
        precompute_amg_state=precompute_amg_state, use_cli=True,
    )
    state.image_shape = image.shape[:-1] if image.ndim == 4 else image.shape

    if viewer is None:
        viewer = napari.Viewer()

    viewer.add_image(image, name="image")
    annotator = AnnotatorTracking(viewer, reset_state=False)

    # Trigger layer update of the annotator so that layers have the correct shape.
    annotator._update_image()

    # Add the annotator widget to the viewer and sync widgets.
    viewer.window.add_dock_widget(annotator)
    vutil._sync_embedding_widget(
        widget=state.widgets["embeddings"],
        model_type=model_type if checkpoint_path is None else state.predictor.model_type,
        save_path=embedding_path,
        checkpoint_path=checkpoint_path,
        device=device,
        tile_shape=tile_shape,
        halo=halo,
    )

    if return_viewer:
        return viewer

    napari.run()


def main():
    """@private"""
    parser = vutil._initialize_parser(
        description="Run interactive segmentation for an image volume.",
        with_segmentation_result=False,
        with_instance_segmentation=False,
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

    annotator_tracking(
        image, embedding_path=args.embedding_path, model_type=args.model_type,
        tile_shape=args.tile_shape, halo=args.halo,
        checkpoint_path=args.checkpoint, device=args.device,
    )
