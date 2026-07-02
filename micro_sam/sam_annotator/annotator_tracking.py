import os
from typing import List, Optional, Tuple, Union

import napari
import numpy as np
import imageio.v3 as imageio
import torch
from magicgui.widgets import ComboBox, Container

from .. import util
from ..v2.util import DEFAULT_MODEL
from . import _widgets as widgets
from . import util as vutil
from ._annotator import _AnnotatorBase
from ._batch import BatchAnnotatorTask, run_batch
from ._state import AnnotatorState
from ._tooltips import get_tooltip

# Cyan (track) and Magenta (division)
STATE_COLOR_CYCLE = [
    "#00FFFF",
    "#FF00FF",
]
"""@private"""


def _validate_tracking_model_type(model_type):
    if not model_type.startswith("hvit_"):
        raise ValueError(
            "The tracking annotator only supports micro-sam2/SAM2 models. "
            f"Got unsupported model '{model_type}'."
        )


# This solution is a bit hacky, so I won't move it to _widgets.py yet.
def create_tracking_menu(
    points_layer, box_layer, states, track_ids, point_labels=None, tracking_widget=None
):
    """@private"""
    state = AnnotatorState()

    def _get_widget_menu(container, label):
        for w in container:
            if isinstance(w, ComboBox) and w.label == label:
                return w
        raise ValueError(f"ComboBox with label '{label}' not found.")

    if tracking_widget is None:
        # The prompt label menu (positive / negative point prompts) shares this container with the
        # track id / track state menus so that all three dropdowns align in a single label column.
        label_menu = ComboBox(
            label="prompt",
            choices=point_labels,
            tooltip=get_tooltip("prompt_menu", "labels"),
        )
        state_menu = ComboBox(
            label="track_state",
            choices=states,
            tooltip=get_tooltip("annotator_tracking", "track_state"),
        )
        track_id_menu = ComboBox(
            label="track_id",
            choices=list(map(str, track_ids)),
            tooltip=get_tooltip("annotator_tracking", "track_id"),
        )
        tracking_widget = Container(widgets=[label_menu, state_menu, track_id_menu])
    else:
        label_menu = _get_widget_menu(tracking_widget, "prompt")
        state_menu = _get_widget_menu(tracking_widget, "track_state")
        track_id_menu = _get_widget_menu(tracking_widget, "track_id")

    # Keep the prompt label menu in sync with the point layer's current label.
    def update_label_menu(event):
        new_label = str(points_layer.current_properties["label"][0])
        if new_label != label_menu.value:
            label_menu.value = new_label

    def label_changed(new_label):
        current_properties = points_layer.current_properties
        current_properties["label"] = np.array([new_label])
        points_layer.current_properties = current_properties
        points_layer.refresh_colors()

    points_layer.events.current_properties.connect(update_label_menu)
    label_menu.changed.connect(label_changed)

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

    def _create_embedding_widget(self):
        return widgets.EmbeddingWidget(sam2_only=True, is_timeseries=True)

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
            if (
                layer_choices and "current_object" in layer_choices
            ):  # Check at 'commit' call button.
                widgets._validation_window_for_missing_layer("current_object")
            self._viewer.add_labels(data=dummy_data, name="current_object")
            if image_scale is not None:
                self._viewer.layers["current_object"].scale = image_scale

        if "auto_segmentation" not in self._viewer.layers:
            if (
                layer_choices and "auto_segmentation" in layer_choices
            ):  # Check at 'commit' call button.
                widgets._validation_window_for_missing_layer(
                    "auto_segmentation"
                )
            self._viewer.add_labels(data=dummy_data, name="auto_segmentation")
            if image_scale is not None:
                self._viewer.layers["auto_segmentation"].scale = image_scale

        if "committed_objects" not in self._viewer.layers:
            if (
                layer_choices and "committed_objects" in layer_choices
            ):  # Check at 'commit' call button.
                widgets._validation_window_for_missing_layer(
                    "committed_objects"
                )
            self._viewer.add_labels(data=dummy_data, name="committed_objects")
            # Randomize colors so it is easy to see when object committed.
            self._viewer.layers["committed_objects"].new_colormap()
            if image_scale is not None:
                self._viewer.layers["committed_objects"].scale = image_scale

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
            curr_property_choices = self._viewer.layers[
                "point_prompts"
            ].property_choices
            point_layer_mismatch = set(curr_property_choices.keys()) != set(
                _point_prompt_property_choices.keys()
            )

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
            curr_property_choices = self._viewer.layers[
                "prompts"
            ].property_choices
            box_layer_mismatch = set(curr_property_choices.keys()) != set(
                _box_prompt_property_choices.keys()
            )

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
                point_labels=self._point_labels,
                tracking_widget=getattr(self, "_tracking_widget", None),
            )

    def _get_widgets(self):
        self._require_layers()

        # Ensure the tracking state menu exists ('_require_layers' creates it when the layers are
        # (re)created; create it here as a fallback otherwise).
        if getattr(self, "_tracking_widget", None) is None:
            self._tracking_widget = create_tracking_menu(
                points_layer=self._point_prompt_layer,
                box_layer=self._box_prompt_layer,
                states=self._track_state_labels,
                track_ids=list(AnnotatorState().lineage.keys()),
                point_labels=self._point_labels,
            )

        # The prompt menu, the track id / track state menus and the segment / clear controls all
        # live in a single merged container, mirroring the segmentation annotator.
        interactive = widgets.InteractiveTrackingWidget(
            self._viewer, tracking_widget=self._tracking_widget,
        )
        autotrack = widgets.AutoTrackWidget(
            self._viewer, with_decoder=self._with_decoder, volumetric=True
        )
        return {
            "interactive": interactive,
            "autosegment": autotrack,
            "commit": widgets.commit_track(),
            "export": widgets.export_track(),
        }

    def _create_keybindings(self):
        interactive = self._widgets["interactive"]

        @self._viewer.bind_key("s", overwrite=True)
        def _segment(viewer):
            interactive.segment(viewer)

        # We also need to over-write the keybindings for the prompt layers.
        # See https://github.com/napari/napari/issues/7302 for details.
        prompt_layer = self._viewer.layers["prompts"]
        point_prompt_layer = self._viewer.layers["point_prompts"]

        @prompt_layer.bind_key("s", overwrite=True)
        def _segment_prompts(event):
            interactive.segment(self._viewer)

        @point_prompt_layer.bind_key("s", overwrite=True)
        def _segment_point_prompts(event):
            interactive.segment(self._viewer)

        @self._viewer.bind_key("c", overwrite=True)
        def _commit(viewer):
            self._widgets["commit"](viewer)

        @self._viewer.bind_key("t", overwrite=True)
        def _toggle_label(event=None):
            vutil.toggle_label(self._point_prompt_layer)

        @self._viewer.bind_key("Shift-C", overwrite=True)
        def _clear_annotations(viewer):
            interactive.clear(viewer)

    def __init__(
        self, viewer: "napari.viewer.Viewer", reset_state: bool = True
    ) -> None:
        # Initialize the state for tracking.
        self._init_track_state()
        # At startup the decoder is not loaded yet; also treat the default model as decoder-capable
        # when it has a registered decoder, so the default mode is correct before 'Compute Embeddings'.
        from ..v2.util import has_registered_decoder
        self._with_decoder = AnnotatorState().decoder is not None or has_registered_decoder(DEFAULT_MODEL)
        super().__init__(viewer=viewer, ndim=3)
        # Go to t=0.
        self._viewer.dims.current_step = (0, 0, 0) + tuple(
            sh // 2 for sh in self._shape[1:]
        )

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
    model_type: str = DEFAULT_MODEL,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    return_viewer: bool = False,
    viewer: Optional["napari.viewer.Viewer"] = None,
    precompute_amg_state: bool = False,
    checkpoint_path: Optional[str] = None,
    decoder_path: Optional[str] = None,
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
        halo: Shape of the overlap between tiles, which is needed to segment objects on tile borders.
        return_viewer: Whether to return the napari viewer to further modify it before starting the tool.
            By default, does not return the napari viewer.
        viewer: The viewer to which the Segment Anything functionality should be added.
            This enables using a pre-initialized viewer.
        precompute_amg_state: Whether to precompute the state for automatic mask generation.
            This will take more time when precomputing embeddings, but will then make
            automatic mask generation much faster. By default, set to 'False'.
        checkpoint_path: Path to a custom checkpoint from which to load the SAM model.
        decoder_path: Path to a custom decoder checkpoint from which to load the 'micro-sam` decoder.
        device: The computational device to use for the SAM model.
            By default, automatically chooses the best available device.

    Returns:
        The napari viewer, only returned if `return_viewer=True`.
    """

    _validate_tracking_model_type(model_type)

    # Initialize the predictor state.
    state = AnnotatorState()
    state.initialize_predictor(
        image,
        model_type=model_type,
        save_path=embedding_path,
        halo=halo,
        tile_shape=tile_shape,
        prefer_decoder=True,
        ndim=3,
        checkpoint_path=checkpoint_path,
        decoder_path=decoder_path,
        device=device,
        precompute_amg_state=precompute_amg_state,
        use_cli=True,
    )
    state.image_shape = image.shape[:-1] if image.ndim == 4 else image.shape

    if viewer is None:
        viewer = napari.Viewer()

    viewer.add_image(image, name="image")
    annotator = AnnotatorTracking(viewer, reset_state=False)

    # Trigger layer update of the annotator so that layers have the correct shape.
    annotator._update_image()

    # Add the annotator widget to the viewer and sync widgets.
    viewer.window.add_dock_widget(annotator, name="Segment Anything for Microscopy (Tracking)")
    vutil._sync_embedding_widget(
        widget=state.widgets["embeddings"],
        model_type=(
            model_type
            if checkpoint_path is None
            else state.predictor.model_type
        ),
        save_path=embedding_path,
        checkpoint_path=checkpoint_path,
        device=device,
        tile_shape=tile_shape,
        halo=halo,
    )

    if return_viewer:
        return viewer

    napari.run()


class TrackingBatchTask(BatchAnnotatorTask):
    """Batch task for tracking: each item is a TYX timeseries tracked independently."""

    empty_item_message = "Nothing is tracked yet. Do you wish to continue to the next timeseries?"

    def __init__(
        self, *, model_type, embedding_path=None, tile_shape=None, halo=None,
        checkpoint_path=None, decoder_path=None, device=None, precompute_amg_state=False,
    ):
        _validate_tracking_model_type(model_type)
        self.model_type = model_type
        self.embedding_path = embedding_path
        self.tile_shape = tile_shape
        self.halo = halo
        self.checkpoint_path = checkpoint_path
        self.decoder_path = decoder_path
        self.device = device
        self.precompute_amg_state = precompute_amg_state

    def result_filename(self, entry, index):
        if self.have_inputs_as_arrays:
            return f"tracks_{index:05}.tif"
        return os.path.splitext(os.path.basename(entry))[0] + "_tracks.tif"

    def precompute(self, images):
        # The SAM2 video embeddings are computed lazily per video in start/advance. When an embedding
        # folder is given, derive one per-video zarr path inside it; otherwise keep them in memory.
        if self.embedding_path is None:
            return [None] * len(images)
        os.makedirs(self.embedding_path, exist_ok=True)
        if self.have_inputs_as_arrays:
            return [os.path.join(self.embedding_path, f"tracking_{i:05}.zarr") for i in range(len(images))]
        return [
            os.path.join(self.embedding_path, os.path.splitext(os.path.basename(p))[0] + ".zarr") for p in images
        ]

    def _init_predictor(self, image, embedding_path, reuse):
        state = AnnotatorState()
        # Reuse the loaded model and decoder on subsequent videos (only the embeddings and the
        # interactive segmenter are rebuilt per video).
        if reuse and state.predictor is not None:
            kwargs = dict(predictor=state.predictor, decoder=state.decoder, prefer_decoder=False)
        else:
            kwargs = dict(prefer_decoder=True)
        state.initialize_predictor(
            image, model_type=self.model_type, save_path=embedding_path, halo=self.halo,
            tile_shape=self.tile_shape, ndim=3, checkpoint_path=self.checkpoint_path,
            decoder_path=self.decoder_path, device=self.device,
            precompute_amg_state=self.precompute_amg_state, use_cli=True, **kwargs,
        )
        state.image_shape = image.shape[:-1] if image.ndim == 4 else image.shape

    def start(self, viewer, entry, image, embedding_path, index):
        self._init_predictor(image, embedding_path, reuse=False)
        viewer.add_image(image, name="image")
        AnnotatorState().image_scale = tuple(viewer.layers["image"].scale)

        annotator = AnnotatorTracking(viewer, reset_state=False)
        annotator._update_image()

        state = AnnotatorState()
        viewer.window.add_dock_widget(annotator, name="Segment Anything for Microscopy (Batch Tracking)")
        vutil._sync_embedding_widget(
            widget=state.widgets["embeddings"],
            model_type=self.model_type if self.checkpoint_path is None else state.predictor.model_type,
            save_path=embedding_path, checkpoint_path=self.checkpoint_path,
            device=self.device, tile_shape=self.tile_shape, halo=self.halo,
        )
        return annotator

    def advance(self, viewer, annotator, entry, image, embedding_path, index):
        viewer.layers["image"].data = image
        self._init_predictor(image, embedding_path, reuse=True)
        AnnotatorState().image_scale = tuple(viewer.layers["image"].scale)
        annotator._update_image()

    def has_unsaved_content(self, viewer):
        return viewer.layers["committed_objects"].data.sum() != 0

    def save_item(self, viewer, entry, index):
        save_path = os.path.join(self.output_folder, self.result_filename(entry, index))
        imageio.imwrite(save_path, viewer.layers["committed_objects"].data, compression="zlib")


def batch_tracking_annotator(
    images: Union[List[Union[os.PathLike, str]], List[np.ndarray]],
    output_folder: str,
    *,
    model_type: str = DEFAULT_MODEL,
    embedding_path: Optional[str] = None,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    checkpoint_path: Optional[str] = None,
    decoder_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    precompute_amg_state: bool = False,
    viewer: Optional["napari.viewer.Viewer"] = None,
    return_viewer: bool = False,
    skip_done: bool = True,
) -> Optional["napari.viewer.Viewer"]:
    """Run the tracking annotation tool for a batch of timeseries (each item is one TYX video).

    Args:
        images: List of timeseries (TYX arrays) or file paths, each tracked independently.
        output_folder: The folder where the per-video tracking results are saved.
        model_type: The micro-sam2/SAM2 model to use (must start with 'hvit_').
        embedding_path: Folder where to save/load the per-video embeddings.
        tile_shape: Shape of tiles for tiled embedding prediction.
            If `None` then the whole image is passed to Segment Anything.
        halo: Shape of the overlap between tiles, which is needed to segment objects on tile borders.
        checkpoint_path: Path to a custom checkpoint from which to load the SAM model.
        decoder_path: Path to a custom decoder checkpoint from which to load the `micro-sam` decoder.
        device: The computational device to use for the SAM model.
            By default, automatically chooses the best available device.
        precompute_amg_state: Whether to precompute the state for automatic mask generation.
        viewer: The viewer to which the functionality should be added.
        return_viewer: Whether to return the napari viewer instead of starting the event loop.
        skip_done: Whether to skip videos whose tracking result already exists in `output_folder`.

    Returns:
        The napari viewer, only returned if `return_viewer=True`.
    """
    have_inputs_as_arrays = isinstance(images[0], np.ndarray)
    task = TrackingBatchTask(
        model_type=model_type, embedding_path=embedding_path, tile_shape=tile_shape, halo=halo,
        checkpoint_path=checkpoint_path, decoder_path=decoder_path, device=device,
        precompute_amg_state=precompute_amg_state,
    )
    return run_batch(
        images, output_folder, task, have_inputs_as_arrays=have_inputs_as_arrays,
        viewer=viewer, return_viewer=return_viewer, skip_done=skip_done,
    )


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
        image,
        embedding_path=args.embedding_path,
        model_type=args.model_type,
        tile_shape=args.tile_shape,
        halo=args.halo,
        checkpoint_path=args.checkpoint,
        decoder_path=args.decoder_path,
        device=args.device,
    )
