import json
import warnings
from pathlib import Path
from typing import Optional, Tuple

import napari
import numpy as np

from magicgui import magicgui
from magicgui.widgets import ComboBox, Container
from napari import Viewer
from napari.utils import progress
from scipy.ndimage import shift
from segment_anything import SamPredictor

# this is more precise for comuting the centers, but slow!
# from vigra.filters import eccentricityCenters

from .. import util
from ..prompt_based_segmentation import segment_from_mask
from ..visualization import project_embeddings_for_visualization
from . import util as vutil
from .gui_utils import show_wrong_file_warning
from ._state import AnnotatorState

# Cyan (track) and Magenta (division)
STATE_COLOR_CYCLE = ["#00FFFF", "#FF00FF", ]
"""@private"""

COMMITTED_LINEAGES = []
"""@private"""


#
# util functionality
#


def _compute_movement(seg, t0, t1):

    def compute_center(t):

        # computation with vigra eccentricity centers (too slow)
        # center = np.array(eccentricityCenters(seg[t].astype("uint32")))
        # assert center.shape == (2, 2)
        # return center[1]

        # computation with center of mass
        center = np.where(seg[t] == 1)
        center = np.array(np.mean(center[0]), np.mean(center[1]))
        return center

    center0 = compute_center(t0)
    center1 = compute_center(t1)

    move = center1 - center0
    return move.astype("float64")


def _shift_object(mask, motion_model):
    mask_shifted = np.zeros_like(mask)
    shift(mask, motion_model, output=mask_shifted, order=0, prefilter=False)
    return mask_shifted


# TODO division classifier
def _track_from_prompts(
    point_prompts, box_prompts, seg, predictor, slices, image_embeddings,
    stop_upper, threshold, projection,
    progress_bar=None, motion_smoothing=0.5, box_extension=0,
):
    assert projection in ("mask", "bounding_box")
    if projection == "mask":
        use_mask, use_box = True, True
    else:
        use_mask, use_box = False, True

    def _update_progress():
        if progress_bar is not None:
            progress_bar.update(1)

    # shift the segmentation based on the motion model and update the motion model
    def _update_motion_model(seg, t, t0, motion_model):
        if t in (t0, t0 + 1):  # this is the first or second frame, we don't have a motion yet
            pass
        elif t == t0 + 2:  # this the third frame, we initialize the motion model
            current_move = _compute_movement(seg, t - 1, t - 2)
            motion_model = current_move
        else:  # we already have a motion model and update it
            current_move = _compute_movement(seg, t - 1, t - 2)
            alpha = motion_smoothing
            motion_model = alpha * motion_model + (1 - alpha) * current_move

        return motion_model

    has_division = False
    motion_model = None
    verbose = False

    t0 = int(slices.min())
    t = t0 + 1
    while True:

        # update the motion model
        motion_model = _update_motion_model(seg, t, t0, motion_model)

        # use the segmentation from prompts if we are in a slice with prompts
        if t in slices:
            seg_prev = None
            seg_t = seg[t]
            # currently using the box layer doesn't work for keeping track of the track state
            # track_state = prompt_layers_to_state(point_prompts, box_prompts, t)
            track_state = vutil.prompt_layer_to_state(point_prompts, t)

        # otherwise project the mask (under the motion model) and segment the next slice from the mask
        else:
            if verbose:
                print(f"Tracking object in frame {t} with movement {motion_model}")

            seg_prev = seg[t - 1]
            # shift the segmentation according to the motion model
            if motion_model is not None:
                seg_prev = _shift_object(seg_prev, motion_model)

            seg_t = segment_from_mask(predictor, seg_prev, image_embeddings=image_embeddings, i=t,
                                      use_mask=use_mask, use_box=use_box, box_extension=box_extension)
            track_state = "track"

            # are we beyond the last slice with prompt?
            # if no: we continue tracking because we know we need to connect to a future frame
            # if yes: we only continue tracking if overlaps are above the threshold
            if t < slices[-1]:
                seg_prev = None

            _update_progress()

        if (threshold is not None) and (seg_prev is not None):
            iou = util.compute_iou(seg_prev, seg_t)
            if iou < threshold:
                msg = f"Segmentation stopped at frame {t} due to IOU {iou} < {threshold}."
                print(msg)
                break

        # stop if we have a division
        if track_state == "division":
            has_division = True
            break

        seg[t] = seg_t
        t += 1

        # stop tracking if we have stop upper set (i.e. single negative point was set to indicate stop track)
        if t == slices[-1] and stop_upper:
            break

        # stop if we are at the last slce
        if t == seg.shape[0]:
            break

    return seg, has_division


def _update_lineage():
    global TRACKING_WIDGET
    state = AnnotatorState()

    mother = state.current_track_id
    assert mother in state.lineage
    assert len(state.lineage[mother]) == 0

    daughter1, daughter2 = state.current_track_id + 1, state.current_track_id + 2
    state.lineage[mother] = [daughter1, daughter2]
    state.lineage[daughter1] = []
    state.lineage[daughter2] = []

    # update the choices in the track_id menu
    track_ids = list(map(str, state.lineage.keys()))
    TRACKING_WIDGET[1].choices = track_ids

    # not sure if this does the right thing.
    # for now the user has to take care of this manually
    # # reset the state to track
    # TRACKING_WIDGET[0].set_choice("track")


#
# the widgets
#


@magicgui(call_button="Segment Frame [S]")
def _segment_frame_wigdet(v: Viewer) -> None:
    state = AnnotatorState()
    shape = v.layers["current_track"].data.shape[1:]
    position = v.cursor.position
    t = int(position[0])

    point_prompts = vutil.point_layer_to_prompts(v.layers["point_prompts"], i=t, track_id=state.current_track_id)
    # this is a stop prompt, we do nothing
    if not point_prompts:
        return

    boxes, masks = vutil.shape_layer_to_prompts(v.layers["prompts"], shape, i=t, track_id=state.current_track_id)
    points, labels = point_prompts

    seg = vutil.prompt_segmentation(
        state.predictor, points, labels, boxes, masks, shape, multiple_box_prompts=False,
        image_embeddings=state.image_embeddings, i=t
    )

    # no prompts were given or prompts were invalid, skip segmentation
    if seg is None:
        print("You either haven't provided any prompts or invalid prompts. The segmentation will be skipped.")
        return

    # clear the old segmentation for this track_id
    old_mask = v.layers["current_track"].data[t] == state.current_track_id
    v.layers["current_track"].data[t][old_mask] = 0
    # set the new segmentation
    new_mask = seg.squeeze() == 1
    v.layers["current_track"].data[t][new_mask] = state.current_track_id
    v.layers["current_track"].refresh()


@magicgui(call_button="Track Object [Shift-S]", projection={"choices": ["default", "bounding_box", "mask"]})
def _track_object_widget(
    v: Viewer, iou_threshold: float = 0.5, projection: str = "default",
    motion_smoothing: float = 0.5, box_extension: float = 0.1,
) -> None:
    state = AnnotatorState()
    shape = state.image_shape

    # we use the bounding box projection method as default which generally seems to work better for larger changes
    # between frames (which is pretty tyipical for tracking compared to 3d segmentation)
    projection_ = "mask" if projection == "default" else projection

    with progress(total=shape[0]) as progress_bar:
        # step 1: segment all slices with prompts
        seg, slices, _, stop_upper = vutil.segment_slices_with_prompts(
            state.predictor, v.layers["point_prompts"], v.layers["prompts"], state.image_embeddings, shape,
            progress_bar=progress_bar, track_id=state.current_track_id
        )

        # step 2: track the object starting from the lowest annotated slice
        seg, has_division = _track_from_prompts(
            v.layers["point_prompts"], v.layers["prompts"], seg,
            state.predictor, slices, state.image_embeddings, stop_upper,
            threshold=iou_threshold, projection=projection_,
            progress_bar=progress_bar, motion_smoothing=motion_smoothing,
            box_extension=box_extension,
        )

    # if a division has occurred and it's the first time it occurred for this track
    # we need to create the two daughter tracks and update the lineage
    if has_division and (len(state.lineage[state.current_track_id]) == 0):
        _update_lineage()

    # clear the old track mask
    v.layers["current_track"].data[v.layers["current_track"].data == state.current_track_id] = 0
    # set the new track mask
    v.layers["current_track"].data[seg == 1] = state.current_track_id
    v.layers["current_track"].refresh()


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


def _reset_tracking_state():
    global TRACKING_WIDGET
    state = AnnotatorState()

    # reset the lineage and track id
    state.current_track_id = 1
    state.lineage = {1: []}

    # reset the choices in the track_id menu
    track_ids = list(map(str, state.lineage.keys()))
    TRACKING_WIDGET[1].choices = track_ids


@magicgui(call_button="Commit [C]", layer={"choices": ["current_track"]})
def _commit_tracking_widget(v: Viewer, layer: str = "current_track") -> None:
    global COMMITTED_LINEAGES
    state = AnnotatorState()

    seg = v.layers[layer].data

    id_offset = int(v.layers["committed_tracks"].data.max())
    mask = seg != 0

    v.layers["committed_tracks"].data[mask] = (seg[mask] + id_offset)
    v.layers["committed_tracks"].refresh()

    shape = v.layers["raw"].data.shape
    v.layers[layer].data = np.zeros(shape, dtype="uint32")
    v.layers[layer].refresh()

    updated_lineage = {
        parent + id_offset: [child + id_offset for child in children] for parent, children in state.lineage.items()
    }
    COMMITTED_LINEAGES.append(updated_lineage)

    _reset_tracking_state()
    vutil.clear_annotations(v, clear_segmentations=False)


@magicgui(call_button="Clear Annotations [Shift + C]")
def _clear_widget_tracking(v: Viewer) -> None:
    _reset_tracking_state()
    vutil.clear_annotations(v)


@magicgui(call_button="Save Lineage")
def _save_lineage_widget(v: Viewer, path: Path) -> None:
    path = path.with_suffix(".json")
    with open(path, "w") as f:
        json.dump(COMMITTED_LINEAGES, f)


def annotator_tracking(
    raw: np.ndarray,
    embedding_path: Optional[str] = None,
    show_embeddings: bool = False,
    tracking_result: Optional[str] = None,
    model_type: str = util._DEFAULT_MODEL,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    return_viewer: bool = False,
    predictor: Optional[SamPredictor] = None,
) -> Optional[Viewer]:
    """The annotation tool for tracking in timeseries data.

    Args:
        raw: The image data.
        embedding_path: Filepath for saving the precomputed embeddings.
        show_embeddings: Show PCA visualization of the image embeddings.
            This can be helpful to judge how well Segment Anything works for your data,
            and which objects can be segmented.
        tracking_result: An initial tracking result to load.
            This can be used to correct tracking with Segment Anything or to save and load progress.
            The segmentation will be loaded as the 'committed_tracks' layer.
        model_type: The Segment Anything model to use. For details on the available models check out
            https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#finetuned-models.
        tile_shape: Shape of tiles for tiled embedding prediction.
            If `None` then the whole image is passed to Segment Anything.
        halo: Shape of the overlap between tiles, which is needed to segment objects on tile boarders.
        return_viewer: Whether to return the napari viewer to further modify it before starting the tool.
        predictor: The Segment Anything model. Passing this enables using fully custom models.
            If you pass `predictor` then `model_type` will be ignored.

    Returns:
        The napari viewer, only returned if `return_viewer=True`.
    """
    # NOTE: the tracking widget is left as global state for now.
    # The fact that it is in the state is quite a hack. When building a plugin for the
    # tracking annotator this needs to be redesigned!
    global TRACKING_WIDGET

    state = AnnotatorState()

    if predictor is None:
        state.predictor = util.get_sam_model(model_type=model_type)
    else:
        state.predictor = predictor
    state.image_embeddings = util.precompute_image_embeddings(
        state.predictor, raw, save_path=embedding_path, tile_shape=tile_shape, halo=halo,
        wrong_file_callback=show_wrong_file_warning,
    )
    state.image_shape = raw.shape

    state.current_track_id = 1
    state.lineage = {1: []}

    #
    # initialize the viewer and add layers
    #

    v = Viewer()

    v.add_image(raw)
    if tracking_result is None:
        v.add_labels(data=np.zeros(raw.shape, dtype="uint32"), name="committed_tracks")
    else:
        assert tracking_result.shape == raw.shape
        v.add_labels(data=tracking_result, name="committed_tracks")
    v.layers["committed_tracks"].new_colormap()  # randomize colors so it is easy to see when object committed
    v.add_labels(data=np.zeros(raw.shape, dtype="uint32"), name="current_track")

    # show the PCA of the image embeddings
    if show_embeddings:
        embedding_vis, scale = project_embeddings_for_visualization(state.image_embeddings)
        v.add_image(embedding_vis, name="embeddings", scale=scale)

    #
    # add the widgets
    #
    labels = ["positive", "negative"]
    state_labels = ["track", "division"]
    prompts = v.add_points(
        data=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],  # FIXME workaround
        name="point_prompts",
        properties={
            "label": labels,
            "state": state_labels,
            "track_id": ["1", "1"],  # NOTE we use string to avoid pandas warnings...
        },
        edge_color="label",
        edge_color_cycle=vutil.LABEL_COLOR_CYCLE,
        symbol="o",
        face_color="state",
        face_color_cycle=STATE_COLOR_CYCLE,
        edge_width=0.4,
        size=12,
        ndim=3,
    )
    prompts.edge_color_mode = "cycle"
    prompts.face_color_mode = "cycle"

    # using the box layer to set divisions currently doesn't work
    # (and setting new track ids also doesn't work, but keeping track of them in the properties is working)
    box_prompts = v.add_shapes(
        data=[
            np.array([[0, 0, 0], [0, 0, 10], [0, 10, 0], [0, 10, 10]]),
            np.array([[0, 0, 0], [0, 0, 11], [0, 11, 0], [0, 11, 11]]),
        ],  # FIXME workaround
        shape_type="rectangle",  # FIXME workaround
        edge_width=4, ndim=3,
        face_color="transparent",
        name="prompts",
        edge_color="green",
        properties={"track_id": ["1", "1"]},
        # properties={"track_id": ["1", "1"], "state": state_labels},
        # edge_color_cycle=STATE_COLOR_CYCLE,
    )
    # box_prompts.edge_color_mode = "cycle"

    #
    # add the widgets
    #

    # TODO add (optional) auto-segmentation and tracking functionality

    prompt_widget = vutil.create_prompt_menu(prompts, labels)
    v.window.add_dock_widget(prompt_widget)

    TRACKING_WIDGET = create_tracking_menu(prompts, box_prompts, state_labels, list(state.lineage.keys()))
    v.window.add_dock_widget(TRACKING_WIDGET)

    v.window.add_dock_widget(_segment_frame_wigdet)
    v.window.add_dock_widget(_track_object_widget)
    v.window.add_dock_widget(_commit_tracking_widget)
    v.window.add_dock_widget(_save_lineage_widget)
    v.window.add_dock_widget(_clear_widget_tracking)

    #
    # key bindings
    #

    @v.bind_key("s")
    def _seg_slice(v):
        _segment_frame_wigdet(v)

    @v.bind_key("Shift-S")
    def _track_object(v):
        _track_object_widget(v)

    @v.bind_key("t")
    def _toggle_label(event=None):
        vutil.toggle_label(prompts)

    @v.bind_key("c")
    def _commit(v):
        _commit_tracking_widget(v)

    @v.bind_key("Shift-C")
    def clear_prompts(v):
        _clear_widget_tracking(v)

    #
    # start the viewer
    #

    # go to t=0
    v.dims.current_step = (0,) + tuple(sh // 2 for sh in raw.shape[1:])

    # clear the initial points needed for workaround
    vutil.clear_annotations(v, clear_segmentations=False)

    if return_viewer:
        return v
    napari.run()


def main():
    """@private"""
    parser = vutil._initialize_parser(
        description="Run interactive segmentation for an image volume.",
        with_segmentation_result=False,
    )
    parser.add_argument(
        "-t", "--tracking_result",
        help="Optional filepath to a precomputed tracking result. If passed this will be used to initialize the "
        "'committed_tracks' layer. This can be useful if you want to correct an existing tracking result or if you "
        "have saved intermediate results from the annotator and want to continue. "
        "Supports the same file formats as 'input'."
    )
    parser.add_argument(
        "-tk", "--tracking_key",
        help="The key for opening the tracking result. Same rules as for 'key' apply."
    )

    args = parser.parse_args()
    raw = util.load_image_data(args.input, key=args.key)

    if args.tracking_result is None:
        tracking_result = None
    else:
        tracking_result = util.load_image_data(args.tracking_result, key=args.tracking_key)

    if args.embedding_path is None:
        warnings.warn("You have not passed an embedding_path. Restarting the annotator may take a long time.")

    annotator_tracking(
        raw, embedding_path=args.embedding_path, show_embeddings=args.show_embeddings,
        tracking_result=tracking_result, model_type=args.model_type,
        tile_shape=args.tile_shape, halo=args.halo,
    )
