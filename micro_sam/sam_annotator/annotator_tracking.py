import napari
import numpy as np

from magicgui import magicgui
from napari import Viewer
from napari.utils import progress
from scipy.ndimage import shift

# this is more precise for comuting the centers, but slow!
# from vigra.filters import eccentricityCenters

from .. import util
from ..segment_from_prompts import segment_from_mask, segment_from_points
from .util import (
    create_prompt_menu, prompt_layer_to_points, prompt_layer_to_state, segment_slices_with_prompts, LABEL_COLOR_CYCLE
)
from ..visualization import project_embeddings_for_visualization

# Magenta and Cyan
STATE_COLOR_CYCLE = ["#FF00FF", "#00FFFF"]


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


def _update_motion_model(motion_model, move, motion_smoothing):
    alpha = motion_smoothing
    motion_model = alpha * motion_model + (1 - alpha) * move
    return motion_model


def _shift_object(mask, motion_model):
    mask_shifted = np.zeros_like(mask)
    shift(mask, motion_model, output=mask_shifted, order=0, prefilter=False)
    return mask_shifted


# TODO handle divison annotations + division classifier
def _track_from_prompts(
    prompt_layer, seg, predictor, slices, image_embeddings,
    stop_upper, threshold, projection,
    progress_bar=None, motion_smoothing=0.5,
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
    def _update_motion(seg, t, t0, motion_model):
        seg_prev = seg[t - 1]

        if t == t0 + 1:  # this is the second frame, we don't have a motion model yet
            pass
        elif t == t0 + 2:  # this the third frame, we initialize the motion model
            current_move = _compute_movement(seg, t - 1, t - 2)
            motion_model = current_move
        else:  # we already have a motion model and update it
            current_move = _compute_movement(seg, t - 1, t - 2)
            motion_model = _update_motion_model(motion_model, current_move, motion_smoothing)

        if motion_model is not None:  # shift the segmentation according to the motion model
            seg_prev = _shift_object(seg_prev, motion_model)

        return seg_prev, motion_model

    motion_model = None
    verbose = False

    t0 = int(slices.min())
    t = t0 + 1
    while True:

        if t in slices:  # this is a slice with prompts
            seg_prev = None
            seg_t = seg[t]
            track_state = prompt_layer_to_state(prompt_layer, t)
            # TODO what do we do with the motion model here?

        else:  # this is a slice without prompts
            seg_prev, motion_model = _update_motion(seg, t, t0, motion_model)
            if verbose:
                print(f"Tracking object in frame {t} with movement {motion_model}")
            seg_t = segment_from_mask(predictor, seg_prev, image_embeddings=image_embeddings, i=t,
                                      use_mask=use_mask, use_box=use_box)
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

        seg[t] = seg_t
        t += 1

        # stop tracking if we have stop upper set (i.e. single negative point was set to indicate stop track)
        if t == slices[-1] and stop_upper:
            break

        # stop if we are at the last slce
        if t == seg.shape[0] - 1:
            break

        # stop if we have a division
        if track_state == "division":
            break

    return seg


#
# the widgets
#


@magicgui(call_button="Segment Frame [S]")
def segment_frame_wigdet(v: Viewer):
    position = v.cursor.position
    t = int(position[0])

    this_prompts = prompt_layer_to_points(v.layers["prompts"], t)
    points, labels = this_prompts
    seg = segment_from_points(PREDICTOR, points, labels, image_embeddings=IMAGE_EMBEDDINGS, i=t)

    v.layers["current_track"].data[t] = seg.squeeze()
    v.layers["current_track"].refresh()


@magicgui(call_button="Track Object [V]", projection={"choices": ["default", "bounding_box", "mask"]})
def track_objet_widget(
    v: Viewer, iou_threshold: float = 0.5, projection: str = "default", motion_smoothing: float = 0.5
):
    shape = v.layers["raw"].data.shape

    # choose mask projection for square images and bounding box projection otherwise
    # (because mask projection does not work properly for non-square images yet)
    if projection == "default":
        projection_ = "mask" if shape[1] == shape[2] else "bounding_box"
    else:
        projection_ = projection

    with progress(total=shape[0]) as progress_bar:
        # step 1: segment all slices with prompts
        seg, slices, _, stop_upper = segment_slices_with_prompts(
            PREDICTOR, v.layers["prompts"], IMAGE_EMBEDDINGS, shape, progress_bar=progress_bar
        )

        # step 2: track the object starting from the lowest annotated slice
        seg = _track_from_prompts(
            v.layers["prompts"], seg, PREDICTOR, slices, IMAGE_EMBEDDINGS, stop_upper, threshold=iou_threshold,
            projection=projection_, progress_bar=progress_bar, motion_smoothing=motion_smoothing,
        )

    v.layers["current_track"].data = seg
    v.layers["current_track"].refresh()


def annotator_tracking(raw, embedding_path=None, show_embeddings=False):
    # for access to the predictor and the image embeddings in the widgets
    global PREDICTOR, IMAGE_EMBEDDINGS, NEXT_ID
    NEXT_ID = 1
    PREDICTOR = util.get_sam_model()
    IMAGE_EMBEDDINGS = util.precompute_image_embeddings(PREDICTOR, raw, save_path=embedding_path)

    #
    # initialize the viewer and add layers
    #

    v = Viewer()

    v.add_image(raw)
    v.add_labels(data=np.zeros(raw.shape, dtype="uint32"), name="committed_tracks")
    v.add_labels(data=np.zeros(raw.shape, dtype="uint32"), name="current_track")

    # show the PCA of the image embeddings
    if show_embeddings:
        embedding_vis, scale = project_embeddings_for_visualization(IMAGE_EMBEDDINGS["features"], raw.shape)
        v.add_image(embedding_vis, name="embeddings", scale=scale)

    #
    # add the widgets
    #
    labels = ["positive", "negative"]
    state_labels = ["division", "track"]
    prompts = v.add_points(
        data=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],  # FIXME workaround
        name="prompts",
        properties={"label": labels, "state": state_labels},
        edge_color="label",
        edge_color_cycle=LABEL_COLOR_CYCLE,
        symbol="o",
        face_color="state",
        face_color_cycle=STATE_COLOR_CYCLE,
        edge_width=0.4,
        size=12,
        ndim=3,
    )
    prompts.edge_color_mode = "cycle"
    prompts.face_color_mode = "cycle"

    #
    # add the widgets
    #

    # TODO add (optional) auto-segmentation functionality

    prompt_widget = create_prompt_menu(prompts, labels)
    v.window.add_dock_widget(prompt_widget)

    state_widget = create_prompt_menu(prompts, state_labels, menu_name="state", label_name="state")
    v.window.add_dock_widget(state_widget)

    v.window.add_dock_widget(segment_frame_wigdet)
    v.window.add_dock_widget(track_objet_widget)

    #
    # key bindings
    #

    @v.bind_key("s")
    def _seg_slice(v):
        segment_frame_wigdet(v)

    @v.bind_key("v")
    def _track_object(v):
        track_objet_widget(v)

    @v.bind_key("t")
    def toggle_label(event=None):
        # get the currently selected label
        current_properties = prompts.current_properties
        current_label = current_properties["label"][0]
        new_label = "negative" if current_label == "positive" else "positive"
        current_properties["label"] = np.array([new_label])
        prompts.current_properties = current_properties
        prompts.refresh()
        prompts.refresh_colors()

    @v.bind_key("Shift-C")
    def clear_prompts(v):
        prompts.data = []
        prompts.refresh()

    #
    # start the viewer
    #

    # go to t=0
    v.dims.current_step = (0,) + tuple(sh // 2 for sh in raw.shape[1:])

    # clear the initial points needed for workaround
    clear_prompts(v)
    napari.run()


def main():
    import argparse
    import warnings

    parser = argparse.ArgumentParser(
        description="Run interactive segmentation for an image volume."
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="The filepath to the image data. Supports all data types that can be read by imageio (e.g. tif, png, ...) "
        "or elf.io.open_file (e.g. hdf5, zarr, mrc) For the latter you also need to pass the 'key' parameter."
    )
    parser.add_argument(
        "-k", "--key",
        help="The key for opening data with elf.io.open_file. This is the internal path for a hdf5 or zarr container, "
        "for a image series it is a wild-card, e.g. '*.png' and for mrc it is 'data'."
    )
    parser.add_argument(
        "-e", "--embedding_path",
        help="The filepath for saving/loading the pre-computed image embeddings. "
        "NOTE: It is recommended to pass this argument and store the embeddings, "
        "otherwise they will be recomputed every time (which can take a long time)."
    )
    # Not implemented for the tracking annotator yet.
    # And we should change the name for it.
    # parser.add_argument(
    #     "-s", "--segmentation",
    #     help="Optional filepath to a precomputed segmentation. If passed this will be used to initialize the "
    #     "'committed_objects' layer. This can be useful if you want to correct an existing segmentation or if you "
    #     "have saved intermediate results from the annotator and want to continue with your annotations. "
    #     "Supports the same file formats as 'input'."
    # )
    # parser.add_argument(
    #     "-sk", "--segmentation_key",
    #     help="The key for opening the segmentation data. Same rules as for 'key' apply."
    # )
    parser.add_argument(
        "--show_embeddings", action="store_true",
        help="Visualize the embeddings computed by SegmentAnything. This can be helpful for debugging."
    )

    args = parser.parse_args()
    raw = util.load_image_data(args.input, ndim=3, key=args.key)

    # if args.segmentation is None:
    #     segmentation = None
    # else:
    #     segmentation = util.load_image_data(args.segmentation, args.segmentation_key)

    if args.embedding_path is None:
        warnings.warn("You have not passed an embedding_path. Restarting the annotator may take a long time.")

    annotator_tracking(
        raw, embedding_path=args.embedding_path, show_embeddings=args.show_embeddings
    )
