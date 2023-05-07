import napari
import numpy as np

from magicgui import magicgui
from napari import Viewer
from napari.utils import progress

from .. import util
from ..segment_from_prompts import segment_from_mask, segment_from_points
from ..visualization import project_embeddings_for_visualization
from .util import create_prompt_menu, prompt_layer_to_points, segment_slices_with_prompts

COLOR_CYCLE = ["#00FF00", "#FF0000"]


#
# util functionality
#


# TODO handle divison annotations + division classifier
def _track_from_prompts(seg, predictor, slices, image_embeddings, stop_upper, threshold, projection, progress_bar=None):
    assert projection in ("mask", "bounding_box")
    if projection == "mask":
        use_mask, use_box = True, True
    else:
        use_mask, use_box = False, True

    def _update_progress():
        if progress_bar is not None:
            progress_bar.update(1)

    t0 = int(slices.min())
    t = t0 + 1
    while True:
        if t in slices:
            seg_prev = None
            seg_t = seg[t]
        else:
            seg_prev = seg[t - 1]
            seg_t = segment_from_mask(predictor, seg_prev, image_embeddings=image_embeddings, i=t,
                                      use_mask=use_mask, use_box=use_box)
            _update_progress()

        if (threshold is not None) and (seg_prev is not None):
            iou = util.compute_iou(seg_prev, seg_t)
            if iou < threshold:
                msg = f"Segmentation stopped at frame {t} due to IOU {iou} < {threshold}."
                print(msg)
                break

        seg[t] = seg_t
        t += 1

        # TODO here we need to also stop once divisions are implemented
        # stop tracking if we have stop upper set (i.e. single negative point was set to indicate stop track)
        if t == slices[-1] and stop_upper:
            break

        # stop if we are at the last slce
        if t == seg.shape[0] - 1:
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
def track_objet_widget(v: Viewer, iou_threshold: float = 0.8, projection: str = "default"):
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
            seg, PREDICTOR, slices, IMAGE_EMBEDDINGS, stop_upper, iou_threshold, projection_, progress_bar=progress_bar
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
    # TODO add the division labels
    labels = ["positive", "negative"]
    prompts = v.add_points(
        data=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],  # FIXME workaround
        name="prompts",
        properties={"label": labels},
        edge_color="label",
        edge_color_cycle=COLOR_CYCLE,
        symbol="o",
        face_color="transparent",
        edge_width=0.5,  # FIXME workaround
        size=12,
        ndim=3,
    )
    prompts.edge_color_mode = "cycle"

    #
    # add the widgets
    #

    # TODO add (optional) auto-segmentation functionality

    prompt_widget = create_prompt_menu(prompts, labels)
    v.window.add_dock_widget(prompt_widget)

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
