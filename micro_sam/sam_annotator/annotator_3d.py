import napari
import numpy as np

from magicgui import magicgui
from napari import Viewer

from .. import util
from ..segment_from_prompts import segment_from_mask, segment_from_points
from ..visualization import compute_pca
from .util import commit_segmentation_widget, create_prompt_menu, prompt_layer_to_points, segment_slices_with_prompts

COLOR_CYCLE = ["#00FF00", "#FF0000"]


#
# utility functionality
# (some of this should be refactored to util.py)
#


# TODO refactor
def _segment_volume(
    seg, predictor, image_embeddings, segmented_slices,
    stop_lower, stop_upper, iou_threshold, method
):
    assert method in ("mask", "bounding_box")
    if method == "mask":
        use_mask, use_box = True, True
    else:
        use_mask, use_box = False, True

    # TODO refactor to utils so that it can be used by other plugins
    def segment_range(z_start, z_stop, increment, stopping_criterion, threshold=None, verbose=False):
        z = z_start + increment
        while True:
            if verbose:
                print(f"Segment {z_start} to {z_stop}: segmenting slice {z}")
            seg_prev = seg[z - increment]
            seg_z = segment_from_mask(predictor, seg_prev, image_embeddings=image_embeddings, i=z,
                                      use_mask=use_mask, use_box=use_box)
            if threshold is not None:
                iou = util.compute_iou(seg_prev, seg_z)
                if iou < threshold:
                    msg = f"Segmentation stopped at slice {z} due to IOU {iou} < {iou_threshold}."
                    print(msg)
                    break
            seg[z] = seg_z
            z += increment
            if stopping_criterion(z, z_stop):
                if verbose:
                    print(f"Segment {z_start} to {z_stop}: stop at slice {z}")
                break

    z0, z1 = int(segmented_slices.min()), int(segmented_slices.max())

    # segment below the min slice
    if z0 > 0 and not stop_lower:
        segment_range(z0, 0, -1, np.less, iou_threshold)

    # segment above the max slice
    if z1 < seg.shape[0] - 1 and not stop_upper:
        segment_range(z1, seg.shape[0] - 1, 1, np.greater, iou_threshold)

    verbose = False
    # segment in between min and max slice
    if z0 != z1:
        for z_start, z_stop in zip(segmented_slices[:-1], segmented_slices[1:]):
            slice_diff = z_stop - z_start
            z_mid = int((z_start + z_stop) // 2)
            if slice_diff == 1:  # the slices are adjacent -> we don't need to do anything
                pass

            elif slice_diff == 2:  # there is only one slice in between -> use combined mask
                z = z_start + 1
                seg_prompt = np.logical_or(seg[z_start] == 1, seg[z_stop] == 1)
                seg[z] = segment_from_mask(predictor, seg_prompt, image_embeddings=image_embeddings, i=z,
                                           use_mask=use_mask, use_box=use_box)

            else:  # there is a range of more than 2 slices in between -> segment ranges
                # segment from bottom
                segment_range(z_start, z_mid, 1, np.greater_equal, verbose=verbose)
                # segment from top
                segment_range(z_stop, z_mid, -1, np.less_equal, verbose=verbose)
                # if the difference between start and stop is even,
                # then we have a slice in the middle that is the same distance from top bottom
                # in this case the slice is not segmented in the ranges above, and we segment it
                # using the combined mask from the adjacent top and bottom slice as prompt
                if slice_diff % 2 == 0:
                    seg_prompt = np.logical_or(seg[z_mid - 1] == 1, seg[z_mid + 1] == 1)
                    seg[z_mid] = segment_from_mask(predictor, seg_prompt, image_embeddings=image_embeddings, i=z_mid,
                                                   use_mask=use_mask, use_box=use_box)

    return seg


#
# the widgets
#


@magicgui(call_button="Segment Slice [S]")
def segment_slice_wigdet(v: Viewer):
    position = v.cursor.position
    z = int(position[0])

    this_prompts = prompt_layer_to_points(v.layers["prompts"], z)
    if this_prompts is None:
        return

    points, labels = this_prompts
    seg = segment_from_points(PREDICTOR, points, labels, image_embeddings=IMAGE_EMBEDDINGS, i=z)

    v.layers["current_object"].data[z] = seg.squeeze()
    v.layers["current_object"].refresh()


@magicgui(call_button="Segment Volume [V]", method={"choices": ["bounding_box", "mask"]})
def segment_volume_widget(v: Viewer, iou_threshold: float = 0.8, method: str = "mask"):
    # step 1: segment all slices with prompts
    shape = v.layers["raw"].data.shape
    seg, slices, stop_lower, stop_upper = segment_slices_with_prompts(
        PREDICTOR, v.layers["prompts"], IMAGE_EMBEDDINGS, shape
    )

    # step 2: segment the rest of the volume based on smart prompting
    seg = _segment_volume(
        seg, PREDICTOR, IMAGE_EMBEDDINGS, slices,
        stop_lower, stop_upper,
        iou_threshold=iou_threshold, method=method,
    )

    v.layers["current_object"].data = seg
    v.layers["current_object"].refresh()


def annotator_3d(raw, embedding_path=None, show_embeddings=False, segmentation_result=None):
    # for access to the predictor and the image embeddings in the widgets
    global PREDICTOR, IMAGE_EMBEDDINGS
    PREDICTOR = util.get_sam_model()
    IMAGE_EMBEDDINGS = util.precompute_image_embeddings(PREDICTOR, raw, save_path=embedding_path)

    #
    # initialize the viewer and add layers
    #

    v = Viewer()

    v.add_image(raw)
    if segmentation_result is None:
        v.add_labels(data=np.zeros(raw.shape, dtype="uint32"), name="committed_objects")
    else:
        assert segmentation_result.shape == raw.shape
        v.add_labels(data=segmentation_result, name="committed_objects")
    v.add_labels(data=np.zeros(raw.shape, dtype="uint32"), name="current_object")

    # show the PCA of the image embeddings
    if show_embeddings:
        embedding_vis = compute_pca(IMAGE_EMBEDDINGS["features"])
        # FIXME determine the scale from data
        v.add_image(embedding_vis, name="embeddings", scale=(1, 8, 8))

    labels = ["positive", "negative"]
    prompts = v.add_points(
        data=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],  # FIXME workaround
        name="prompts",
        properties={"label": labels},
        edge_color="label",
        edge_color_cycle=COLOR_CYCLE,
        symbol="o",
        face_color="transparent",
        edge_width=0.5,
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

    v.window.add_dock_widget(segment_slice_wigdet)
    # v.bind_key("s", segment_slice_wigdet)  FIXME this causes an issue with all shortcuts

    v.window.add_dock_widget(segment_volume_widget)
    v.window.add_dock_widget(commit_segmentation_widget)

    #
    # key bindings
    #

    @v.bind_key("s")
    def _seg_slice(v):
        segment_slice_wigdet(v)

    @v.bind_key("v")
    def _seg_volume(v):
        segment_volume_widget(v)

    @v.bind_key("c")
    def _commit(v):
        commit_segmentation_widget(v)

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

    # clear the initial points needed for workaround
    clear_prompts(v)
    napari.run()
