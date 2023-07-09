import napari
import numpy as np

from magicgui import magicgui
from napari import Viewer
from napari.utils import progress

from .. import util
from ..prompt_based_segmentation import segment_from_mask
from ..visualization import project_embeddings_for_visualization
from .util import (
    clear_all_prompts, commit_segmentation_widget, create_prompt_menu,
    prompt_layer_to_boxes, prompt_layer_to_points, prompt_segmentation,
    segment_slices_with_prompts, toggle_label, LABEL_COLOR_CYCLE,
    _initialize_parser
)


#
# utility functionality
# (some of this should be refactored to util.py)
#


# TODO refactor
def _segment_volume(
    seg, predictor, image_embeddings, segmented_slices,
    stop_lower, stop_upper, iou_threshold, projection,
    progress_bar=None, box_extension=0,
):
    assert projection in ("mask", "bounding_box", "points")
    if projection == "mask":
        use_box, use_mask, use_points = True, True, False
    elif projection == "points":
        use_box, use_mask, use_points = True, True, True
    else:
        use_box, use_mask, use_points = True, False, False

    def _update_progress():
        if progress_bar is not None:
            progress_bar.update(1)

    # TODO refactor to utils so that it can be used by other plugins
    def segment_range(z_start, z_stop, increment, stopping_criterion, threshold=None, verbose=False):
        z = z_start + increment
        while True:
            if verbose:
                print(f"Segment {z_start} to {z_stop}: segmenting slice {z}")
            seg_prev = seg[z - increment]
            seg_z = segment_from_mask(predictor, seg_prev, image_embeddings=image_embeddings, i=z,
                                      use_mask=use_mask, use_box=use_box, use_points=use_points,
                                      box_extension=box_extension)
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
            _update_progress()

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

            elif z_start == z0 and stop_lower:  # the lower slice is stop: we just segment from upper
                segment_range(z_stop, z_start, -1, np.less_equal, verbose=verbose)

            elif z_stop == z1 and stop_upper:  # the upper slice is stop: we just segment from lower
                segment_range(z_start, z_stop, 1, np.greater_equal, verbose=verbose)

            elif slice_diff == 2:  # there is only one slice in between -> use combined mask
                z = z_start + 1
                seg_prompt = np.logical_or(seg[z_start] == 1, seg[z_stop] == 1)
                seg[z] = segment_from_mask(predictor, seg_prompt, image_embeddings=image_embeddings, i=z,
                                           use_mask=use_mask, use_box=use_box, use_points=use_points,
                                           box_extension=box_extension)
                _update_progress()

            else:  # there is a range of more than 2 slices in between -> segment ranges
                # segment from bottom
                segment_range(
                    z_start, z_mid, 1, np.greater_equal if slice_diff % 2 == 0 else np.greater, verbose=verbose
                )
                # segment from top
                segment_range(z_stop, z_mid, -1, np.less_equal, verbose=verbose)
                # if the difference between start and stop is even,
                # then we have a slice in the middle that is the same distance from top bottom
                # in this case the slice is not segmented in the ranges above, and we segment it
                # using the combined mask from the adjacent top and bottom slice as prompt
                if slice_diff % 2 == 0:
                    seg_prompt = np.logical_or(seg[z_mid - 1] == 1, seg[z_mid + 1] == 1)
                    seg[z_mid] = segment_from_mask(predictor, seg_prompt, image_embeddings=image_embeddings, i=z_mid,
                                                   use_mask=use_mask, use_box=use_box, use_points=use_points,
                                                   box_extension=box_extension)
                    _update_progress()

    return seg


#
# the widgets
#


@magicgui(call_button="Segment Slice [S]")
def segment_slice_wigdet(v: Viewer):
    position = v.cursor.position
    z = int(position[0])

    point_prompts = prompt_layer_to_points(v.layers["prompts"], z)
    # this is a stop prompt, we do nothing
    if not point_prompts:
        return

    boxes = prompt_layer_to_boxes(v.layers["box_prompts"], z)
    points, labels = point_prompts

    shape = v.layers["current_object"].data.shape[1:]
    seg = prompt_segmentation(
        PREDICTOR, points, labels, boxes, shape, multiple_box_prompts=False,
        image_embeddings=IMAGE_EMBEDDINGS, i=z
    )

    # no prompts were given or prompts were invalid, skip segmentation
    if seg is None:
        print("You either haven't provided any prompts or invalid prompts. The segmentation will be skipped.")
        return

    v.layers["current_object"].data[z] = seg
    v.layers["current_object"].refresh()


@magicgui(call_button="Segment Volume [V]", projection={"choices": ["default", "bounding_box", "mask", "points"]})
def segment_volume_widget(
    v: Viewer, iou_threshold: float = 0.8, projection: str = "default", box_extension: float = 0.1
):
    # step 1: segment all slices with prompts
    shape = v.layers["raw"].data.shape

    # we have the following projection modes:
    # bounding_box: uses only the bounding box as prompt
    # mask: uses the bounding box and the mask
    # points: uses the bounding box, mask and points derived from the mask
    # by default we choose mask, which qualitatively seems to work the best
    if projection == "default":
        projection_ = "mask"
    else:
        projection_ = projection

    with progress(total=shape[0]) as progress_bar:

        seg, slices, stop_lower, stop_upper = segment_slices_with_prompts(
            PREDICTOR, v.layers["prompts"], v.layers["box_prompts"], IMAGE_EMBEDDINGS, shape, progress_bar=progress_bar,
        )

        # step 2: segment the rest of the volume based on smart prompting
        seg = _segment_volume(
            seg, PREDICTOR, IMAGE_EMBEDDINGS, slices,
            stop_lower, stop_upper,
            iou_threshold=iou_threshold, projection=projection_,
            progress_bar=progress_bar, box_extension=box_extension,
        )

    v.layers["current_object"].data = seg
    v.layers["current_object"].refresh()


def annotator_3d(
    raw, embedding_path=None, show_embeddings=False, segmentation_result=None,
    model_type="vit_h", tile_shape=None, halo=None, return_viewer=False
):
    # for access to the predictor and the image embeddings in the widgets
    global PREDICTOR, IMAGE_EMBEDDINGS
    PREDICTOR = util.get_sam_model(model_type=model_type)
    IMAGE_EMBEDDINGS = util.precompute_image_embeddings(
        PREDICTOR, raw, save_path=embedding_path, tile_shape=tile_shape, halo=halo
    )

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
    v.layers["committed_objects"].new_colormap()  # randomize colors so it is easy to see when object committed
    v.add_labels(data=np.zeros(raw.shape, dtype="uint32"), name="current_object")

    # show the PCA of the image embeddings
    if show_embeddings:
        embedding_vis, scale = project_embeddings_for_visualization(IMAGE_EMBEDDINGS)
        v.add_image(embedding_vis, name="embeddings", scale=scale)

    labels = ["positive", "negative"]
    prompts = v.add_points(
        data=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],  # FIXME workaround
        name="prompts",
        properties={"label": labels},
        edge_color="label",
        edge_color_cycle=LABEL_COLOR_CYCLE,
        symbol="o",
        face_color="transparent",
        edge_width=0.5,
        size=12,
        ndim=3,
    )
    prompts.edge_color_mode = "cycle"

    v.add_shapes(
        face_color="transparent", edge_color="green", edge_width=4, name="box_prompts", ndim=3
    )

    #
    # add the widgets
    #

    # TODO add (optional) auto-segmentation functionality

    prompt_widget = create_prompt_menu(prompts, labels)
    v.window.add_dock_widget(prompt_widget)

    v.window.add_dock_widget(segment_slice_wigdet)

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
    def _toggle_label(event=None):
        toggle_label(prompts)

    @v.bind_key("Shift-C")
    def clear_prompts(v):
        clear_all_prompts(v)

    #
    # start the viewer
    #

    # clear the initial points needed for workaround
    clear_prompts(v)

    if return_viewer:
        return v
    napari.run()


def main():
    import warnings

    parser = _initialize_parser(description="Run interactive segmentation for an image volume.")
    args = parser.parse_args()
    raw = util.load_image_data(args.input, ndim=3, key=args.key)

    if args.segmentation_result is None:
        segmentation_result = None
    else:
        segmentation_result = util.load_image_data(args.segmentation_result, args.segmentation_key)

    if args.embedding_path is None:
        warnings.warn("You have not passed an embedding_path. Restarting the annotator may take a long time.")

    annotator_3d(
        raw, embedding_path=args.embedding_path,
        show_embeddings=args.show_embeddings, segmentation_result=segmentation_result,
        model_type=args.model_type, tile_shape=args.tile_shape, halo=args.halo,
    )
