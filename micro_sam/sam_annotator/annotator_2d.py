import warnings

import napari
import numpy as np

from magicgui import magicgui
from napari import Viewer

from .. import util
from .. import segment_instances
from ..visualization import project_embeddings_for_visualization
from .util import (
    clear_all_prompts, commit_segmentation_widget, create_prompt_menu,
    prompt_layer_to_boxes, prompt_layer_to_points, prompt_segmentation, toggle_label, LABEL_COLOR_CYCLE,
    _initialize_parser,
)


@magicgui(call_button="Segment Object [S]")
def segment_wigdet(v: Viewer):
    # get the current box and point prompts
    boxes = prompt_layer_to_boxes(v.layers["box_prompts"])
    points, labels = prompt_layer_to_points(v.layers["prompts"])

    shape = v.layers["current_object"].data.shape
    if IMAGE_EMBEDDINGS["original_size"] is None:  # tiled prediction
        seg = prompt_segmentation(
            PREDICTOR, points, labels, boxes, shape, image_embeddings=IMAGE_EMBEDDINGS, multiple_box_prompts=True
        )
    else:  # normal prediction and we have set the precomputed embeddings already
        seg = prompt_segmentation(PREDICTOR, points, labels, boxes, shape, multiple_box_prompts=True)

    # no prompts were given or prompts were invalid, skip segmentation
    if seg is None:
        print("You either haven't provided any prompts or invalid prompts. The segmentation will be skipped.")
        return

    v.layers["current_object"].data = seg
    v.layers["current_object"].refresh()


# TODO expose more parameters:
# - min initial size
# - advanced params???
@magicgui(call_button="Automatic Segmentation")
def autosegment_widget(
    v: Viewer, with_background: bool = True, box_extension: float = 0.1, pred_iou_thresh: float = 0.88
):
    is_tiled = IMAGE_EMBEDDINGS["input_size"] is None
    if is_tiled:
        seg = segment_instances.segment_instances_from_embeddings_with_tiling(
            PREDICTOR, IMAGE_EMBEDDINGS, with_background=with_background,
            box_extension=box_extension, pred_iou_thresh=pred_iou_thresh,
        )
    else:
        seg = segment_instances.segment_instances_from_embeddings(
            PREDICTOR, IMAGE_EMBEDDINGS, with_background=with_background,
            box_extension=box_extension, pred_iou_thresh=pred_iou_thresh,
        )
    v.layers["auto_segmentation"].data = seg
    v.layers["auto_segmentation"].refresh()


def _get_shape(raw):
    if raw.ndim == 2:
        shape = raw.shape
    elif raw.ndim == 3 and raw.shape[-1] == 3:
        shape = raw.shape[:2]
    else:
        raise ValueError(f"Invalid input image of shape {raw.shape}. Expect either 2D grayscale or 3D RGB image.")
    return shape


def _initialize_viewer(raw, segmentation_result, tile_shape, show_embeddings):
    v = Viewer()

    #
    # initialize the viewer and add layers
    #

    v.add_image(raw)
    shape = _get_shape(raw)

    v.add_labels(data=np.zeros(shape, dtype="uint32"), name="auto_segmentation")
    if segmentation_result is None:
        v.add_labels(data=np.zeros(shape, dtype="uint32"), name="committed_objects")
    else:
        v.add_labels(segmentation_result, name="committed_objects")
    v.layers["committed_objects"].new_colormap()  # randomize colors so it is easy to see when object committed
    v.add_labels(data=np.zeros(shape, dtype="uint32"), name="current_object")

    # show the PCA of the image embeddings
    if show_embeddings:
        embedding_vis, scale = project_embeddings_for_visualization(IMAGE_EMBEDDINGS)
        v.add_image(embedding_vis, name="embeddings", scale=scale)

    labels = ["positive", "negative"]
    prompts = v.add_points(
        data=[[0.0, 0.0], [0.0, 0.0]],  # FIXME workaround
        name="prompts",
        properties={"label": labels},
        edge_color="label",
        edge_color_cycle=LABEL_COLOR_CYCLE,
        symbol="o",
        face_color="transparent",
        edge_width=0.5,
        size=12,
        ndim=2,
    )
    prompts.edge_color_mode = "cycle"

    v.add_shapes(
        face_color="transparent", edge_color="green", edge_width=4, name="box_prompts"
    )

    #
    # add the widgets
    #

    prompt_widget = create_prompt_menu(prompts, labels)
    v.window.add_dock_widget(prompt_widget)

    # (optional) auto-segmentation functionality
    v.window.add_dock_widget(autosegment_widget)

    v.window.add_dock_widget(segment_wigdet)
    v.window.add_dock_widget(commit_segmentation_widget)

    #
    # key bindings
    #

    @v.bind_key("s")
    def _segmet(v):
        segment_wigdet(v)

    @v.bind_key("c")
    def _commit(v):
        commit_segmentation_widget(v)

    @v.bind_key("t")
    def _toggle_label(event=None):
        toggle_label(prompts)

    @v.bind_key("Shift-C")
    def clear_prompts(v):
        clear_all_prompts(v)

    return v


def _update_viewer(v, raw, show_embeddings, segmentation_result):
    if show_embeddings or segmentation_result is not None:
        raise NotImplementedError

    # update the image layer
    v.layers["raw"].data = raw
    shape = _get_shape(raw)

    # update the segmentation layers
    v.layers["auto_segmentation"].data = np.zeros(shape, dtype="uint32")
    v.layers["committed_objects"].data = np.zeros(shape, dtype="uint32")
    v.layers["current_object"].data = np.zeros(shape, dtype="uint32")


def annotator_2d(
    raw, embedding_path=None, show_embeddings=False, segmentation_result=None,
    model_type="vit_h", tile_shape=None, halo=None, return_viewer=False, v=None,
    predictor=None,
):
    # for access to the predictor and the image embeddings in the widgets
    global PREDICTOR, IMAGE_EMBEDDINGS

    if predictor is None:
        PREDICTOR = util.get_sam_model(model_type=model_type)
    else:
        PREDICTOR = predictor
    IMAGE_EMBEDDINGS = util.precompute_image_embeddings(
        PREDICTOR, raw, save_path=embedding_path, ndim=2, tile_shape=tile_shape, halo=halo
    )

    # we set the pre-computed image embeddings if we don't use tiling
    # (if we use tiling we cannot directly set it because the tile will be chosen dynamically)
    if tile_shape is None:
        util.set_precomputed(PREDICTOR, IMAGE_EMBEDDINGS)

    # viewer is freshly initialized
    if v is None:
        v = _initialize_viewer(raw, segmentation_result, tile_shape, show_embeddings)
    # we use an existing viewer and just update all the layers
    else:
        _update_viewer(v, raw, show_embeddings, segmentation_result)

    #
    # start the viewer
    #
    clear_all_prompts(v)

    if return_viewer:
        return v

    napari.run()


def main():
    parser = _initialize_parser(description="Run interactive segmentation for an image.")
    args = parser.parse_args()
    raw = util.load_image_data(args.input, ndim=2, key=args.key)

    if args.segmentation_result is None:
        segmentation_result = None
    else:
        segmentation_result = util.load_image_data(args.segmentation_result, args.segmentation_key)

    if args.embedding_path is None:
        warnings.warn("You have not passed an embedding_path. Restarting the annotator may take a long time.")

    annotator_2d(
        raw, embedding_path=args.embedding_path,
        show_embeddings=args.show_embeddings, segmentation_result=segmentation_result,
        model_type=args.model_type, tile_shape=args.tile_shape, halo=args.halo,
    )
