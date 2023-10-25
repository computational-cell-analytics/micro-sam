import warnings
from typing import Optional, Tuple

import napari
import numpy as np

from magicgui import magicgui
from napari import Viewer
from segment_anything import SamPredictor

from .. import instance_segmentation, util
from ..precompute_state import cache_amg_state
from ..visualization import project_embeddings_for_visualization
from . import util as vutil
from .gui_utils import show_wrong_file_warning


@magicgui(call_button="Segment Object [S]")
def _segment_widget(v: Viewer, box_extension: float = 0.1) -> None:
    shape = v.layers["current_object"].data.shape

    # get the current box and point prompts
    boxes, masks = vutil.shape_layer_to_prompts(v.layers["prompts"], shape)
    points, labels = vutil.point_layer_to_prompts(v.layers["point_prompts"], with_stop_annotation=False)

    if IMAGE_EMBEDDINGS["original_size"] is None:  # tiled prediction
        seg = vutil.prompt_segmentation(
            PREDICTOR, points, labels, boxes, masks, shape, image_embeddings=IMAGE_EMBEDDINGS,
            multiple_box_prompts=True, box_extension=box_extension,
        )
    else:  # normal prediction and we have set the precomputed embeddings already
        seg = vutil.prompt_segmentation(
            PREDICTOR, points, labels, boxes, masks, shape, multiple_box_prompts=True, box_extension=box_extension,
        )

    # no prompts were given or prompts were invalid, skip segmentation
    if seg is None:
        print("You either haven't provided any prompts or invalid prompts. The segmentation will be skipped.")
        return

    v.layers["current_object"].data = seg
    v.layers["current_object"].refresh()


def _changed_param(amg, **params):
    if amg is None:
        return None
    for name, val in params.items():
        if hasattr(amg, f"_{name}") and getattr(amg, f"_{name}") != val:
            return name
    return None


@magicgui(
    call_button="Automatic Segmentation",
    min_object_size={"min": 0, "max": 10000},
)
def _autosegment_widget(
    v: Viewer,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    min_object_size: int = 100,
    with_background: bool = True,
) -> None:
    global AMG
    is_tiled = IMAGE_EMBEDDINGS["input_size"] is None
    if AMG is None:
        AMG = instance_segmentation.get_amg(PREDICTOR, is_tiled)

    print("initializing autosegment")
    if not AMG.is_initialized:
        AMG.initialize(v.layers["raw"].data, image_embeddings=IMAGE_EMBEDDINGS, verbose=True)

    seg = AMG.generate(pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh)

    shape = v.layers["raw"].data.shape[:2]
    seg = instance_segmentation.mask_data_to_segmentation(
        seg, shape, with_background=with_background, min_object_size=min_object_size
    )
    assert isinstance(seg, np.ndarray)

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

    v.add_image(raw, name="raw")
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
        name="point_prompts",
        properties={"label": labels},
        edge_color="label",
        edge_color_cycle=vutil.LABEL_COLOR_CYCLE,
        symbol="o",
        face_color="transparent",
        edge_width=0.5,
        size=12,
        ndim=2,
    )
    prompts.edge_color_mode = "cycle"

    v.add_shapes(
        face_color="transparent", edge_color="green", edge_width=4, name="prompts"
    )

    #
    # add the widgets
    #

    prompt_widget = vutil.create_prompt_menu(prompts, labels)
    v.window.add_dock_widget(prompt_widget)

    v.window.add_dock_widget(_autosegment_widget)
    v.window.add_dock_widget(_segment_widget)
    v.window.add_dock_widget(vutil._commit_segmentation_widget)
    v.window.add_dock_widget(vutil._clear_widget)

    #
    # key bindings
    #

    @v.bind_key("s")
    def _segmet(v):
        _segment_widget(v)

    @v.bind_key("c")
    def _commit(v):
        vutil._commit_segmentation_widget(v)

    @v.bind_key("t")
    def _toggle_label(event=None):
        vutil.toggle_label(prompts)

    @v.bind_key("Shift-C")
    def clear_prompts(v):
        vutil.clear_annotations(v)

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
    raw: np.ndarray,
    embedding_path: Optional[str] = None,
    show_embeddings: bool = False,
    segmentation_result: Optional[np.ndarray] = None,
    model_type: str = util._DEFAULT_MODEL,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    return_viewer: bool = False,
    v: Optional[Viewer] = None,
    predictor: Optional[SamPredictor] = None,
    precompute_amg_state: bool = False,
) -> Optional[Viewer]:
    """The 2d annotation tool.

    Args:
        raw: The image data.
        embedding_path: Filepath where to save the embeddings.
        show_embeddings: Show PCA visualization of the image embeddings.
            This can be helpful to judge how well Segment Anything works for your data,
            and which objects can be segmented.
        segmentation_result: An initial segmentation to load.
            This can be used to correct segmentations with Segment Anything or to save and load progress.
            The segmentation will be loaded as the 'committed_objects' layer.
        model_type: The Segment Anything model to use. For details on the available models check out
            https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#finetuned-models.
        tile_shape: Shape of tiles for tiled embedding prediction.
            If `None` then the whole image is passed to Segment Anything.
        halo: Shape of the overlap between tiles, which is needed to segment objects on tile boarders.
        return_viewer: Whether to return the napari viewer to further modify it before starting the tool.
        v: The viewer to which the SegmentAnything functionality should be added.
            This enables using a pre-initialized viewer, for example in `sam_annotator.image_series_annotator`.
        predictor: The Segment Anything model. Passing this enables using fully custom models.
            If you pass `predictor` then `model_type` will be ignored.
        precompute_amg_state: Whether to precompute the state for automatic mask generation.
            This will take more time when precomputing embeddings, but will then make
            automatic mask generation much faster.

    Returns:
        The napari viewer, only returned if `return_viewer=True`.
    """
    # for access to the predictor and the image embeddings in the widgets
    global PREDICTOR, IMAGE_EMBEDDINGS, AMG
    AMG = None

    if predictor is None:
        PREDICTOR = util.get_sam_model(model_type=model_type)
    else:
        PREDICTOR = predictor

    IMAGE_EMBEDDINGS = util.precompute_image_embeddings(
        PREDICTOR, raw, save_path=embedding_path, ndim=2, tile_shape=tile_shape, halo=halo,
        wrong_file_callback=show_wrong_file_warning
    )
    if precompute_amg_state and (embedding_path is not None):
        AMG = cache_amg_state(PREDICTOR, raw, IMAGE_EMBEDDINGS, embedding_path)

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
    vutil.clear_annotations(v, clear_segmentations=False)

    if return_viewer:
        return v

    napari.run()


def main():
    """@private"""
    parser = vutil._initialize_parser(description="Run interactive segmentation for an image.")
    parser.add_argument("--precompute_amg_state", action="store_true")
    args = parser.parse_args()
    raw = util.load_image_data(args.input, key=args.key)

    if args.segmentation_result is None:
        segmentation_result = None
    else:
        segmentation_result = util.load_image_data(args.segmentation_result, key=args.segmentation_key)

    if args.embedding_path is None:
        warnings.warn("You have not passed an embedding_path. Restarting the annotator may take a long time.")

    annotator_2d(
        raw, embedding_path=args.embedding_path,
        show_embeddings=args.show_embeddings, segmentation_result=segmentation_result,
        model_type=args.model_type, tile_shape=args.tile_shape, halo=args.halo,
        precompute_amg_state=args.precompute_amg_state,
    )
