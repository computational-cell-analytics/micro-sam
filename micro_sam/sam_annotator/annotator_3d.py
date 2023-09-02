import warnings
from typing import Optional, Tuple

import napari
import numpy as np

from magicgui import magicgui
from napari import Viewer
from napari.utils import progress
from segment_anything import SamPredictor

from .. import util
from ..multi_dimensional_segmentation import segment_mask_in_volume
from ..visualization import project_embeddings_for_visualization
from . import util as vutil
from .gui_utils import show_wrong_file_warning


#
# the widgets
#


@magicgui(call_button="Segment Slice [S]")
def _segment_slice_wigdet(v: Viewer) -> None:
    position = v.cursor.position
    z = int(position[0])

    point_prompts = vutil.prompt_layer_to_points(v.layers["prompts"], z)
    # this is a stop prompt, we do nothing
    if not point_prompts:
        return

    boxes = vutil.prompt_layer_to_boxes(v.layers["box_prompts"], z)
    points, labels = point_prompts

    shape = v.layers["current_object"].data.shape[1:]
    seg = vutil.prompt_segmentation(
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
def _segment_volume_widget(
    v: Viewer, iou_threshold: float = 0.8, projection: str = "default", box_extension: float = 0.05
) -> None:
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

        seg, slices, stop_lower, stop_upper = vutil.segment_slices_with_prompts(
            PREDICTOR, v.layers["prompts"], v.layers["box_prompts"], IMAGE_EMBEDDINGS, shape, progress_bar=progress_bar,
        )

        # step 2: segment the rest of the volume based on smart prompting
        seg = segment_mask_in_volume(
            seg, PREDICTOR, IMAGE_EMBEDDINGS, slices,
            stop_lower, stop_upper,
            iou_threshold=iou_threshold, projection=projection_,
            progress_bar=progress_bar, box_extension=box_extension,
        )

    v.layers["current_object"].data = seg
    v.layers["current_object"].refresh()


def annotator_3d(
    raw: np.ndarray,
    embedding_path: Optional[str] = None,
    show_embeddings: bool = False,
    segmentation_result: Optional[np.ndarray] = None,
    model_type: str = util._DEFAULT_MODEL,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    return_viewer: bool = False,
    predictor: Optional[SamPredictor] = None,
) -> Optional[Viewer]:
    """The 3d annotation tool.

    Args:
        raw: The image data.
        embedding_path: Filepath for saving the precomputed embeddings.
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
        predictor: The Segment Anything model. Passing this enables using fully custom models.
            If you pass `predictor` then `model_type` will be ignored.

    Returns:
        The napari viewer, only returned if `return_viewer=True`.
    """
    # for access to the predictor and the image embeddings in the widgets
    global PREDICTOR, IMAGE_EMBEDDINGS

    if predictor is None:
        PREDICTOR = util.get_sam_model(model_type=model_type)
    else:
        PREDICTOR = predictor
    IMAGE_EMBEDDINGS = util.precompute_image_embeddings(
        PREDICTOR, raw, save_path=embedding_path, tile_shape=tile_shape, halo=halo,
        wrong_file_callback=show_wrong_file_warning,
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
        edge_color_cycle=vutil.LABEL_COLOR_CYCLE,
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

    prompt_widget = vutil.create_prompt_menu(prompts, labels)
    v.window.add_dock_widget(prompt_widget)

    v.window.add_dock_widget(_segment_slice_wigdet)

    v.window.add_dock_widget(_segment_volume_widget)
    v.window.add_dock_widget(vutil._commit_segmentation_widget)
    v.window.add_dock_widget(vutil._clear_widget)

    #
    # key bindings
    #

    @v.bind_key("s")
    def _seg_slice(v):
        _segment_slice_wigdet(v)

    @v.bind_key("v")
    def _seg_volume(v):
        _segment_volume_widget(v)

    @v.bind_key("c")
    def _commit(v):
        vutil._commit_segmentation_widget(v)

    @v.bind_key("t")
    def _toggle_label(event=None):
        vutil.toggle_label(prompts)

    @v.bind_key("Shift-C")
    def clear_prompts(v):
        vutil.clear_annotations(v)

    #
    # start the viewer
    #

    # clear the initial points needed for workaround
    vutil.clear_annotations(v, clear_segmentations=False)

    if return_viewer:
        return v
    napari.run()


def main():
    """@private"""
    parser = vutil._initialize_parser(description="Run interactive segmentation for an image volume.")
    args = parser.parse_args()
    raw = util.load_image_data(args.input, key=args.key)

    if args.segmentation_result is None:
        segmentation_result = None
    else:
        segmentation_result = util.load_image_data(args.segmentation_result, key=args.segmentation_key)

    if args.embedding_path is None:
        warnings.warn("You have not passed an embedding_path. Restarting the annotator may take a long time.")

    annotator_3d(
        raw, embedding_path=args.embedding_path,
        show_embeddings=args.show_embeddings, segmentation_result=segmentation_result,
        model_type=args.model_type, tile_shape=args.tile_shape, halo=args.halo,
    )
