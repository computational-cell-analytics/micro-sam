import napari
import numpy as np
from napari import Viewer
from magicgui import magicgui

from .annotator_2d import _get_shape
from ..import util
from ..import segment_instances
from ..visualization import project_embeddings_for_visualization


@magicgui(call_button="Automatic Segmentation [S]")
def autosegment_widget(
    v: Viewer,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    min_initial_size: int = 10,
    box_extension: float = 0.1,
    with_background: bool = True,
    use_box: bool = True,
    use_mask: bool = True,
    use_points: bool = False,
):
    is_tiled = IMAGE_EMBEDDINGS["input_size"] is None
    if is_tiled:
        seg, initial_seg = segment_instances.segment_instances_from_embeddings_with_tiling(
            PREDICTOR, IMAGE_EMBEDDINGS, with_background=with_background,
            box_extension=box_extension, pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_initial_size=min_initial_size, return_initial_segmentation=True, verbose=2,
            use_box=use_box, use_mask=use_mask, use_points=use_points,
        )
    else:
        seg, initial_seg = segment_instances.segment_instances_from_embeddings(
            PREDICTOR, IMAGE_EMBEDDINGS, with_background=with_background,
            box_extension=box_extension, pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_initial_size=min_initial_size, return_initial_segmentation=True, verbose=2,
            use_box=use_box, use_mask=use_mask, use_points=use_points,
        )

    v.layers["auto_segmentation"].data = seg
    v.layers["auto_segmentation"].refresh()

    v.layers["initial_segmentation"].data = initial_seg
    v.layers["initial_segmentation"].refresh()


def interactive_instance_segmentation(
    raw, embedding_path=None, model_type="vit_h", tile_shape=None, halo=None, checkpoint=None,
):
    """Visualizing and debugging automatic instance segmentation.
    """
    global PREDICTOR, IMAGE_EMBEDDINGS

    PREDICTOR = util.get_sam_model(model_type=model_type, checkpoint_path=checkpoint)
    IMAGE_EMBEDDINGS = util.precompute_image_embeddings(
        PREDICTOR, raw, save_path=embedding_path, ndim=2, tile_shape=tile_shape, halo=halo,
    )

    shape = _get_shape(raw)

    v = napari.Viewer()

    v.add_image(raw)
    embedding_vis, scale = project_embeddings_for_visualization(IMAGE_EMBEDDINGS)
    v.add_image(embedding_vis, name="embeddings", scale=scale, visible=False)
    v.add_labels(data=np.zeros(shape, dtype="uint32"), name="initial_segmentation")
    v.add_labels(data=np.zeros(shape, dtype="uint32"), name="auto_segmentation")

    v.window.add_dock_widget(autosegment_widget)

    napari.run()
