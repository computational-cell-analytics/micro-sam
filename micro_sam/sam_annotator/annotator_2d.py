import napari
import numpy as np

from magicgui import magicgui
from napari import Viewer

from .. import util
from .. import segment_instances
from ..visualization import project_embeddings_for_visualization
from ..segment_from_prompts import segment_from_box, segment_from_box_and_points, segment_from_points
from .util import (
    commit_segmentation_widget, create_prompt_menu, prompt_layer_to_points, toggle_label, LABEL_COLOR_CYCLE
)


@magicgui(call_button="Segment Object [S]")
def segment_wigdet(v: Viewer):
    # get the current point prompts
    points, labels = prompt_layer_to_points(v.layers["prompts"])
    assert len(points) == len(labels)
    have_points = len(points) > 0

    # get the current box prompts
    box_layer = v.layers["box_prompts"]
    have_boxes = box_layer.nshapes > 0

    # segment only with points
    if have_points and not have_boxes:
        seg = segment_from_points(PREDICTOR, points, labels).squeeze()

    # segment only with boxes
    elif not have_points and have_boxes:
        shape = v.layers["current_object"].data.shape
        seg = np.zeros(shape, dtype="uint32")

        seg_id = 1
        for prompt_id in range(box_layer.nshapes):
            shape_type = box_layer.shape_type[prompt_id]

            # for now we only support segmentation from rectangles.
            # supporting other shapes would be possible by casting the shape to a mask
            # and then segmenting from mask and bounding box.
            # but for this we need to fix issue with resizing the mask for non-square shapes.
            if shape_type != "rectangle":
                print(f"You have provided a {shape_type} shape.")
                print("We currently only support rectangle shapes for prompts and this prompt will be skipped.")
                continue

            box = box_layer.data[prompt_id]
            prompt_box = np.array([box[:, 0].min(), box[:, 1].min(), box[:, 0].max(), box[:, 1].max()])
            mask = segment_from_box(PREDICTOR, prompt_box).squeeze()
            seg[mask] = seg_id
            seg_id += 1

    # segment with points and box (currently only one box supported)
    elif have_points and have_boxes:
        if box_layer.nshapes > 1:
            print("You have provided point prompts and more than one box prompt.")
            print("This setting is currently not supported.")
            print("When providing both points and prompts you can only segment one object at a time.")
            return

        box = box_layer.data[0]
        prompt_box = np.array([box[:, 0].min(), box[:, 1].min(), box[:, 0].max(), box[:, 1].max()])
        seg = segment_from_box_and_points(PREDICTOR, prompt_box, points, labels).squeeze()

    # no prompts were given, skip segmentation
    else:
        print("You haven't given any prompts.")
        print("Please provide point and/or box prompts.")
        return

    v.layers["current_object"].data = seg
    v.layers["current_object"].refresh()


# TODO expose more parameters
@magicgui(call_button="Segment All Objects", method={"choices": ["default", "sam", "embeddings"]})
def autosegment_widget(v: Viewer, method: str = "default"):
    if method in ("default", "sam"):
        print("Run automatic segmentation with SAM. This can take a few minutes ...")
        image = v.layers["raw"].data
        seg = segment_instances.segment_instances_sam(SAM, image)
    elif method == "embeddings":
        seg = segment_instances.segment_instances_from_embeddings(PREDICTOR, IMAGE_EMBEDDINGS)
    else:
        raise ValueError
    v.layers["auto_segmentation"].data = seg
    v.layers["auto_segmentation"].refresh()


def annotator_2d(raw, embedding_path=None, show_embeddings=False, segmentation_result=None, model_type="vit_h"):
    # for access to the predictor and the image embeddings in the widgets
    global PREDICTOR, IMAGE_EMBEDDINGS, SAM

    PREDICTOR, SAM = util.get_sam_model(model_type=model_type, return_sam=True)
    IMAGE_EMBEDDINGS = util.precompute_image_embeddings(PREDICTOR, raw, save_path=embedding_path, ndim=2)
    util.set_precomputed(PREDICTOR, IMAGE_EMBEDDINGS)

    #
    # initialize the viewer and add layers
    #

    v = Viewer()

    v.add_image(raw)
    if raw.ndim == 2:
        shape = raw.shape
    elif raw.ndim == 3 and raw.shape[-1] == 3:
        shape = raw.shape[:2]
    else:
        raise ValueError(f"Invalid input image of shape {raw.shape}. Expect either 2D grayscale or 3D RGB image.")

    v.add_labels(data=np.zeros(shape, dtype="uint32"), name="auto_segmentation")
    if segmentation_result is None:
        v.add_labels(data=np.zeros(shape, dtype="uint32"), name="committed_objects")
    else:
        v.add_labels(segmentation_result, name="committed_objects")
    v.add_labels(data=np.zeros(shape, dtype="uint32"), name="current_object")

    # show the PCA of the image embeddings
    if show_embeddings:
        embedding_vis, scale = project_embeddings_for_visualization(IMAGE_EMBEDDINGS["features"], shape)
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

    box_prompts = v.add_shapes(
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
        prompts.data = []
        prompts.refresh()
        box_prompts.data = []
        box_prompts.refresh()

    #
    # start the viewer
    #

    # clear the initial points needed for workaround
    clear_prompts(v)
    napari.run()


def main():
    import argparse
    import warnings

    parser = argparse.ArgumentParser(
        description="Run interactive segmentation for an image."
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
    parser.add_argument(
        "-s", "--segmentation_result",
        help="Optional filepath to a precomputed segmentation. If passed this will be used to initialize the "
        "'committed_objects' layer. This can be useful if you want to correct an existing segmentation or if you "
        "have saved intermediate results from the annotator and want to continue with your annotations. "
        "Supports the same file formats as 'input'."
    )
    parser.add_argument(
        "-sk", "--segmentation_key",
        help="The key for opening the segmentation data. Same rules as for 'key' apply."
    )
    parser.add_argument(
        "--show_embeddings", action="store_true",
        help="Visualize the embeddings computed by SegmentAnything. This can be helpful for debugging."
    )
    parser.add_argument(
        "--model_type", default="vit_h", help="The segment anything model that will be used, one of vit_h,l,b."
    )

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
        model_type=args.model_type,
    )
