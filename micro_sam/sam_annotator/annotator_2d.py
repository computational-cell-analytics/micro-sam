# TODO make napari imports optional so we can use micro_sam as pure library
import napari
import numpy as np

from magicgui import magicgui
from napari import Viewer

from .. import util
from ..visualization import compute_pca
from ..segment_from_prompts import segment_from_points
from .util import create_prompt_menu, prompt_layer_to_points

COLOR_CYCLE = ["#00FF00", "#FF0000"]


@magicgui(call_button="Segment Object [S]")
def segment_wigdet(v: Viewer):
    points, labels = prompt_layer_to_points(v.layers["prompts"])
    seg = segment_from_points(PREDICTOR, points, labels)
    v.layers["current_object"].data = seg.squeeze()
    v.layers["current_object"].refresh()


def annotator_2d(raw, embedding_path=None, show_embeddings=False):
    # for access to the predictor and the image embeddings in the widgets
    global PREDICTOR, NEXT_ID
    NEXT_ID = 1

    _, PREDICTOR = util.get_sam_model()
    image_embeddings = util.precompute_image_embeddings(PREDICTOR, raw, save_path=embedding_path)
    util.set_precomputed(PREDICTOR, image_embeddings)

    #
    # initialize the viewer and add layers
    #

    v = Viewer()

    v.add_image(raw)
    v.add_labels(data=np.zeros(raw.shape, dtype="uint32"), name="committed_objects")
    v.add_labels(data=np.zeros(raw.shape, dtype="uint32"), name="current_object")

    # show the PCA of the image embeddings
    if show_embeddings:
        embedding_vis = compute_pca(image_embeddings["features"])
        # FIXME we need to set the scale from data
        v.add_image(embedding_vis, name="embeddings", scale=(8, 8))

    labels = ["positive", "negative"]
    prompts = v.add_points(
        data=[[0.0, 0.0], [0.0, 0.0]],  # FIXME workaround
        name="prompts",
        properties={"label": labels},
        edge_color="label",
        edge_color_cycle=COLOR_CYCLE,
        symbol="o",
        face_color="transparent",
        edge_width=0.5,
        size=12,
        ndim=2,
    )
    prompts.edge_color_mode = "cycle"

    #
    # add the widgets
    #

    # TODO add (optional) auto-segmentation functionality

    prompt_widget = create_prompt_menu(prompts, labels)
    v.window.add_dock_widget(prompt_widget)

    v.window.add_dock_widget(segment_wigdet)

    #
    # start the viewer
    #
    #
    # key bindings
    #

    @v.bind_key("s")
    def _segmet(v):
        segment_wigdet(v)

    # @v.bind_key("c")
    # def _commit(v):
    #     commit_widget(v)

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

    # clear the initial points needed for workaround
    clear_prompts(v)
    napari.run()
