# Example for a small application implemented using napari and the micro_sam library:
# Iterate over a series of images in a folder and provide annotations with SAM.

import os
from glob import glob

import imageio
import micro_sam.util as util
import napari
import numpy as np

from magicgui import magicgui
from micro_sam.segment_from_prompts import segment_from_points
from micro_sam.sam_annotator.util import create_prompt_menu, prompt_layer_to_points
from napari import Viewer


@magicgui(call_button="Segment Object [S]")
def segment_wigdet(v: Viewer):
    points, labels = prompt_layer_to_points(v.layers["prompts"])
    seg = segment_from_points(PREDICTOR, points, labels)
    v.layers["segmented_object"].data = seg.squeeze()
    v.layers["segmented_object"].refresh()


def image_series_annotator(image_paths, embedding_save_path, output_folder):
    global PREDICTOR

    os.makedirs(output_folder, exist_ok=True)

    # get the sam predictor and precompute the image embeddings
    PREDICTOR = util.get_sam_model()
    images = np.stack([imageio.imread(p) for p in image_paths])
    image_embeddings = util.precompute_image_embeddings(PREDICTOR, images, save_path=embedding_save_path)
    util.set_precomputed(PREDICTOR, image_embeddings, i=0)

    v = napari.Viewer()

    # add the first image
    next_image_id = 0
    v.add_image(images[0], name="image")

    # add a layer for the segmented object
    v.add_labels(data=np.zeros(images.shape[1:], dtype="uint32"), name="segmented_object")

    # create the point layer for the sam prompts and add the widget for toggling the points
    labels = ["positive", "negative"]
    prompts = v.add_points(
        data=[[0.0, 0.0], [0.0, 0.0]],  # FIXME workaround
        name="prompts",
        properties={"label": labels},
        edge_color="label",
        edge_color_cycle=["green", "red"],
        symbol="o",
        face_color="transparent",
        edge_width=0.5,
        size=12,
        ndim=2,
    )
    prompts.data = []
    prompts.edge_color_mode = "cycle"
    prompt_widget = create_prompt_menu(prompts, labels)
    v.window.add_dock_widget(prompt_widget)

    # toggle the points between positive / negative
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

    # bind the segmentation to a key 's'
    @v.bind_key("s")
    def _segmet(v):
        segment_wigdet(v)

    #
    # the functionality for saving segmentations and going to the next image
    #

    def _save_segmentation(seg, output_folder, image_path):
        fname = os.path.basename(image_path)
        save_path = os.path.join(output_folder, os.path.splitext(fname)[0] + ".tif")
        imageio.imwrite(save_path, seg)

    def _next(v):
        nonlocal next_image_id
        v.layers["image"].data = images[next_image_id]
        util.set_precomputed(PREDICTOR, image_embeddings, i=next_image_id)

        v.layers["segmented_object"].data = np.zeros(images[0].shape, dtype="uint32")
        v.layers["prompts"].data = []

        next_image_id += 1
        if next_image_id >= images.shape[0]:
            print("Last image!")

    @v.bind_key("n")
    def next_image(v):
        seg = v.layers["segmented_object"].data
        if seg.max() == 0:
            print("This image has not been segmented yet, doing nothing!")
            return

        _save_segmentation(seg, output_folder, image_paths[next_image_id - 1])
        _next(v)

    napari.run()


# this uses data from the cell tracking challenge as example data
# see 'sam_annotator_tracking' for examples
def main():
    image_paths = sorted(glob("./data/DIC-C2DH-HeLa/train/01/*.tif"))[:50]
    image_series_annotator(image_paths, "./embeddings/embeddings-ctc.zarr", "segmented-series")


if __name__ == "__main__":
    main()
