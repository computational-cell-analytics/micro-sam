# Example for a small application implemented using napari and the micro_sam library:
# Iterate over a series of images in a folder and provide annotations with SAM.

import os
import argparse

from glob import glob

from skimage.io import imread, imsave
from skimage.color import rgb2gray
import micro_sam.util as util
import napari
import numpy as np

from magicgui import magicgui
from micro_sam.segment_from_prompts import segment_from_points
from micro_sam.sam_annotator.util import create_prompt_menu, prompt_layer_to_points, toggle_label
from napari import Viewer
from qtpy.QtWidgets import QPushButton


@magicgui(call_button="Segment Object [S]")
def segment_wigdet(v: Viewer):
    points, labels = prompt_layer_to_points(v.layers["prompts"])
    seg = segment_from_points(PREDICTOR, points, labels)
    v.layers["segmented_object"].data = seg.squeeze()
    v.layers["segmented_object"].refresh()


def increase_brush_size(v):
    """
    Change brush size in layer segmented object to correct the segmentation result manually
    """
    if v.layers.selection.active == v.layers["segmented_object"]:
        v.layers["segmented_object"].brush_size += 1


def decrease_brush_size(v):
    """
    Change brush size in layer segmented object to correct the segmentation result manually
    """
    if v.layers.selection.active == v.layers["segmented_object"]:
        current_size = v.layers["segmented_object"].brush_size
        if current_size > 1:
            v.layers["segmented_object"].brush_size -= 1


def activate_segmented_object():
    v.layers.selection.active = v.layers["segmented_object"]


def image_series_annotator(embedding_save_path, output_folder):
    global PREDICTOR, v, image_paths

    os.makedirs(output_folder, exist_ok=True)

    # get the sam predictor and precompute the image embeddings
    PREDICTOR = util.get_sam_model()
    image = imread(image_paths[0])

    if len(image.shape) == 3:
        image_embeddings = util.precompute_image_embeddings(PREDICTOR, rgb2gray(image))
    else:
        image_embeddings = util.precompute_image_embeddings(PREDICTOR, image)
    util.set_precomputed(PREDICTOR, image_embeddings)

    v = napari.Viewer()

    # add the first image
    next_image_id = 0
    v.add_image(image, name="image")

    # add a layer for the segmented object
    v.add_labels(data=np.zeros(image.shape[:2], dtype="uint32"), name="segmented_object")
    # default brush size
    v.layers["segmented_object"].brush_size = 100

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
    prompt_widget = create_prompt_menu(prompts, labels, viewer=v)
    v.window.add_dock_widget(prompt_widget)

    # toggle the points between positive / negative
    @v.bind_key("t")
    def _toggle_label(event=None):
        toggle_label(prompts)

    # bind the segmentation to a key 's'
    @v.bind_key("s")
    def _segment(v):
        segment_wigdet(v)

    #
    # the functionality for saving segmentations and going to the next image
    #

    # bind increase brush size to up arrow key
    @v.bind_key("Up")
    def _increase_brush_size(v):
        increase_brush_size(v)

    @v.bind_key("Down")
    def _decrease_brush_size(v):
        decrease_brush_size(v)

    def _save_segmentation(seg, output_folder, image_path):
        fname = os.path.basename(image_path)
        save_path = os.path.join(output_folder, os.path.splitext(fname)[0] + ".tif")
        imsave(save_path, seg)

    def _next(v):
        nonlocal next_image_id
        global image_path
        if next_image_id == 0:
            next_image_id += 1
        v.layers["image"].data = imread(image_paths[next_image_id])

        if len(v.layers["image"].data.shape) == 3:
            image_embeddings = util.precompute_image_embeddings(PREDICTOR, rgb2gray(v.layers["image"].data))
        else:
            image_embeddings = util.precompute_image_embeddings(PREDICTOR, v.layers["image"].data)
        util.set_precomputed(PREDICTOR, image_embeddings)

        v.layers["segmented_object"].data = np.zeros(v.layers["image"].data.shape[:2], dtype="uint32")
        v.layers["prompts"].data = []

        next_image_id += 1
        if next_image_id >= len(image_paths):
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

parser = argparse.ArgumentParser(description='Segmentation of cells images')
parser.add_argument('-p', '--path', metavar='N', type=str, help='Path to the folder containing the images to process')
parser.add_argument('-e', '--extension', type=str, help='Extension of images (default is tif)', default="tif")

args = parser.parse_args()


def main():
    global image_paths
    image_paths = sorted(glob(os.path.join(args.path, "*." + args.extension)))
    output = os.path.join(os.path.split(image_paths[0])[0], "masks")
    image_series_annotator("./embeddings/embeddings-ctc.zarr", output)


if __name__ == "__main__":
    main()
