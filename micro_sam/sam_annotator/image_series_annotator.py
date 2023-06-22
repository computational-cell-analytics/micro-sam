import os
from glob import glob

import imageio.v3 as imageio
import napari

from magicgui import magicgui
from .annotator_2d import annotator_2d
from .. import util


# TODO implement this
def precompute_embeddings_for_image_series(predictor, image_files):
    pass


def image_series_annotator(image_files, output_folder, embedding_path=None, **kwargs):
    # make sure we don't set incompatible kwargs
    assert kwargs.get("show_embeddings", False) is False
    assert kwargs.get("segmentation_results", None) is None
    assert "return_viewer" not in kwargs
    assert "v" not in kwargs

    os.makedirs(output_folder, exist_ok=True)
    next_image_id = 0

    predictor = util.get_sam_model(model_type=kwargs.get("model_type", "vit_h"))
    if embedding_path is None:
        embedding_paths = None
    else:
        embedding_paths = precompute_embeddings_for_image_series(predictor, image_files)

    def _save_segmentation(image_path, segmentation):
        fname = os.path.basename(image_path)
        fname = os.path.splitext(fname)[0] + ".tif"
        out_path = os.path.join(output_folder, fname)
        imageio.imwrite(out_path, segmentation)

    image = imageio.imread(image_files[next_image_id])
    image_embedding_path = None if embedding_paths is None else embedding_paths[next_image_id]
    v = annotator_2d(image, embedding_path=image_embedding_path, return_viewer=True, predictor=predictor, **kwargs)

    @magicgui(call_button="Next Image [N]")
    def next_image(*args):
        nonlocal next_image_id

        segmentation = v.layers["committed_objects"].data
        if segmentation.sum() == 0:
            print("Nothing is segmented yet, skipping next image.")
            return

        # save the current segmentation
        _save_segmentation(image_files[next_image_id], segmentation)

        # load the next image
        next_image_id += 1
        print("Loading next image from:", image_files[next_image_id])
        image = imageio.imread(image_files[next_image_id])
        image_embedding_path = None if embedding_paths is None else embedding_paths[next_image_id]
        annotator_2d(image, embedding_path=image_embedding_path, v=v, return_viewer=True, predictor=predictor, **kwargs)

    v.window.add_dock_widget(next_image)

    @v.bind_key("n")
    def _next_image(v):
        next_image(v)

    napari.run()


def image_folder_annotator(root_folder, output_folder, pattern="*", embedding_path=None, **kwargs):
    image_files = sorted(glob(os.path.join(root_folder, pattern)))
    image_series_annotator(image_files, output_folder, embedding_path, **kwargs)


# TODO implement the CLI
def main():
    import argparse
    image_folder_annotator()
