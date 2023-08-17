import os
import warnings

from glob import glob
from pathlib import Path
from typing import List, Optional, Union

import imageio.v3 as imageio
import napari

from magicgui import magicgui
from segment_anything import SamPredictor

from .. import util
from ..precompute_state import _precompute_state_for_files
from .annotator_2d import annotator_2d


def image_series_annotator(
    image_files: List[Union[os.PathLike, str]],
    output_folder: str,
    embedding_path: Optional[str] = None,
    predictor: Optional[SamPredictor] = None,
    **kwargs
) -> None:
    """Run the 2d annotation tool for a series of images.

    Args:
        input_files: List of the file paths for the images to be annotated.
        output_folder: The folder where the segmentation results are saved.
        pattern: The glob patter for loading files from `input_folder`.
            By default all files will be loaded.
        embedding_path: Filepath where to save the embeddings.
        predictor: The Segment Anything model. Passing this enables using fully custom models.
            If you pass `predictor` then `model_type` will be ignored.
        kwargs: The keywored arguments for `micro_sam.sam_annotator.annotator_2d`.
    """
    # make sure we don't set incompatible kwargs
    assert kwargs.get("show_embeddings", False) is False
    assert kwargs.get("segmentation_results", None) is None
    assert "return_viewer" not in kwargs
    assert "v" not in kwargs

    os.makedirs(output_folder, exist_ok=True)
    next_image_id = 0

    if predictor is None:
        predictor = util.get_sam_model(model_type=kwargs.get("model_type", util._DEFAULT_MODEL))
    if embedding_path is None:
        embedding_paths = None
    else:
        _precompute_state_for_files(
            predictor, image_files, embedding_path, ndim=2,
            tile_shape=kwargs.get("tile_shape", None),
            halo=kwargs.get("halo", None),
            precompute_amg_state=kwargs.get("precompute_amg_state", False),
        )
        embedding_paths = [
            os.path.join(embedding_path, f"{Path(path).stem}.zarr") for path in image_files
        ]
        assert all(os.path.exists(emb_path) for emb_path in embedding_paths)

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
        if next_image_id == len(image_files):
            print("You have annotated the last image.")
            v.close()
            return

        print("Loading next image from:", image_files[next_image_id])
        image = imageio.imread(image_files[next_image_id])
        image_embedding_path = None if embedding_paths is None else embedding_paths[next_image_id]
        annotator_2d(image, embedding_path=image_embedding_path, v=v, return_viewer=True, predictor=predictor, **kwargs)

    v.window.add_dock_widget(next_image)

    @v.bind_key("n")
    def _next_image(v):
        next_image(v)

    napari.run()


def image_folder_annotator(
    input_folder: str,
    output_folder: str,
    pattern: str = "*",
    embedding_path: Optional[str] = None,
    predictor: Optional[SamPredictor] = None,
    **kwargs
) -> None:
    """Run the 2d annotation tool for a series of images in a folder.

    Args:
        input_folder: The folder with the images to be annotated.
        output_folder: The folder where the segmentation results are saved.
        pattern: The glob patter for loading files from `input_folder`.
            By default all files will be loaded.
        embedding_path: Filepath where to save the embeddings.
        predictor: The Segment Anything model. Passing this enables using fully custom models.
            If you pass `predictor` then `model_type` will be ignored.
        kwargs: The keywored arguments for `micro_sam.sam_annotator.annotator_2d`.
    """
    image_files = sorted(glob(os.path.join(input_folder, pattern)))
    image_series_annotator(image_files, output_folder, embedding_path, predictor, **kwargs)


def main():
    """@private"""
    import argparse

    available_models = list(util.get_model_names())
    available_models = ", ".join(available_models)

    parser = argparse.ArgumentParser(description="Annotate a series of images from a folder.")
    parser.add_argument(
        "-i", "--input_folder", required=True,
        help="The folder containing the image data. The data can be stored in any common format (tif, jpg, png, ...)."
    )
    parser.add_argument(
        "-o", "--output_folder", required=True,
        help="The folder where the segmentation results will be stored."
    )
    parser.add_argument(
        "-p", "--pattern", default="*",
        help="The pattern to select the images to annotator from the input folder. E.g. *.tif to annotate all tifs."
        "By default all files in the folder will be loaded and annotated."
    )
    parser.add_argument(
        "-e", "--embedding_path",
        help="The filepath for saving/loading the pre-computed image embeddings. "
        "NOTE: It is recommended to pass this argument and store the embeddings, "
        "otherwise they will be recomputed every time (which can take a long time)."
    )
    parser.add_argument(
        "--model_type", default=util._DEFAULT_MODEL,
        help=f"The segment anything model that will be used, one of {available_models}."
    )
    parser.add_argument(
        "--tile_shape", nargs="+", type=int, help="The tile shape for using tiled prediction", default=None
    )
    parser.add_argument(
        "--halo", nargs="+", type=int, help="The halo for using tiled prediction", default=None
    )
    parser.add_argument("--precompute_amg_state", action="store_true")

    args = parser.parse_args()

    if args.embedding_path is None:
        warnings.warn("You have not passed an embedding_path. Restarting the annotator may take a long time.")

    image_folder_annotator(
        args.input_folder, args.output_folder, args.pattern,
        embedding_path=args.embedding_path, model_type=args.model_type,
        tile_shape=args.tile_shape, halo=args.halo,
        precompute_amg_state=args.precompute_amg_state,
    )
