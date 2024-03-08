import os
import warnings

from glob import glob
from pathlib import Path
from typing import List, Optional, Union, Tuple

import imageio.v3 as imageio
import napari
import torch.nn as nn

from magicgui import magicgui
from segment_anything import SamPredictor

from .. import util
from ..precompute_state import _precompute_state_for_files
from .annotator_2d import Annotator2d
from ._state import AnnotatorState


def _precompute(
    image_files, model_type, predictor, embedding_path,
    tile_shape, halo, precompute_amg_state, decoder,
):
    if predictor is None:
        predictor = util.get_sam_model(model_type=model_type)

    if embedding_path is None:
        embedding_paths = [None] * len(image_files)
    else:
        _precompute_state_for_files(
            predictor, image_files, embedding_path, ndim=2,
            tile_shape=tile_shape, halo=halo,
            precompute_amg_state=precompute_amg_state,
            decoder=decoder,
        )
        embedding_paths = [
            os.path.join(embedding_path, f"{Path(path).stem}.zarr") for path in image_files
        ]
        assert all(os.path.exists(emb_path) for emb_path in embedding_paths)

    return predictor, embedding_paths


def image_series_annotator(
    image_files: List[Union[os.PathLike, str]],
    output_folder: str,
    model_type: str = util._DEFAULT_MODEL,
    embedding_path: Optional[str] = None,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    viewer: Optional["napari.viewer.Viewer"] = None,
    return_viewer: bool = False,
    predictor: Optional[SamPredictor] = None,
    decoder: Optional["nn.Module"] = None,
    precompute_amg_state: bool = False,
) -> Optional["napari.viewer.Viewer"]:
    """Run the 2d annotation tool for a series of images.

    Args:
        input_files: List of the file paths for the images to be annotated.
        output_folder: The folder where the segmentation results are saved.
        model_type: The Segment Anything model to use. For details on the available models check out
            https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#finetuned-models.
        embedding_path: Filepath where to save the embeddings.
        tile_shape: Shape of tiles for tiled embedding prediction.
            If `None` then the whole image is passed to Segment Anything.
        halo: Shape of the overlap between tiles, which is needed to segment objects on tile boarders.
        viewer: The viewer to which the SegmentAnything functionality should be added.
            This enables using a pre-initialized viewer.
        return_viewer: Whether to return the napari viewer to further modify it before starting the tool.
        predictor: The Segment Anything model. Passing this enables using fully custom models.
            If you pass `predictor` then `model_type` will be ignored.
        decoder: The instance segmentation decoder.
        precompute_amg_state: Whether to precompute the state for automatic mask generation.
            This will take more time when precomputing embeddings, but will then make
            automatic mask generation much faster.

    Returns:
        The napari viewer, only returned if `return_viewer=True`.
    """

    os.makedirs(output_folder, exist_ok=True)
    next_image_id = 0

    # Precompute embeddings and amg state (if corresponding options set).
    predictor, embedding_paths = _precompute(
        image_files, model_type, predictor,
        embedding_path, tile_shape, halo, precompute_amg_state,
        decoder=decoder,
    )

    # Load the first image and intialize the viewer, annotator and state.
    image = imageio.imread(image_files[next_image_id])
    image_embedding_path = embedding_paths[next_image_id]

    if viewer is None:
        viewer = napari.Viewer()
    viewer.add_image(image, name="image")

    state = AnnotatorState()
    state.decoder = decoder
    state.initialize_predictor(
        image, model_type=model_type, save_path=image_embedding_path,
        halo=halo, tile_shape=tile_shape, predictor=predictor, ndim=2,
        precompute_amg_state=precompute_amg_state,
    )
    state.image_shape = image.shape[:-1] if image.ndim == 3 else image.shape

    annotator = Annotator2d(viewer)
    annotator._update_image()

    viewer.window.add_dock_widget(annotator)

    def _save_segmentation(image_path, segmentation):
        fname = os.path.basename(image_path)
        fname = os.path.splitext(fname)[0] + ".tif"
        out_path = os.path.join(output_folder, fname)
        imageio.imwrite(out_path, segmentation)

    # Add functionality for going to the next image.
    @magicgui(call_button="Next Image [N]")
    def next_image(*args):
        nonlocal next_image_id

        segmentation = viewer.layers["committed_objects"].data
        if segmentation.sum() == 0:
            print("Nothing is segmented yet. Not advancing to next image.")
            return

        # Save the current segmentation.
        _save_segmentation(image_files[next_image_id], segmentation)

        # Load the next image.
        next_image_id += 1
        if next_image_id == len(image_files):
            print("You have annotated the last image.")
            viewer.close()
            return

        print("Loading next image from:", image_files[next_image_id])
        image = imageio.imread(image_files[next_image_id])
        image_embedding_path = embedding_paths[next_image_id]

        # Set the new image in the viewer, state and annotator.
        viewer.layers["image"].data = image

        state.amg.clear_state()
        state.initialize_predictor(
            image, model_type=model_type, ndim=2, save_path=image_embedding_path,
            halo=halo, tile_shape=tile_shape, predictor=predictor,
            precompute_amg_state=precompute_amg_state,
        )
        state.image_shape = image.shape[:-1] if image.ndim == 3 else image.shape

        annotator._update_image()

    viewer.window.add_dock_widget(next_image)

    @viewer.bind_key("n", overwrite=True)
    def _next_image(viewer):
        next_image(viewer)

    if return_viewer:
        return viewer
    napari.run()


def image_folder_annotator(
    input_folder: str,
    output_folder: str,
    pattern: str = "*",
    viewer: Optional["napari.viewer.Viewer"] = None,
    return_viewer: bool = False,
    **kwargs
) -> Optional["napari.viewer.Viewer"]:
    """Run the 2d annotation tool for a series of images in a folder.

    Args:
        input_folder: The folder with the images to be annotated.
        output_folder: The folder where the segmentation results are saved.
        pattern: The glob patter for loading files from `input_folder`.
            By default all files will be loaded.
        viewer: The viewer to which the SegmentAnything functionality should be added.
            This enables using a pre-initialized viewer.
        return_viewer: Whether to return the napari viewer to further modify it before starting the tool.
        kwargs: The keyword arguments for `micro_sam.sam_annotator.image_series_annotator`.

    Returns:
        The napari viewer, only returned if `return_viewer=True`.
    """
    image_files = sorted(glob(os.path.join(input_folder, pattern)))
    return image_series_annotator(
        image_files, output_folder, viewer=viewer, return_viewer=return_viewer, **kwargs
    )


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
