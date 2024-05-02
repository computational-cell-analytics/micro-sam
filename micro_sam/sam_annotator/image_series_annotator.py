import os

from glob import glob
from pathlib import Path
from typing import List, Optional, Union, Tuple

import numpy as np
import imageio.v3 as imageio
import napari
import torch

from magicgui import magicgui
from qtpy import QtWidgets

from .. import util
from . import _widgets as widgets
from ..precompute_state import _precompute_state_for_files
from ..instance_segmentation import get_decoder
from .annotator_2d import Annotator2d
from .annotator_3d import Annotator3d
from ._state import AnnotatorState
from .util import _sync_embedding_widget
from ._tooltips import get_tooltip


def _precompute(
    images, model_type, embedding_path,
    tile_shape, halo, precompute_amg_state,
    checkpoint_path, device, ndim, prefer_decoder,
):
    device = util.get_device(device)
    predictor, state = util.get_sam_model(
        model_type=model_type, checkpoint_path=checkpoint_path, device=device, return_state=True
    )
    if prefer_decoder and "decoder_state" in state:
        decoder = get_decoder(predictor.model.image_encoder, state["decoder_state"], device)
    else:
        decoder = None

    if embedding_path is None:
        embedding_paths = [None] * len(images)
    else:
        _precompute_state_for_files(
            predictor, images, embedding_path, ndim=ndim, tile_shape=tile_shape, halo=halo,
            precompute_amg_state=precompute_amg_state, decoder=decoder,
        )
        if isinstance(images[0], np.ndarray):
            embedding_paths = [
                os.path.join(embedding_path, f"embedding_{i:05}.zarr") for i, path in enumerate(images)
            ]
        else:
            embedding_paths = [
                os.path.join(embedding_path, f"{Path(path).stem}.zarr") for path in images
            ]
        assert all(os.path.exists(emb_path) for emb_path in embedding_paths)

    return predictor, decoder, embedding_paths


def _get_input_shape(image, is_volumetric=False):
    if image.ndim == 2:
        image_shape = image.shape
    elif image.ndim == 3:
        if is_volumetric:
            image_shape = image.shape
        else:
            image_shape = image.shape[:-1]
    elif image.ndim == 4:
        image_shape = image.shape[:-1]

    return image_shape


def image_series_annotator(
    images: Union[List[Union[os.PathLike, str]], List[np.ndarray]],
    output_folder: str,
    model_type: str = util._DEFAULT_MODEL,
    embedding_path: Optional[str] = None,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    viewer: Optional["napari.viewer.Viewer"] = None,
    return_viewer: bool = False,
    precompute_amg_state: bool = False,
    checkpoint_path: Optional[str] = None,
    is_volumetric: bool = False,
    device: Optional[Union[str, torch.device]] = None,
    prefer_decoder: bool = True,
) -> Optional["napari.viewer.Viewer"]:
    """Run the annotation tool for a series of images (supported for both 2d and 3d images).

    Args:
        images: List of the file paths or list of (set of) slices for the images to be annotated.
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
        precompute_amg_state: Whether to precompute the state for automatic mask generation.
            This will take more time when precomputing embeddings, but will then make
            automatic mask generation much faster.
        checkpoint_path: Path to a custom checkpoint from which to load the SAM model.
        is_volumetric: Whether to use the 3d annotator.
        prefer_decoder: Whether to use decoder based instance segmentation if
            the model used has an additional decoder for instance segmentation.

    Returns:
        The napari viewer, only returned if `return_viewer=True`.
    """

    os.makedirs(output_folder, exist_ok=True)
    next_image_id = 0

    # Precompute embeddings and amg state (if corresponding options set).
    predictor, decoder, embedding_paths = _precompute(
        images, model_type,
        embedding_path, tile_shape, halo, precompute_amg_state,
        checkpoint_path=checkpoint_path, device=device,
        ndim=3 if is_volumetric else 2, prefer_decoder=prefer_decoder,
    )

    # Load the first image and intialize the viewer, annotator and state.
    if isinstance(images[next_image_id], np.ndarray):
        image = images[next_image_id]
        have_inputs_as_arrays = True
    else:
        image = imageio.imread(images[next_image_id])
        have_inputs_as_arrays = False

    image_embedding_path = embedding_paths[next_image_id]

    if viewer is None:
        viewer = napari.Viewer()
    viewer.add_image(image, name="image")

    state = AnnotatorState()
    state.initialize_predictor(
        image, model_type=model_type, save_path=image_embedding_path, halo=halo, tile_shape=tile_shape,
        predictor=predictor, decoder=decoder,
        ndim=3 if is_volumetric else 2, precompute_amg_state=precompute_amg_state,
        checkpoint_path=checkpoint_path, device=device,
    )
    state.image_shape = _get_input_shape(image, is_volumetric)

    if is_volumetric:
        if image.ndim not in [3, 4]:
            raise ValueError(f"Invalid image dimensions for 3d annotator, expect 3 or 4 dimensions, got {image.ndim}")
        annotator = Annotator3d(viewer)
    else:
        if image.ndim not in (2, 3):
            raise ValueError(f"Invalid image dimensions for 2d annotator, expect 2 or 3 dimensions, got {image.ndim}")
        annotator = Annotator2d(viewer)

    annotator._update_image()

    # Add the annotator widget to the viewer and sync widgets.
    viewer.window.add_dock_widget(annotator)
    _sync_embedding_widget(
        state.widgets["embeddings"], model_type,
        save_path=embedding_path, checkpoint_path=checkpoint_path,
        device=device, tile_shape=tile_shape, halo=halo
    )

    def _save_segmentation(image_path, current_idx, segmentation):
        if have_inputs_as_arrays:
            fname = f"seg_{current_idx:05}.tif"
        else:
            fname = os.path.basename(image_path)
            fname = os.path.splitext(fname)[0] + ".tif"

        out_path = os.path.join(output_folder, fname)
        imageio.imwrite(out_path, segmentation)

    # Add functionality for going to the next image.
    @magicgui(call_button="Next Image [N]")
    def next_image(*args):
        nonlocal next_image_id

        segmentation = viewer.layers["committed_objects"].data
        abort = False
        if segmentation.sum() == 0:
            msg = "Nothing is segmented yet. Do you wish to continue to the next image?"
            abort = widgets._generate_message("info", msg)
            if abort:
                return

        # Save the current segmentation.
        _save_segmentation(images[next_image_id], next_image_id, segmentation)

        # Clear the segmentation already to avoid lagging removal.
        viewer.layers["committed_objects"].data = np.zeros_like(viewer.layers["committed_objects"].data)

        # Load the next image.
        next_image_id += 1
        if next_image_id == len(images):
            msg = "You have annotated the last image. Do you wish to close napari?"
            print(msg)
            abort = False
            # inform the user via dialog
            abort = widgets._generate_message("info", msg)
            if not abort:
                viewer.close()
            return

        print(
            "Loading next image:",
            images[next_image_id] if not have_inputs_as_arrays else f"at index {next_image_id}"
        )

        if have_inputs_as_arrays:
            image = images[next_image_id]
        else:
            image = imageio.imread(images[next_image_id])

        image_embedding_path = embedding_paths[next_image_id]

        # Set the new image in the viewer, state and annotator.
        viewer.layers["image"].data = image

        if state.amg is not None:
            state.amg.clear_state()
        state.initialize_predictor(
            image, model_type=model_type, ndim=3 if is_volumetric else 2,
            save_path=image_embedding_path,
            tile_shape=tile_shape, halo=halo,
            predictor=predictor, decoder=decoder,
            precompute_amg_state=precompute_amg_state, device=device,
        )
        state.image_shape = _get_input_shape(image, is_volumetric)

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


class ImageSeriesAnnotator(widgets._WidgetBase):
    def __init__(self, viewer: napari.Viewer, parent=None):
        super().__init__(parent=parent)
        self._viewer = viewer

        # Create the UI: the general options.
        self._create_options()

        # Add the settings (collapsible).
        self.layout().addWidget(self._create_settings())

        # Add the run button to trigger the embedding computation.
        self.run_button = QtWidgets.QPushButton("Annotate Images")
        self.run_button.clicked.connect(self.__call__)
        self.layout().addWidget(self.run_button)

    # model_type: str = util._DEFAULT_MODEL,
    def _create_options(self):
        self.folder = None
        _, layout = self._add_path_param(
            "folder", self.folder, "directory",
            title="Input Folder", placeholder="Folder with images ...",
            tooltip=get_tooltip("image_series_annotator", "folder")
        )
        self.layout().addLayout(layout)

        self.output_folder = None
        _, layout = self._add_path_param(
            "output_folder", self.output_folder, "directory",
            title="Output Folder", placeholder="Folder to save the results ...",
            tooltip=get_tooltip("image_series_annotator", "output_folder")
        )
        self.layout().addLayout(layout)

        self.model_type = util._DEFAULT_MODEL
        model_options = list(util.models().urls.keys())
        model_options = [model for model in model_options if not model.endswith("decoder")]
        _, layout = self._add_choice_param(
            "model_type", self.model_type, model_options, title="Model:",
            tooltip=get_tooltip("embedding", "model")
        )
        self.layout().addLayout(layout)

    def _create_settings(self):
        setting_values = QtWidgets.QWidget()
        setting_values.setLayout(QtWidgets.QVBoxLayout())

        self.pattern = "*"
        _, layout = self._add_string_param(
            "pattern", self.pattern, tooltip=get_tooltip("image_series_annotator", "pattern")
        )
        setting_values.layout().addLayout(layout)

        self.is_volumetric = False
        setting_values.layout().addWidget(self._add_boolean_param(
            "is_volumetric", self.is_volumetric, tooltip=get_tooltip("image_series_annotator", "is_volumetric")
        ))

        self.device = "auto"
        device_options = ["auto"] + util._available_devices()
        self.device_dropdown, layout = self._add_choice_param(
            "device", self.device, device_options, tooltip=get_tooltip("embedding", "device")
        )
        setting_values.layout().addLayout(layout)

        self.embeddings_save_path = None
        _, layout = self._add_path_param(
            "embeddings_save_path", self.embeddings_save_path, "directory", title="embeddings save path:",
            tooltip=get_tooltip("embedding", "embeddings_save_path")
        )
        setting_values.layout().addLayout(layout)

        self.custom_weights = None  # select_file
        _, layout = self._add_path_param(
            "custom_weights", self.custom_weights, "file", title="custom weights path:",
            tooltip=get_tooltip("embedding", "custom_weights")
        )
        setting_values.layout().addLayout(layout)

        self.tile_x, self.tile_y = 0, 0
        self.tile_x_param, self.tile_y_param, layout = self._add_shape_param(
            ("tile_x", "tile_y"), (self.tile_x, self.tile_y), min_val=0, max_val=2048, step=16,
            tooltip=get_tooltip("embedding", "tiling")
        )
        setting_values.layout().addLayout(layout)

        self.halo_x, self.halo_y = 0, 0
        self.halo_x_param, self.halo_y_param, layout = self._add_shape_param(
            ("halo_x", "halo_y"), (self.halo_x, self.halo_y), min_val=0, max_val=512,
            tooltip=get_tooltip("embedding", "halo")
        )
        setting_values.layout().addLayout(layout)

        settings = widgets._make_collapsible(setting_values, title="Advanced Settings")
        return settings

    def _validate_inputs(self):
        missing_data = self.folder is None or len(glob(os.path.join(self.folder, self.pattern))) == 0
        missing_output = self.output_folder is None
        if missing_data or missing_output:
            msg = ""
            if missing_data:
                msg += "The input folder is missing or empty. "
            if missing_output:
                msg += "The output folder is missing."
            return widgets._generate_message("error", msg)
        return False

    def __call__(self, skip_validate=False):
        if not skip_validate and self._validate_inputs():
            return
        tile_shape, halo = widgets._process_tiling_inputs(self.tile_x, self.tile_y, self.halo_x, self.halo_y)

        image_folder_annotator(
            self.folder, self.output_folder, self.pattern,
            model_type=self.model_type,
            embedding_path=self.embeddings_save_path,
            tile_shape=tile_shape, halo=halo, checkpoint_path=self.custom_weights,
            device=self.device, is_volumetric=self.is_volumetric,
            viewer=self._viewer, return_viewer=True,
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
        "-m", "--model_type", default=util._DEFAULT_MODEL,
        help=f"The segment anything model that will be used, one of {available_models}."
    )
    parser.add_argument(
        "-c", "--checkpoint", default=None,
        help="Checkpoint from which the SAM model will be loaded loaded."
    )
    parser.add_argument(
        "-d", "--device", default=None,
        help="The device to use for the predictor. Can be one of 'cuda', 'cpu' or 'mps' (only MAC)."
        "By default the most performant available device will be selected."
    )
    parser.add_argument(
        "--is_volumetric", action="store_true", help="Whether to use the 3d annotator for a set of 3d volumes."
    )

    parser.add_argument(
        "--tile_shape", nargs="+", type=int, help="The tile shape for using tiled prediction", default=None
    )
    parser.add_argument(
        "--halo", nargs="+", type=int, help="The halo for using tiled prediction", default=None
    )
    parser.add_argument("--precompute_amg_state", action="store_true")
    parser.add_argument("--prefer_decoder", action="store_false")

    args = parser.parse_args()

    image_folder_annotator(
        args.input_folder, args.output_folder, args.pattern,
        embedding_path=args.embedding_path, model_type=args.model_type,
        tile_shape=args.tile_shape, halo=args.halo, precompute_amg_state=args.precompute_amg_state,
        checkpoint_path=args.checkpoint, device=args.device, is_volumetric=args.is_volumetric,
        prefer_decoder=args.prefer_decoder,
    )
