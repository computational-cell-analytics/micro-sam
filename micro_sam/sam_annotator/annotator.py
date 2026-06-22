from typing import Optional, Tuple, Union

import napari
import numpy as np
import torch

from .. import util
from . import _widgets as widgets
from . import util as vutil
from ._annotator import _AnnotatorBase
from ._state import AnnotatorState
from .util import (
    _initialize_parser,
    _load_amg_state,
    _load_is_state,
    _sync_embedding_widget,
)


def detect_ndim(image: np.ndarray) -> int:
    """Auto-detect dimensionality from image shape.

    Args:
        image: The input image array.

    Returns:
        The detected number of spatial dimensions (2 or 3).

    Raises:
        ValueError: If the image shape is invalid or ambiguous.

    Rules:
        - ndim=2: (H, W) or (H, W, 3) for RGB
        - ndim=3: (Z, H, W) or (Z, H, W, 3) for RGB volumes
    """
    if image.ndim == 2:
        return 2
    elif image.ndim == 3:
        # RGB 2D vs grayscale 3D - assume last dimension is RGB if size is 3
        return 2 if image.shape[-1] == 3 else 3
    elif image.ndim == 4:
        if image.shape[-1] == 3:
            return 3  # RGB 3D volume
        raise ValueError(
            f"Invalid 4D shape: {image.shape}. Expected shape (Z, H, W, 3) for RGB volumes."
        )
    else:
        raise ValueError(
            f"Invalid image shape: {image.shape}. Expected 2D or 3D image."
        )


def detect_ndim_from_viewer(viewer: "napari.viewer.Viewer") -> int:
    """Detect the dimensionality from image layers already loaded in the viewer.

    Used when the annotator is launched as a napari plugin widget without an explicit
    ndim. Falls back to 2 when no image has been loaded yet, e.g. when the widget is
    opened from the napari Plugins menu before any image is added.

    Args:
        viewer: The napari viewer.

    Returns:
        The detected number of spatial dimensions (2 or 3).
    """
    image_layers = [layer for layer in viewer.layers if isinstance(layer, napari.layers.Image)]
    if image_layers:
        return detect_ndim(image_layers[0].data)
    return 2


class Annotator(_AnnotatorBase):
    """Unified annotator for 2D and 3D images.

    This class handles both 2D and 3D annotation, with dimensionality
    controlled by the `ndim` parameter or auto-detected from the image.
    """

    def _get_widgets(self):
        """Create the widgets for the segmentation annotator.

        The interactive segmentation widget merges the prompt menu, the segment and the
        clear controls into a single ndim-aware widget placed right after the embeddings.
        """
        with_decoder = AnnotatorState().decoder is not None
        return {
            "interactive": widgets.InteractiveSegmentationWidget(
                self._viewer, ndim=self._ndim, prompt_widget=self._prompt_widget,
            ),
            "autosegment": widgets.AutoSegmentWidget(
                self._viewer, with_decoder=with_decoder, volumetric=(self._ndim == 3),
            ),
            "commit": widgets.commit(),
        }

    def _create_keybindings(self):
        """Bind the keys to the merged interactive segmentation widget."""
        interactive = self._widgets["interactive"]

        @self._viewer.bind_key("s", overwrite=True)
        def _segment(viewer):
            interactive.segment(viewer)

        # We also need to over-write the keybindings for the prompt layers.
        # See https://github.com/napari/napari/issues/7302 for details.
        prompt_layer = self._viewer.layers["prompts"]
        point_prompt_layer = self._viewer.layers["point_prompts"]

        @prompt_layer.bind_key("s", overwrite=True)
        def _segment_prompts(event):
            interactive.segment(self._viewer)

        @point_prompt_layer.bind_key("s", overwrite=True)
        def _segment_point_prompts(event):
            interactive.segment(self._viewer)

        @self._viewer.bind_key("c", overwrite=True)
        def _commit(viewer):
            self._widgets["commit"](viewer)

        @self._viewer.bind_key("t", overwrite=True)
        def _toggle_label(event=None):
            vutil.toggle_label(self._point_prompt_layer)

        @self._viewer.bind_key("Shift-C", overwrite=True)
        def _clear_annotations(viewer):
            interactive.clear(viewer)

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        ndim: Optional[int] = None,
        reset_state: bool = True,
    ) -> None:
        """Create the annotator GUI.

        Args:
            viewer: The napari viewer.
            ndim: The number of spatial dimensions (2 or 3). If None, auto-detected from the image.
            reset_state: Whether to reset the annotator state.

        Raises:
            ValueError: If ndim is invalid or doesn't match the image shape.
        """
        # Auto-detect ndim when launched as a napari widget without an explicit ndim.
        # Detect from an already-loaded image layer, defaulting to 2D when none is present
        # (e.g. when the widget is opened from the napari Plugins menu before loading an image).
        if ndim is None:
            ndim = detect_ndim_from_viewer(viewer)

        # Validate ndim
        if ndim not in (2, 3):
            raise ValueError(f"Invalid ndim: {ndim}. Expected 2 or 3.")

        super().__init__(viewer=viewer, ndim=ndim)

        # Set the expected annotator class to the state.
        state = AnnotatorState()

        # Reset the state.
        if reset_state:
            state.reset_state()

        state.annotator = self

        # Rebuild the annotator if an image with a different dimensionality is selected as input.
        # This handles loading e.g. a 3D image after the widget was opened from the Plugins menu.
        self._embedding_widget.image_selection.changed.connect(self._on_image_selection_changed)

        # Normalize and align to any image that is already selected (e.g. opened from the
        # Plugins menu after an image was loaded, so no selection-changed event will fire).
        self._on_image_selection_changed()

    def _on_image_selection_changed(self, *args):
        """Normalize the selected image and rebuild the annotator if its dimensionality changed."""
        # Skip while we are replacing the image layer ourselves during normalization.
        if self._suppress_selection_rebuild:
            return
        image_layer = self._embedding_widget.image_selection.get_value()
        if image_layer is None:
            return
        # Squeeze singletons and map channels to RGB, replacing the image layer if needed,
        # so the image, segmentation and prompt layers all stay aligned.
        image_layer, ndim = self._maybe_normalize_image_layer(image_layer)
        if ndim != self._ndim:
            self._rebuild_for_ndim(ndim)

    def _update_image(self, segmentation_result=None):
        """Update the image and load AMG state for 3D."""
        super()._update_image(segmentation_result=segmentation_result)

        # Load the AMG state from the embedding path (3D only)
        if self._ndim == 3:
            state = AnnotatorState()
            if state.decoder is not None:
                state.amg_state = _load_is_state(state.embedding_path)
            else:
                state.amg_state = _load_amg_state(state.embedding_path)


def annotator(
    image: np.ndarray,
    *,
    ndim: Optional[int] = None,
    embedding_path: Optional[Union[str, util.ImageEmbeddings]] = None,
    segmentation_result: Optional[np.ndarray] = None,
    model_type: str = util._DEFAULT_MODEL,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    return_viewer: bool = False,
    viewer: Optional["napari.viewer.Viewer"] = None,
    precompute_amg_state: bool = False,
    checkpoint_path: Optional[str] = None,
    decoder_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    prefer_decoder: bool = True,
) -> Optional["napari.viewer.Viewer"]:
    """Start the annotation tool for a given image.

    Args:
        image: The image data (2D or 3D).
        ndim: The number of spatial dimensions (2 or 3). If None, auto-detected from image shape.
        embedding_path: Filepath where to save the embeddings
            or the precompted image embeddings computed by `precompute_image_embeddings`.
        segmentation_result: An initial segmentation to load.
            This can be used to correct segmentations with Segment Anything or to save and load progress.
            The segmentation will be loaded as the 'committed_objects' layer.
        model_type: The Segment Anything model to use. For details on the available models check out
            https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#finetuned-models.
        tile_shape: Shape of tiles for tiled embedding prediction.
            If `None` then the whole image is passed to Segment Anything.
        halo: Shape of the overlap between tiles, which is needed to segment objects on tile borders.
        return_viewer: Whether to return the napari viewer to further modify it before starting the tool.
            By default, does not return the napari viewer.
        viewer: The viewer to which the Segment Anything functionality should be added.
            This enables using a pre-initialized viewer.
        precompute_amg_state: Whether to precompute the state for automatic mask generation.
            This will take more time when precomputing embeddings, but will then make
            automatic mask generation much faster. By default, set to 'False'.
        checkpoint_path: Path to a custom checkpoint from which to load the SAM model.
        decoder_path: Path to a custom decoder checkpoint from which to load the `micro-sam` decoder.
        device: The computational device to use for the SAM model.
            By default, automatically chooses the best available device.
        prefer_decoder: Whether to use decoder based instance segmentation if
            the model used has an additional decoder for instance segmentation.
            By default, set to 'True'.

    Returns:
        The napari viewer, only returned if `return_viewer=True`.

    Raises:
        ValueError: If ndim is invalid or doesn't match the image shape.
    """
    # Normalize the image: squeeze singletons and map the channel axis to RGB. This keeps
    # the dimensionality decision consistent with the GUI and the layer shapes aligned.
    image, detected_ndim, rgb = vutil.prepare_annotation_image(image)

    # Auto-detect ndim if not provided
    if ndim is None:
        ndim = detected_ndim

    # Validate ndim
    if ndim not in (2, 3):
        raise ValueError(f"Invalid ndim: {ndim}. Expected 2 or 3.")

    # Validate ndim matches image shape
    if ndim != detected_ndim:
        raise ValueError(
            f"Provided ndim={ndim} does not match detected ndim={detected_ndim} from image shape {image.shape}."
        )

    # Extract image shape (strip RGB channel if present)
    state = AnnotatorState()
    state.image_shape = image.shape[:-1] if rgb else image.shape

    # Initialize the predictor state
    state.initialize_predictor(
        image,
        model_type=model_type,
        save_path=embedding_path,
        halo=halo,
        tile_shape=tile_shape,
        precompute_amg_state=precompute_amg_state,
        ndim=ndim,
        checkpoint_path=checkpoint_path,
        decoder_path=decoder_path,
        device=device,
        prefer_decoder=prefer_decoder,
        skip_load=False,
        use_cli=True,
    )

    # Create or get viewer
    if viewer is None:
        viewer = napari.Viewer()

    viewer.add_image(image, name="image", rgb=rgb)
    annotator_instance = Annotator(viewer, ndim=ndim, reset_state=False)

    # Trigger layer update of the annotator so that layers have the correct shape.
    # And initialize the 'committed_objects' with the segmentation result if it was given.
    annotator_instance._update_image(segmentation_result=segmentation_result)

    # Add the annotator widget to the viewer and sync widgets.
    viewer.window.add_dock_widget(annotator_instance)
    _sync_embedding_widget(
        widget=state.widgets["embeddings"],
        model_type=(
            model_type
            if checkpoint_path is None
            else state.predictor.model_type
        ),
        save_path=embedding_path,
        checkpoint_path=checkpoint_path,
        device=device,
        tile_shape=tile_shape,
        halo=halo,
    )

    if return_viewer:
        return viewer

    napari.run()


def main():
    """@private"""
    parser = _initialize_parser(
        description="Start the μSAM GUI for image segmentation (2D or 3D)."
    )
    parser.add_argument(
        "--ndim", type=int,
        help="The number of spatial dimensions (2 or 3). If None, auto-detected from image shape."
    )
    args = parser.parse_args()
    image = util.load_image_data(args.input, key=args.key)

    if args.segmentation_result is None:
        segmentation_result = None
    else:
        segmentation_result = util.load_image_data(
            args.segmentation_result, key=args.segmentation_key
        )

    annotator(
        image,
        ndim=args.ndim,
        embedding_path=args.embedding_path,
        segmentation_result=segmentation_result,
        model_type=args.model_type,
        tile_shape=args.tile_shape,
        halo=args.halo,
        precompute_amg_state=args.precompute_amg_state,
        checkpoint_path=args.checkpoint,
        decoder_path=args.decoder_path,
        device=args.device,
        prefer_decoder=args.prefer_decoder,
    )
