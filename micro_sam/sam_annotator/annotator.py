from typing import Optional, Tuple, Union

import napari
import numpy as np
import torch
from napari.utils.notifications import show_info

from .. import util
from ..v2.util import DEFAULT_MODEL
from . import _widgets as widgets
from . import util as vutil
from ._annotator import _AnnotatorBase
from ._state import AnnotatorState
from .util import (
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
        # Use the normalizer so singletons/channels are accounted for. Unsupported inputs
        # fall back to 2D here so the widget can open; '_on_image_selection_changed' then
        # reports the issue to the user instead of crashing construction.
        try:
            return vutil.prepare_annotation_image(image_layers[0].data)[1]
        except ValueError:
            return 2
    return 2


class Annotator(_AnnotatorBase):
    """Unified annotator for 2D and 3D images.

    This class handles both 2D and 3D annotation, with dimensionality
    controlled by the `ndim` parameter or auto-detected from the image.
    """

    def _create_embedding_widget(self):
        # Expose the 'image dimensions' (ndim) override here: the segmentation annotator is the only
        # one that wires it into image normalization (it handles both 2d and 3d data).
        return widgets.EmbeddingWidget(ndim_choice=True)

    def _get_widgets(self):
        """Create the widgets for the segmentation annotator.

        The interactive segmentation widget merges the prompt menu, the segment and the
        clear controls into a single ndim-aware widget placed right after the embeddings.
        """
        # The default automatic-segmentation mode depends on whether a UniSAM2 decoder is available.
        # At startup the decoder is not loaded yet (only on 'Compute Embeddings'), so also treat the
        # default model as decoder-capable when it has a registered decoder - otherwise the Microscopy
        # default would wrongly start in 'amg'. The mode is re-synced after compute via
        # '_sync_autosegment_widget' once the actual decoder is known.
        from ..v2.util import DEFAULT_MODEL, has_registered_decoder
        with_decoder = AnnotatorState().decoder is not None or has_registered_decoder(DEFAULT_MODEL)
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

        @prompt_layer.bind_key("t", overwrite=True)
        def _toggle_shape_prompt_label(event=None):
            vutil.toggle_label(self._point_prompt_layer, self._shape_prompt_layer)

        @point_prompt_layer.bind_key("t", overwrite=True)
        def _toggle_point_prompt_label(event=None):
            vutil.toggle_label(self._point_prompt_layer, self._shape_prompt_layer)

        @self._viewer.bind_key("c", overwrite=True)
        def _commit(viewer):
            self._widgets["commit"](viewer)

        @self._viewer.bind_key("t", overwrite=True)
        def _toggle_label(event=None):
            vutil.toggle_label(self._point_prompt_layer, self._shape_prompt_layer)

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

        # Re-normalize when the user changes the 'image dimensions' (ndim) override.
        if getattr(self._embedding_widget, "ndim_choice", False):
            self._embedding_widget.image_ndim_dropdown.currentTextChanged.connect(self._on_ndim_mode_changed)

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
        # so the image, segmentation and prompt layers all stay aligned. The 'image dimensions'
        # override (auto/2d/3d) disambiguates multi-channel inputs. Unsupported inputs (e.g. 3D
        # volumes with a channel axis) are reported instead of crashing the widget.
        try:
            image_layer, ndim = self._maybe_normalize_image_layer(
                image_layer, ndim=self._embedding_widget._ndim_override()
            )
        except ValueError as e:
            show_info(str(e))
            return

        # Detect an actual change of the selected image, tracked by layer identity (the state's
        # 'image_name' is not reliably set on every code path, so we don't depend on it). The first
        # call (during setup) just records the image and does not reset; a later switch to a
        # different image layer triggers the reset below.
        previous_layer = getattr(self, "_last_image_layer", None)
        image_changed = previous_layer is not None and image_layer is not previous_layer
        self._last_image_layer = image_layer

        # When the selected image changes, reset everything so the tool behaves as if it was just
        # opened on the new image: the precomputed embeddings, the model and everything derived from
        # them belong to the previous image and must not be reused (they can even differ in
        # dimensionality, e.g. 3D volume -> 2D image). 'reset_state' clears the state; resetting the
        # (shared, kept) embedding widget inputs restores the default model / tiling / save path; and
        # the forced rebuild recreates the dimension-specific widgets and napari layers, so all
        # checkboxes are back to defaults, the autosegment cache is gone and the prompt / segmentation
        # layers are cleared. The user recomputes embeddings for the new image via 'Compute Embeddings'.
        if image_changed:
            AnnotatorState().reset_state()
            self._embedding_widget._reset_inputs_to_defaults()
            self._rebuild_for_ndim(ndim, force=True)
            return

        if ndim != self._ndim:
            self._rebuild_for_ndim(ndim)

    def _on_ndim_mode_changed(self, *args):
        """Re-interpret the current image when the 'image dimensions' override changes.

        Unlike selecting a different image, this keeps the embedding-widget inputs (model, tiling and
        the override itself); it only re-normalizes, clears the now-invalid embeddings / prompts and
        rebuilds the layers for the resulting dimensionality.
        """
        if self._suppress_selection_rebuild:
            return
        image_layer = self._embedding_widget.image_selection.get_value()
        if image_layer is None:
            return
        try:
            image_layer, ndim = self._maybe_normalize_image_layer(
                image_layer, ndim=self._embedding_widget._ndim_override()
            )
        except ValueError as e:
            # The chosen override (e.g. '3d' on a 2D image) cannot be applied: warn in a modal dialog,
            # recommend 'auto', and revert the dropdown to 'auto' so the tool is not stuck on the
            # invalid choice.
            from qtpy import QtWidgets
            QtWidgets.QMessageBox.warning(
                self, "Invalid image dimensions",
                f"{e}\n\nThis dimensionality cannot be applied to the selected image. "
                "Switching back to 'auto', which is recommended for automatic dimensionality detection.",
                QtWidgets.QMessageBox.Ok,
            )
            dropdown = self._embedding_widget.image_ndim_dropdown
            dropdown.blockSignals(True)
            dropdown.setCurrentText("auto")
            self._embedding_widget.image_ndim_mode = "auto"
            dropdown.blockSignals(False)
            return

        # No effective change (same layer and dimensionality): keep the current work.
        if image_layer is getattr(self, "_last_image_layer", None) and ndim == self._ndim:
            return

        self._last_image_layer = image_layer
        AnnotatorState().reset_state()
        self._rebuild_for_ndim(ndim, force=True)

    def _update_image(self, segmentation_result=None):
        """Update the image and load AMG state for 3D."""
        super()._update_image(segmentation_result=segmentation_result)

        # Load the AMG state from the embedding path (3D only)
        if self._ndim == 3:
            state = AnnotatorState()
            if state.decoder is not None:
                state.autoseg_state = _load_is_state(state.embedding_path)
            else:
                state.autoseg_state = _load_amg_state(state.embedding_path)


def annotator(
    image: np.ndarray,
    *,
    ndim: Optional[int] = None,
    embedding_path: Optional[Union[str, util.ImageEmbeddings]] = None,
    segmentation_result: Optional[np.ndarray] = None,
    model_type: str = DEFAULT_MODEL,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    return_viewer: bool = False,
    viewer: Optional["napari.viewer.Viewer"] = None,
    precompute_autoseg_state: bool = False,
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
        precompute_autoseg_state: Whether to precompute the automatic segmentation state (AMG masks, or
            decoder predictions if the model has a decoder). Requires an embedding path.
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
    # Normalize the image: squeeze singletons and map the channel axis to RGB. The optional 'ndim'
    # override disambiguates multi-channel inputs (e.g. reads a channels-first (C, H, W) array as a
    # 2d image), consistent with the GUI's 'image dimensions' control; with ndim=None it is
    # auto-detected. 'prepare_annotation_image' raises if the override cannot be applied to the shape.
    image, ndim, rgb = vutil.prepare_annotation_image(image, ndim=ndim)

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
        precompute_autoseg_state=precompute_autoseg_state,
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
    viewer.window.add_dock_widget(annotator_instance, name="Segment Anything for Microscopy (Segmentation)")
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
