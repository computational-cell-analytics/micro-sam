"""Independent segmentation of the frames in a 2D time series."""

from typing import Optional, Tuple, Union

import napari
import numpy as np
import torch
from magicgui import magic_factory

from .. import util
from . import _widgets as widgets
from . import util as vutil
from ._annotator import _AnnotatorBase
from ._state import AnnotatorState
from ._tooltips import get_tooltip


@magic_factory(call_button="Segment Frame [S]")
def segment_timeseries_frame(viewer: "napari.viewer.Viewer") -> None:
    """Segment the current frame's prompts, assigning independent instance IDs."""
    return widgets._segment_slice(viewer, unique_ids=True)


@magic_factory(call_button="Clear Annotations [Shift + C]")
def clear_frames(viewer: "napari.viewer.Viewer", all_frames: bool = True) -> None:
    """Clear prompts and interactive segmentations for all or the current frame."""
    if all_frames:
        vutil.clear_annotations(viewer)
    else:
        layer = viewer.layers["current_object"]
        frame = int(round(layer.world_to_data(viewer.dims.point)[0]))
        vutil.clear_annotations_slice(viewer, i=frame)


class AutoSegmentTimeSeriesWidget(widgets.AutoSegmentWidget):
    """Apply 2D automatic segmentation without merging or tracking across time."""

    timeseries = True

    def _create_volumetric_switch(self):
        # Reuse the stack controls of AutoSegmentWidget, with temporal semantics.
        self.apply_to_volume = True
        return self._add_boolean_param(
            "apply_to_volume", self.apply_to_volume, title="Apply to All Frames",
            tooltip=get_tooltip("timeseries", "apply_to_all_frames"),
        )

    def _run_frames(self, kwargs, i=None):
        widgets._validate_layers(self._viewer, automatic_segmentation=True)
        layer = self._viewer.layers["auto_segmentation"]
        segmentation = np.zeros_like(layer.data) if i is None else layer.data.copy()
        frames = range(segmentation.shape[0]) if i is None else [i]
        offset = int(segmentation.max())
        kwargs = dict(kwargs)
        if not self.with_decoder:
            kwargs["min_mask_region_area"] = self.min_object_size

        pbar, signals = widgets._create_pbar_for_threadworker()
        signals.pbar_total.emit(len(frames))
        signals.pbar_description.emit("Segment time series")
        try:
            for frame in frames:
                seg = widgets._instance_segmentation_impl(self.min_object_size, i=frame, **kwargs)
                # Each frame's instances have distinct IDs, including after correcting a single frame.
                seg = seg.astype(segmentation.dtype, copy=True)
                seg_max = int(seg.max())
                seg[seg != 0] += offset
                offset += seg_max
                segmentation[frame] = seg
                signals.pbar_update.emit(1)
        finally:
            signals.pbar_stop.emit()

        if not np.any(segmentation if i is None else segmentation[i]):
            self._empty_segmentation_warning()
        layer.data = segmentation
        layer.refresh()

    def _run_segmentation_2d(self, kwargs, i=None):
        return self._run_frames(kwargs, i=i)

    def _run_segmentation_3d(self, kwargs):
        return self._run_frames(kwargs)


class AnnotatorTimeSeries(_AnnotatorBase):
    """Napari annotator for grayscale (T, Y, X) or RGB (T, Y, X, C) stacks."""

    def _get_widgets(self):
        return {
            "segment": segment_timeseries_frame(),
            "autosegment": AutoSegmentTimeSeriesWidget(
                self._viewer, with_decoder=AnnotatorState().decoder is not None, volumetric=True,
            ),
            "commit": widgets.commit(),
            "clear": clear_frames(),
        }

    def __init__(self, viewer: "napari.viewer.Viewer", reset_state: bool = True) -> None:
        state = AnnotatorState()
        if reset_state:
            state.reset_state()
            state.skip_recomputing_embeddings = False
        super().__init__(viewer=viewer, ndim=3)
        state.annotator = self
        self._set_time_axis()

    def _set_time_axis(self):
        self._viewer.dims.ndisplay = 2
        for axis, label in zip(range(self._viewer.dims.ndim - 3, self._viewer.dims.ndim), ("time", "y", "x")):
            self._viewer.dims.set_axis_label(axis, label)

    def _update_image(self, segmentation_result=None):
        state = AnnotatorState()
        if state.skip_recomputing_embeddings or state.image_shape is None:
            return
        super()._update_image(segmentation_result=segmentation_result)
        # Cache predictions per frame so settings can be adjusted without repeating inference.
        if state.decoder is not None:
            state.amg_state = vutil._load_is_state(state.embedding_path)
        else:
            state.amg_state = vutil._load_amg_state(state.embedding_path)
        image = self._viewer.layers[state.get_image_name(self._viewer)]
        for name in ("current_object", "auto_segmentation", "committed_objects", "point_prompts", "prompts"):
            self._viewer.layers[name].translate = image.translate
        self._set_time_axis()


def annotator_timeseries(
    image: np.ndarray,
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
    """Start independent segmentation of a 2D time series.

    Args:
        image: Image stack in (T, Y, X) or RGB(A) (T, Y, X, C) order.
        embedding_path: Path for caching embeddings, or precomputed image embeddings.
        segmentation_result: Optional initial labels in (T, Y, X) order, loaded into committed_objects.
        model_type: The Segment Anything model to use.
        tile_shape: Optional tile shape for large images.
        halo: Overlap between tiles.
        return_viewer: Return the viewer instead of starting the Qt event loop.
        viewer: An existing napari viewer to use.
        precompute_amg_state: Precompute automatic segmentation predictions in embedding_path.
        checkpoint_path: Path to custom SAM weights.
        decoder_path: Path to custom instance segmentation decoder weights.
        device: Computational device. By default, select the best available device.
        prefer_decoder: Use the instance segmentation decoder when available.

    Returns:
        The napari viewer if return_viewer is True.

    Labels are unique across frames and represent independent instances, without track identities.
    """
    rgb = image.ndim == 4 and image.shape[-1] in (3, 4)
    if (image.ndim != 3 and not rgb) or any(size == 0 for size in image.shape):
        raise ValueError("Expected a time series of shape (T, Y, X) or RGB(A) (T, Y, X, C).")
    shape = image.shape[:-1] if rgb else image.shape
    if segmentation_result is not None and segmentation_result.shape != shape:
        raise ValueError(f"Expected segmentation shape {shape}, got {segmentation_result.shape}.")

    state = AnnotatorState()
    state.reset_state()
    state.skip_recomputing_embeddings = False
    state.image_shape = shape
    state.initialize_predictor(
        image, model_type=model_type, save_path=embedding_path, ndim=3,
        tile_shape=tile_shape, halo=halo, precompute_amg_state=precompute_amg_state,
        checkpoint_path=checkpoint_path, decoder_path=decoder_path,
        device=device, prefer_decoder=prefer_decoder, use_cli=True,
    )

    if viewer is None:
        viewer = napari.Viewer()
    layer = viewer.add_image(image, name="image", rgb=rgb)
    state.image_name = layer.name
    state.image_scale = tuple(layer.scale)
    annotator = AnnotatorTimeSeries(viewer, reset_state=False)
    annotator._update_image(segmentation_result=segmentation_result)
    viewer.window.add_dock_widget(annotator)
    vutil._sync_embedding_widget(
        widget=state.widgets["embeddings"],
        model_type=model_type if checkpoint_path is None else state.predictor.model_type,
        save_path=embedding_path, checkpoint_path=checkpoint_path,
        device=device, tile_shape=tile_shape, halo=halo,
    )
    vutil._sync_autosegment_widget(state.widgets["autosegment"], model_type, checkpoint_path)
    if return_viewer:
        return viewer
    napari.run()


def main():
    """@private"""
    parser = vutil._initialize_parser(description="Segment the frames of a 2D time series independently.")
    args = parser.parse_args()
    image = util.load_image_data(args.input, key=args.key)
    segmentation = None if args.segmentation_result is None else util.load_image_data(
        args.segmentation_result, key=args.segmentation_key,
    )
    annotator_timeseries(
        image, embedding_path=args.embedding_path, segmentation_result=segmentation,
        model_type=args.model_type, tile_shape=args.tile_shape, halo=args.halo,
        checkpoint_path=args.checkpoint, decoder_path=args.decoder_path, device=args.device,
        precompute_amg_state=args.precompute_amg_state, prefer_decoder=args.prefer_decoder,
    )


if __name__ == "__main__":
    main()
