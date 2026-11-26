from typing import Optional, Tuple, Union

import numpy as np

import napari

import torch

from .. import util
from . import _widgets as widgets
from ._tooltips import get_tooltip
from ._state import AnnotatorState
from .annotator_3d import Annotator3d
from .util import _initialize_parser, _sync_embedding_widget


class AutoSegmentTimeSeriesWidget(widgets.AutoSegmentWidget):
    timeseries = True

    def _create_volumetric_switch(self):
        self.apply_to_volume = True
        return self._add_boolean_param(
            "apply_to_volume", self.apply_to_volume, title="Apply to All Frames",
            tooltip=get_tooltip("timeseries", "apply_to_all_frames"),
        )

    def _run_segmentation_3d(self, kwargs):
        # Segment each frame on its own, without merging objects across frames.
        layer = self._viewer.layers["auto_segmentation"]
        segmentation = np.zeros_like(layer.data)
        pbar, pbar_signals = widgets._create_pbar_for_threadworker()
        pbar_signals.pbar_total.emit(segmentation.shape[0])
        pbar_signals.pbar_description.emit("Segment time series")

        offset = 0
        for i in range(segmentation.shape[0]):
            seg = widgets._instance_segmentation_impl(self.min_object_size, i=i, **kwargs)
            segmentation[i] = seg
            segmentation[i][seg != 0] += offset
            offset = max(offset, int(segmentation[i].max()))
            pbar_signals.pbar_update.emit(1)
        pbar_signals.pbar_stop.emit()

        if offset == 0:
            self._empty_segmentation_warning()
        layer.data = segmentation
        layer.refresh()


class AnnotatorTimeSeries(Annotator3d):
    def _get_widgets(self):
        autosegment = AutoSegmentTimeSeriesWidget(self._viewer, with_decoder=self._with_decoder, volumetric=True)
        return {
            "segment": widgets.segment_slice(),
            "autosegment": autosegment,
            "commit": widgets.commit(),
            "clear": widgets.clear_volume(),
        }


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
    """Start the annotation tool for segmenting each frame of a 2d time series independently.

    Args:
        image: The time series in (T, Y, X) or RGB (T, Y, X, C) order.
        embedding_path: Filepath where to save the embeddings
            or the precomputed image embeddings computed by `precompute_image_embeddings`.
        segmentation_result: An initial segmentation to load as the 'committed_objects' layer.
        model_type: The Segment Anything model to use.
        tile_shape: Shape of tiles for tiled embedding prediction.
        halo: Shape of the overlap between tiles.
        return_viewer: Whether to return the napari viewer to further modify it before starting the tool.
        viewer: The viewer to which the Segment Anything functionality should be added.
        precompute_amg_state: Whether to precompute the state for automatic segmentation.
        checkpoint_path: Path to a custom checkpoint from which to load the SAM model.
        decoder_path: Path to a custom decoder checkpoint from which to load the `micro-sam` decoder.
        device: The computational device to use for the SAM model.
        prefer_decoder: Whether to use decoder based instance segmentation if the model has a decoder.

    Returns:
        The napari viewer, only returned if `return_viewer=True`.
    """
    state = AnnotatorState()
    state.image_shape = image.shape[:-1] if image.ndim == 4 else image.shape
    state.initialize_predictor(
        image, model_type=model_type, save_path=embedding_path,
        halo=halo, tile_shape=tile_shape, ndim=3, precompute_amg_state=precompute_amg_state,
        checkpoint_path=checkpoint_path, decoder_path=decoder_path,
        device=device, prefer_decoder=prefer_decoder, use_cli=True,
    )

    if viewer is None:
        viewer = napari.Viewer()

    viewer.add_image(image, name="image")
    annotator = AnnotatorTimeSeries(viewer, reset_state=False)
    annotator._update_image(segmentation_result=segmentation_result)

    viewer.window.add_dock_widget(annotator)
    _sync_embedding_widget(
        widget=state.widgets["embeddings"],
        model_type=model_type if checkpoint_path is None else state.predictor.model_type,
        save_path=embedding_path, checkpoint_path=checkpoint_path,
        device=device, tile_shape=tile_shape, halo=halo,
    )

    if return_viewer:
        return viewer

    napari.run()


def main():
    """@private"""
    parser = _initialize_parser(description="Segment each frame of a 2d time series independently.")
    args = parser.parse_args()
    image = util.load_image_data(args.input, key=args.key)

    if args.segmentation_result is None:
        segmentation_result = None
    else:
        segmentation_result = util.load_image_data(args.segmentation_result, key=args.segmentation_key)

    annotator_timeseries(
        image, embedding_path=args.embedding_path, segmentation_result=segmentation_result,
        model_type=args.model_type, tile_shape=args.tile_shape, halo=args.halo,
        checkpoint_path=args.checkpoint, device=args.device,
        precompute_amg_state=args.precompute_amg_state, prefer_decoder=args.prefer_decoder,
        decoder_path=args.decoder_path,
    )


if __name__ == "__main__":
    main()
