import platform

import numpy as np
import pytest
from qtpy import QtWidgets

from micro_sam.sam_annotator._state import AnnotatorState
from micro_sam.sam_annotator._widgets import ClassificationEmbeddingWidget, EmbeddingWidget

# The popup only makes sense for computations that are actually slow on the CPU: 3d data, or a 2d
# image split into many tiles. Everything else (plain 2d, few tiles, GPU, the lightweight
# classification tools) must stay silent.

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows."),
]


@pytest.fixture
def popups(monkeypatch):
    """Record the info popups instead of opening a modal dialog that would block the test."""
    calls = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "information",
        lambda parent, title, text, *args, **kwargs: calls.append((title, text)),
    )
    return calls


def make_widget(viewer, classifier=False, device="cpu"):
    widget = ClassificationEmbeddingWidget() if classifier else EmbeddingWidget(ndim_choice=True)
    widget.device = device
    return widget


@pytest.mark.parametrize(
    "ndim, tile_shape, shape, expected",
    [
        (2, None, (256, 256), False),  # plain 2d: a single forward pass.
        (2, (512, 512), (2048, 2048), False),  # 16 tiles.
        (2, (256, 256), (1792, 1792), False),  # 49 tiles: just below the threshold.
        (2, (256, 256), (2048, 2048), True),  # 64 tiles: at the threshold.
        (2, (256, 256), (2304, 2304), True),  # 81 tiles: above the threshold.
        (3, None, (8, 256, 256), True),  # 3d runs the encoder per slice.
        (3, (512, 512), (8, 2048, 2048), True),
    ],
)
def test_cpu_popup_gating(make_napari_viewer_proxy, popups, ndim, tile_shape, shape, expected):
    widget = make_widget(make_napari_viewer_proxy())
    widget._maybe_warn_cpu(ndim, tile_shape, shape)
    assert bool(popups) is expected


@pytest.mark.parametrize("tile_shape", [None, (512, 512)])
def test_cpu_popup_always_shown_for_tracking(make_napari_viewer_proxy, popups, tile_shape):
    # The tracking annotator always runs on a (T, Y, X) timeseries, i.e. ndim is always 3, so it warns
    # on the CPU regardless of tiling. The message says 'timeseries' rather than '3D', matching the
    # relabelled embedding progress bar.
    make_napari_viewer_proxy()
    widget = EmbeddingWidget(sam2_only=True, is_timeseries=True)
    widget.device = "cpu"
    widget._maybe_warn_cpu(3, tile_shape, (8, 2048, 2048))

    assert len(popups) == 1
    title, text = popups[0]
    assert title == "Running on CPU"
    assert "timeseries" in text and "3D" not in text


def test_cpu_popup_says_3d_outside_tracking(make_napari_viewer_proxy, popups):
    widget = make_widget(make_napari_viewer_proxy())
    widget._maybe_warn_cpu(3, None, (8, 256, 256))

    title, text = popups[0]
    assert "3D" in text and "timeseries" not in text


def test_cpu_popup_not_shown_on_gpu(make_napari_viewer_proxy, popups, monkeypatch):
    # A 3d volume, which would warn on the CPU, must stay silent on a GPU.
    monkeypatch.setattr("micro_sam.sam_annotator._widgets.util.get_device", lambda device: "cuda")
    widget = make_widget(make_napari_viewer_proxy(), device="cuda")
    widget._maybe_warn_cpu(3, None, (8, 256, 256))
    assert not popups


def test_cpu_popup_shown_once_per_session(make_napari_viewer_proxy, popups):
    widget = make_widget(make_napari_viewer_proxy())
    widget._maybe_warn_cpu(3, None, (8, 256, 256))
    assert len(popups) == 1
    assert AnnotatorState().cpu_info_shown

    # A second (also expensive) computation, and a freshly created widget, must not warn again.
    widget._maybe_warn_cpu(3, None, (8, 256, 256))
    make_widget(make_napari_viewer_proxy())._maybe_warn_cpu(2, (256, 256), (2304, 2304))
    assert len(popups) == 1


@pytest.mark.parametrize(
    "ndim, tile_shape, shape",
    [(3, None, (8, 256, 256)), (2, (256, 256), (2304, 2304))],
)
def test_no_cpu_popup_for_classifiers(make_napari_viewer_proxy, popups, ndim, tile_shape, shape):
    # The classification tools are lightweight, so they never warn, not even for the expensive cases.
    widget = make_widget(make_napari_viewer_proxy(), classifier=True)
    widget._maybe_warn_cpu(ndim, tile_shape, shape)
    assert not popups


@pytest.mark.parametrize(
    "shape, tile, expected_tile_shape, expected_n_tiles",
    [
        ((288, 288), 32, (256, 256), 4),  # the tile input is floored at 256, giving 4 tiles, not 81.
        ((2048, 2048), 256, (256, 256), 64),
    ],
)
def test_cpu_popup_counts_normalized_tiles(
    make_napari_viewer_proxy, monkeypatch, shape, tile, expected_tile_shape, expected_n_tiles
):
    # '_process_tiling_inputs' floors the tile shape at 256, so the tile count must be derived from the
    # normalized shape that is actually computed, not from the raw value the user typed.
    viewer = make_napari_viewer_proxy()
    layer = viewer.add_image(np.zeros(shape, dtype="uint8"), name="image")

    widget = make_widget(viewer)
    widget.image_selection.reset_choices()
    widget.image_selection.value = layer
    widget.tiling = "yes"
    widget.tile_x, widget.tile_y = tile, tile
    widget.halo_x, widget.halo_y = 64, 64

    # Record what '__call__' hands to the gating, and skip the (expensive) embedding computation.
    seen = {}
    monkeypatch.setattr(
        EmbeddingWidget, "_maybe_warn_cpu",
        lambda self, ndim, tile_shape, shape: seen.update(ndim=ndim, tile_shape=tile_shape, shape=shape),
    )
    monkeypatch.setattr(AnnotatorState, "initialize_predictor", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(EmbeddingWidget, "_update_model", lambda self, state: None)

    widget(skip_validate=True)

    assert tuple(seen["tile_shape"]) == expected_tile_shape
    assert widget._n_tiles(seen["tile_shape"], seen["shape"]) == expected_n_tiles
