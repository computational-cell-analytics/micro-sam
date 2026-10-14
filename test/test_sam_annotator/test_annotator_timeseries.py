import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import napari
import numpy as np
import pytest
from qtpy.QtWidgets import QApplication, QCheckBox

from micro_sam.sam_annotator import annotator_timeseries
from micro_sam.sam_annotator import _widgets as widgets
from micro_sam.sam_annotator._state import AnnotatorState
from micro_sam.sam_annotator.annotator_timeseries import AnnotatorTimeSeries


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def viewer(app):
    state = AnnotatorState()
    state.reset_state()
    state.skip_recomputing_embeddings = False
    state.widgets = {}
    viewer = napari.Viewer(show=False)
    yield viewer
    viewer.close()
    state.reset_state()
    state.widgets = {}
    state.annotator = None


def _initialize(viewer, with_decoder=False, scale=(1, 1, 1), translate=(0, 0, 0)):
    image = viewer.add_image(np.ones((3, 16, 16), dtype="uint16"), name="time series", rgb=False,
                             scale=scale, translate=translate)
    annotator = AnnotatorTimeSeries(viewer)
    viewer.window.add_dock_widget(annotator)
    state = AnnotatorState()
    state.image_shape = image.data.shape
    state.image_scale = tuple(image.scale)
    state.image_name = image.name
    state.image_embeddings = {"input_size": (16, 16)}
    state.decoder = object() if with_decoder else None
    annotator._widgets["autosegment"]._reset_segmentation_mode(with_decoder)
    annotator._update_image()
    annotator._embedding_widget.image_selection.reset_choices()
    annotator._embedding_widget.image_selection.value = image
    return annotator


def _frame(label=1):
    seg = np.zeros((16, 16), dtype="uint32")
    seg[4:12, 4:12] = label
    return seg


def _progress(monkeypatch):
    names = ("pbar_total", "pbar_description", "pbar_update", "pbar_stop")
    signals = SimpleNamespace(**{name: Mock() for name in names})
    monkeypatch.setattr(widgets, "_create_pbar_for_threadworker", lambda: (Mock(), signals))
    return signals


@pytest.mark.parametrize("with_decoder", [False, True])
def test_segment_all_frames_independently(viewer, monkeypatch, with_decoder):
    annotator = _initialize(viewer, with_decoder=with_decoder)
    widget = annotator._widgets["autosegment"]
    frames = [_frame(), np.zeros((16, 16), dtype="uint32"), _frame()]
    segment = Mock(side_effect=lambda min_object_size, i, **kwargs: frames[i])
    monkeypatch.setattr(widgets, "_instance_segmentation_impl", segment)
    # Volume merging and tracking must never be called by this mode.
    monkeypatch.setattr(widgets, "merge_instance_segmentation_3d", Mock(side_effect=AssertionError))
    monkeypatch.setattr(widgets, "track_across_frames", Mock(side_effect=AssertionError))
    signals = _progress(monkeypatch)
    viewer.layers["auto_segmentation"].data[:] = 9

    widget()

    result = viewer.layers["auto_segmentation"].data
    np.testing.assert_array_equal(result[0], _frame(1))
    assert not result[1].any()
    np.testing.assert_array_equal(result[2], _frame(2))
    assert [call.kwargs["i"] for call in segment.call_args_list] == [0, 1, 2]
    for call in segment.call_args_list:
        key = "min_size" if with_decoder else "min_mask_region_area"
        assert call.kwargs[key] == widget.min_object_size
        if with_decoder:
            assert call.kwargs["foreground_smoothing"] == 1.0
            assert call.kwargs["distance_smoothing"] == 1.6
            assert call.kwargs["foreground_threshold"] == 0.5
    signals.pbar_total.emit.assert_called_once_with(3)
    assert signals.pbar_update.emit.call_count == 3  # Includes empty frames.
    signals.pbar_stop.emit.assert_called_once()
    assert not hasattr(widget, "gap_closing_param")
    assert not hasattr(widget, "min_extent_param")
    assert "tracking" not in annotator._widgets
    assert "segment_nd" not in annotator._widgets


def test_current_frame_uses_data_coordinates_and_preserves_other_frames(viewer, monkeypatch):
    annotator = _initialize(viewer, scale=(2.5, 1, 1), translate=(10, 0, 0))
    widget = annotator._widgets["autosegment"]
    widget.findChild(QCheckBox).setChecked(False)
    before = np.stack([_frame(1), _frame(2), _frame(3)])
    viewer.layers["auto_segmentation"].data = before.copy()
    viewer.dims.set_point(0, 12.5)  # Frame 1, in world coordinates.
    segment = Mock(return_value=_frame())
    monkeypatch.setattr(widgets, "_instance_segmentation_impl", segment)
    _progress(monkeypatch)

    widget()
    result = viewer.layers["auto_segmentation"].data
    assert segment.call_args.kwargs["i"] == 1
    np.testing.assert_array_equal(result[0], before[0])
    np.testing.assert_array_equal(result[2], before[2])
    np.testing.assert_array_equal(result[1], _frame(4))
    widget()
    np.testing.assert_array_equal(viewer.layers["auto_segmentation"].data[1], _frame(5))


@pytest.mark.parametrize("with_decoder", [False, True])
def test_predictions_are_cached_per_frame(viewer, monkeypatch, with_decoder):
    annotator = _initialize(viewer, with_decoder=with_decoder)
    state = AnnotatorState()

    class Segmenter:
        def __init__(self):
            self.initialized = []

        def initialize(self, image, image_embeddings, i, **kwargs):
            self.frame = i
            self.initialized.append(i)

        def get_state(self):
            return {"frame": self.frame}

        def set_state(self, prediction):
            self.frame = prediction["frame"]

        def generate(self, **kwargs):
            return _frame(self.frame + 1)

    state.amg = Segmenter()
    _progress(monkeypatch)
    widget = annotator._widgets["autosegment"]
    widget()
    assert state.amg.initialized == [0, 1, 2]
    assert all(i in state.amg_state for i in range(3))
    widget.findChild(QCheckBox).setChecked(False)
    viewer.dims.set_point(0, 1)
    widget()
    assert state.amg.initialized == [0, 1, 2]
    assert state.amg.frame == 1


def test_failure_closes_progress_and_preserves_previous_results(viewer, monkeypatch):
    annotator = _initialize(viewer)
    before = np.stack([_frame(1), _frame(2), _frame(3)])
    viewer.layers["auto_segmentation"].data = before.copy()
    monkeypatch.setattr(widgets, "_instance_segmentation_impl", Mock(side_effect=[_frame(), RuntimeError("failed")]))
    signals = _progress(monkeypatch)
    with pytest.raises(RuntimeError, match="failed"):
        annotator._widgets["autosegment"]()
    signals.pbar_stop.emit.assert_called_once()
    np.testing.assert_array_equal(viewer.layers["auto_segmentation"].data, before)


def test_empty_stack_clears_previous_results(viewer, monkeypatch):
    annotator = _initialize(viewer)
    widget = annotator._widgets["autosegment"]
    viewer.layers["auto_segmentation"].data[:] = 1
    monkeypatch.setattr(widgets, "_instance_segmentation_impl", Mock(return_value=np.zeros((16, 16), dtype="uint32")))
    warning = Mock()
    monkeypatch.setattr(widget, "_empty_segmentation_warning", warning)
    signals = _progress(monkeypatch)
    widget()
    assert not viewer.layers["auto_segmentation"].data.any()
    warning.assert_called_once()
    assert signals.pbar_update.emit.call_count == 3


def test_commit_preserves_independent_instances(viewer):
    annotator = _initialize(viewer)
    segmentation = np.stack([_frame(1), _frame(2), _frame(3)])
    viewer.layers["auto_segmentation"].data = segmentation.copy()
    annotator._widgets["commit"](viewer, layer="auto_segmentation")
    np.testing.assert_array_equal(viewer.layers["committed_objects"].data, segmentation)
    assert not viewer.layers["auto_segmentation"].data.any()
    options = widgets._get_auto_segmentation_options(AnnotatorState(), [1, 2, 3])
    assert options["apply_to_all_frames"] is True
    assert "gap_closing" not in options
    assert "min_extent" not in options
    assert "apply_to_volume" not in options


def test_clear_current_frame(viewer):
    annotator = _initialize(viewer, scale=(2.5, 1, 1), translate=(10, 0, 0))
    before = np.stack([_frame(1), _frame(2), _frame(3)])
    viewer.layers["current_object"].data = before.copy()
    viewer.layers["point_prompts"].data = [[0, 8, 8], [1, 8, 8], [2, 8, 8]]
    viewer.dims.set_point(0, 12.5)
    annotator._widgets["clear"](viewer, all_frames=False)
    np.testing.assert_array_equal(viewer.layers["current_object"].data[0], before[0])
    assert not viewer.layers["current_object"].data[1].any()
    np.testing.assert_array_equal(viewer.layers["current_object"].data[2], before[2])
    np.testing.assert_array_equal(viewer.layers["point_prompts"].data[:, 0], [0, 2])


@pytest.mark.parametrize("dtype", ["bool", "uint8", "uint32"])
def test_prompt_segmentation_assigns_independent_ids(viewer, monkeypatch, dtype):
    annotator = _initialize(viewer, scale=(2.5, 1, 1))
    viewer.layers["point_prompts"].data = [[0, 8, 8], [1, 8, 8]]
    segment = Mock(return_value=_frame().astype(dtype))
    monkeypatch.setattr(widgets.vutil, "prompt_segmentation", segment)
    for frame in (0, 1):
        viewer.dims.set_point(0, frame * 2.5)
        annotator._widgets["segment"](viewer)
        assert segment.call_args.kwargs["i"] == frame
    np.testing.assert_array_equal(viewer.layers["current_object"].data[0], _frame(1))
    np.testing.assert_array_equal(viewer.layers["current_object"].data[1], _frame(2))
    assert not viewer.layers["current_object"].data[2].any()
    annotator._widgets["commit"](viewer)
    np.testing.assert_array_equal(viewer.layers["committed_objects"].data[0], _frame(1))
    np.testing.assert_array_equal(viewer.layers["committed_objects"].data[1], _frame(2))


def test_rgb_precomputation_uses_time_axis(monkeypatch):
    module = importlib.import_module("micro_sam.sam_annotator._state")
    state = AnnotatorState()
    state.reset_state()
    image = np.arange(3 * 16 * 16 * 3, dtype="uint16").reshape(3, 16, 16, 3)
    monkeypatch.setattr(module.util, "precompute_image_embeddings", Mock(return_value={"input_size": (16, 16)}))
    monkeypatch.setattr(module.zarr, "open", Mock(return_value=SimpleNamespace(attrs={"data_signature": "test"})))
    cache = Mock()
    monkeypatch.setattr(module, "cache_amg_state", cache)
    state.initialize_predictor(image, "vit_b_lm", ndim=3, predictor=object(), save_path="embeddings.zarr",
                               precompute_amg_state=True)
    assert cache.call_count == 3
    for i, call in enumerate(cache.call_args_list):
        assert call.kwargs["i"] == i
        np.testing.assert_array_equal(call.kwargs["raw"], image[i])
    state.reset_state()


@pytest.mark.parametrize("cancel", [False, True])
def test_embedding_validation_preserves_results(viewer, monkeypatch, cancel):
    annotator = _initialize(viewer)
    state = AnnotatorState()
    before = np.stack([_frame(1), _frame(2), _frame(3)])
    viewer.layers["committed_objects"].data = before.copy()
    old_shape, old_name, old_embeddings = state.image_shape, state.image_name, state.image_embeddings
    if not cancel:
        image = viewer.add_image(np.ones((16, 16)), name="2D image")
        annotator._embedding_widget.image_selection.value = image
    initialize = Mock(side_effect=AssertionError("Validation should prevent inference"))
    monkeypatch.setattr(state, "initialize_predictor", initialize)
    message = Mock(return_value=True)
    monkeypatch.setattr(widgets, "_generate_message", message)

    annotator._embedding_widget.run_button.click()

    initialize.assert_not_called()
    assert state.image_shape == old_shape
    assert state.image_name == old_name
    assert state.image_embeddings is old_embeddings
    np.testing.assert_array_equal(viewer.layers["committed_objects"].data, before)
    assert message.called


@pytest.mark.parametrize("rgb", [False, True])
def test_compute_embeddings_button_updates_selected_stack(viewer, monkeypatch, rgb):
    annotator = _initialize(viewer)
    image = np.ones((4, 16, 16, 3) if rgb else (4, 16, 16), dtype="uint16")
    selected = viewer.add_image(image, name="second stack", rgb=rgb, scale=(2.5, 0.5, 0.5))
    annotator._embedding_widget.image_selection.value = selected
    state = AnnotatorState()

    def initialize(image_data, **kwargs):
        assert image_data is image
        assert kwargs["ndim"] == 3
        state.image_embeddings = {"input_size": (16, 16)}
        state.decoder = object()

    monkeypatch.setattr(state, "initialize_predictor", initialize)
    monkeypatch.setattr(widgets, "_generate_message", Mock(return_value=False))
    annotator._embedding_widget.run_button.click()
    assert state.image_name == "second stack"
    assert state.image_shape == (4, 16, 16)
    for name in ("auto_segmentation", "current_object", "committed_objects"):
        assert viewer.layers[name].data.shape == (4, 16, 16)
        np.testing.assert_array_equal(viewer.layers[name].scale, selected.scale)
    assert annotator._widgets["autosegment"].with_decoder
    assert annotator._widgets["autosegment"].apply_to_volume


@pytest.mark.parametrize("rgb", [False, True])
def test_python_entry_point_initializes_stack(viewer, monkeypatch, rgb):
    shape = (3, 16, 16, 3) if rgb else (3, 16, 16)
    image = np.ones(shape, dtype="uint16")
    initial_labels = np.stack([_frame(1), _frame(2), _frame(3)])
    state = AnnotatorState()
    initialize = Mock()
    monkeypatch.setattr(state, "initialize_predictor", initialize)

    result = annotator_timeseries(image, segmentation_result=initial_labels, viewer=viewer, return_viewer=True)

    assert result is viewer
    assert initialize.call_args.kwargs["ndim"] == 3
    assert viewer.layers["image"].rgb is rgb
    assert tuple(viewer.dims.axis_labels) == ("time", "y", "x")
    assert viewer.layers["auto_segmentation"].data.shape == (3, 16, 16)
    assert viewer.layers["point_prompts"].ndim == 3
    assert viewer.layers["prompts"].ndim == 3
    np.testing.assert_array_equal(viewer.layers["committed_objects"].data, initial_labels)
    assert state.widgets["segment"].call_button.text == "Segment Frame [S]"
    assert state.widgets["autosegment"].apply_to_volume


@pytest.mark.parametrize("shape", [(16, 16), (2, 3, 16, 16, 3), (2, 16, 16, 2), (0, 16, 16)])
def test_invalid_stack_is_rejected_before_model_loading(monkeypatch, shape):
    initialize = Mock(side_effect=AssertionError)
    monkeypatch.setattr(AnnotatorState(), "initialize_predictor", initialize)
    with pytest.raises(ValueError, match="Expected a time series"):
        annotator_timeseries(np.zeros(shape))
    initialize.assert_not_called()


def test_invalid_initial_segmentation_is_rejected():
    with pytest.raises(ValueError, match="Expected segmentation shape"):
        annotator_timeseries(np.zeros((3, 16, 16)), segmentation_result=np.zeros((16, 16)))


def test_cli_loads_stack_and_initial_labels(monkeypatch):
    module = importlib.import_module("micro_sam.sam_annotator.annotator_timeseries")
    image, labels = np.ones((3, 16, 16)), np.ones((3, 16, 16), dtype="uint32")
    load = Mock(side_effect=[image, labels])
    run = Mock()
    monkeypatch.setattr(module.util, "load_image_data", load)
    monkeypatch.setattr(module, "annotator_timeseries", run)
    monkeypatch.setattr("sys.argv", ["micro_sam.annotator_timeseries", "-i", "stack.tif", "-s", "labels.tif"])
    module.main()
    assert run.call_args.args[0] is image
    assert run.call_args.kwargs["segmentation_result"] is labels
