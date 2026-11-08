import platform
from types import SimpleNamespace

import pytest
import numpy as np

from micro_sam.sam_annotator import _widgets as widgets
from micro_sam.sam_annotator._state import AnnotatorState
from micro_sam.sam_annotator.annotator_3d import Annotator3d
from micro_sam.sam_annotator.annotator_tracking import AnnotatorTracking
from micro_sam.sam_annotator.annotator_timeseries import AnnotatorTimeSeries


pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(platform.system() == "Windows", reason="GUI tests require working OpenGL."),
]


@pytest.fixture(params=[AnnotatorTimeSeries])
def annotator_viewer(request, make_napari_viewer, qtbot):
    state = AnnotatorState()
    state.reset_state()
    state.widgets = {}
    state.image_shape = (3, 32, 32)
    state.image_scale = (2, 1, 1)
    state.image_embeddings = {}
    state.decoder = object()
    state.skip_recomputing_embeddings = False
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros(state.image_shape), name="image", scale=state.image_scale)
    annotator = request.param(viewer, reset_state=False)
    qtbot.addWidget(annotator)
    annotator._update_image()
    yield viewer
    state.reset_state()
    state.widgets = {}
    state.annotator = None


def test_interactive_frame_ids_remain_unique_after_commit(annotator_viewer, monkeypatch):
    viewer = annotator_viewer
    frame = np.zeros((32, 32), dtype=bool)
    frame[2:6, 2:6] = True
    monkeypatch.setattr(widgets.vutil, "prompt_segmentation", lambda *args, **kwargs: frame.copy())
    segment = AnnotatorState().widgets["segment"]
    for i in range(3):
        viewer.layers["point_prompts"].add([i, 4, 4])
        viewer.dims.set_point(0, 2 * i)
        segment(viewer)
    segmentation = viewer.layers["current_object"].data.copy()
    assert np.unique(segmentation[segmentation != 0]).size == 3
    np.testing.assert_array_equal(segmentation != 0, np.broadcast_to(frame, segmentation.shape))

    AnnotatorState().widgets["commit"](viewer, preserve_mode="none")
    np.testing.assert_array_equal(viewer.layers["committed_objects"].data, segmentation)


def test_automatic_frame_ids_remain_unique_after_rerun(annotator_viewer, monkeypatch):
    viewer = annotator_viewer
    frame = np.zeros((32, 32), dtype="uint32")
    frame[2:6, 2:6] = 1
    frame[10:14, 10:14] = 2
    monkeypatch.setattr(widgets, "_instance_segmentation_impl", lambda *args, **kwargs: frame.copy())
    autosegment = AnnotatorState().widgets["autosegment"]
    autosegment()
    previous = viewer.layers["auto_segmentation"].data.copy()

    autosegment.apply_to_volume = False
    viewer.dims.set_point(0, 2)
    autosegment()
    segmentation = viewer.layers["auto_segmentation"].data
    assert np.unique(segmentation[segmentation != 0]).size == 6
    np.testing.assert_array_equal(segmentation[[0, 2]], previous[[0, 2]])
    np.testing.assert_array_equal(segmentation[1] != 0, frame != 0)


@pytest.mark.parametrize("annotator_viewer", [Annotator3d, AnnotatorTracking, AnnotatorTimeSeries], indirect=True)
def test_current_slice_recovers_missing_segmentation_layer(annotator_viewer, monkeypatch):
    viewer = annotator_viewer
    frame = np.ones((32, 32), dtype="uint32")
    monkeypatch.setattr(widgets, "_instance_segmentation_impl", lambda *args, **kwargs: frame.copy())
    autosegment = AnnotatorState().widgets["autosegment"]
    autosegment.apply_to_volume = False
    del viewer.layers["auto_segmentation"]
    viewer.dims.set_point(0, 2)
    autosegment()

    layer = viewer.layers["auto_segmentation"]
    np.testing.assert_array_equal(layer.scale, [2, 1, 1])
    np.testing.assert_array_equal(layer.data[1], frame)
    assert not layer.data[[0, 2]].any()


def test_commit_history_options_include_ais_controls(qtbot):
    autosegment = widgets.AutoSegmentWidget(viewer=None, with_decoder=True, volumetric=False)
    qtbot.addWidget(autosegment)
    settings = {"distance_smoothing": 2.3, "foreground_smoothing": 1.7, "foreground_threshold": 0.65}
    for name, value in settings.items():
        getattr(autosegment, f"{name}_param").setValue(value)
    state = SimpleNamespace(widgets={"autosegment": autosegment})
    options = widgets._get_auto_segmentation_options(state, [1])
    assert {name: options[name] for name in settings} == settings
