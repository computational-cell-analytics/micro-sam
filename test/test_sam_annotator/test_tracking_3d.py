"""Tests for the tracking annotator on a 3d timeseries, (T, Z, Y, X), without a model."""

from types import SimpleNamespace

import numpy as np

from napari.layers import Points, Shapes

SHAPE = (5, 4, 32, 32)


class Tracker:
    """Records the prompts routed to the 3d tracker and returns a full mask for each timepoint."""

    def __init__(self):
        self.point_calls, self.box_calls, self.predict_calls, self.timepoint_calls = [], [], [], []

    def reset_predictor(self):
        self.point_calls, self.box_calls = [], []

    def add_point_prompts(self, t, z, points, point_labels):
        self.point_calls.append((t, z, np.asarray(points).tolist(), np.asarray(point_labels).tolist()))

    def add_box_prompts(self, t, z, boxes):
        self.box_calls.append((t, z, [np.asarray(box).tolist() for box in boxes]))

    def get_progress_total(self, t_range):
        return t_range[1] - t_range[0] + 1

    def predict(self, t_range, update_progress=None):
        self.predict_calls.append(t_range)
        return {t: np.ones(SHAPE[1:], dtype=bool) for t in range(t_range[0], t_range[1] + 1)}

    def segment_timepoint(self, t):
        self.timepoint_calls.append(t)
        return np.ones(SHAPE[1:], dtype=bool)


def point_layer(data, labels=None, states=None, track_ids=None):
    n = len(data)
    properties = {
        "label": np.array(labels or ["positive"] * n), "state": np.array(states or ["track"] * n),
        "track_id": np.array(track_ids or ["1"] * n),
    }
    return Points(data=np.array(data, dtype="float64").reshape(n, 4), properties=properties, ndim=4)


def box_layer(boxes=()):
    """Build the shape layer from boxes given as (t, z, y0, x0, y1, x1)."""
    data = [
        np.array([[t, z, y0, x0], [t, z, y0, x1], [t, z, y1, x1], [t, z, y1, x0]], dtype="float64")
        for t, z, y0, x0, y1, x1 in boxes
    ]
    properties = {"label": np.array(["positive"] * len(data)), "track_id": np.array(["1"] * len(data))}
    return Shapes(data=data or None, shape_type="rectangle", properties=properties, ndim=4)


def make_widget(monkeypatch, points, shapes, track_id=1, lineage=None, current_t=0):
    from micro_sam.sam_annotator import _widgets

    current_object = SimpleNamespace(data=np.zeros(SHAPE, dtype="uint32"), refresh=lambda: None)
    viewer = SimpleNamespace(layers={"current_object": current_object, "point_prompts": points, "prompts": shapes})
    tracker = Tracker()
    state = SimpleNamespace(
        image_shape=SHAPE, current_track_id=track_id, lineage=lineage or {1: []}, interactive_segmenter=tracker,
    )
    signal = SimpleNamespace(emit=lambda *args: None)
    signals = SimpleNamespace(pbar_total=signal, pbar_description=signal, pbar_update=signal, pbar_stop=signal)
    mothers, messages = [], []
    monkeypatch.setattr(_widgets, "AnnotatorState", lambda: state)
    monkeypatch.setattr(_widgets, "_create_pbar_for_threadworker", lambda: (None, signals))
    monkeypatch.setattr(_widgets, "show_info", messages.append)
    monkeypatch.setattr(_widgets, "_update_lineage", lambda viewer, mother: mothers.append(mother))
    monkeypatch.setattr(_widgets.vutil, "current_slice", lambda viewer, layer: current_t)

    widget = SimpleNamespace(_viewer=viewer)
    for name in ("_push_prompts_3d", "_run_tracking_3d", "_run_timepoint_segmentation"):
        method = getattr(_widgets.UnifiedSegmentWidget, name)
        setattr(widget, name, lambda *args, method=method, **kwargs: method(widget, *args, **kwargs))
    return widget, tracker, current_object, mothers, messages


def test_prompt_readers_select_a_plane():
    from micro_sam.sam_annotator import util as vutil

    points = point_layer([[1, 2, 10, 12], [1, 3, 20, 22], [2, 2, 5, 6]], labels=["positive", "negative", "positive"])
    shapes = box_layer([(1, 2, 4, 5, 14, 15), (3, 0, 1, 2, 3, 4)])

    assert vutil.prompted_planes(points, shapes) == [(1, 2), (1, 3), (2, 2), (3, 0)]
    prompts = vutil.collect_frame_prompts(
        points, shapes, SHAPE[2:], i=(1, 2), track_id=1, with_stop_annotation=False,
    )
    assert prompts.points.tolist() == [[10, 12]]
    assert prompts.labels.tolist() == [1]
    assert [box.tolist() for box in prompts.boxes] == [[4, 5, 14, 15]]


def test_tracking_routes_the_prompts_of_each_plane(monkeypatch):
    points = point_layer([[1, 3, 20, 22]], labels=["negative"])
    shapes = box_layer([(1, 2, 4, 5, 14, 15)])
    widget, tracker, current_object, _, _ = make_widget(monkeypatch, points, shapes)

    widget._run_tracking_3d()

    assert tracker.box_calls == [(1, 2, [[4, 5, 14, 15]])]
    # The negative point lies on a timepoint with a box, so it is a correction and not a stop.
    assert tracker.point_calls == [(1, 3, [[20, 22]], [0])]
    assert tracker.predict_calls == [(1, SHAPE[0] - 1)]
    assert not current_object.data[0].any()
    assert (current_object.data[1:] == 1).all()


def test_tracking_stops_at_a_lone_negative_point(monkeypatch):
    points = point_layer([[0, 1, 10, 12], [2, 1, 10, 12], [3, 1, 5, 5]], labels=["positive", "positive", "negative"])
    widget, tracker, _, _, _ = make_widget(monkeypatch, points, box_layer())

    widget._run_tracking_3d()

    assert tracker.predict_calls == [(0, 2)]
    assert [call[0] for call in tracker.point_calls] == [0, 2]


def test_tracking_needs_a_positive_prompt(monkeypatch):
    points = point_layer([[0, 1, 10, 12], [0, 2, 10, 12]], labels=["negative", "negative"])
    widget, tracker, _, _, messages = make_widget(monkeypatch, points, box_layer())

    widget._run_tracking_3d()

    assert tracker.predict_calls == []
    assert len(messages) == 1


def test_division_bounds_the_mother_and_seeds_the_daughters(monkeypatch):
    points = point_layer([[0, 1, 10, 12], [2, 1, 10, 12]], states=["track", "division"])
    widget, tracker, current_object, mothers, _ = make_widget(monkeypatch, points, box_layer())

    widget._run_tracking_3d()

    # The division marker is not a prompt.
    assert [call[0] for call in tracker.point_calls] == [0]
    assert tracker.predict_calls == [(0, 2)]
    assert mothers == [1]
    assert not current_object.data[3:].any()


def test_division_of_a_segmented_mother_only_truncates_it(monkeypatch):
    points = point_layer([[0, 1, 10, 12], [2, 1, 10, 12]], states=["track", "division"])
    widget, tracker, current_object, mothers, _ = make_widget(monkeypatch, points, box_layer())
    current_object.data[:] = 1

    widget._run_tracking_3d()

    assert tracker.predict_calls == []
    assert mothers == [1]
    assert (current_object.data[:3] == 1).all()
    assert not current_object.data[3:].any()


def test_daughter_starts_after_the_division_of_its_mother(monkeypatch):
    points = point_layer(
        [[2, 1, 10, 12], [1, 1, 10, 12]], states=["division", "track"], track_ids=["1", "2"],
    )
    widget, _, current_object, _, _ = make_widget(
        monkeypatch, points, box_layer(), track_id=2, lineage={1: [2, 3], 2: [], 3: []},
    )

    widget._run_tracking_3d()

    assert not current_object.data[:3].any()
    assert (current_object.data[3:] == 2).all()


def test_timepoint_segmentation_only_changes_the_current_timepoint(monkeypatch):
    points = point_layer([[1, 2, 10, 12], [3, 2, 10, 12]])
    widget, tracker, current_object, _, _ = make_widget(monkeypatch, points, box_layer(), current_t=3)

    widget._run_timepoint_segmentation()

    assert tracker.timepoint_calls == [3]
    assert [call[0] for call in tracker.point_calls] == [3]
    assert (current_object.data[3] == 1).all()
    assert not current_object.data[:3].any() and not current_object.data[4:].any()
