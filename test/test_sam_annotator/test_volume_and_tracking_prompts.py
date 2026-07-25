"""Prompt routing for SAM2 volume segmentation and tracking propagation."""

from types import SimpleNamespace

import numpy as np
from napari.layers import Points, Shapes


class Signal:
    def emit(self, *args):
        pass


class CurrentObject:
    def __init__(self, shape=(4, 32, 32)):
        self.data = np.zeros(shape, dtype="uint32")
        self.refresh_count = 0

    def refresh(self):
        self.refresh_count += 1


class Segmenter:
    """Records the prompts routed to the video predictor, keeping the real sync semantics."""

    def __init__(self, shape=(4, 32, 32)):
        self.shape = shape
        self.reset_count = 0
        self.point_calls = []
        self.box_calls = []
        self.mask_calls = []
        self.predict_calls = []
        self.signatures = set()

    def reset_predictor(self):
        self.reset_count += 1

    def sync_prompt_state(self, signatures):
        signatures = set(signatures)
        if not self.signatures.issubset(signatures):
            self.reset_predictor()
        self.signatures = signatures

    def add_point_prompts(self, **kwargs):
        self.point_calls.append(kwargs)

    def add_box_prompts(self, **kwargs):
        self.box_calls.append(kwargs)

    def add_mask_prompts(self, **kwargs):
        self.mask_calls.append(kwargs)

    def get_progress_total(self, z_range):
        return self.shape[0]

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        return np.ones(self.shape, dtype="uint32")


def make_signals():
    return SimpleNamespace(
        pbar_total=Signal(), pbar_description=Signal(), pbar_update=Signal(), pbar_stop=Signal(),
    )


def prompt_objects(segmenter):
    """The object id each point call was routed to, with its labels."""
    return [
        (call["object_id"], np.asarray(call["point_labels"]).tolist())
        for call in segmenter.point_calls
    ]


def run_volume_segmentation(monkeypatch, point_layer, prompt_layer, batched, shape=(4, 32, 32)):
    from micro_sam.sam_annotator import _widgets

    current_object = CurrentObject(shape)
    viewer = SimpleNamespace(layers={
        "current_object": current_object,
        "point_prompts": point_layer,
        "prompts": prompt_layer,
    })
    segmenter = Segmenter(shape)
    state = SimpleNamespace(
        is_sam2=True, image_shape=shape,
        image_embeddings={"input_size": (1024, 1024)}, interactive_segmenter=segmenter,
    )
    widget = SimpleNamespace(_viewer=viewer, batched=batched, early_stop_patience=0, z_range=None)
    messages = []
    monkeypatch.setattr(_widgets, "AnnotatorState", lambda: state)
    monkeypatch.setattr(_widgets, "_create_pbar_for_threadworker", lambda: (None, make_signals()))
    monkeypatch.setattr(_widgets, "show_info", messages.append)
    monkeypatch.setattr(_widgets, "_generate_message", lambda kind, msg: messages.append(msg))

    _widgets.UnifiedSegmentWidget._run_volumetric_segmentation(widget)
    return segmenter, messages


def empty_shape_layer(with_track_id=False):
    properties = {"label": np.empty((0,), dtype=str)}
    if with_track_id:
        properties["track_id"] = np.empty((0,), dtype=str)
    return Shapes(ndim=3, properties=properties)


def test_batched_volume_shares_negative_points_with_the_positive_object(monkeypatch):
    """A negative point corrects the positive object instead of becoming an object of its own."""
    point_layer = Points(
        data=np.array([[1.0, 10.0, 12.0], [1.0, 20.0, 22.0]]),
        properties={"label": np.array(["positive", "negative"])},
    )

    segmenter, _ = run_volume_segmentation(monkeypatch, point_layer, empty_shape_layer(), batched=True)

    # One object, prompted with its positive point followed by the shared negative one.
    assert prompt_objects(segmenter) == [([1, 1], [1, 0])]


def test_batched_volume_gives_each_positive_point_its_own_object(monkeypatch):
    point_layer = Points(
        data=np.array([[1.0, 10.0, 12.0], [1.0, 20.0, 22.0], [1.0, 5.0, 5.0]]),
        properties={"label": np.array(["positive", "positive", "negative"])},
    )

    segmenter, _ = run_volume_segmentation(monkeypatch, point_layer, empty_shape_layer(), batched=True)

    assert prompt_objects(segmenter) == [([1, 1], [1, 0]), ([2, 2], [1, 0])]


def test_batched_volume_shares_negative_points_with_box_objects(monkeypatch):
    # Two negative points, because a lone one on a slice is the established 'stop' annotation.
    point_layer = Points(
        data=np.array([[2.0, 5.0, 5.0], [2.0, 25.0, 25.0]]),
        properties={"label": np.array(["negative", "negative"])},
    )
    prompt_layer = Shapes(
        data=[np.array([[2.0, 2.0, 3.0], [2.0, 2.0, 8.0], [2.0, 7.0, 8.0], [2.0, 7.0, 3.0]])],
        shape_type=["rectangle"], properties={"label": np.array(["positive"])},
    )

    segmenter, _ = run_volume_segmentation(monkeypatch, point_layer, prompt_layer, batched=True)

    assert len(segmenter.box_calls) == 1
    assert segmenter.box_calls[0]["object_id"] == [1]
    # The negative points correct the box object; they never become objects themselves.
    assert prompt_objects(segmenter) == [([1, 1], [0, 0])]


def test_batched_volume_rejects_a_negative_only_prompt_set(monkeypatch):
    point_layer = Points(
        data=np.array([[1.0, 10.0, 12.0], [1.0, 20.0, 22.0]]),
        properties={"label": np.array(["negative", "negative"])},
    )

    segmenter, messages = run_volume_segmentation(
        monkeypatch, point_layer, empty_shape_layer(), batched=True
    )

    assert segmenter.point_calls == [] and segmenter.predict_calls == []
    assert any("positive point, box or mask prompt" in msg for msg in messages)


def test_non_batched_volume_feeds_every_point_to_one_object(monkeypatch):
    point_layer = Points(
        data=np.array([[1.0, 10.0, 12.0], [1.0, 20.0, 22.0]]),
        properties={"label": np.array(["positive", "negative"])},
    )

    segmenter, _ = run_volume_segmentation(monkeypatch, point_layer, empty_shape_layer(), batched=False)

    assert prompt_objects(segmenter) == [(None, [1, 0])]


def run_tracking(monkeypatch, point_layer, prompt_layer, shape=(6, 32, 32)):
    from micro_sam.sam_annotator import _widgets

    current_object = CurrentObject(shape)
    viewer = SimpleNamespace(layers={
        "current_object": current_object,
        "point_prompts": point_layer,
        "prompts": prompt_layer,
    })
    segmenter = Segmenter(shape)
    state = SimpleNamespace(
        is_sam2=True, image_shape=shape, current_track_id=1, lineage={1: {}},
        image_embeddings={"input_size": (1024, 1024)}, interactive_segmenter=segmenter,
    )
    widget = SimpleNamespace(_viewer=viewer)
    messages = []
    monkeypatch.setattr(_widgets, "AnnotatorState", lambda: state)
    monkeypatch.setattr(_widgets, "_create_pbar_for_threadworker", lambda: (None, make_signals()))
    monkeypatch.setattr(_widgets, "show_info", messages.append)

    _widgets.UnifiedSegmentWidget._run_tracking(widget)
    return segmenter, messages


def track_points(n_points):
    return {
        "label": np.array(["positive"] * n_points),
        "track_id": np.array(["1"] * n_points),
        "state": np.array(["track"] * n_points),
    }


def test_tracking_negative_point_beside_a_scribble_is_a_correction_not_a_stop(monkeypatch):
    """A negative point on a scribbled frame corrects the scribble; it must not stop propagation."""
    properties = track_points(2)
    properties["label"] = np.array(["positive", "negative"])
    point_layer = Points(
        data=np.array([[1.0, 10.0, 12.0], [3.0, 20.0, 22.0]]), properties=properties,
    )
    prompt_layer = Shapes(
        data=[np.array([[3.0, 8.0, 8.0], [3.0, 16.0, 16.0]])],
        shape_type=["path"],
        properties={"label": np.array(["positive"]), "track_id": np.array(["1"])},
    )

    segmenter, _ = run_tracking(monkeypatch, point_layer, prompt_layer)

    # Frame 3 is a prompted frame, not a stop, so propagation runs to the end of the volume.
    assert segmenter.predict_calls[0]["z_range"] == (1, 5)
    frames = [call["frame_ids"] for call in segmenter.point_calls]
    assert 3 in frames
    # The negative point reaches the predictor instead of being discarded as a stop annotation.
    labels = np.concatenate([np.asarray(c["point_labels"]) for c in segmenter.point_calls])
    assert 0 in labels.tolist()


def test_tracking_lone_negative_point_still_stops_propagation(monkeypatch):
    properties = track_points(2)
    properties["label"] = np.array(["positive", "negative"])
    point_layer = Points(
        data=np.array([[1.0, 10.0, 12.0], [3.0, 20.0, 22.0]]), properties=properties,
    )

    segmenter, _ = run_tracking(monkeypatch, point_layer, empty_shape_layer(with_track_id=True))

    # No scribble on frame 3, so the lone negative point is a stop. Following the v1 convention
    # ('stop_upper' -> stop at the topmost prompted slice), propagation ends at the last prompted
    # frame rather than running to the end of the timeseries.
    assert segmenter.predict_calls[0]["z_range"] == (1, 1)


def test_tracking_without_a_stop_propagates_to_the_end(monkeypatch):
    properties = track_points(1)
    point_layer = Points(data=np.array([[1.0, 10.0, 12.0]]), properties=properties)

    segmenter, _ = run_tracking(monkeypatch, point_layer, empty_shape_layer(with_track_id=True))

    assert segmenter.predict_calls[0]["z_range"] == (1, 5)


def test_tracking_stop_below_a_later_prompt_is_ignored(monkeypatch):
    """A stop only bounds propagation when it sits above every prompted frame."""
    properties = track_points(3)
    properties["label"] = np.array(["positive", "negative", "positive"])
    point_layer = Points(
        data=np.array([[1.0, 10.0, 12.0], [3.0, 20.0, 22.0], [4.0, 11.0, 13.0]]), properties=properties,
    )

    segmenter, _ = run_tracking(monkeypatch, point_layer, empty_shape_layer(with_track_id=True))

    assert segmenter.predict_calls[0]["z_range"] == (1, 5)


def test_tracking_rejects_a_negative_only_prompt_set(monkeypatch):
    properties = track_points(1)
    properties["label"] = np.array(["negative"])
    point_layer = Points(data=np.array([[1.0, 10.0, 12.0]]), properties=properties)
    prompt_layer = Shapes(
        data=[np.array([[1.0, 8.0, 8.0], [1.0, 16.0, 16.0]])],
        shape_type=["path"],
        properties={"label": np.array(["negative"]), "track_id": np.array(["1"])},
    )

    segmenter, messages = run_tracking(monkeypatch, point_layer, prompt_layer)

    assert segmenter.predict_calls == []
    assert any("cannot track an object" in msg for msg in messages)
