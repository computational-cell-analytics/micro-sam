"""Prompt routing for SAM2 volume segmentation and tracking propagation."""

from types import SimpleNamespace

import numpy as np
import torch
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


def run_volume_segmentation(monkeypatch, point_layer, prompt_layer, batched, shape=(4, 32, 32), segmenter=None):
    from micro_sam.sam_annotator import _widgets

    current_object = CurrentObject(shape)
    viewer = SimpleNamespace(layers={
        "current_object": current_object,
        "point_prompts": point_layer,
        "prompts": prompt_layer,
    })
    if segmenter is None:
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


def test_batched_volume_gives_each_positive_point_its_own_object(monkeypatch):
    point_layer = Points(
        data=np.array([[1.0, 10.0, 12.0], [1.0, 20.0, 22.0], [1.0, 5.0, 5.0]]),
        properties={"label": np.array(["positive", "positive", "negative"])},
    )

    segmenter, _ = run_volume_segmentation(monkeypatch, point_layer, empty_shape_layer(), batched=True)

    assert prompt_objects(segmenter) == [([1, 1], [1, 0]), ([2, 2], [1, 0])]


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


def polygon_layer(frame=2.0):
    return Shapes(
        data=[np.array([[frame, 2.0, 3.0], [frame, 2.0, 9.0], [frame, 8.0, 9.0], [frame, 8.0, 3.0]])],
        shape_type=["polygon"], properties={"label": np.array(["positive"])},
    )


def single_negative_layer(frame=2.0):
    return Points(data=np.array([[frame, 5.0, 5.0]]), properties={"label": np.array(["negative"])})


def test_batched_volume_shares_a_single_negative_with_a_box_object(monkeypatch):
    """A lone negative is only a stop when nothing else on the slice defines an object."""
    segmenter, _ = run_volume_segmentation(
        monkeypatch, single_negative_layer(), rectangle_layer(frame=2.0), batched=True
    )

    assert segmenter.box_calls[0]["object_id"] == [1]
    # The negative corrects the box object instead of being discarded as a stop annotation.
    assert prompt_objects(segmenter) == [([1], [0])]


def test_batched_volume_shares_a_single_negative_with_a_mask_object(monkeypatch):
    segmenter, _ = run_volume_segmentation(
        monkeypatch, single_negative_layer(), polygon_layer(), batched=True
    )

    assert segmenter.mask_calls[0]["object_id"] == [1]
    assert prompt_objects(segmenter) == [([1], [0])]


def test_standard_volume_keeps_a_single_negative_beside_a_box(monkeypatch):
    segmenter, _ = run_volume_segmentation(
        monkeypatch, single_negative_layer(), rectangle_layer(frame=2.0), batched=False
    )

    assert len(segmenter.box_calls) == 1
    assert prompt_objects(segmenter) == [(None, [0])]


class SliceSegmenter:
    def __init__(self, shape=(32, 32)):
        self.shape = shape
        self.calls = []

    def segment_slice(self, **kwargs):
        self.calls.append(kwargs)
        return np.zeros(self.shape, dtype="uint32")


def run_slice_segmentation(monkeypatch, point_layer, prompt_layer, batched, shape=(4, 32, 32), z=2):
    from micro_sam.sam_annotator import _widgets

    viewer = SimpleNamespace(
        layers={
            "current_object": CurrentObject(shape),
            "point_prompts": point_layer,
            "prompts": prompt_layer,
        },
        dims=SimpleNamespace(point=(float(z), 0.0, 0.0)),
    )
    segmenter = SliceSegmenter(shape[1:])
    state = SimpleNamespace(
        is_sam2=True, image_shape=shape,
        image_embeddings={"input_size": (1024, 1024)}, interactive_segmenter=segmenter,
    )
    widget = SimpleNamespace(_viewer=viewer, batched=batched)
    widget._segment_slice_batched = lambda *args: _widgets.UnifiedSegmentWidget._segment_slice_batched(
        widget, *args
    )
    messages = []
    monkeypatch.setattr(_widgets, "AnnotatorState", lambda: state)
    monkeypatch.setattr(_widgets, "show_info", messages.append)
    monkeypatch.setattr(_widgets, "_generate_message", lambda kind, msg: messages.append(msg))

    _widgets.UnifiedSegmentWidget._run_slice_segmentation(widget)
    return segmenter, messages


def test_batched_slice_shares_a_single_negative_with_the_box_object(monkeypatch):
    segmenter, _ = run_slice_segmentation(
        monkeypatch, single_negative_layer(), rectangle_layer(frame=2.0), batched=True
    )

    assert len(segmenter.calls) == 1
    call = segmenter.calls[0]
    assert call["object_id"] == 1 and len(call["boxes"]) == 1
    # The negative reaches the box object instead of being discarded as a stop annotation.
    assert np.asarray(call["labels"]).tolist() == [0]


def test_batched_slice_ignores_a_lone_negative_without_a_shape(monkeypatch):
    segmenter, _ = run_slice_segmentation(
        monkeypatch, single_negative_layer(), empty_shape_layer(), batched=True
    )

    assert segmenter.calls == []


def test_a_lone_negative_without_a_shape_stays_a_stop(monkeypatch):
    """Without another cue on the slice the point keeps its stop meaning and is not pushed."""
    point_layer = Points(
        data=np.array([[1.0, 10.0, 12.0], [2.0, 5.0, 5.0]]),
        properties={"label": np.array(["positive", "negative"])},
    )

    segmenter, _ = run_volume_segmentation(monkeypatch, point_layer, empty_shape_layer(), batched=True)

    assert prompt_objects(segmenter) == [([1], [1])]


class RecordingPredictor:
    """A stand-in SAM2 predictor recording the prompts that reach the persistent state."""

    def __init__(self):
        self.calls = []
        self.reset_count = 0

    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, clear_old_points=False,
                              points=None, labels=None, box=None):
        self.calls.append({
            "frame_idx": frame_idx, "obj_id": obj_id,
            "points": None if points is None else np.asarray(points).tolist(),
            "labels": None if labels is None else np.asarray(labels).tolist(),
            "box": None if box is None else np.asarray(box).tolist(),
        })
        return None, [obj_id], torch.zeros((1, 1, 32, 32))

    def reset_state(self, inference_state):
        self.reset_count += 1


def make_stateful_segmenter(shape=(4, 32, 32)):
    """A segmenter with the real dedup and sync bookkeeping, so a second run sees the first one's state.

    The routing tests above build a fresh recorder per call and therefore cannot observe the persistent
    SAM2 state carried between rounds, which is where a changed prompt-to-object mapping does its damage.
    """
    from micro_sam.v2.prompt_based_segmentation import PromptableSegmentation3D

    segmenter = PromptableSegmentation3D.__new__(PromptableSegmentation3D)
    segmenter.predictor = RecordingPredictor()
    segmenter.volume = np.zeros(shape, dtype="uint8")
    segmenter.inference_state = {}
    segmenter._clear_pushed_prompts()
    segmenter.get_progress_total = lambda z_range=None: shape[0]
    segmenter.predict = lambda **kwargs: np.ones(shape, dtype="uint32")
    return segmenter


def pushed_prompts(segmenter):
    """The (object id, point) of every point push, and the (object id, frame) of every box push."""
    points = [
        (call["obj_id"], tuple(call["points"][0])) for call in segmenter.predictor.calls if call["points"]
    ]
    boxes = [(call["obj_id"], call["frame_idx"]) for call in segmenter.predictor.calls if call["box"]]
    return points, boxes


def rectangle_layer(frame=1.0):
    return Shapes(
        data=[np.array([[frame, 2.0, 3.0], [frame, 2.0, 8.0], [frame, 7.0, 8.0], [frame, 7.0, 3.0]])],
        shape_type=["rectangle"], properties={"label": np.array(["positive"])},
    )


def make_point_layer(coords, labels):
    data = np.array(coords, dtype="float64").reshape(-1, 3)
    return Points(data=data, properties={"label": np.array(labels, dtype=str)})


def test_adding_a_box_renumbers_the_objects_and_replays_from_scratch(monkeypatch):
    """A box takes object id 1 from the point behind it, so the state it was pushed to is gone."""
    segmenter = make_stateful_segmenter()
    points = make_point_layer([[1.0, 10.0, 12.0]], ["positive"])

    run_volume_segmentation(monkeypatch, points, empty_shape_layer(), batched=True, segmenter=segmenter)
    assert pushed_prompts(segmenter) == ([(1, (12.0, 10.0))], [])

    run_volume_segmentation(monkeypatch, points, rectangle_layer(), batched=True, segmenter=segmenter)

    assert segmenter.predictor.reset_count == 1
    pushed_points, pushed_boxes = pushed_prompts(segmenter)
    # The box owns object 1 now and the point moved to object 2. The point must not also be replayed
    # into the box's object, which is what the box push does with a stale dedup state.
    assert pushed_boxes == [(1, 1)]
    assert pushed_points[1:] == [(2, (12.0, 10.0))]


def test_adding_a_point_behind_a_box_keeps_the_state(monkeypatch):
    """The routing of the existing box is unchanged, so the round stays incremental."""
    segmenter = make_stateful_segmenter()
    boxes = rectangle_layer()

    run_volume_segmentation(
        monkeypatch, make_point_layer([], []), boxes, batched=True, segmenter=segmenter
    )
    run_volume_segmentation(
        monkeypatch, make_point_layer([[1.0, 10.0, 12.0]], ["positive"]), boxes, batched=True, segmenter=segmenter
    )

    assert segmenter.predictor.reset_count == 0
    pushed_points, pushed_boxes = pushed_prompts(segmenter)
    assert pushed_boxes == [(1, 1)]  # the box is pushed once, not again on the second run
    assert pushed_points == [(2, (12.0, 10.0))]


def test_adding_a_negative_correction_keeps_the_state(monkeypatch):
    segmenter = make_stateful_segmenter()

    run_volume_segmentation(
        monkeypatch, make_point_layer([[1.0, 10.0, 12.0]], ["positive"]), empty_shape_layer(),
        batched=True, segmenter=segmenter,
    )
    run_volume_segmentation(
        monkeypatch, make_point_layer([[1.0, 10.0, 12.0], [1.0, 20.0, 22.0]], ["positive", "negative"]),
        empty_shape_layer(), batched=True, segmenter=segmenter,
    )

    assert segmenter.predictor.reset_count == 0
    pushed_points, _ = pushed_prompts(segmenter)
    # The positive point is deduped and the negative joins its object as a correction.
    assert pushed_points == [(1, (12.0, 10.0)), (1, (22.0, 20.0))]


def test_switching_from_batched_to_standard_replays_from_scratch(monkeypatch):
    """The same two points define two objects in batched mode and one in standard mode."""
    segmenter = make_stateful_segmenter()
    points = make_point_layer([[1.0, 10.0, 12.0], [1.0, 20.0, 22.0]], ["positive", "positive"])

    run_volume_segmentation(monkeypatch, points, empty_shape_layer(), batched=True, segmenter=segmenter)
    assert pushed_prompts(segmenter)[0] == [(1, (12.0, 10.0)), (2, (22.0, 20.0))]

    run_volume_segmentation(monkeypatch, points, empty_shape_layer(), batched=False, segmenter=segmenter)

    assert segmenter.predictor.reset_count == 1
    assert pushed_prompts(segmenter)[0][2:] == [(1, (12.0, 10.0)), (1, (22.0, 20.0))]


def test_deleting_a_prompt_replays_from_scratch(monkeypatch):
    segmenter = make_stateful_segmenter()

    run_volume_segmentation(
        monkeypatch, make_point_layer([[1.0, 10.0, 12.0], [1.0, 20.0, 22.0]], ["positive", "positive"]),
        empty_shape_layer(), batched=True, segmenter=segmenter,
    )
    run_volume_segmentation(
        monkeypatch, make_point_layer([[1.0, 20.0, 22.0]], ["positive"]), empty_shape_layer(),
        batched=True, segmenter=segmenter,
    )

    assert segmenter.predictor.reset_count == 1
    assert pushed_prompts(segmenter)[0][2:] == [(1, (22.0, 20.0))]


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
        is_sam2=True, image_shape=shape, current_track_id=1, lineage={1: {}}, seed_masks={},
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


def test_tracking_negative_point_beside_a_box_is_a_correction_not_a_stop(monkeypatch):
    """A negative point on a frame that also carries a box corrects that box, it does not stop the track."""
    properties = track_points(2)
    properties["label"] = np.array(["positive", "negative"])
    points = Points(data=np.array([[1.0, 10.0, 12.0], [3.0, 20.0, 22.0]]), properties=properties)
    prompt_layer = Shapes(
        data=[np.array([[3.0, 2.0, 3.0], [3.0, 2.0, 8.0], [3.0, 7.0, 8.0], [3.0, 7.0, 3.0]])],
        shape_type=["rectangle"],
        properties={"label": np.array(["positive"]), "track_id": np.array(["1"])},
    )

    segmenter, _ = run_tracking(monkeypatch, points, prompt_layer)

    # Frame 3 is prompted, not a stop, so propagation runs to the end of the timeseries.
    assert segmenter.predict_calls[0]["z_range"] == (1, 5)
    assert len(segmenter.box_calls) == 1
    labels = np.concatenate([np.asarray(call["point_labels"]) for call in segmenter.point_calls])
    assert 0 in labels.tolist()  # the negative reaches the predictor instead of being discarded


def test_tracking_frame_segmentation_keeps_a_single_negative_beside_a_box(monkeypatch):
    """'_segment_track_on_frame' reads the same stop rule, so the box on the frame is still segmented."""
    from micro_sam.sam_annotator import _widgets

    properties = track_points(1)
    properties["label"] = np.array(["negative"])
    points = Points(data=np.array([[3.0, 20.0, 22.0]]), properties=properties)
    prompt_layer = Shapes(
        data=[np.array([[3.0, 2.0, 3.0], [3.0, 2.0, 8.0], [3.0, 7.0, 8.0], [3.0, 7.0, 3.0]])],
        shape_type=["rectangle"],
        properties={"label": np.array(["positive"]), "track_id": np.array(["1"])},
    )
    segmenter = SliceSegmenter((32, 32))
    widget = SimpleNamespace(_viewer=SimpleNamespace(layers={"point_prompts": points, "prompts": prompt_layer}))
    state = SimpleNamespace(interactive_segmenter=segmenter)
    monkeypatch.setattr(_widgets, "show_info", lambda msg: None)

    _widgets.UnifiedSegmentWidget._segment_track_on_frame(widget, state, t=3, track_id=1, shape=(32, 32))

    assert len(segmenter.calls) == 1
    assert len(segmenter.calls[0]["boxes"]) == 1
    assert np.asarray(segmenter.calls[0]["labels"]).tolist() == [0]


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
