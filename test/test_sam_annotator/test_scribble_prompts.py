from types import SimpleNamespace

import numpy as np
import pytest
from napari.layers import Points, Shapes

from micro_sam.sam_annotator import util as annotator_util


class _PromptShapesLayer:
    def __init__(self, data, shape_type, labels):
        self.data = data
        self.shape_type = shape_type
        self.properties = {"label": np.asarray(labels)}


def test_positive_and_negative_scribbles_are_resampled():
    layer = _PromptShapesLayer(
        data=[
            np.array([[0.0, 0.0], [0.0, 64.0]]),
            np.array([[0.0, 0.0], [32.0, 0.0]]),
        ],
        shape_type=["path", "line"],
        labels=["positive", "negative"],
    )

    points, labels = annotator_util.scribble_layer_to_prompts(
        layer, image_shape=(256, 256), spacing=64.0
    )

    np.testing.assert_allclose(points[:5], [[0, 0], [0, 16], [0, 32], [0, 48], [0, 64]])
    np.testing.assert_allclose(points[5:], [[0, 0], [16, 0], [32, 0]])
    np.testing.assert_array_equal(labels, [1, 1, 1, 1, 1, 0, 0, 0])


def test_scribble_sampling_is_resolution_independent_and_capped():
    low_res = _PromptShapesLayer(
        data=[np.array([[0.0, 0.0], [0.0, 256.0]])],
        shape_type=["path"],
        labels=["positive"],
    )
    high_res = _PromptShapesLayer(
        data=[np.array([[0.0, 0.0], [0.0, 2048.0]])],
        shape_type=["path"],
        labels=["positive"],
    )

    low_points, _ = annotator_util.scribble_layer_to_prompts(
        low_res, image_shape=(256, 256), spacing=32.0, max_points_per_stroke=8
    )
    high_points, _ = annotator_util.scribble_layer_to_prompts(
        high_res, image_shape=(2048, 2048), spacing=32.0, max_points_per_stroke=8
    )

    assert len(low_points) == len(high_points) == 8
    np.testing.assert_allclose(low_points / 256.0, high_points / 2048.0)


def test_scribble_sampling_shares_a_global_budget_by_demand():
    layer = _PromptShapesLayer(
        data=[
            np.array([[0.0, 0.0], [0.0, 64.0]]),
            np.array([[128.0, 0.0], [128.0, 256.0]]),
            np.array([[256.0, 0.0], [256.0, 512.0]]),
        ],
        shape_type=["line", "line", "line"],
        labels=["positive", "negative", "positive"],
    )

    points, _ = annotator_util.scribble_layer_to_prompts(
        layer, image_shape=(1024, 1024), spacing=32.0, max_points=12
    )

    counts = [np.count_nonzero(points[:, 0] == row) for row in (0.0, 128.0, 256.0)]
    assert counts == [2, 4, 6]
    assert len(points) == 12


def test_curved_scribble_gets_more_samples_and_preserves_bends():
    layer = _PromptShapesLayer(
        data=[
            np.array([[0.0, 0.0], [0.0, 256.0]]),
            np.array([[200.0, 0.0], [136.0, 0.0], [136.0, 128.0], [200.0, 128.0]]),
        ],
        shape_type=["line", "path"],
        labels=["positive", "negative"],
    )

    points, labels = annotator_util.scribble_layer_to_prompts(
        layer, image_shape=(1024, 1024), spacing=32.0
    )

    assert np.count_nonzero(labels == 0) > np.count_nonzero(labels == 1)
    negative_points = points[labels == 0]
    assert any(np.allclose(point, [136.0, 0.0]) for point in negative_points)
    assert any(np.allclose(point, [136.0, 128.0]) for point in negative_points)


def test_nearby_same_label_strokes_are_deduplicated_but_conflicts_remain():
    same_label = _PromptShapesLayer(
        data=[
            np.array([[0.0, 0.0], [0.0, 64.0]]),
            np.array([[2.0, 0.0], [2.0, 64.0]]),
        ],
        shape_type=["line", "line"],
        labels=["positive", "positive"],
    )
    conflicting = _PromptShapesLayer(
        data=same_label.data,
        shape_type=same_label.shape_type,
        labels=["positive", "negative"],
    )

    deduplicated, _ = annotator_util.scribble_layer_to_prompts(
        same_label, image_shape=(1024, 1024), spacing=32.0, deduplication_distance=4.0
    )
    conflict_points, conflict_labels = annotator_util.scribble_layer_to_prompts(
        conflicting, image_shape=(1024, 1024), spacing=32.0, deduplication_distance=4.0
    )

    assert len(deduplicated) == 3
    assert len(conflict_points) == 6
    np.testing.assert_array_equal(conflict_labels, [1, 1, 1, 0, 0, 0])


def test_short_scribble_preserves_both_endpoints():
    layer = _PromptShapesLayer(
        data=[np.array([[10.0, 10.0], [10.0, 12.0]])],
        shape_type=["line"],
        labels=["positive"],
    )

    points, labels = annotator_util.scribble_layer_to_prompts(
        layer, image_shape=(1024, 1024), spacing=32.0
    )

    np.testing.assert_allclose(points, [[10.0, 10.0], [10.0, 12.0]])
    np.testing.assert_array_equal(labels, [1, 1])


def test_scribbles_are_selected_per_volume_slice():
    layer = _PromptShapesLayer(
        data=[
            np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 64.0]]),
            np.array([[3.0, 0.0, 0.0], [3.0, 64.0, 0.0]]),
        ],
        shape_type=["path", "path"],
        labels=["positive", "negative"],
    )

    points, labels = annotator_util.scribble_layer_to_prompts(
        layer, image_shape=(256, 256), i=3, spacing=64.0
    )

    np.testing.assert_allclose(points, [[0, 0], [16, 0], [32, 0], [48, 0], [64, 0]])
    np.testing.assert_array_equal(labels, np.zeros(5, dtype="int64"))


def test_scribble_layer_to_prompts_filters_by_track_id():
    layer = _PromptShapesLayer(
        data=[
            np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 64.0]]),
            np.array([[2.0, 0.0, 0.0], [2.0, 64.0, 0.0]]),
        ],
        shape_type=["path", "path"],
        labels=["positive", "negative"],
    )
    layer.properties["track_id"] = np.array(["1", "2"])

    points_1, labels_1 = annotator_util.scribble_layer_to_prompts(
        layer, image_shape=(256, 256), i=2, spacing=64.0, track_id=1
    )
    points_2, labels_2 = annotator_util.scribble_layer_to_prompts(
        layer, image_shape=(256, 256), i=2, spacing=64.0, track_id=2
    )

    assert len(points_1) > 0 and len(points_2) > 0
    np.testing.assert_array_equal(labels_1, np.ones(len(labels_1), dtype="int64"))
    np.testing.assert_array_equal(labels_2, np.zeros(len(labels_2), dtype="int64"))


def test_get_scribble_slices_filters_by_track_id():
    layer = _PromptShapesLayer(
        data=[
            np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 32.0]]),
            np.array([[3.0, 0.0, 0.0], [3.0, 32.0, 0.0]]),
        ],
        shape_type=["path", "path"],
        labels=["positive", "positive"],
    )
    layer.properties["track_id"] = np.array(["1", "2"])

    np.testing.assert_array_equal(annotator_util.get_scribble_slices(layer, track_id=1), [1])
    np.testing.assert_array_equal(annotator_util.get_scribble_slices(layer, track_id=2), [3])
    np.testing.assert_array_equal(annotator_util.get_scribble_slices(layer), [1, 3])


def test_closed_shape_in_shared_prompt_layer_is_ignored_by_scribble_converter():
    layer = _PromptShapesLayer(
        data=[np.array([[0.0, 0.0], [0.0, 8.0], [8.0, 8.0]])],
        shape_type=["polygon"],
        labels=["positive"],
    )

    points, labels = annotator_util.scribble_layer_to_prompts(layer, image_shape=(16, 16))

    assert points.shape == (0, 2)
    assert labels.shape == (0,)


def test_shared_layer_routes_closed_and_open_shapes_separately():
    layer = _PromptShapesLayer(
        data=[
            np.array([[1.0, 1.0], [1.0, 8.0], [8.0, 8.0], [8.0, 1.0]]),
            np.array([[2.0, 2.0], [12.0, 12.0]]),
        ],
        shape_type=["rectangle", "path"],
        labels=["positive", "negative"],
    )

    boxes, masks = annotator_util.shape_layer_to_prompts(layer, shape=(16, 16))
    points, labels = annotator_util.scribble_layer_to_prompts(
        layer, image_shape=(16, 16), spacing=128.0
    )

    assert len(boxes) == len(masks) == 1
    np.testing.assert_array_equal(boxes[0], [1, 1, 8, 8])
    assert masks[0] is None
    assert len(points) > 1
    np.testing.assert_array_equal(labels, np.zeros(len(points), dtype="int64"))


def test_existing_point_and_box_prompt_conversion_in_2d_and_3d():
    point_layer_2d = Points(
        data=np.array([[2.0, 3.0], [4.0, 5.0]]),
        properties={"label": np.array(["positive", "negative"])},
    )
    points, labels = annotator_util.point_layer_to_prompts(
        point_layer_2d, with_stop_annotation=False
    )
    np.testing.assert_array_equal(points, [[2.0, 3.0], [4.0, 5.0]])
    np.testing.assert_array_equal(labels, [1, 0])

    shape_layer_2d = Shapes(
        data=[np.array([[1.0, 2.0], [1.0, 8.0], [7.0, 8.0], [7.0, 2.0]])],
        shape_type=["rectangle"],
    )
    boxes, masks = annotator_util.shape_layer_to_prompts(shape_layer_2d, shape=(16, 16))
    np.testing.assert_array_equal(boxes, [[1.0, 2.0, 7.0, 8.0]])
    assert masks == [None]

    point_layer_3d = Points(
        data=np.array([[0.0, 1.0, 2.0], [1.0, 3.0, 4.0], [1.0, 5.0, 6.0]]),
        properties={"label": np.array(["negative", "positive", "negative"])},
    )
    points, labels = annotator_util.point_layer_to_prompts(
        point_layer_3d, i=1, with_stop_annotation=False
    )
    np.testing.assert_array_equal(points, [[3.0, 4.0], [5.0, 6.0]])
    np.testing.assert_array_equal(labels, [1, 0])

    shape_layer_3d = Shapes(
        data=[
            np.array(
                [[1.0, 2.0, 3.0], [1.0, 2.0, 8.0], [1.0, 7.0, 8.0], [1.0, 7.0, 3.0]]
            )
        ],
        shape_type=["rectangle"],
    )
    boxes, masks = annotator_util.shape_layer_to_prompts(shape_layer_3d, shape=(16, 16), i=1)
    np.testing.assert_array_equal(boxes, [[2.0, 3.0, 7.0, 8.0]])
    assert masks == [None]


def test_clear_slice_keeps_shape_properties_aligned():
    point_layer = Points(
        data=np.array([[1.0, 3.0, 4.0], [2.0, 5.0, 6.0]]),
        properties={"label": np.array(["negative", "positive"])},
    )
    prompt_layer = Shapes(
        data=[
            np.array([[1.0, 2.0, 2.0], [1.0, 8.0, 8.0]]),
            np.array(
                [[2.0, 2.0, 3.0], [2.0, 2.0, 8.0], [2.0, 7.0, 8.0], [2.0, 7.0, 3.0]]
            ),
        ],
        shape_type=["path", "rectangle"],
        properties={"label": np.array(["negative", "positive"])},
    )
    viewer = SimpleNamespace(layers={"point_prompts": point_layer, "prompts": prompt_layer})

    annotator_util.clear_annotations_slice(viewer, i=1, clear_segmentations=False)

    np.testing.assert_array_equal(point_layer.data, [[2.0, 5.0, 6.0]])
    assert prompt_layer.shape_type == ["rectangle"]
    np.testing.assert_array_equal(prompt_layer.properties["label"], ["positive"])


def test_closed_shapes_stay_green_while_negative_scribbles_are_red():
    layer = Shapes(
        ndim=2,
        property_choices={"label": ["positive", "negative"]},
        edge_color="label",
        edge_color_cycle=annotator_util.LABEL_COLOR_CYCLE,
    )
    layer.edge_color_mode = "cycle"
    current_properties = layer.current_properties
    current_properties["label"] = np.array(["negative"])
    layer.current_properties = current_properties
    layer.mode = "add_polyline"
    annotator_util.sync_prompt_shape_current_color(layer)
    assert layer.current_edge_color == "red"

    layer.add_rectangles(np.array([[1, 1], [8, 8]]))
    annotator_util.normalize_prompt_shape_labels(layer)
    layer.add_paths(np.array([[2, 2], [12, 12]]))
    annotator_util.normalize_prompt_shape_labels(layer)

    np.testing.assert_array_equal(layer.properties["label"], ["positive", "negative"])
    np.testing.assert_allclose(layer.edge_color[0], [0, 1, 0, 1])
    np.testing.assert_allclose(layer.edge_color[1], [1, 0, 0, 1])
    assert layer.current_properties["label"][0] == "negative"

    points, labels = annotator_util.scribble_layer_to_prompts(layer, image_shape=(16, 16))
    assert len(points) > 0
    np.testing.assert_array_equal(labels, np.zeros(len(points), dtype="int64"))


def test_selected_scribble_is_relabelled_while_polyline_tool_is_active():
    layer = Shapes(
        ndim=2,
        property_choices={"label": ["positive", "negative"]},
        edge_color="label",
        edge_color_cycle=annotator_util.LABEL_COLOR_CYCLE,
    )
    layer.edge_color_mode = "cycle"
    annotator_util.set_prompt_label(layer, "negative")
    layer.add_paths(np.array([[2.0, 2.0], [12.0, 12.0]]))
    layer.mode = "add_polyline"
    layer.selected_data = {0}

    annotator_util.set_prompt_label(layer, "positive")
    assert layer.current_properties["label"][0] == "positive"
    assert layer.properties["label"][0] == "positive"
    np.testing.assert_allclose(layer.edge_color[0], [0, 1, 0, 1])
    _, labels = annotator_util.scribble_layer_to_prompts(layer, image_shape=(16, 16))
    np.testing.assert_array_equal(labels, np.ones(len(labels), dtype="int64"))

    annotator_util.set_prompt_label(layer, "negative")
    assert layer.properties["label"][0] == "negative"
    np.testing.assert_allclose(layer.edge_color[0], [1, 0, 0, 1])
    _, labels = annotator_util.scribble_layer_to_prompts(layer, image_shape=(16, 16))
    np.testing.assert_array_equal(labels, np.zeros(len(labels), dtype="int64"))


def scribble_layer_with_one_positive_path():
    """A shape layer holding one drawn positive scribble, still selected as napari leaves it."""
    layer = Shapes(
        ndim=2,
        property_choices={"label": ["positive", "negative"]},
        edge_color="label",
        edge_color_cycle=annotator_util.LABEL_COLOR_CYCLE,
    )
    layer.edge_color_mode = "cycle"
    annotator_util.set_prompt_label(layer, "positive")
    layer.add_paths(np.array([[2.0, 2.0], [12.0, 12.0]]))
    layer.selected_data = {0}
    return layer


def point_layer_2d():
    return Points(ndim=2, property_choices={"label": ["positive", "negative"]})


def test_toggling_on_the_point_layer_does_not_relabel_a_drawn_scribble():
    """A polarity change aimed at the next point must not reach back into the shape layer."""
    scribbles = scribble_layer_with_one_positive_path()
    points = point_layer_2d()
    annotator_util.set_prompt_label(points, "positive")

    annotator_util.toggle_label(points, scribbles)

    # Both layers take the new drawing default, so the shared prompt menu stays truthful.
    assert points.current_properties["label"][0] == "negative"
    assert scribbles.current_properties["label"][0] == "negative"
    # The scribble that was already drawn keeps its own label.
    assert scribbles.properties["label"][0] == "positive"
    _, labels = annotator_util.scribble_layer_to_prompts(scribbles, image_shape=(16, 16))
    np.testing.assert_array_equal(labels, np.ones(len(labels), dtype="int64"))


def test_toggling_on_the_shape_layer_does_not_relabel_a_placed_point():
    """The mirror case: a polarity change aimed at the next scribble must not relabel a placed point."""
    scribbles = scribble_layer_with_one_positive_path()
    points = point_layer_2d()
    annotator_util.set_prompt_label(points, "positive")
    points.add(np.array([[4.0, 4.0]]))
    points.selected_data = {0}

    annotator_util.toggle_label(scribbles, points)

    assert points.current_properties["label"][0] == "negative"
    # The point that was already placed keeps its own label.
    np.testing.assert_array_equal(points.properties["label"], ["positive"])


def test_toggling_on_the_shape_layer_still_relabels_its_selected_scribble():
    """Fixing a scribble's polarity right after drawing it has to keep working."""
    scribbles = scribble_layer_with_one_positive_path()
    points = point_layer_2d()

    annotator_util.toggle_label(scribbles, points)

    assert scribbles.properties["label"][0] == "negative"
    assert points.current_properties["label"][0] == "negative"
    _, labels = annotator_util.scribble_layer_to_prompts(scribbles, image_shape=(16, 16))
    np.testing.assert_array_equal(labels, np.zeros(len(labels), dtype="int64"))


def test_set_prompt_label_can_leave_the_selection_alone():
    scribbles = scribble_layer_with_one_positive_path()

    annotator_util.set_prompt_label(scribbles, "negative", relabel_selected=False)

    assert scribbles.current_properties["label"][0] == "negative"
    assert scribbles.properties["label"][0] == "positive"


def test_prompt_label_change_drops_stale_shape_selection():
    layer = Shapes(
        ndim=3,
        property_choices={"label": ["positive", "negative"]},
        edge_color="label",
        edge_color_cycle=annotator_util.LABEL_COLOR_CYCLE,
    )
    # Simulate the transient state seen after napari removes the final shape: geometry/features are
    # already empty, while Selection has not emitted its clearing event yet.
    layer._selected_data._data = {0: None}

    annotator_util.set_prompt_label(layer, "negative")

    assert layer.selected_data == set()
    assert layer.current_properties["label"][0] == "negative"


def test_merge_point_and_scribble_prompts():
    points, labels = annotator_util.merge_point_prompts(
        (np.array([[4, 5]]), np.array([1])),
        (np.array([[8, 9], [10, 11]]), np.array([1, 0])),
    )

    np.testing.assert_array_equal(points, [[4, 5], [8, 9], [10, 11]])
    np.testing.assert_array_equal(labels, [1, 1, 0])


def test_volume_scribbles_pass_layer_validation(monkeypatch):
    from micro_sam.sam_annotator import _widgets

    prompt_layer = _PromptShapesLayer(
        data=[np.array([[2, 2, 2], [2, 8, 8]])],
        shape_type=["path"],
        labels=["positive"],
    )
    viewer = SimpleNamespace(layers={
        "current_object": SimpleNamespace(data=np.zeros((4, 16, 16), dtype="uint32")),
        "prompts": prompt_layer,
        "point_prompts": SimpleNamespace(data=[]),
    })
    annotator = SimpleNamespace(_require_layers=lambda: None)
    monkeypatch.setattr(_widgets, "AnnotatorState", lambda: SimpleNamespace(annotator=annotator))
    result = _widgets._validate_layers(viewer)

    assert result is False


def test_slice_segmentation_merges_3d_scribbles_with_points(monkeypatch):
    from micro_sam.sam_annotator import _widgets

    class _CurrentObject:
        def __init__(self):
            self.data = np.zeros((3, 32, 32), dtype="uint32")
            self.refresh_count = 0

        def refresh(self):
            self.refresh_count += 1

    class _Segmenter:
        def __init__(self):
            self.captured = None

        def segment_slice(self, **kwargs):
            self.captured = kwargs
            return np.ones((32, 32), dtype="uint32")

    current_object = _CurrentObject()
    point_layer = Points(
        data=np.array([[1.0, 3.0, 4.0]]),
        properties={"label": np.array(["positive"])},
    )
    prompt_layer = Shapes(
        data=[np.array([[1.0, 8.0, 8.0], [1.0, 16.0, 16.0]])],
        shape_type=["path"],
        properties={"label": np.array(["negative"])},
    )
    viewer = SimpleNamespace(
        layers={
            "current_object": current_object,
            "point_prompts": point_layer,
            "prompts": prompt_layer,
        },
        dims=SimpleNamespace(point=(1.0, 0.0, 0.0)),
    )
    segmenter = _Segmenter()
    state = SimpleNamespace(
        is_sam2=True,
        image_embeddings={"input_size": (1024, 1024)},
        interactive_segmenter=segmenter,
    )
    widget = SimpleNamespace(_viewer=viewer, batched=True)
    messages = []
    monkeypatch.setattr(_widgets, "AnnotatorState", lambda: state)
    monkeypatch.setattr(_widgets, "show_info", messages.append)

    _widgets.UnifiedSegmentWidget._run_slice_segmentation(widget)

    captured = segmenter.captured
    assert captured["frame_idx"] == 1
    np.testing.assert_array_equal(captured["labels"][0], 1)
    np.testing.assert_array_equal(captured["labels"][1:], 0)
    np.testing.assert_array_equal(captured["points"][0], [4.0, 3.0])
    assert captured["boxes"] == []
    assert "not supported with scribble prompts" in messages[0]
    np.testing.assert_array_equal(current_object.data[1], 1)
    assert current_object.refresh_count == 1


def test_track_frame_segmentation_merges_scribbles_for_the_active_track(monkeypatch):
    """The per-frame tracking path merges a track's scribble into its point prompts. It ignores
    the scribbles that belong to a different track."""
    from micro_sam.sam_annotator import _widgets

    class _Segmenter:
        def __init__(self):
            self.captured = None

        def segment_slice(self, **kwargs):
            self.captured = kwargs
            return np.ones((1, 32, 32), dtype="uint32")

    point_layer = Points(
        data=np.array([[1.0, 3.0, 4.0]]),
        properties={"label": np.array(["positive"]), "track_id": np.array(["1"])},
    )
    prompt_layer = Shapes(
        data=[
            np.array([[1.0, 8.0, 8.0], [1.0, 16.0, 16.0]]),
            np.array([[1.0, 20.0, 20.0], [1.0, 28.0, 28.0]]),
        ],
        shape_type=["path", "path"],
        properties={"label": np.array(["negative", "negative"]), "track_id": np.array(["1", "2"])},
    )
    viewer = SimpleNamespace(layers={"point_prompts": point_layer, "prompts": prompt_layer})
    segmenter = _Segmenter()
    state = SimpleNamespace(interactive_segmenter=segmenter)
    widget = SimpleNamespace(_viewer=viewer)
    monkeypatch.setattr(_widgets, "show_info", lambda message: None)

    mask = _widgets.UnifiedSegmentWidget._segment_track_on_frame(
        widget, state, t=1, track_id=1, shape=(32, 32)
    )

    # The merge keeps only the track-1 scribble. It drops the track-2 stroke.
    exp_scribble, exp_scribble_labels = annotator_util.scribble_layer_to_prompts(
        prompt_layer, image_shape=(32, 32), i=1, track_id=1
    )
    expected_points = np.concatenate([[[3.0, 4.0]], exp_scribble])[:, ::-1]
    expected_labels = np.concatenate([[1], exp_scribble_labels])

    captured = segmenter.captured
    np.testing.assert_allclose(captured["points"], expected_points)
    np.testing.assert_array_equal(captured["labels"], expected_labels)
    assert captured["boxes"] == []
    assert mask is not None


def test_sam2_volume_propagation_merges_3d_scribbles_points_and_boxes(monkeypatch):
    from micro_sam.sam_annotator import _widgets

    class _Signal:
        def emit(self, *args):
            pass

    class _CurrentObject:
        def __init__(self):
            self.data = np.zeros((3, 32, 32), dtype="uint32")
            self.refresh_count = 0

        def refresh(self):
            self.refresh_count += 1

    class _Segmenter:
        def __init__(self):
            self.reset_count = 0
            self.point_calls = []
            self.box_calls = []
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
            raise AssertionError("No mask prompt was expected.")

        def get_progress_total(self, z_range):
            return 3

        def predict(self, **kwargs):
            return np.ones((3, 32, 32), dtype="uint32")

    current_object = _CurrentObject()
    point_layer = Points(
        data=np.array([[1.0, 3.0, 4.0]]),
        properties={"label": np.array(["positive"])},
    )
    prompt_layer = Shapes(
        data=[
            np.array([[1.0, 8.0, 8.0], [1.0, 16.0, 16.0]]),
            np.array(
                [[2.0, 2.0, 3.0], [2.0, 2.0, 8.0], [2.0, 7.0, 8.0], [2.0, 7.0, 3.0]]
            ),
        ],
        shape_type=["path", "rectangle"],
        properties={"label": np.array(["negative", "positive"])},
    )
    viewer = SimpleNamespace(layers={
        "current_object": current_object,
        "point_prompts": point_layer,
        "prompts": prompt_layer,
    })
    segmenter = _Segmenter()
    state = SimpleNamespace(
        is_sam2=True,
        image_shape=(3, 32, 32),
        image_embeddings={"input_size": (1024, 1024)},
        interactive_segmenter=segmenter,
    )
    widget = SimpleNamespace(
        _viewer=viewer,
        batched=True,
        early_stop_patience=0,
        z_range=None,
    )
    signals = SimpleNamespace(
        pbar_total=_Signal(),
        pbar_description=_Signal(),
        pbar_update=_Signal(),
        pbar_stop=_Signal(),
    )
    messages = []
    monkeypatch.setattr(_widgets, "AnnotatorState", lambda: state)
    monkeypatch.setattr(_widgets, "_create_pbar_for_threadworker", lambda: (None, signals))
    monkeypatch.setattr(_widgets, "show_info", messages.append)

    _widgets.UnifiedSegmentWidget._run_volumetric_segmentation(widget)

    # Nothing was pushed before, so the first run has no stale prompts to discard.
    assert segmenter.reset_count == 0
    # The signatures cover the scribble-derived points, the point prompt and the rectangle, each
    # tagged with the object it is routed to ('point', object_id, frame_id, y, x, label).
    assert ("point", 1, 1, 3, 4, 1) in segmenter.signatures
    assert sum(1 for sig in segmenter.signatures if sig[0] == "point") > 1
    assert sum(1 for sig in segmenter.signatures if sig[0] == "box") == 1
    assert len(segmenter.box_calls) == 1
    assert segmenter.box_calls[0]["frame_ids"] == 2
    assert len(segmenter.point_calls) == 1
    point_call = segmenter.point_calls[0]
    assert point_call["frame_ids"] == 1
    np.testing.assert_array_equal(point_call["point_labels"][0], 1)
    np.testing.assert_array_equal(point_call["point_labels"][1:], 0)
    assert point_call["object_id"] is None
    assert "not supported with scribble prompts" in messages[0]
    np.testing.assert_array_equal(current_object.data, 1)
    assert current_object.refresh_count == 1


def test_legacy_volume_segmentation_merges_slice_scribbles(monkeypatch):
    point_layer = Points(ndim=3, properties={"label": np.empty((0,), dtype=str)})
    prompt_layer = Shapes(
        data=[np.array([[1.0, 4.0, 5.0], [1.0, 12.0, 13.0]])],
        shape_type=["path"],
        properties={"label": np.array(["positive"])},
    )
    captured = {}

    def _prompt_segmentation(predictor, points, labels, boxes, masks, shape, **kwargs):
        captured.update(points=points, labels=labels, boxes=boxes, masks=masks, shape=shape, kwargs=kwargs)
        return np.ones(shape, dtype="uint32")

    monkeypatch.setattr(annotator_util, "prompt_segmentation", _prompt_segmentation)

    seg, slices, stop_lower, stop_upper = annotator_util.segment_slices_with_prompts(
        predictor=object(),
        point_prompts=point_layer,
        box_prompts=prompt_layer,
        image_embeddings=object(),
        shape=(3, 32, 32),
    )

    np.testing.assert_array_equal(slices, [1])
    assert stop_lower is False
    assert stop_upper is False
    assert len(captured["points"]) > 1
    np.testing.assert_array_equal(captured["labels"], np.ones(len(captured["labels"]), dtype="int64"))
    assert captured["boxes"] == []
    np.testing.assert_array_equal(seg[1], 1)


def test_negative_only_scribble_is_rejected_before_prediction(monkeypatch):
    from micro_sam.sam_annotator import _widgets

    prompt_layer = _PromptShapesLayer(
        data=[np.array([[2.0, 2.0], [16.0, 16.0]])],
        shape_type=["path"],
        labels=["negative"],
    )
    viewer = SimpleNamespace(layers={
        "current_object": SimpleNamespace(data=np.zeros((32, 32), dtype="uint32")),
        "prompts": prompt_layer,
        "point_prompts": SimpleNamespace(data=[]),
    })
    monkeypatch.setattr(_widgets, "_validate_embeddings", lambda viewer: False)
    monkeypatch.setattr(_widgets, "_validate_layers", lambda viewer: False)
    monkeypatch.setattr(
        _widgets.vutil, "point_layer_to_prompts",
        lambda layer, **kwargs: (np.empty((0, 2)), np.empty((0,), dtype="int64")),
    )
    monkeypatch.setattr(_widgets, "_generate_message", lambda message_type, message: (message_type, message))

    result = _widgets._segment_object_2d(viewer)

    assert result == (
        "error",
        "A negative scribble needs a positive point, positive scribble, box or mask prompt.",
    )


def test_2d_segmentation_merges_scribbles_with_clicks(monkeypatch):
    from micro_sam.sam_annotator import _widgets
    from micro_sam.v2 import prompt_based_segmentation

    class _Layer:
        def __init__(self, data):
            self.data = data
            self.refresh_count = 0

        def refresh(self):
            self.refresh_count += 1

    current_object = _Layer(np.zeros((32, 32), dtype="uint32"))
    prompt_layer = _Layer([np.array([[2, 2], [20, 20]])])
    viewer = SimpleNamespace(layers={
        "current_object": current_object,
        "prompts": prompt_layer,
        "point_prompts": _Layer(np.empty((0, 2))),
    })
    state = SimpleNamespace(
        predictor=object(), image_embeddings={"input_size": (1024, 1024)}, is_sam2=True,
    )

    monkeypatch.setattr(_widgets, "_validate_embeddings", lambda viewer: False)
    monkeypatch.setattr(_widgets, "_validate_layers", lambda viewer: False)
    monkeypatch.setattr(_widgets, "AnnotatorState", lambda: state)
    converted_layers = {}

    def _shape_prompts(layer, shape, **kwargs):
        converted_layers["shape"] = layer
        return [], []

    monkeypatch.setattr(_widgets.vutil, "shape_layer_to_prompts", _shape_prompts)
    monkeypatch.setattr(
        _widgets.vutil, "point_layer_to_prompts",
        lambda layer, **kwargs: (np.array([[3.0, 4.0]]), np.array([1])),
    )

    def _scribble_prompts(layer, image_shape=None, **kwargs):
        converted_layers["scribble"] = layer
        return np.array([[8.0, 9.0], [10.0, 11.0]]), np.array([1, 0])

    monkeypatch.setattr(_widgets.vutil, "scribble_layer_to_prompts", _scribble_prompts)
    captured = {}

    def _predict(**kwargs):
        captured.update(kwargs)
        return np.ones((32, 32), dtype="uint8")

    monkeypatch.setattr(prompt_based_segmentation, "promptable_segmentation_2d", _predict)
    monkeypatch.setattr(_widgets, "show_info", lambda message: captured.setdefault("message", message))

    _widgets._segment_object_2d(viewer, batched=True)

    np.testing.assert_array_equal(captured["points"], [[3, 4], [8, 9], [10, 11]])
    np.testing.assert_array_equal(captured["labels"], [1, 1, 0])
    assert captured["batched"] is False
    assert "not supported with scribble prompts" in captured["message"]
    assert converted_layers == {"shape": prompt_layer, "scribble": prompt_layer}
    np.testing.assert_array_equal(current_object.data, 1)
    assert current_object.refresh_count == 1


@pytest.mark.parametrize("is_sam2", [False, True])
def test_point_and_box_only_segmentation_is_unchanged(monkeypatch, is_sam2):
    """The scribble integration must be a no-op for the existing point/box-only workflow."""
    from micro_sam.sam_annotator import _widgets
    from micro_sam.v2 import prompt_based_segmentation

    class _Layer:
        def __init__(self, data):
            self.data = data
            self.refresh_count = 0

        def refresh(self):
            self.refresh_count += 1

    current_object = _Layer(np.zeros((32, 32), dtype="uint32"))
    prompt_layer = _Layer([np.array([[2, 3], [20, 21]])])
    point_layer = _Layer(np.array([[7.0, 8.0]]))
    viewer = SimpleNamespace(layers={
        "current_object": current_object,
        "prompts": prompt_layer,
        "point_prompts": point_layer,
    })
    state = SimpleNamespace(
        predictor=object(), image_embeddings={"input_size": (1024, 1024)}, is_sam2=is_sam2,
    )
    boxes = np.array([[2.0, 3.0, 20.0, 21.0]])
    points = np.array([[7.0, 8.0]])
    labels = np.array([1])

    monkeypatch.setattr(_widgets, "_validate_embeddings", lambda viewer: False)
    monkeypatch.setattr(_widgets, "_validate_layers", lambda viewer: False)
    monkeypatch.setattr(_widgets, "AnnotatorState", lambda: state)
    monkeypatch.setattr(
        _widgets.vutil, "shape_layer_to_prompts", lambda layer, shape, **kwargs: (boxes, [None]),
    )
    monkeypatch.setattr(
        _widgets.vutil, "point_layer_to_prompts",
        lambda layer, **kwargs: (points, labels),
    )
    monkeypatch.setattr(
        _widgets.vutil, "scribble_layer_to_prompts",
        lambda layer, image_shape=None, **kwargs: (np.empty((0, 2)), np.empty((0,), dtype="int64")),
    )
    captured = {}

    if is_sam2:
        def _predict(**kwargs):
            captured.update(kwargs)
            return np.ones((32, 32), dtype="uint8")

        monkeypatch.setattr(prompt_based_segmentation, "promptable_segmentation_2d", _predict)
    else:
        def _predict(predictor, points, labels, boxes, masks, shape, **kwargs):
            captured.update(
                points=points, labels=labels, boxes=boxes, masks=masks, shape=shape, **kwargs
            )
            return np.ones(shape, dtype="uint8")

        monkeypatch.setattr(_widgets.vutil, "prompt_segmentation", _predict)

    _widgets._segment_object_2d(viewer, batched=True)

    np.testing.assert_array_equal(captured["points"], points)
    np.testing.assert_array_equal(captured["labels"], labels)
    np.testing.assert_array_equal(captured["boxes"], boxes)
    assert captured["masks"] == [None]
    assert captured["batched"] is True
    np.testing.assert_array_equal(current_object.data, 1)
    assert current_object.refresh_count == 1


def point_layer_3d(coords, labels):
    return Points(data=np.asarray(coords, dtype=float), properties={"label": np.asarray(labels)})


def shape_layer_3d(data, shape_type, labels):
    return Shapes(data=data, shape_type=shape_type, properties={"label": np.asarray(labels)})


def empty_shapes():
    return Shapes(ndim=3, properties={"label": np.empty((0,), dtype=str)})


def test_collect_frame_prompts_reads_a_lone_negative_point_as_a_stop():
    layer = point_layer_3d([[1.0, 10.0, 12.0]], ["negative"])

    prompts = annotator_util.collect_frame_prompts(layer, empty_shapes(), (32, 32), i=1)

    assert prompts.is_stop
    assert len(prompts.points) == 0 and not prompts.boxes


def test_collect_frame_prompts_scribble_suppresses_the_stop():
    """The unified rule: a lone negative point is only a stop when it is the frame's sole cue."""
    points = point_layer_3d([[1.0, 10.0, 12.0]], ["negative"])
    scribble = shape_layer_3d(
        [np.array([[1.0, 2.0, 2.0], [1.0, 8.0, 8.0]])], ["path"], ["positive"],
    )

    prompts = annotator_util.collect_frame_prompts(points, scribble, (32, 32), i=1)

    assert not prompts.is_stop
    assert prompts.have_scribbles
    # The scribble points are merged in alongside the negative point.
    assert np.any(prompts.labels == 1) and np.any(prompts.labels == 0)


def test_collect_frame_prompts_box_suppresses_the_stop():
    """A box on the frame defines an object, so the lone negative point corrects it instead of stopping."""
    points = point_layer_3d([[1.0, 10.0, 12.0]], ["negative"])
    boxes = shape_layer_3d(
        [np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 8.0], [1.0, 7.0, 8.0], [1.0, 7.0, 3.0]])],
        ["rectangle"], ["positive"],
    )

    prompts = annotator_util.collect_frame_prompts(points, boxes, (32, 32), i=1)

    assert not prompts.is_stop
    assert len(prompts.boxes) == 1
    np.testing.assert_array_equal(prompts.labels, [0])


def test_collect_frame_prompts_mask_shape_suppresses_the_stop():
    points = point_layer_3d([[1.0, 10.0, 12.0]], ["negative"])
    shapes = shape_layer_3d(
        [np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 9.0], [1.0, 8.0, 9.0], [1.0, 8.0, 3.0]])],
        ["polygon"], ["positive"],
    )

    prompts = annotator_util.collect_frame_prompts(points, shapes, (32, 32), i=1)

    assert not prompts.is_stop
    assert prompts.split_shapes()[1]  # the polygon carries a filled mask
    np.testing.assert_array_equal(prompts.labels, [0])


def test_collect_frame_prompts_can_disable_the_stop_annotation():
    layer = point_layer_3d([[1.0, 10.0, 12.0]], ["negative"])

    prompts = annotator_util.collect_frame_prompts(
        layer, empty_shapes(), (32, 32), i=1, with_stop_annotation=False,
    )

    assert not prompts.is_stop
    np.testing.assert_array_equal(prompts.labels, [0])


def test_collect_frame_prompts_splits_rectangles_from_mask_shapes():
    points = point_layer_3d(np.zeros((0, 3)), [])
    shapes = shape_layer_3d(
        [
            np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 8.0], [1.0, 7.0, 8.0], [1.0, 7.0, 3.0]]),
            np.array([[1.0, 12.0, 13.0], [1.0, 12.0, 18.0], [1.0, 17.0, 18.0], [1.0, 17.0, 13.0]]),
        ],
        ["rectangle", "polygon"], ["positive", "positive"],
    )

    prompts = annotator_util.collect_frame_prompts(points, shapes, (32, 32), i=1)
    rect_boxes, poly_masks = prompts.split_shapes()

    assert len(rect_boxes) == 1 and len(poly_masks) == 1
    assert len(prompts.boxes) == 2  # every shape contributes a box, the polygon also a mask
