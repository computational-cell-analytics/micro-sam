from types import SimpleNamespace

import numpy as np
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

        def reset_predictor(self):
            self.reset_count += 1

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

    assert segmenter.reset_count == 1
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
        lambda layer, with_stop_annotation: (np.empty((0, 2)), np.empty((0,), dtype="int64")),
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

    def _shape_prompts(layer, shape):
        converted_layers["shape"] = layer
        return [], []

    monkeypatch.setattr(_widgets.vutil, "shape_layer_to_prompts", _shape_prompts)
    monkeypatch.setattr(
        _widgets.vutil, "point_layer_to_prompts",
        lambda layer, with_stop_annotation: (np.array([[3.0, 4.0]]), np.array([1])),
    )

    def _scribble_prompts(layer, image_shape):
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
