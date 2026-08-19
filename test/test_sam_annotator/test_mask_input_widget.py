import platform

import numpy as np
import pytest

from qtpy import QtWidgets
from qtpy.QtCore import Qt

from micro_sam.sam_annotator import _widgets
from micro_sam.sam_annotator._state import AnnotatorState
from micro_sam.sam_annotator._widgets import _commit_impl
from micro_sam.sam_annotator.annotator import Annotator


pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(
        platform.system() == "Windows",
        reason="GUI tests are not reliable on Windows.",
    ),
]


def _item_for_layer(widget, layer):
    """Return the checklist item for a layer without depending on its display name."""
    for candidate, item in widget._mask_layer_items.values():
        if candidate is layer or candidate.name == layer.name:
            return item
    raise AssertionError(f"No checklist item found for layer {layer.name!r}.")


def _set_only_checked(widget, layer):
    for candidate, item in widget._mask_layer_items.values():
        is_target = candidate is layer or candidate.name == layer.name
        item.setCheckState(Qt.Checked if is_target else Qt.Unchecked)


def _listed_mask_layers(widget):
    return [layer for layer, _ in widget._mask_layer_items.values()]


def test_mask_layer_checklist_tracks_compatible_layers(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    image = viewer.add_image(np.zeros((16, 16), dtype="uint8"), name="image")
    compatible = viewer.add_labels(
        np.zeros(image.data.shape, dtype="uint32"),
        name="cellpose",
    )
    incompatible_shape = viewer.add_labels(
        np.zeros((8, 8), dtype="uint32"),
        name="wrong shape",
    )
    incompatible_transform = viewer.add_labels(
        np.zeros(image.data.shape, dtype="uint32"),
        name="wrong transform",
        translate=(1, 0),
    )

    annotator = Annotator(viewer)
    widget = annotator._widgets["interactive"]

    listed_names = {layer.name for layer in _listed_mask_layers(widget)}
    assert listed_names.isdisjoint(
        {"auto_segmentation", "committed_objects", "current_object"}
    )
    internal_current_object = viewer.layers["current_object"]
    internal_current_object.name = "renamed micro-sam output"
    viewer.add_labels(np.zeros(image.data.shape, dtype="uint32"), name="trigger refresh")
    assert internal_current_object not in _listed_mask_layers(widget)
    compatible_item = _item_for_layer(widget, compatible)
    incompatible_shape_item = _item_for_layer(widget, incompatible_shape)
    incompatible_transform_item = _item_for_layer(widget, incompatible_transform)
    assert compatible_item.flags() & Qt.ItemIsEnabled
    assert not incompatible_shape_item.flags() & Qt.ItemIsEnabled
    assert not incompatible_transform_item.flags() & Qt.ItemIsEnabled
    assert compatible_item.checkState() == Qt.Unchecked

    compatible_item.setCheckState(Qt.Checked)
    compatible.name = "renamed cellpose"
    assert _item_for_layer(widget, compatible).checkState() == Qt.Checked

    widget._set_all_mask_layers_checked(False)
    assert all(
        item.checkState() == Qt.Unchecked
        for _, item in widget._mask_layer_items.values()
    )
    widget._set_all_mask_layers_checked(True)
    assert _item_for_layer(widget, compatible).checkState() == Qt.Checked
    assert _item_for_layer(widget, incompatible_shape).checkState() == Qt.Unchecked
    assert _item_for_layer(widget, incompatible_transform).checkState() == Qt.Unchecked

    viewer.close()


def test_mask_layer_checklist_accepts_and_crops_napari_trailing_surplus(
    make_napari_viewer_proxy,
):
    viewer = make_napari_viewer_proxy()
    image = viewer.add_image(np.zeros((12, 12), dtype="uint8"), name="image")
    annotator = Annotator(viewer)
    widget = annotator._widgets["interactive"]

    labels = np.zeros((13, 13), dtype="uint32")
    labels[2:5, 3:6] = 11
    labels[12, 12] = 99
    napari_labels = viewer.add_labels(labels, name="Labels")

    item = _item_for_layer(widget, napari_labels)
    assert item.flags() & Qt.ItemIsEnabled
    _set_only_checked(widget, napari_labels)

    mask_inputs = widget._collect_mask_inputs()
    assert mask_inputs.labels.shape == image.data.shape
    assert mask_inputs.object_ids == (11,)
    assert mask_inputs.cropped_source_names == ("Labels",)
    assert "Cropped one trailing pixel on one or more axes" in widget.mask_input_summary.text()
    assert "Labels" in widget.mask_input_summary.text()

    widget._commit_selected_masks_unchanged()
    committed = viewer.layers["committed_objects"].data
    assert committed.shape == image.data.shape
    assert set(np.unique(committed)) == {0, 11}

    viewer.close()


def test_segment_object_routes_selected_masks_to_refinement(
    make_napari_viewer_proxy,
    monkeypatch,
):
    viewer = make_napari_viewer_proxy()
    viewer.add_image(np.zeros((12, 12), dtype="uint8"), name="image")
    annotator = Annotator(viewer)
    widget = annotator._widgets["interactive"]
    calls = []

    monkeypatch.setattr(
        _widgets,
        "_segment_object_2d",
        lambda viewer, batched: calls.append(("prompts", batched)),
    )
    widget.segment()
    assert calls == [("prompts", False)]

    labels = np.zeros((12, 12), dtype="uint32")
    labels[2:5, 3:6] = 11
    cellpose = viewer.add_labels(labels, name="cellpose")
    _set_only_checked(widget, cellpose)
    monkeypatch.setattr(
        widget,
        "_refine_selected_masks",
        lambda: calls.append(("masks", False)),
    )
    widget.segment()
    assert calls[-1] == ("masks", False)

    button_texts = {
        button.text() for button in widget.findChildren(QtWidgets.QPushButton)
    }
    assert "Segment Object [S]" in button_texts
    assert "Refine with SAM" not in button_texts

    viewer.close()


def test_prompt_targets_are_stored_per_point_and_shape(
    make_napari_viewer_proxy,
):
    viewer = make_napari_viewer_proxy()
    viewer.add_image(np.zeros((20, 20), dtype="uint8"), name="image")
    annotator = Annotator(viewer)
    widget = annotator._widgets["interactive"]

    labels = np.zeros((20, 20), dtype="uint32")
    labels[1:8, 1:8] = 7
    labels[11:18, 11:18] = 19
    input_layer = viewer.add_labels(labels, name="cellpose")
    _set_only_checked(widget, input_layer)

    assert not widget.correction_target_row.isHidden()
    assert not widget.correction_target_help.isHidden()
    assert widget.correction_target.currentIndex() == -1
    assert [
        widget.correction_target.itemData(index)
        for index in range(widget.correction_target.count())
    ] == [7, 19]

    points = viewer.layers["point_prompts"]
    shapes = viewer.layers["prompts"]
    assert "object_id" in points.properties
    assert "object_id" in shapes.properties

    widget.correction_target.setCurrentIndex(widget.correction_target.findData(7))
    points.add(np.array([[3.0, 3.0]]))
    points.selected_data = set()
    widget.correction_target.setCurrentIndex(widget.correction_target.findData(19))
    points.add(np.array([[14.0, 14.0]]))
    points.selected_data = set()
    shapes.add_rectangles(np.array([[12.0, 12.0], [16.0, 16.0]]))

    np.testing.assert_array_equal(points.properties["object_id"], ["7", "19"])
    np.testing.assert_array_equal(shapes.properties["object_id"], ["19"])

    _, _, prompts_7 = widget._collect_mask_prompts(labels == 7, object_id=7)
    _, _, prompts_19 = widget._collect_mask_prompts(labels == 19, object_id=19)
    np.testing.assert_array_equal(prompts_7.points, [[3.0, 3.0]])
    np.testing.assert_array_equal(prompts_19.points, [[14.0, 14.0]])
    assert len(prompts_7.boxes) == 0
    assert len(prompts_19.boxes) == 1

    points.selected_data = {0}
    assert widget.correction_target.currentData() == 7
    assert shapes.current_properties["object_id"][0] == "7"

    points.selected_data = {0, 1}
    assert widget.correction_target.currentIndex() == -1
    assert widget.correction_target.placeholderText() == "Multiple target IDs"
    points.selected_data = set()
    assert widget.correction_target.currentData() == 7
    assert points.current_properties["object_id"][0] == "7"
    assert shapes.current_properties["object_id"][0] == "7"

    points.selected_data = {0}
    widget.correction_target.setCurrentIndex(widget.correction_target.findData(19))
    np.testing.assert_array_equal(points.properties["object_id"], ["19", "19"])

    widget._validate_prompt_target_ids(widget._collect_mask_inputs())

    button_texts = {
        button.text() for button in widget.findChildren(QtWidgets.QPushButton)
    }
    assert "Commit selected masks unchanged" in button_texts
    assert "Commit input masks unchanged" not in button_texts
    assert widget._prompt_widget[0].label == "Prompt type"

    viewer.close()


def test_single_mask_id_assignment_preserves_selected_prompt_types(
    make_napari_viewer_proxy,
):
    viewer = make_napari_viewer_proxy()
    viewer.add_image(np.zeros((12, 12), dtype="uint8"), name="image")
    annotator = Annotator(viewer)
    widget = annotator._widgets["interactive"]
    points = viewer.layers["point_prompts"]
    points.add(np.array([[2.0, 2.0], [8.0, 8.0]]))
    properties = dict(points.properties)
    properties["label"] = np.array(["positive", "negative"])
    points.properties = properties
    points.selected_data = {0, 1}

    labels = np.zeros((12, 12), dtype="uint32")
    labels[1:5, 1:5] = 7
    input_layer = viewer.add_labels(labels, name="cellpose")
    _set_only_checked(widget, input_layer)

    np.testing.assert_array_equal(points.properties["label"], ["positive", "negative"])
    np.testing.assert_array_equal(points.properties["object_id"], ["7", "7"])
    assert points.selected_data == {0, 1}

    viewer.close()


def test_multiple_mask_ids_require_every_correction_to_have_a_target(
    make_napari_viewer_proxy,
):
    viewer = make_napari_viewer_proxy()
    viewer.add_image(np.zeros((12, 12), dtype="uint8"), name="image")
    annotator = Annotator(viewer)
    widget = annotator._widgets["interactive"]

    labels = np.zeros((12, 12), dtype="uint32")
    labels[1:4, 1:4] = 7
    labels[7:10, 7:10] = 19
    input_layer = viewer.add_labels(labels, name="cellpose")
    _set_only_checked(widget, input_layer)
    viewer.layers["point_prompts"].add(np.array([[2.0, 2.0]]))

    with pytest.raises(ValueError, match="Invalid or unassigned targets"):
        widget._validate_prompt_target_ids(widget._collect_mask_inputs())

    widget.correction_target.setCurrentIndex(widget.correction_target.findData(7))
    widget._validate_prompt_target_ids(widget._collect_mask_inputs())

    viewer.close()


def test_direct_commit_preserves_ids_unions_same_id_and_is_atomic(
    make_napari_viewer_proxy,
    monkeypatch,
):
    monkeypatch.setattr(_widgets, "_generate_message", lambda *args: True)
    viewer = make_napari_viewer_proxy()
    viewer.add_image(np.zeros((12, 12), dtype="uint8"), name="image")
    annotator = Annotator(viewer)
    widget = annotator._widgets["interactive"]

    incoming = np.zeros((12, 12), dtype="uint32")
    incoming[1:4, 1:4] = 7
    incoming[7:9, 7:9] = 23
    cellpose = viewer.add_labels(incoming, name="cellpose")
    _set_only_checked(widget, cellpose)
    widget._commit_selected_masks_unchanged()
    np.testing.assert_array_equal(viewer.layers["committed_objects"].data, incoming)

    same_id = np.zeros_like(incoming)
    same_id[3:6, 3:6] = 7
    extension = viewer.add_labels(same_id, name="cellpose extension")
    _set_only_checked(widget, extension)
    widget._commit_selected_masks_unchanged()
    expected = incoming.copy()
    expected[3:6, 3:6] = 7
    np.testing.assert_array_equal(viewer.layers["committed_objects"].data, expected)

    conflict = np.zeros_like(incoming)
    conflict[2, 2] = 99
    conflicting_layer = viewer.add_labels(conflict, name="conflict")
    _set_only_checked(widget, conflicting_layer)
    before = viewer.layers["committed_objects"].data.copy()
    widget._commit_selected_masks_unchanged()
    np.testing.assert_array_equal(viewer.layers["committed_objects"].data, before)

    viewer.close()


def test_direct_commit_keeps_the_selected_image_transform(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    image = viewer.add_image(
        np.zeros((12, 12), dtype="uint8"),
        name="image",
        scale=(2, 3),
        translate=(4, 5),
    )
    annotator = Annotator(viewer)
    widget = annotator._widgets["interactive"]

    labels = np.zeros((12, 12), dtype="uint32")
    labels[2:5, 3:6] = 11
    cellpose = viewer.add_labels(
        labels,
        name="cellpose",
        scale=image.scale,
        translate=image.translate,
    )
    _set_only_checked(widget, cellpose)
    widget._commit_selected_masks_unchanged()

    committed = viewer.layers["committed_objects"]
    np.testing.assert_allclose(
        committed.data_to_world((0, 0)),
        image.data_to_world((0, 0)),
    )
    np.testing.assert_allclose(
        committed.data_to_world((1, 1)),
        image.data_to_world((1, 1)),
    )

    viewer.close()


def test_direct_commit_survives_same_image_embedding_initialization(
    make_napari_viewer_proxy,
):
    viewer = make_napari_viewer_proxy()
    image = viewer.add_image(np.zeros((10, 10), dtype="uint8"), name="image")
    annotator = Annotator(viewer)
    widget = annotator._widgets["interactive"]

    labels = np.zeros(image.data.shape, dtype="uint32")
    labels[2:6, 3:7] = 17
    cellpose = viewer.add_labels(labels, name="cellpose")
    _set_only_checked(widget, cellpose)
    widget._commit_selected_masks_unchanged()

    state = AnnotatorState()
    state.image_shape = image.data.shape
    state.image_scale = image.scale
    state.skip_recomputing_embeddings = False
    annotator._update_image()

    np.testing.assert_array_equal(
        viewer.layers["committed_objects"].data,
        labels,
    )

    viewer.close()


def test_refined_commit_replaces_source_ids_without_offset(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    viewer.add_image(np.zeros((12, 12), dtype="uint8"), name="image")
    Annotator(viewer)

    committed = np.zeros((12, 12), dtype="uint32")
    committed[1:4, 1:4] = 7
    committed[8:10, 8:10] = 31
    viewer.layers["committed_objects"].data = committed

    refined = np.zeros_like(committed)
    refined[4:7, 4:7] = 7
    current = viewer.layers["current_object"]
    current.data = refined
    current.metadata["micro_sam_mask_refinement"] = {
        "object_ids": (7,),
        "source_layers": ("cellpose",),
    }

    _commit_impl(viewer, "current_object", "pixels", 0.75)
    expected = np.zeros_like(committed)
    expected[4:7, 4:7] = 7
    expected[8:10, 8:10] = 31
    np.testing.assert_array_equal(viewer.layers["committed_objects"].data, expected)

    viewer.close()


def test_3d_seed_uses_largest_cross_section_and_lowest_tie(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    image = viewer.add_image(np.zeros((5, 12, 12), dtype="uint8"), name="volume")
    annotator = Annotator(viewer, ndim=3)
    widget = annotator._widgets["interactive"]

    labels = np.zeros(image.data.shape, dtype="uint32")
    labels[1, 2:5, 2:5] = 13
    labels[3, 6:9, 6:9] = 13
    cellpose = viewer.add_labels(labels, name="cellpose")
    _set_only_checked(widget, cellpose)
    mask_inputs = widget._collect_mask_inputs()

    assert widget._seed_slice(mask_inputs, 13) == 1
    assert widget.mask_3d_strategy.currentText() == "Refine existing slices only"
    assert "z-extent is preserved" in widget.mask_3d_help.text()
    assert not widget.batched_checkbox.isEnabled()
    assert "already processed independently" in widget.batched_checkbox.toolTip()
    assert "does not change mask refinement" in widget.apply_to_volume_checkbox.toolTip()

    widget.mask_3d_strategy.setCurrentText("Refine and extend through z")
    assert "largest cross-section" in widget.mask_3d_help.text()
    assert "not used as anchors" in widget.mask_3d_help.text()
    assert not widget._propagation_settings.isHidden()

    widget._set_all_mask_layers_checked(False)
    assert widget.batched_checkbox.isEnabled()
    assert "Choose whether to segment" in widget.apply_to_volume_checkbox.toolTip()

    viewer.close()


def test_sam_v1_mask_propagation_respects_z_range(
    make_napari_viewer_proxy,
    monkeypatch,
):
    viewer = make_napari_viewer_proxy()
    image = viewer.add_image(np.zeros((5, 12, 12), dtype="uint8"), name="volume")
    annotator = Annotator(viewer, ndim=3)
    widget = annotator._widgets["interactive"]

    labels = np.zeros(image.data.shape, dtype="uint32")
    labels[2, 3:7, 3:7] = 13
    source = viewer.add_labels(labels, name="cellpose")
    _set_only_checked(widget, source)
    mask_inputs = widget._collect_mask_inputs()
    widget._segment_widget.z_range = (1, 3)
    monkeypatch.setattr(widget, "_refine_mask_slice", lambda mask, object_id, z: mask)

    state = AnnotatorState()
    monkeypatch.setattr(state, "is_sam2", False)
    monkeypatch.setattr(state, "predictor", object())
    monkeypatch.setattr(
        state,
        "image_embeddings",
        {
            "features": np.zeros((5, 1, 1, 1, 1), dtype="float32"),
            "input_size": (12, 12),
            "original_size": (12, 12),
        },
    )

    def _fake_segment_mask_in_volume(
        segmentation,
        predictor,
        image_embeddings,
        segmented_slices,
        **kwargs,
    ):
        assert segmentation.shape == (3, 12, 12)
        assert image_embeddings["features"].shape[0] == 3
        np.testing.assert_array_equal(segmented_slices, np.array([1]))
        return segmentation, (1, 1)

    monkeypatch.setattr(_widgets, "segment_mask_in_volume", _fake_segment_mask_in_volume)
    candidates = widget._refine_masks_propagated(mask_inputs)

    assert not candidates[13][0].any()
    assert candidates[13][2].any()
    assert not candidates[13][4].any()

    viewer.close()


def test_refined_commit_conflict_keeps_result_prompts_and_provenance(
    make_napari_viewer_proxy,
    monkeypatch,
):
    monkeypatch.setattr(_widgets, "_generate_message", lambda *args: True)
    viewer = make_napari_viewer_proxy()
    viewer.add_image(np.zeros((12, 12), dtype="uint8"), name="image")
    annotator = Annotator(viewer)

    committed = np.zeros((12, 12), dtype="uint32")
    committed[5:8, 5:8] = 31
    viewer.layers["committed_objects"].data = committed

    refined = np.zeros_like(committed)
    refined[6:9, 6:9] = 7
    current = viewer.layers["current_object"]
    current.data = refined
    provenance = {"object_ids": (7,), "source_layers": ("cellpose",)}
    current.metadata["micro_sam_mask_refinement"] = provenance.copy()
    viewer.layers["point_prompts"].add(np.array([[2, 2]]))

    before_destination = viewer.layers["committed_objects"].data.copy()
    before_result = current.data.copy()
    annotator._widgets["commit"](viewer)

    np.testing.assert_array_equal(
        viewer.layers["committed_objects"].data,
        before_destination,
    )
    np.testing.assert_array_equal(current.data, before_result)
    assert current.metadata["micro_sam_mask_refinement"] == provenance
    assert len(viewer.layers["point_prompts"].data) == 1

    viewer.close()
