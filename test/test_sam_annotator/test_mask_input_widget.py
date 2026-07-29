import platform

import numpy as np
import pytest

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
    assert widget.mask_3d_strategy.currentText() == "Refine all occupied slices"

    widget.mask_3d_strategy.setCurrentText("Propagate from seed slices")
    assert not widget._propagation_settings.isHidden()

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
