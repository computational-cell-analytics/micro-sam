import pytest
import numpy as np

from micro_sam.sam_annotator.util import toggle_label, clear_annotations, clear_annotations_slice


@pytest.mark.gui
def test_toggle_label_after_clear(make_napari_viewer):
    viewer = make_napari_viewer()
    points = viewer.add_points(
        [[10, 20]], name="point_prompts", properties={"label": ["positive"]},
        property_choices={"label": ["positive", "negative"]},
    )
    points.selected_data = {0}
    clear_annotations(viewer, clear_segmentations=False)
    assert not points.selected_data
    assert len(points.data) == 0
    toggle_label(points)
    points.add([30, 40])
    assert points.properties["label"].tolist() == ["negative"]


@pytest.mark.gui
def test_clear_slice_preserves_point_properties(make_napari_viewer):
    viewer = make_napari_viewer()
    points = viewer.add_points(
        [[0, 10, 20], [1, 30, 40], [2, 50, 60]], name="point_prompts",
        properties={"label": ["positive", "negative", "positive"], "track_id": ["1", "2", "3"]},
        property_choices={"label": ["positive", "negative"], "track_id": ["1", "2", "3"]},
    )
    points.selected_data = {2}
    clear_annotations_slice(viewer, 0, clear_segmentations=False)
    assert not points.selected_data
    np.testing.assert_array_equal(points.data, [[1, 30, 40], [2, 50, 60]])
    assert points.properties["label"].tolist() == ["negative", "positive"]
    assert points.properties["track_id"].tolist() == ["2", "3"]
    toggle_label(points)
