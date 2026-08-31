import platform

import numpy as np
import pytest
from napari.layers import Points, Shapes
from skimage.data import binary_blobs

from micro_sam.v2.util import DEFAULT_MODEL
from micro_sam.sam_annotator import annotator_tracking
from micro_sam.sam_annotator._widgets import (
    AutoSegmentV1Widget,
    AutoSegmentWidget,
    AutoTrackWidget,
    EmbeddingWidget,
)
from micro_sam.sam_annotator.annotator_tracking import (
    AnnotatorTracking,
    _validate_tracking_model_type,
)
from micro_sam._test_util import check_layer_initialization


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_annotator_tracking(make_napari_viewer_proxy):
    """Integration test for annotator_tracking.
    """

    image = np.stack(4 * [binary_blobs(512)])

    viewer = make_napari_viewer_proxy()
    # test generating image embedding, then adding micro-sam dock widgets to the GUI
    viewer = annotator_tracking(
        image,
        model_type=DEFAULT_MODEL,
        viewer=viewer,
        return_viewer=True
    )

    check_layer_initialization(viewer, image.shape)
    viewer.close()  # must close the viewer at the end of tests


def test_tracking_uses_timeseries_layer_label(qtbot):
    tracking_widget = AnnotatorTracking._create_embedding_widget(None)
    image_widget = EmbeddingWidget()
    qtbot.addWidget(tracking_widget)
    qtbot.addWidget(image_widget)

    assert tracking_widget.image_layer_label.text() == "Timeseries Layer:"
    assert image_widget.image_layer_label.text() == "Image Layer:"


def test_box_selection_updates_current_track_id(qtbot):
    from micro_sam.sam_annotator._state import AnnotatorState
    from micro_sam.sam_annotator.annotator_tracking import create_tracking_menu

    property_choices = {
        "label": ["positive", "negative"],
        "state": ["track", "division"],
        "track_id": ["1", "2"],
    }
    points = Points(ndim=2, property_choices=property_choices)
    boxes = Shapes(
        data=[np.array([[0, 0], [8, 8]]), np.array([[16, 16], [24, 24]])],
        shape_type="rectangle",
        properties={"track_id": np.array(["1", "2"])},
        ndim=2,
    )
    tracking_widget = create_tracking_menu(
        points_layer=points,
        box_layer=boxes,
        states=property_choices["state"],
        track_ids=property_choices["track_id"],
        point_labels=property_choices["label"],
    )
    qtbot.addWidget(tracking_widget.native)

    boxes.selected_data = {1}

    assert tracking_widget[2].value == "2"
    assert AnnotatorState().current_track_id == 2


def test_division_frame_detection():
    # A point tagged with the 'division' track-state marks the frame where the track divides. Build
    # the points layer directly (no viewer) so the test does not need a GL context on any platform.
    from micro_sam.sam_annotator._widgets import _division_frame_for_track

    data = np.array([[1, 10, 10], [3, 20, 20], [2, 5, 5]])
    properties = {
        "state": np.array(["track", "division", "track"]),
        "track_id": np.array(["1", "1", "2"]),
    }
    layer = Points(data, properties=properties, ndim=3)

    assert _division_frame_for_track(layer, 1) == 3  # track 1 divides at frame 3
    assert _division_frame_for_track(layer, 2) is None  # track 2 has no division
    assert _division_frame_for_track(layer, 3) is None  # unknown track


def test_mother_division_frame():
    # A daughter's mask must start the frame after its mother divides. This resolves the bound.
    from micro_sam.sam_annotator._widgets import _mother_division_frame

    data = np.array([[10, 10, 10]])
    properties = {"state": np.array(["division"]), "track_id": np.array(["1"])}
    layer = Points(data, properties=properties, ndim=3)

    lineage = {1: [2, 3], 2: [], 3: []}
    assert _mother_division_frame(layer, lineage, 2) == 10  # daughter of track 1 (divides at 10)
    assert _mother_division_frame(layer, lineage, 3) == 10
    assert _mother_division_frame(layer, lineage, 1) is None  # the mother itself is not a daughter
    assert _mother_division_frame(layer, lineage, 4) is None  # unknown track


def test_division_marker_excluded_from_prompts():
    # A 'division' point bounds propagation but must not be fed to the predictor as a prompt,
    # otherwise it adds a second conditioning frame that wipes the mother track's earlier frames.
    from micro_sam.sam_annotator.util import point_layer_to_prompts

    data = np.array([[0, 10, 10], [3, 20, 20], [3, 21, 21]])
    properties = {
        "label": np.array(["positive", "positive", "positive"]),
        "state": np.array(["track", "division", "track"]),
        "track_id": np.array(["1", "1", "1"]),
    }
    layer = Points(data, properties=properties, ndim=3)

    # Frame 3 has one division point and one regular point: only the regular one is a prompt.
    points, labels = point_layer_to_prompts(layer, i=3, track_id=1, exclude_states=("division",))
    assert len(points) == 1
    np.testing.assert_array_equal(points[0], [21, 21])

    # Without the filter both points come through (the buggy behavior).
    points_all, _ = point_layer_to_prompts(layer, i=3, track_id=1)
    assert len(points_all) == 2


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_update_lineage_seeds_daughters(make_napari_viewer_proxy):
    # Recording a division seeds two daughter tracks and refreshes the track-id menu.
    from micro_sam.sam_annotator._state import AnnotatorState
    from micro_sam.sam_annotator._widgets import _update_lineage
    from micro_sam.sam_annotator.annotator_tracking import AnnotatorTracking

    viewer = make_napari_viewer_proxy()
    AnnotatorTracking(viewer)
    state = AnnotatorState()
    state.lineage = {1: []}
    state.current_track_id = 1

    _update_lineage(viewer, mother=1)
    assert state.lineage == {1: [2, 3], 2: [], 3: []}
    assert set(state.annotator._tracking_widget[2].choices) == {"1", "2", "3"}

    # Re-dividing the same mother is a no-op.
    _update_lineage(viewer, mother=1)
    assert state.lineage == {1: [2, 3], 2: [], 3: []}

    # Dividing a daughter allocates fresh, non-colliding ids.
    _update_lineage(viewer, mother=2)
    assert state.lineage[2] == [4, 5]
    viewer.close()


def test_tracking_rejects_sam1_models():
    with pytest.raises(ValueError, match="only supports micro-sam2/SAM2"):
        _validate_tracking_model_type("vit_b_lm")


def test_auto_tracking_uses_sam2_widget():
    assert issubclass(AutoTrackWidget, AutoSegmentWidget)
    assert not issubclass(AutoTrackWidget, AutoSegmentV1Widget)


def test_auto_tracking_offers_apg(qtbot):
    widget = AutoTrackWidget(viewer=None, with_decoder=True, volumetric=True)
    qtbot.addWidget(widget)

    choices = [widget.mode_dropdown.itemText(i) for i in range(widget.mode_dropdown.count())]
    assert choices == ["sparse", "dense", "apg"]
    widget.mode_dropdown.setCurrentText("apg")
    assert widget.mode == "apg"
    assert hasattr(widget, "candidate_threshold_param")
