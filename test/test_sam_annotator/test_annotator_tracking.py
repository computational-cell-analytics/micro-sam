import platform

import numpy as np
import pytest
from skimage.data import binary_blobs

from micro_sam.v2.util import DEFAULT_MODEL
from micro_sam.sam_annotator import annotator_tracking
from micro_sam.sam_annotator._widgets import AutoSegmentV1Widget, AutoSegmentWidget, AutoTrackWidget
from micro_sam.sam_annotator.annotator_tracking import _validate_tracking_model_type
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


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_division_frame_detection(make_napari_viewer_proxy):
    # A point tagged with the 'division' track-state marks the frame where the track divides.
    from micro_sam.sam_annotator._widgets import _division_frame_for_track

    viewer = make_napari_viewer_proxy()
    data = np.array([[1, 10, 10], [3, 20, 20], [2, 5, 5]])
    properties = {
        "state": np.array(["track", "division", "track"]),
        "track_id": np.array(["1", "1", "2"]),
    }
    layer = viewer.add_points(data, properties=properties, ndim=3)

    assert _division_frame_for_track(layer, 1) == 3  # track 1 divides at frame 3
    assert _division_frame_for_track(layer, 2) is None  # track 2 has no division
    assert _division_frame_for_track(layer, 3) is None  # unknown track
    viewer.close()


def test_mother_division_frame(make_napari_viewer_proxy):
    # A daughter's mask must start the frame after its mother divides; this resolves the bound.
    from micro_sam.sam_annotator._widgets import _mother_division_frame

    viewer = make_napari_viewer_proxy()
    data = np.array([[10, 10, 10]])
    properties = {"state": np.array(["division"]), "track_id": np.array(["1"])}
    layer = viewer.add_points(data, properties=properties, ndim=3)

    lineage = {1: [2, 3], 2: [], 3: []}
    assert _mother_division_frame(layer, lineage, 2) == 10  # daughter of track 1 (divides at 10)
    assert _mother_division_frame(layer, lineage, 3) == 10
    assert _mother_division_frame(layer, lineage, 1) is None  # the mother itself is not a daughter
    assert _mother_division_frame(layer, lineage, 4) is None  # unknown track
    viewer.close()


def test_division_marker_excluded_from_prompts(make_napari_viewer_proxy):
    # A 'division' point bounds propagation but must not be fed to the predictor as a prompt,
    # otherwise it adds a second conditioning frame that wipes the mother track's earlier frames.
    from micro_sam.sam_annotator.util import point_layer_to_prompts

    viewer = make_napari_viewer_proxy()
    data = np.array([[0, 10, 10], [3, 20, 20], [3, 21, 21]])
    properties = {
        "label": np.array(["positive", "positive", "positive"]),
        "state": np.array(["track", "division", "track"]),
        "track_id": np.array(["1", "1", "1"]),
    }
    layer = viewer.add_points(data, properties=properties, ndim=3)

    # Frame 3 has one division point and one regular point: only the regular one is a prompt.
    points, labels = point_layer_to_prompts(layer, i=3, track_id=1, exclude_states=("division",))
    assert len(points) == 1
    np.testing.assert_array_equal(points[0], [21, 21])

    # Without the filter both points come through (the buggy behavior).
    points_all, _ = point_layer_to_prompts(layer, i=3, track_id=1)
    assert len(points_all) == 2
    viewer.close()


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
