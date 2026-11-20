"""Clicking 'Compute embeddings' must only reset the annotation layers for a new image."""

import platform
from types import SimpleNamespace

import pytest
import numpy as np

from micro_sam import util
from micro_sam.sam_annotator import _widgets
from micro_sam.sam_annotator._state import AnnotatorState
from micro_sam.sam_annotator.annotator import Annotator
from micro_sam.sam_annotator.annotator_tracking import AnnotatorTracking
from micro_sam.sam_annotator.pixel_classifier import PixelClassifier

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows."),
]


def _fake_initialize_predictor(self, image_data, model_type, ndim, **kwargs):
    """Set what the annotators read from the state without loading a model."""
    self.image_embeddings = {"features": None}
    self.predictor = SimpleNamespace(model_type=model_type)
    self.data_signature = util._compute_data_signature(image_data)


@pytest.fixture
def fake_compute(monkeypatch):
    monkeypatch.setattr(AnnotatorState, "initialize_predictor", _fake_initialize_predictor)
    monkeypatch.setattr(_widgets.EmbeddingWidget, "warn_on_cpu", False)
    # Answer OK to the 'embeddings have already been precomputed' dialog.
    monkeypatch.setattr(_widgets, "_generate_message", lambda kind, msg: None)


def _image(shape=(64, 64), seed=0):
    return np.random.default_rng(seed).integers(0, 255, size=shape, dtype="uint8")


def _annotate(viewer):
    committed = np.zeros(viewer.layers["committed_objects"].data.shape, dtype="uint32")
    committed[..., 10:20, 10:20] = 1
    viewer.layers["committed_objects"].data = committed
    viewer.layers["point_prompts"].add([[0] * (committed.ndim - 2) + [30, 30]])


def _is_annotated(viewer):
    return viewer.layers["committed_objects"].data.max() == 1 and len(viewer.layers["point_prompts"].data) == 1


def _segmentation_annotator(viewer, shape=(64, 64)):
    viewer.add_image(_image(shape), name="image")
    annotator = Annotator(viewer)
    embeddings = annotator._embedding_widget
    embeddings()
    _annotate(viewer)
    return annotator, embeddings


def test_recompute_same_image_keeps_annotations(make_napari_viewer_proxy, fake_compute):
    viewer = make_napari_viewer_proxy()
    _, embeddings = _segmentation_annotator(viewer)

    embeddings.run_button.click()
    assert _is_annotated(viewer)
    viewer.close()


def test_cancelled_recompute_keeps_annotations(make_napari_viewer_proxy, fake_compute, monkeypatch):
    viewer = make_napari_viewer_proxy()
    _, embeddings = _segmentation_annotator(viewer)

    monkeypatch.setattr(_widgets, "_generate_message", lambda kind, msg: True)
    embeddings.run_button.click()
    assert _is_annotated(viewer)
    viewer.close()


def test_failed_validation_keeps_annotations(make_napari_viewer_proxy, fake_compute, monkeypatch):
    viewer = make_napari_viewer_proxy()
    _, embeddings = _segmentation_annotator(viewer)

    monkeypatch.setattr(_widgets.EmbeddingWidget, "_validate_inputs", lambda self: True)
    embeddings.run_button.click()
    assert _is_annotated(viewer)
    viewer.close()


def test_failed_computation_keeps_annotations(make_napari_viewer_proxy, fake_compute, monkeypatch):
    viewer = make_napari_viewer_proxy()
    _, embeddings = _segmentation_annotator(viewer)

    def fail(self, *args, **kwargs):
        raise RuntimeError("download failed")

    monkeypatch.setattr(AnnotatorState, "initialize_predictor", fail)
    with pytest.raises(RuntimeError):
        embeddings()
    assert _is_annotated(viewer)

    # The next successful computation for the same image keeps them as well.
    monkeypatch.setattr(AnnotatorState, "initialize_predictor", _fake_initialize_predictor)
    embeddings()
    assert _is_annotated(viewer)
    viewer.close()


def test_recompute_drops_cached_volume_segmentation(make_napari_viewer_proxy, fake_compute):
    viewer = make_napari_viewer_proxy()
    annotator, embeddings = _segmentation_annotator(viewer, shape=(4, 64, 64))
    segment_widget = annotator._widgets["interactive"]._segment_widget
    segment_widget._last_volume_run = ("key", np.zeros((1,)))

    embeddings()
    assert segment_widget._last_volume_run is None
    viewer.close()


def test_tracking_recompute_keeps_lineage(make_napari_viewer_proxy, fake_compute):
    viewer = make_napari_viewer_proxy()
    viewer.add_image(_image((4, 64, 64)), name="timeseries")
    annotator = AnnotatorTracking(viewer)
    embeddings = annotator._embedding_widget
    embeddings()
    _annotate(viewer)

    state = AnnotatorState()
    state.current_track_id = 2
    state.lineage = {1: [2, 3], 2: [], 3: []}
    state.committed_lineages = [{1: []}]

    embeddings()
    assert _is_annotated(viewer)
    assert state.current_track_id == 2
    assert state.lineage == {1: [2, 3], 2: [], 3: []}
    assert state.committed_lineages == [{1: []}]
    viewer.close()


def test_classifier_recompute_keeps_scribbles(make_napari_viewer_proxy, fake_compute):
    viewer = make_napari_viewer_proxy()
    viewer.add_image(_image(), name="image")
    annotator = PixelClassifier(viewer)
    embeddings = annotator._embedding_widget
    embeddings()

    scribbles = np.zeros((64, 64), dtype="uint32")
    scribbles[5:10, 5:10] = 1
    viewer.layers["annotations"].data = scribbles
    state = AnnotatorState()
    state.pixel_features = np.zeros((1,))
    annotator._trained_annotations = (scribbles.copy(), annotator._training_settings())

    embeddings()
    np.testing.assert_array_equal(viewer.layers["annotations"].data, scribbles)
    # The features come from the old embeddings, so they are discarded and the prediction is stale.
    assert state.pixel_features is None
    assert annotator.is_prediction_stale()
    viewer.close()


def test_classifier_new_image_resets_scribbles(make_napari_viewer_proxy, fake_compute):
    viewer = make_napari_viewer_proxy()
    first = viewer.add_image(_image(seed=0), name="first")
    annotator = PixelClassifier(viewer)
    embeddings = annotator._embedding_widget
    embeddings.image_selection.value = first
    embeddings()

    scribbles = np.zeros((64, 64), dtype="uint32")
    scribbles[5:10, 5:10] = 1
    viewer.layers["annotations"].data = scribbles

    second = viewer.add_image(_image(seed=1), name="second")
    embeddings.image_selection.reset_choices()
    embeddings.image_selection.value = second
    embeddings()
    assert viewer.layers["annotations"].data.max() == 0
    viewer.close()
