import platform

import numpy as np
import pytest

from micro_sam.sam_annotator import _widgets as widgets
from micro_sam.sam_annotator._state import AnnotatorState
from micro_sam.sam_annotator.object_classifier import ObjectClassifier
from micro_sam.sam_annotator.pixel_classifier import PixelClassifier


pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(platform.system() == "Windows", reason="GUI tests require working OpenGL."),
]


@pytest.mark.parametrize("classifier_cls", [PixelClassifier, ObjectClassifier])
def test_resolve_feature_inputs_uses_selected_image_layer(make_napari_viewer_proxy, monkeypatch, classifier_cls):
    state = AnnotatorState()
    state.reset_state()
    viewer = make_napari_viewer_proxy()
    try:
        other_img = np.zeros((32, 32), dtype="uint8")
        target_img = np.full((32, 32), 7, dtype="uint8")
        viewer.add_image(other_img, name="other")
        viewer.add_image(target_img, name="T33")

        annotator = classifier_cls(viewer)
        state.image_name = "T33"

        sentinel_upsampler = object()
        monkeypatch.setattr(annotator, "_get_cached_upsampler", lambda: sentinel_upsampler)

        # AnyUp enabled with a custom image layer name (issue #1444).
        annotator._set_options(False, None, True, "fixed", None)
        image, upsampler, classical, ok = annotator._resolve_feature_inputs()
        assert ok
        assert upsampler is sentinel_upsampler
        assert classical is None
        np.testing.assert_array_equal(image, target_img)

        # Classical features enabled without AnyUp also resolve the selected image layer.
        choices = list(classifier_cls.classical_feature_choices)[:1]
        annotator._set_options(False, None, False, "fixed", choices)
        image, upsampler, classical, ok = annotator._resolve_feature_inputs()
        assert ok
        assert upsampler is None
        assert classical == choices
        np.testing.assert_array_equal(image, target_img)

        # When image_name is unset (e.g. precomputed embeddings), fall back to the viewer's image layer.
        state.image_name = None
        del viewer.layers["other"]
        image, upsampler, classical, ok = annotator._resolve_feature_inputs()
        assert ok
        np.testing.assert_array_equal(image, target_img)

        # When the referenced image layer is gone, report an error cleanly instead of raising KeyError.
        messages = []
        monkeypatch.setattr(widgets, "_generate_message", lambda kind, msg: messages.append((kind, msg)))
        state.image_name = "deleted_layer"
        image, upsampler, classical, ok = annotator._resolve_feature_inputs()
        assert not ok
        assert image is None
        assert len(messages) == 1 and messages[0][0] == "error"
    finally:
        state.reset_state()
        viewer.close()
