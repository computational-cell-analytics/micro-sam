import platform

import numpy as np
import pytest
from napari.utils.key_bindings import coerce_keybinding
from qtpy import QtWidgets
from skimage.data import binary_blobs

from micro_sam.sam_annotator._state import AnnotatorState
from micro_sam.sam_annotator.pixel_classifier import PixelClassifier
from micro_sam.sam_annotator.object_classifier import ObjectClassifier


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
@pytest.mark.parametrize("classifier_cls", [PixelClassifier, ObjectClassifier])
def test_classifier_shortcuts_and_button_labels(make_napari_viewer_proxy, classifier_cls):
    viewer = make_napari_viewer_proxy()
    annotator = classifier_cls(viewer)

    assert coerce_keybinding("Shift-C") in viewer.keymap
    assert coerce_keybinding("C") not in viewer.keymap

    buttons = annotator._train_and_predict_widget.native.findChildren(QtWidgets.QPushButton)
    button_labels = {button.text() for button in buttons}
    assert "Train and Predict [Shift + T]" in button_labels
    assert "Clear Annotations [Shift + C]" in button_labels
    viewer.close()


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
@pytest.mark.parametrize("classifier_cls", [PixelClassifier, ObjectClassifier])
def test_classifier_recreates_label_layers_for_3d(make_napari_viewer_proxy, classifier_cls):
    # Regression for the pixel/object classifier 3d crash: the tool opens with 2d placeholder label
    # layers; loading a 3d image must recreate them at ndim=3 rather than reassigning 3d data + a
    # 3-element scale onto stale 2d layers (which crashes napari in Affine.set_slice).
    state = AnnotatorState()

    viewer = make_napari_viewer_proxy()
    annotator = classifier_cls(viewer)
    assert viewer.layers["annotations"].ndim == 2

    # Simulate a 3d image being loaded and selected.
    volume = np.stack(4 * [binary_blobs(64)])
    state.image_shape = volume.shape
    state.image_scale = (1.0, 1.0, 1.0)
    annotator._update_image()

    for name in ("annotations", "prediction"):
        layer = viewer.layers[name]
        assert layer.ndim == 3
        assert layer.data.ndim == 3
        assert tuple(layer.data.shape) == volume.shape
        assert len(layer.scale) == 3

    viewer.close()
