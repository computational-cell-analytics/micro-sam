import platform
from types import SimpleNamespace

import numpy as np
import pytest
from qtpy import QtWidgets

from micro_sam.sam_annotator import _widgets as widgets
from micro_sam.sam_annotator._state import AnnotatorState
from micro_sam.sam_annotator.pixel_classifier import PixelClassifier
from micro_sam.sam_annotator.object_classifier import ObjectClassifier


pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(platform.system() == "Windows", reason="GUI tests require working OpenGL."),
]


@pytest.fixture
def annotator(make_napari_viewer_proxy, request):
    state = AnnotatorState()
    state.batch_training_data, state.batch_index = {}, None
    viewer = make_napari_viewer_proxy()
    yield request.param(viewer), viewer
    state.batch_training_data, state.batch_index = {}, None
    viewer.close()


def _row_ids(annotator):
    layout = annotator._class_layout
    rows = [layout.itemAt(i).widget() for i in range(layout.count())]
    return [int(row.layout().itemAt(0).widget().text()) for row in rows]


both = pytest.mark.parametrize("annotator", [PixelClassifier, ObjectClassifier], indirect=True)


@both
def test_class_rows_and_selection(annotator):
    annotator, viewer = annotator
    layer = viewer.layers["annotations"]

    # The selected label has a row from the start: id, radio button, swatch, remove button and name.
    assert _row_ids(annotator) == [1]
    row_layout = annotator._class_rows[1][0].layout()
    row_widgets = [row_layout.itemAt(i).widget() for i in range(row_layout.count())]
    types = [type(widget) for widget in row_widgets if widget is not None]
    assert types == [QtWidgets.QLabel, QtWidgets.QRadioButton, QtWidgets.QPushButton,
                     QtWidgets.QToolButton, QtWidgets.QLineEdit]
    assert annotator._class_group.button(1).isChecked()

    # 'Add Class' paints with the next free id.
    add_button = [b for b in annotator._label_widget.parentWidget().findChildren(QtWidgets.QPushButton)
                  if b.text() == "Add Class"][0]
    add_button.click()
    assert layer.selected_label == 2 and layer.mode == "paint"
    assert annotator._class_group.button(2).isChecked()

    # A label id selected in the napari layer controls gets a row, sorted by id.
    layer.selected_label = 5
    layer.selected_label = 3
    assert _row_ids(annotator) == [1, 2, 3, 5]
    assert annotator._class_group.button(3).isChecked()

    # Clicking a radio button selects its class.
    annotator._class_group.button(5).click()
    assert layer.selected_label == 5

    # A new color applies to the annotations and the prediction.
    annotator._set_class_color(2, "#ff0000")
    for name in ("annotations", "prediction"):
        np.testing.assert_allclose(viewer.layers[name].colormap.map(2)[:3], [1, 0, 0])

    # Annotations that are set as layer data (e.g. restored for a revisited image) get rows.
    data = np.zeros_like(layer.data)
    data[:4, :4] = 7
    layer.data = data
    assert 7 in _row_ids(annotator)


@both
def test_remove_class(annotator, monkeypatch):
    annotator, viewer = annotator
    layer = viewer.layers["annotations"]
    annotator._select_class(2)
    data = np.zeros_like(layer.data)
    data[:4, :4], data[4:8, 4:8] = 1, 2
    layer.data = data

    # Removing a class removes its row and its annotations, and selects the remaining class with the lowest id.
    annotator._remove_class(2, confirm=False)
    assert _row_ids(annotator) == [1]
    assert not (layer.data == 2).any() and (layer.data == 1).sum() == 16
    assert layer.selected_label == 1

    # Removing the last class creates class 1 again, so there is always a class to paint with.
    annotator._remove_class(1, confirm=False)
    assert _row_ids(annotator) == [1] and not layer.data.any()

    # A class that is annotated in another image of a batch is not removed.
    messages = []
    monkeypatch.setattr(widgets, "_generate_message", lambda kind, msg: messages.append(kind))
    state = AnnotatorState()
    state.batch_index = 1
    state.batch_training_data = {0: (np.zeros((2, 3)), np.array([1, 4]), {})}
    annotator._refresh_label_widget()
    annotator._remove_class(4)
    assert messages == ["error"] and 4 in _row_ids(annotator)


@both
def test_class_names(annotator):
    annotator, viewer = annotator
    annotator._select_class(2)
    annotator._set_class_name(2, "tumor")
    rf = SimpleNamespace(classes_=np.array([1, 2]))
    assert annotator._gather_class_names(rf) == {2: "tumor"}

    # Names stored with a classifier are restored, including the names of classes without a row.
    annotator._restore_from_spec({"class_names": {"3": "stroma"}})
    assert annotator._classes[3]["name"] == "stroma"
    assert annotator._class_rows[3][2].text() == "stroma"
