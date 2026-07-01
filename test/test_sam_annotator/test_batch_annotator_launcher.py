import os
import platform
import tempfile
import importlib

import imageio.v3 as imageio
import numpy as np
import pytest
from skimage.data import binary_blobs

from micro_sam.sam_annotator.batch_annotator import BatchAnnotator, TASKS


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_launcher_task_selector_toggles_segmentation_folder(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    widget = BatchAnnotator(viewer)

    # Continuing from existing outputs is enabled by default.
    assert widget.continue_annotation is True
    assert widget.continue_annotation_checkbox.isChecked()

    # The task dropdown offers all tasks and defaults to segmentation, with the segmentation folder hidden.
    assert widget.task == "Segmentation"
    assert widget.continue_annotation_checkbox.isVisibleTo(widget)
    items = [widget.task_dropdown.itemText(i) for i in range(widget.task_dropdown.count())]
    assert items == TASKS
    assert not widget._seg_folder_container.isVisibleTo(widget)

    # The segmentation folder is shown only for object classification.
    widget.task_dropdown.setCurrentText("Object Classification")
    assert widget.task == "Object Classification"
    assert widget._seg_folder_container.isVisibleTo(widget)
    assert not widget.continue_annotation_checkbox.isVisibleTo(widget)

    widget.task_dropdown.setCurrentText("Pixel Classification")
    assert not widget._seg_folder_container.isVisibleTo(widget)

    widget.task_dropdown.setCurrentText("Segmentation")
    assert widget.continue_annotation_checkbox.isVisibleTo(widget)


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_launcher_embedding_widget_swaps_per_task(make_napari_viewer_proxy):
    from micro_sam.sam_annotator._widgets import EmbeddingWidget, ClassificationEmbeddingWidget

    viewer = make_napari_viewer_proxy()
    widget = BatchAnnotator(viewer)

    # Segmentation: a plain embedding widget with the image-dimensions dropdown, no 'Advanced Models'.
    assert isinstance(widget._embedding_widget, EmbeddingWidget)
    assert not isinstance(widget._embedding_widget, ClassificationEmbeddingWidget)
    assert widget._embedding_widget.ndim_choice
    assert not hasattr(widget._embedding_widget, "advanced_checkbox")
    # The image selector and the Compute button are hidden (the launcher works on a folder).
    assert widget._embedding_widget.image_selection.native.isHidden()
    assert widget._embedding_widget.run_button.isHidden()

    # Classifier tasks: the classification embedding widget, which has the 'Advanced Models' selector.
    widget.task_dropdown.setCurrentText("Object Classification")
    assert isinstance(widget._embedding_widget, ClassificationEmbeddingWidget)
    assert hasattr(widget._embedding_widget, "advanced_checkbox")

    # Tracking: the SAM2-only timeseries widget (no image-dimensions dropdown, always 3d).
    widget.task_dropdown.setCurrentText("Tracking")
    assert isinstance(widget._embedding_widget, EmbeddingWidget)
    assert widget._embedding_widget.sam2_only
    assert not widget._embedding_widget.ndim_choice


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_launcher_model_dropdown_above_task(make_napari_viewer_proxy):
    # The model dropdown is relocated out of the embedding widget into the model row (above Task), and
    # re-relocated (pointing at the new widget's dropdown) whenever the task changes and the widget is
    # rebuilt.
    viewer = make_napari_viewer_proxy()
    widget = BatchAnnotator(viewer)

    def _row_widgets():
        row = widget._model_row
        return [row.itemAt(i).widget() for i in range(row.count()) if row.itemAt(i).widget() is not None]

    assert widget._relocated_model_dropdown is widget._embedding_widget.model_family_dropdown
    assert widget._relocated_model_dropdown in _row_widgets()

    # After switching tasks (which rebuilds the embedding widget), the row tracks the new dropdown.
    widget.task_dropdown.setCurrentText("Pixel Classification")
    assert widget._relocated_model_dropdown is widget._embedding_widget.model_family_dropdown
    assert widget._relocated_model_dropdown in _row_widgets()


# Each task must dispatch to its batch function. The launcher imports these lazily from their home
# modules, so patching the module attribute intercepts the call.
DISPATCH = [
    ("Segmentation", "micro_sam.sam_annotator.batch_annotator", "image_folder_annotator"),
    ("Tracking", "micro_sam.sam_annotator.annotator_tracking", "batch_tracking_annotator"),
    ("Pixel Classification", "micro_sam.sam_annotator.pixel_classifier", "batch_pixel_classifier"),
    ("Object Classification", "micro_sam.sam_annotator.object_classifier", "batch_object_classifier"),
]


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
@pytest.mark.parametrize("task, module_path, func_name", DISPATCH)
def test_launcher_dispatches_to_the_selected_task(
    make_napari_viewer_proxy, monkeypatch, task, module_path, func_name
):
    module = importlib.import_module(module_path)
    calls = []

    def launch(*args, **kwargs):
        calls.append((args, kwargs))
        return kwargs["viewer"]

    monkeypatch.setattr(module, func_name, launch)

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(2):
            imageio.imwrite(os.path.join(tmpdir, f"image-{i}.tif"), binary_blobs(64).astype(np.uint8) * 255)

        viewer = make_napari_viewer_proxy()
        widget = BatchAnnotator(viewer)
        widget.folder = tmpdir
        widget.output_folder = os.path.join(tmpdir, "out")
        widget.pattern = "*.tif"
        widget.task_dropdown.setCurrentText(task)

        widget(skip_validate=True)

        assert len(calls) == 1, f"expected exactly one dispatch for task '{task}'"
        # The output folder is forwarded to the selected batch function (positionally or by keyword).
        args, kwargs = calls[0]
        assert widget.output_folder in args or widget.output_folder in kwargs.values()
        if task == "Segmentation":
            assert kwargs["skip_segmented"] is True


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_launcher_can_restart_segmentation_from_first_image(make_napari_viewer_proxy, monkeypatch):
    isa = importlib.import_module("micro_sam.sam_annotator.batch_annotator")
    calls = []

    def launch(*args, **kwargs):
        calls.append((args, kwargs))
        return kwargs["viewer"]

    monkeypatch.setattr(isa, "image_folder_annotator", launch)

    with tempfile.TemporaryDirectory() as tmpdir:
        imageio.imwrite(os.path.join(tmpdir, "image.tif"), binary_blobs(64).astype(np.uint8) * 255)

        viewer = make_napari_viewer_proxy()
        widget = BatchAnnotator(viewer)
        widget.folder = tmpdir
        widget.output_folder = os.path.join(tmpdir, "out")
        widget.pattern = "*.tif"
        widget.continue_annotation_checkbox.setChecked(False)

        widget(skip_validate=True)

        assert not widget.continue_annotation
        assert len(calls) == 1
        assert calls[0][1]["skip_segmented"] is False


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_launcher_removes_itself_after_launch(make_napari_viewer_proxy, monkeypatch):
    # Once the task + settings are locked in and a launch happens, the console dock removes itself so
    # the annotator has the screen to itself.
    from qtpy.QtWidgets import QApplication, QDockWidget
    isa = importlib.import_module("micro_sam.sam_annotator.batch_annotator")

    def launch(*args, **kwargs):
        return kwargs["viewer"]

    monkeypatch.setattr(isa, "image_folder_annotator", launch)

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(2):
            imageio.imwrite(os.path.join(tmpdir, f"image-{i}.tif"), binary_blobs(64).astype(np.uint8) * 255)

        viewer = make_napari_viewer_proxy()
        widget = BatchAnnotator(viewer)
        widget.folder = tmpdir
        widget.output_folder = os.path.join(tmpdir, "out")
        widget.pattern = "*.tif"
        dock = viewer.window.add_dock_widget(widget, name="Batch Annotator")
        assert dock in viewer.window._qt_window.findChildren(QDockWidget)

        widget(skip_validate=True)
        # The removal is deferred to the event loop; flush it.
        for _ in range(3):
            QApplication.processEvents()

        assert dock not in viewer.window._qt_window.findChildren(QDockWidget)


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_launcher_stays_open_when_all_images_are_annotated(make_napari_viewer_proxy, monkeypatch):
    from qtpy.QtWidgets import QApplication, QDockWidget
    isa = importlib.import_module("micro_sam.sam_annotator.batch_annotator")
    messages = []
    monkeypatch.setattr(isa, "image_folder_annotator", lambda *args, **kwargs: None)
    monkeypatch.setattr(isa.widgets, "_generate_message", lambda *args: messages.append(args))

    with tempfile.TemporaryDirectory() as tmpdir:
        imageio.imwrite(os.path.join(tmpdir, "image.tif"), binary_blobs(64).astype(np.uint8) * 255)

        viewer = make_napari_viewer_proxy()
        widget = BatchAnnotator(viewer)
        widget.folder = tmpdir
        widget.output_folder = os.path.join(tmpdir, "out")
        widget.pattern = "*.tif"
        dock = viewer.window.add_dock_widget(widget, name="Batch Annotator")

        widget(skip_validate=True)
        for _ in range(3):
            QApplication.processEvents()

        assert dock in viewer.window._qt_window.findChildren(QDockWidget)
        assert len(messages) == 1
        assert messages[0][0] == "info"
        assert "All images have already been annotated" in messages[0][1]
        assert "Continue Annotation" in messages[0][1]
