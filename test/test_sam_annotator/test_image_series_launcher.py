import os
import platform
import tempfile
import importlib

import imageio.v3 as imageio
import numpy as np
import pytest
from skimage.data import binary_blobs

from micro_sam.sam_annotator.image_series_annotator import ImageSeriesAnnotator, TASKS


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_launcher_task_selector_toggles_segmentation_folder(make_napari_viewer_proxy):
    viewer = make_napari_viewer_proxy()
    widget = ImageSeriesAnnotator(viewer)

    # The task dropdown offers all tasks and defaults to segmentation, with the segmentation folder hidden.
    assert widget.task == "Segmentation"
    items = [widget.task_dropdown.itemText(i) for i in range(widget.task_dropdown.count())]
    assert items == TASKS
    assert not widget._seg_folder_container.isVisibleTo(widget)

    # The segmentation folder is shown only for object classification.
    widget.task_dropdown.setCurrentText("Object Classification")
    assert widget.task == "Object Classification"
    assert widget._seg_folder_container.isVisibleTo(widget)

    widget.task_dropdown.setCurrentText("Pixel Classification")
    assert not widget._seg_folder_container.isVisibleTo(widget)


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_launcher_embedding_widget_swaps_per_task(make_napari_viewer_proxy):
    from micro_sam.sam_annotator._widgets import EmbeddingWidget, ClassificationEmbeddingWidget

    viewer = make_napari_viewer_proxy()
    widget = ImageSeriesAnnotator(viewer)

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
    widget = ImageSeriesAnnotator(viewer)

    def _row_widgets():
        row = widget._model_row
        return [row.itemAt(i).widget() for i in range(row.count()) if row.itemAt(i).widget() is not None]

    assert widget._relocated_model_dropdown is widget._embedding_widget.model_family_dropdown
    assert widget._relocated_model_dropdown in _row_widgets()

    # After switching tasks (which rebuilds the embedding widget), the row tracks the new dropdown.
    widget.task_dropdown.setCurrentText("Pixel Classification")
    assert widget._relocated_model_dropdown is widget._embedding_widget.model_family_dropdown
    assert widget._relocated_model_dropdown in _row_widgets()


# Each task must dispatch to its series function. The launcher imports these lazily from their home
# modules, so patching the module attribute intercepts the call.
DISPATCH = [
    ("Segmentation", "micro_sam.sam_annotator.image_series_annotator", "image_folder_annotator"),
    ("Tracking", "micro_sam.sam_annotator.annotator_tracking", "image_series_tracking_annotator"),
    ("Pixel Classification", "micro_sam.sam_annotator.pixel_classifier", "image_series_pixel_classifier"),
    ("Object Classification", "micro_sam.sam_annotator.object_classifier", "image_series_object_classifier"),
]


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
@pytest.mark.parametrize("task, module_path, func_name", DISPATCH)
def test_launcher_dispatches_to_the_selected_task(
    make_napari_viewer_proxy, monkeypatch, task, module_path, func_name
):
    module = importlib.import_module(module_path)
    calls = []
    monkeypatch.setattr(module, func_name, lambda *args, **kwargs: calls.append((args, kwargs)))

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(2):
            imageio.imwrite(os.path.join(tmpdir, f"image-{i}.tif"), binary_blobs(64).astype(np.uint8) * 255)

        viewer = make_napari_viewer_proxy()
        widget = ImageSeriesAnnotator(viewer)
        widget.folder = tmpdir
        widget.output_folder = os.path.join(tmpdir, "out")
        widget.pattern = "*.tif"
        widget.task_dropdown.setCurrentText(task)

        widget(skip_validate=True)

        assert len(calls) == 1, f"expected exactly one dispatch for task '{task}'"
        # The output folder is forwarded to the selected series function (positionally or by keyword).
        args, kwargs = calls[0]
        assert widget.output_folder in args or widget.output_folder in kwargs.values()
