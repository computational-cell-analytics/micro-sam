import os
import platform
import tempfile

import numpy as np
import imageio.v3 as imageio
import pytest
from qtpy import QtWidgets
from skimage.data import binary_blobs

from micro_sam.v2.util import DEFAULT_MODEL
from micro_sam.sam_annotator import image_series_annotator, image_folder_annotator
from micro_sam.sam_annotator._state import AnnotatorState
from micro_sam._test_util import check_layer_initialization


def _create_images(tmpdir, n_images):
    image_paths = []
    for i in range(n_images):
        im_path = os.path.join(tmpdir, f"image-{i}.png")
        image_data = binary_blobs(512)
        imageio.imwrite(im_path, image_data)
        image_paths.append(im_path)
    return image_paths


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_image_series_annotator(make_napari_viewer_proxy):
    """Integration test for `image_series_annotator`.
    """
    n_images = 3
    model_type = DEFAULT_MODEL

    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = _create_images(tmpdir, n_images)
        output_folder = os.path.join(tmpdir, "segmentation_results")

        viewer = make_napari_viewer_proxy()
        # test generating image embedding, then adding micro-sam dock widgets to the GUI
        viewer = image_series_annotator(
            image_paths,
            output_folder,
            model_type=model_type,
            viewer=viewer,
            return_viewer=True,
        )

        check_layer_initialization(viewer, (512, 512))
        viewer.close()  # must close the viewer at the end of tests


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_image_series_navigation(make_napari_viewer_proxy):
    """Drive the Next/Previous navigation harness: forward saves and advances, backward saves,
    steps back and reloads the previously saved result.
    """
    n_images = 3
    model_type = DEFAULT_MODEL

    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = _create_images(tmpdir, n_images)
        output_folder = os.path.join(tmpdir, "segmentation_results")

        viewer = make_napari_viewer_proxy()
        viewer = image_series_annotator(
            image_paths, output_folder, model_type=model_type, viewer=viewer, return_viewer=True,
        )

        next_image = AnnotatorState().widgets["series_next"]
        prev_image = AnnotatorState().widgets["series_prev"]

        # In a series session the launcher owns the embedding settings, so the docked annotator's
        # embedding section is hidden (its wrapping group box is explicitly hidden).
        embedding_widget = AnnotatorState().annotator._embedding_widget
        frame = embedding_widget
        while frame is not None and not isinstance(frame, QtWidgets.QGroupBox):
            frame = frame.parentWidget()
        assert frame is not None and frame.isHidden()

        def _result_path(index):
            return os.path.join(output_folder, os.path.splitext(os.path.basename(image_paths[index]))[0] + ".tif")

        # Item 0: paint a segmentation and advance. It should be saved and the next image loaded.
        seg0 = np.full((512, 512), 1, dtype="uint32")
        viewer.layers["committed_objects"].data = seg0
        next_image()
        assert os.path.exists(_result_path(0))
        np.testing.assert_array_equal(imageio.imread(_result_path(0)), seg0)
        # The freshly loaded item starts from a cleared committed layer.
        assert viewer.layers["committed_objects"].data.sum() == 0

        # Item 1: paint a distinct segmentation and advance.
        seg1 = np.full((512, 512), 5, dtype="uint32")
        viewer.layers["committed_objects"].data = seg1
        next_image()
        assert os.path.exists(_result_path(1))
        np.testing.assert_array_equal(imageio.imread(_result_path(1)), seg1)

        # Step back from item 2 to item 1: the previously saved segmentation should be reloaded.
        prev_image()
        np.testing.assert_array_equal(viewer.layers["committed_objects"].data, seg1)

        viewer.close()  # must close the viewer at the end of tests


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_image_folder_annotator(make_napari_viewer_proxy):
    """Integration test for `image_folder_annotator`.
    """
    n_images = 3
    model_type = DEFAULT_MODEL

    with tempfile.TemporaryDirectory() as tmpdir:
        _create_images(tmpdir, n_images)
        output_folder = os.path.join(tmpdir, "segmentation_results")

        viewer = make_napari_viewer_proxy()
        # test generating image embedding, then adding micro-sam dock widgets to the GUI
        viewer = image_folder_annotator(
            tmpdir,
            output_folder,
            pattern="*.png",
            model_type=model_type,
            viewer=viewer,
            return_viewer=True,
        )

        check_layer_initialization(viewer, (512, 512))
        viewer.close()  # must close the viewer at the end of tests
