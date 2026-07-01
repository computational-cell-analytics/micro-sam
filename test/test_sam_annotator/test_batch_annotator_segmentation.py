import os
import platform
import tempfile

import numpy as np
import imageio.v3 as imageio
import pytest
from qtpy import QtWidgets
from skimage.data import binary_blobs

import micro_sam.util as util
from micro_sam.v2.util import DEFAULT_MODEL
from micro_sam.sam_annotator import batch_annotator, image_folder_annotator
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
def test_batch_annotator(make_napari_viewer_proxy):
    """Integration test for `batch_annotator`.
    """
    n_images = 3
    model_type = DEFAULT_MODEL

    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = _create_images(tmpdir, n_images)
        output_folder = os.path.join(tmpdir, "segmentation_results")

        viewer = make_napari_viewer_proxy()
        # test generating image embedding, then adding micro-sam dock widgets to the GUI
        viewer = batch_annotator(
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
def test_batch_navigation(make_napari_viewer_proxy):
    """Drive the forward-only navigation harness: advancing saves and loads the next image."""
    n_images = 3
    model_type = DEFAULT_MODEL

    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = _create_images(tmpdir, n_images)
        output_folder = os.path.join(tmpdir, "segmentation_results")

        viewer = make_napari_viewer_proxy()
        viewer = batch_annotator(
            image_paths, output_folder, model_type=model_type, viewer=viewer, return_viewer=True,
        )

        state = AnnotatorState()
        next_image = state.widgets["batch_next"]
        assert "batch_prev" not in state.widgets

        # In a batch session the launcher owns the embedding settings, so the docked annotator's
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

        viewer.close()  # must close the viewer at the end of tests


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_batch_lazy_embeddings(make_napari_viewer_proxy):
    """The segmentation batch computes embeddings lazily per item (saved to a per-item zarr) and
    reuses the loaded model across items, rather than precomputing everything up front.
    """
    model_type = "vit_t" if util.VIT_T_SUPPORT else "vit_b"
    n_images = 3

    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = _create_images(tmpdir, n_images)
        output_folder = os.path.join(tmpdir, "seg")
        embedding_folder = os.path.join(tmpdir, "emb")

        viewer = make_napari_viewer_proxy()
        viewer = batch_annotator(
            image_paths, output_folder, model_type=model_type,
            embedding_path=embedding_folder, viewer=viewer, return_viewer=True,
        )

        def _zarr(index):
            stem = os.path.splitext(os.path.basename(image_paths[index]))[0]
            return os.path.join(embedding_folder, stem + ".zarr")

        state = AnnotatorState()
        # Only the first item's embeddings are computed at launch (lazy), not the whole batch.
        assert os.path.exists(_zarr(0))
        assert not os.path.exists(_zarr(1))

        predictor = state.predictor
        assert predictor is not None

        # Advancing computes the next item's embeddings now, reusing the already-loaded model.
        viewer.layers["committed_objects"].data = np.ones((512, 512), dtype="uint32")
        state.widgets["batch_next"]()
        assert os.path.exists(_zarr(1))
        assert state.predictor is predictor

        viewer.close()


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


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_batch_annotator_continue_or_restart(make_napari_viewer_proxy):
    """Existing outputs are completion markers when continuing and editable inputs when restarting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = _create_images(tmpdir, 3)
        output_folder = os.path.join(tmpdir, "segmentation_results")
        os.makedirs(output_folder)

        completed = np.full((512, 512), 7, dtype="uint32")
        imageio.imwrite(os.path.join(output_folder, "image-0.tif"), completed)

        # Continue mode skips the completed first image and starts at the first missing output.
        viewer = make_napari_viewer_proxy()
        viewer = batch_annotator(
            image_paths, output_folder, model_type=DEFAULT_MODEL,
            viewer=viewer, return_viewer=True, skip_segmented=True,
        )
        np.testing.assert_array_equal(viewer.layers["image"].data, imageio.imread(image_paths[1]))
        assert viewer.layers["committed_objects"].data.sum() == 0
        viewer.close()

        # Restart mode begins at image 0 and loads its saved segmentation for review or editing.
        viewer = make_napari_viewer_proxy()
        viewer = batch_annotator(
            image_paths, output_folder, model_type=DEFAULT_MODEL,
            viewer=viewer, return_viewer=True, skip_segmented=False,
        )
        np.testing.assert_array_equal(viewer.layers["image"].data, imageio.imread(image_paths[0]))
        np.testing.assert_array_equal(viewer.layers["committed_objects"].data, completed)
        viewer.close()


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_batch_annotator_all_images_done(make_napari_viewer_proxy, capsys):
    """A napari launch reports an exhausted batch to its caller instead of writing to the terminal."""
    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = _create_images(tmpdir, 2)
        output_folder = os.path.join(tmpdir, "segmentation_results")
        os.makedirs(output_folder)
        for image_path in image_paths:
            result_path = os.path.join(output_folder, os.path.splitext(os.path.basename(image_path))[0] + ".tif")
            imageio.imwrite(result_path, np.zeros((512, 512), dtype="uint32"))

        viewer = make_napari_viewer_proxy()
        result = batch_annotator(
            image_paths, output_folder, model_type=DEFAULT_MODEL,
            viewer=viewer, return_viewer=True, skip_segmented=True,
        )

        assert result is None
        assert capsys.readouterr().out == ""
        viewer.close()
