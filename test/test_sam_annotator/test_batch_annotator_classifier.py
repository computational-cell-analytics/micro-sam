import os
import platform
import tempfile

import numpy as np
import pytest
import imageio.v3 as imageio
from joblib import load
from skimage.data import binary_blobs
from skimage.measure import label

from micro_sam.v2.util import DEFAULT_MODEL
from micro_sam.sam_annotator._state import AnnotatorState
from micro_sam.sam_annotator.object_classifier import batch_object_classifier
from micro_sam.sam_annotator.pixel_classifier import batch_pixel_classifier

MODEL_TYPE = DEFAULT_MODEL


def _images(n, size=256):
    return [binary_blobs(size).astype("float32") for _ in range(n)]


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_batch_object_classifier_navigation(make_napari_viewer_proxy):
    """Drive the object-classifier batch harness: train, then advance to save the prediction, the
    classifier and the accumulated features/labels, and load the next image.
    """
    images = _images(2)
    segmentations = [label(binary_blobs(256)).astype("uint32") for _ in range(2)]
    # Annotate two objects with two classes so the random forest has something to train on.
    ann = np.zeros((256, 256), dtype="uint32")
    object_ids = np.unique(segmentations[0])[1:3]
    ann[segmentations[0] == object_ids[0]] = 1
    ann[segmentations[0] == object_ids[1]] = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        output_folder = os.path.join(tmpdir, "results")
        viewer = make_napari_viewer_proxy()
        viewer = batch_object_classifier(
            images, segmentations, output_folder, model_type=MODEL_TYPE, viewer=viewer, return_viewer=True,
        )

        state = AnnotatorState()
        # The batch starts on the first image with image, segmentation, annotations and prediction layers.
        for name in ("image", "segmentation", "annotations", "prediction"):
            assert name in viewer.layers

        # Paint the annotations and train+predict, then advance.
        viewer.layers["annotations"].data = ann
        state.annotator._run_train_and_predict(True)
        assert state.object_rf is not None
        assert viewer.layers["prediction"].data.sum() > 0

        state.widgets["batch_next"]()

        # The prediction, the classifier and the accumulated features/labels are saved.
        assert os.path.exists(os.path.join(output_folder, "prediction_00000.tif"))
        assert os.path.exists(os.path.join(output_folder, "features.npy"))
        assert os.path.exists(os.path.join(output_folder, "labels.npy"))
        rf_path = os.path.join(output_folder, "rf.joblib")
        assert os.path.exists(rf_path)
        stored = load(rf_path)
        assert "rf" in stored and "model_spec" in stored
        # Two labeled objects were accumulated into the running training set.
        assert np.load(os.path.join(output_folder, "labels.npy")).shape[0] == 2

        # We advanced to the second image. With state forwarding on (the default), the carried-forward
        # classifier auto-predicts on it without any new annotations.
        np.testing.assert_array_equal(viewer.layers["image"].data, images[1])
        assert viewer.layers["prediction"].data.sum() > 0

        viewer.close()


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_batch_pixel_classifier_navigation(make_napari_viewer_proxy):
    """Drive the pixel-classifier batch harness through one train + advance cycle."""
    images = _images(2)
    ann = np.zeros((256, 256), dtype="uint32")
    ann[10:60, 10:60] = 1
    ann[120:170, 120:170] = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        output_folder = os.path.join(tmpdir, "results")
        viewer = make_napari_viewer_proxy()
        viewer = batch_pixel_classifier(
            images, output_folder, model_type=MODEL_TYPE, viewer=viewer, return_viewer=True,
        )

        state = AnnotatorState()
        assert "segmentation" not in viewer.layers  # pixel classification needs no segmentation layer
        for name in ("image", "annotations", "prediction"):
            assert name in viewer.layers

        viewer.layers["annotations"].data = ann
        state.annotator._run_train_and_predict(True)
        assert state.pixel_rf is not None
        assert viewer.layers["prediction"].data.sum() > 0

        state.widgets["batch_next"]()

        assert os.path.exists(os.path.join(output_folder, "prediction_00000.tif"))
        assert os.path.exists(os.path.join(output_folder, "rf.joblib"))
        assert os.path.exists(os.path.join(output_folder, "features.npy"))
        # With state forwarding on (the default), advancing auto-predicts on the second image.
        np.testing.assert_array_equal(viewer.layers["image"].data, images[1])
        assert viewer.layers["prediction"].data.sum() > 0

        viewer.close()


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_object_classifier_forwards_state_by_default(make_napari_viewer_proxy):
    """With 'Forward Classifier State' on (default), the classifier trained on image 0 is applied to
    image 1 automatically (no annotations there) and the accumulated training set keeps growing.
    """
    images = _images(2)
    segmentations = [label(binary_blobs(256)).astype("uint32") for _ in range(2)]
    ann = np.zeros((256, 256), dtype="uint32")
    object_ids = np.unique(segmentations[0])[1:3]
    ann[segmentations[0] == object_ids[0]] = 1
    ann[segmentations[0] == object_ids[1]] = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        viewer = make_napari_viewer_proxy()
        viewer = batch_object_classifier(
            images, segmentations, os.path.join(tmpdir, "results"),
            model_type=MODEL_TYPE, viewer=viewer, return_viewer=True,
        )
        state = AnnotatorState()
        assert state.widgets["batch_forward_state"].value is True  # on by default

        viewer.layers["annotations"].data = ann
        state.annotator._run_train_and_predict(True)

        # Advance without annotating image 1: the forwarded classifier predicts on it.
        state.widgets["batch_next"]()
        assert viewer.layers["annotations"].data.sum() == 0  # no annotations on image 1
        assert viewer.layers["prediction"].data.sum() > 0    # ...but it is predicted
        assert state.batch_training_data[0][1].shape[0] == 2

        viewer.close()


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_object_classifier_independent_when_forward_off(make_napari_viewer_proxy):
    """With 'Forward Classifier State' off, advancing resets the classifier state, so the next image
    is classified independently (no carried-over training set / RF, no auto-prediction).
    """
    images = _images(2)
    segmentations = [label(binary_blobs(256)).astype("uint32") for _ in range(2)]
    ann = np.zeros((256, 256), dtype="uint32")
    object_ids = np.unique(segmentations[0])[1:3]
    ann[segmentations[0] == object_ids[0]] = 1
    ann[segmentations[0] == object_ids[1]] = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        viewer = make_napari_viewer_proxy()
        viewer = batch_object_classifier(
            images, segmentations, os.path.join(tmpdir, "results"),
            model_type=MODEL_TYPE, viewer=viewer, return_viewer=True,
        )
        state = AnnotatorState()
        viewer.layers["annotations"].data = ann
        state.annotator._run_train_and_predict(True)

        # Turn off forwarding, then advance.
        state.widgets["batch_forward_state"].value = False
        state.widgets["batch_next"]()

        # The accumulated training set and the classifier are reset. Image 1 is not auto-predicted.
        assert state.batch_training_data == {}
        assert state.object_rf is None
        assert viewer.layers["prediction"].data.sum() == 0

        viewer.close()


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_pixel_classifier_forwards_state_by_default(make_napari_viewer_proxy):
    """The forwarding also works for the pixel classifier: image 1 is auto-predicted from image 0."""
    images = _images(2)
    ann = np.zeros((256, 256), dtype="uint32")
    ann[10:60, 10:60] = 1
    ann[120:170, 120:170] = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        viewer = make_napari_viewer_proxy()
        viewer = batch_pixel_classifier(
            images, os.path.join(tmpdir, "results"), model_type=MODEL_TYPE, viewer=viewer, return_viewer=True,
        )
        state = AnnotatorState()
        assert state.widgets["batch_forward_state"].value is True

        viewer.layers["annotations"].data = ann
        state.annotator._run_train_and_predict(True)

        state.widgets["batch_next"]()
        assert viewer.layers["annotations"].data.sum() == 0
        assert viewer.layers["prediction"].data.sum() > 0

        viewer.close()


def _object_annotations(segmentation, n_objects=2):
    ann = np.zeros(segmentation.shape, dtype="uint32")
    object_ids = np.unique(segmentation)[1:n_objects + 1]
    for i, object_id in enumerate(object_ids):
        ann[segmentation == object_id] = i % 2 + 1
    return ann


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_object_classifier_go_back(make_napari_viewer_proxy):
    """Going back restores the annotations and segmentation of an image. Leaving it again replaces its
    training data instead of adding it twice, so the classifier is trained once on each image's latest annotations.
    """
    images = _images(3)
    segmentations = [label(binary_blobs(256)).astype("uint32") for _ in range(3)]
    ann0, ann1 = _object_annotations(segmentations[0], 4), _object_annotations(segmentations[1], 2)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_folder = os.path.join(tmpdir, "results")
        viewer = make_napari_viewer_proxy()
        viewer = batch_object_classifier(
            images, segmentations, output_folder, model_type=MODEL_TYPE, viewer=viewer, return_viewer=True,
        )
        state = AnnotatorState()
        assert "batch_previous" in state.widgets

        # Going back from the first image does nothing.
        state.widgets["batch_previous"]()
        assert state.batch_index == 0

        # Annotate image 0 without training: leaving it retrains, so the saved classifier includes it.
        viewer.layers["annotations"].data = ann0
        state.widgets["batch_next"]()
        assert os.path.exists(os.path.join(output_folder, "annotations_00000.tif"))
        assert load(os.path.join(output_folder, "rf.joblib"))["rf"] is not None
        assert imageio.imread(os.path.join(output_folder, "prediction_00000.tif")).any()
        assert state.batch_training_data[0][1].shape[0] == 4

        viewer.layers["annotations"].data = ann1
        state.widgets["batch_next"]()
        assert state.batch_index == 2

        # Go back twice: image 0 shows its own annotations and segmentation again.
        state.widgets["batch_previous"]()
        state.widgets["batch_previous"]()
        assert state.batch_index == 0
        np.testing.assert_array_equal(viewer.layers["image"].data, images[0])
        np.testing.assert_array_equal(viewer.layers["segmentation"].data, segmentations[0])
        np.testing.assert_array_equal(viewer.layers["annotations"].data, ann0)
        assert viewer.layers["prediction"].data.any()

        # Erase two of the four objects. Leaving replaces the entry of image 0 (no duplicates).
        viewer.layers["annotations"].data = _object_annotations(segmentations[0], 2)
        state.widgets["batch_next"]()
        assert state.batch_index == 1  # visited items are not skipped
        assert sorted(state.batch_training_data) == [0, 1]
        assert state.batch_training_data[0][1].shape[0] == 2
        assert np.load(os.path.join(output_folder, "labels.npy")).shape[0] == 4
        np.testing.assert_array_equal(viewer.layers["annotations"].data, ann1)

        viewer.close()


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_pixel_classifier_go_back(make_napari_viewer_proxy):
    """Going back in the pixel classifier restores the annotations and predicts on the image again."""
    images = _images(2)
    ann = np.zeros((256, 256), dtype="uint32")
    ann[10:60, 10:60] = 1
    ann[120:170, 120:170] = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        viewer = make_napari_viewer_proxy()
        viewer = batch_pixel_classifier(
            images, os.path.join(tmpdir, "results"), model_type=MODEL_TYPE, viewer=viewer, return_viewer=True,
        )
        state = AnnotatorState()
        viewer.layers["annotations"].data = ann
        state.annotator._run_train_and_predict(True)

        state.widgets["batch_next"]()
        n_rows = state.batch_training_data[0][1].shape[0]
        state.widgets["batch_previous"]()
        np.testing.assert_array_equal(viewer.layers["image"].data, images[0])
        np.testing.assert_array_equal(viewer.layers["annotations"].data, ann)
        assert viewer.layers["prediction"].data.any()

        # Leaving again without changes keeps the same training data.
        state.widgets["batch_next"]()
        assert state.batch_training_data[0][1].shape[0] == n_rows

        viewer.close()


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_classifier_saves_on_close(make_napari_viewer_proxy):
    """Closing the viewer saves the untrained annotations and a classifier trained on them, but not the
    prediction, so a resumed session starts on this image and restores its annotations.
    """
    images = _images(2)
    segmentations = [label(binary_blobs(256)).astype("uint32") for _ in range(2)]
    ann = _object_annotations(segmentations[0], 2)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_folder = os.path.join(tmpdir, "results")
        viewer = make_napari_viewer_proxy()
        viewer = batch_object_classifier(
            images, segmentations, output_folder, model_type=MODEL_TYPE, viewer=viewer, return_viewer=True,
        )
        viewer.layers["annotations"].data = ann
        # Close the main window like the user does. 'viewer.close()' removes the layers first.
        AnnotatorState().annotator.window().close()
        viewer.close()

        assert os.path.exists(os.path.join(output_folder, "rf.joblib"))
        assert not os.path.exists(os.path.join(output_folder, "prediction_00000.tif"))
        np.testing.assert_array_equal(imageio.imread(os.path.join(output_folder, "annotations_00000.tif")), ann)

        viewer = make_napari_viewer_proxy()
        viewer = batch_object_classifier(
            images, segmentations, output_folder, model_type=MODEL_TYPE, viewer=viewer, return_viewer=True,
        )
        np.testing.assert_array_equal(viewer.layers["annotations"].data, ann)
        assert viewer.layers["prediction"].data.any()
        viewer.close()


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_classifier_retrains_on_changed_settings(make_napari_viewer_proxy):
    """Changing a training setting after the last training run retrains the classifier when leaving the image."""
    images = _images(2)
    ann = np.zeros((256, 256), dtype="uint32")
    ann[10:60, 10:60] = 1
    ann[120:170, 120:170] = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        viewer = make_napari_viewer_proxy()
        viewer = batch_pixel_classifier(
            images, os.path.join(tmpdir, "results"), model_type=MODEL_TYPE, viewer=viewer, return_viewer=True,
        )
        state = AnnotatorState()
        viewer.layers["annotations"].data = ann
        state.annotator._run_train_and_predict(True)
        assert not state.annotator.is_prediction_stale()

        # Reduce the features to 8 PCA components without retraining.
        state.annotator._set_options(True, 8, False)
        assert state.annotator.is_prediction_stale()
        state.widgets["batch_next"]()
        assert load(os.path.join(tmpdir, "results", "rf.joblib"))["rf"].named_steps["pca"].n_components == 8

        viewer.close()
