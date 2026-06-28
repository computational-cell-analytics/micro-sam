import os
import platform
import tempfile

import numpy as np
import pytest
from joblib import load
from skimage.data import binary_blobs
from skimage.measure import label

from micro_sam.v2.util import DEFAULT_MODEL
from micro_sam.sam_annotator._state import AnnotatorState
from micro_sam.sam_annotator.object_classifier import image_series_object_classifier
from micro_sam.sam_annotator.pixel_classifier import image_series_pixel_classifier

MODEL_TYPE = DEFAULT_MODEL


def _images(n, size=256):
    return [binary_blobs(size).astype("float32") for _ in range(n)]


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_image_series_object_classifier_navigation(make_napari_viewer_proxy):
    """Drive the object-classifier series harness: train, then advance to save the prediction, the
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
        viewer = image_series_object_classifier(
            images, segmentations, output_folder, model_type=MODEL_TYPE, viewer=viewer, return_viewer=True,
        )

        state = AnnotatorState()
        # The series starts on the first image with image, segmentation, annotations and prediction layers.
        for name in ("image", "segmentation", "annotations", "prediction"):
            assert name in viewer.layers

        # Paint the annotations and train+predict, then advance.
        viewer.layers["annotations"].data = ann
        state.annotator._run_train_and_predict(True)
        assert state.object_rf is not None
        assert viewer.layers["prediction"].data.sum() > 0

        state.widgets["series_next"]()

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

        # We advanced to the second image and the cached features were dropped.
        assert state.object_features is None
        np.testing.assert_array_equal(viewer.layers["image"].data, images[1])

        viewer.close()


@pytest.mark.gui
@pytest.mark.skipif(platform.system() in ("Windows",), reason="Gui test is not working on windows.")
def test_image_series_pixel_classifier_navigation(make_napari_viewer_proxy):
    """Drive the pixel-classifier series harness through one train + advance cycle."""
    images = _images(2)
    ann = np.zeros((256, 256), dtype="uint32")
    ann[10:60, 10:60] = 1
    ann[120:170, 120:170] = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        output_folder = os.path.join(tmpdir, "results")
        viewer = make_napari_viewer_proxy()
        viewer = image_series_pixel_classifier(
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

        state.widgets["series_next"]()

        assert os.path.exists(os.path.join(output_folder, "prediction_00000.tif"))
        assert os.path.exists(os.path.join(output_folder, "rf.joblib"))
        assert os.path.exists(os.path.join(output_folder, "features.npy"))
        assert state.pixel_features is None
        np.testing.assert_array_equal(viewer.layers["image"].data, images[1])

        viewer.close()
