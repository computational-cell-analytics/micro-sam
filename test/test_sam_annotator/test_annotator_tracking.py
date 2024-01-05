import platform

import numpy as np
import pytest
from skimage.data import binary_blobs

import micro_sam.util as util
from micro_sam.sam_annotator import annotator_tracking


def _check_layer_initialization(viewer, expected_shape):
    """Utility function to check the initial layer setup is correct."""

    assert len(viewer.layers) == 6
    expected_layer_names = [
        "image", "auto_segmentation", "committed_objects", "current_object", "point_prompts", "prompts"
    ]

    for layer_name in expected_layer_names:
        assert layer_name in viewer.layers

    # Check prompt layers
    assert viewer.layers["prompts"].data == []  # shape data is list, not numpy array
    np.testing.assert_equal(viewer.layers["point_prompts"].data, 0)

    # Check segmentation layers.
    for layer_name in ["auto_segmentation", "committed_objects", "current_object"]:
        assert viewer.layers[layer_name].data.shape == expected_shape
        np.testing.assert_equal(viewer.layers[layer_name].data, 0)


@pytest.mark.gui
@pytest.mark.skipif(platform.system() == "Windows", reason="Gui test is not working on windows.")
def test_annotator_tracking(make_napari_viewer_proxy):
    """Integration test for annotator_2d widget with automatic mask generation.

    * Creates 3D image embedding
    * Opens annotator_3d widget in napari
    """

    image = np.stack(4 * [binary_blobs(512)])
    model_type = "vit_t" if util.VIT_T_SUPPORT else "vit_b"

    viewer = make_napari_viewer_proxy()
    # test generating image embedding, then adding micro-sam dock widgets to the GUI
    viewer = annotator_tracking(
        image,
        model_type=model_type,
        viewer=viewer,
        return_viewer=True
    )

    _check_layer_initialization(viewer, image.shape)
    viewer.close()  # must close the viewer at the end of tests
