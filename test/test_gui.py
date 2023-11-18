import platform

import numpy as np
import pytest

from micro_sam.sam_annotator import annotator_2d
from micro_sam.sam_annotator.annotator_2d import _initialize_viewer


def _check_layer_initialization(viewer):
    """Utility function to check the initial layer setup is correct."""
    assert len(viewer.layers) == 6
    expected_layer_names = ['raw', 'auto_segmentation', 'committed_objects', 'current_object', 'point_prompts', 'prompts']
    for layername in expected_layer_names:
        assert layername in viewer.layers
    # Check layers are empty before beginning tests
    np.testing.assert_equal(viewer.layers["auto_segmentation"].data, 0)
    np.testing.assert_equal(viewer.layers["current_object"].data, 0)
    np.testing.assert_equal(viewer.layers["committed_objects"].data, 0)
    np.testing.assert_equal(viewer.layers["point_prompts"].data, 0)
    assert viewer.layers["prompts"].data == []  # shape data is list, not numpy array


@pytest.mark.gui
@pytest.mark.skipif(platform.system() == "Windows", reason="Gui test is not working on windows.")
def test_annotator_2d(make_napari_viewer_proxy, tmp_path):
    """Integration test for annotator_2d widget with automatic mask generation.

    * Creates 2D image embedding
    * Opens annotator_2d widget in napari
    """
    model_type = "vit_b"
    embedding_path = tmp_path / "test-embedding.zarr"
    # example data - a basic checkerboard pattern
    image = np.zeros((16,16))
    image[:8,:8] = 1
    image[8:,8:] = 1

    viewer = make_napari_viewer_proxy()
    viewer = _initialize_viewer(image, None, None, None)  # TODO: fix hacky workaround
    # test generating image embedding, then adding micro-sam dock widgets to the GUI
    viewer = annotator_2d(
        image,
        embedding_path,
        show_embeddings=False,
        model_type=model_type,
        v=viewer,
        return_viewer=True
    )
    _check_layer_initialization(viewer)
    viewer.close()  # must close the viewer at the end of tests
