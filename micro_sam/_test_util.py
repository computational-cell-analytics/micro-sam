import numpy as np


def check_layer_initialization(viewer, expected_shape):
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
