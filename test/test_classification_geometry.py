"""Geometry tests for the classification feature extraction.

The image encoders (SAM1, SAM2 and the VFM encoders) all resize the longest side and zero-pad the
bottom/right, so a non-square image has its content in the top-left sub-rectangle of the square
embedding. These tests pin that the padded region is discarded instead of being treated as image
content, which silently misplaces every feature for non-square inputs.
"""

import numpy as np

from micro_sam.object_classification import compute_object_features
from micro_sam.pixel_classification import compute_pixel_features

IMAGE_SHAPE = (48, 64)  # Non-square, so a quarter of the square embedding is padding.
EMBEDDING_SIZE = 16
N_CHANNELS = 8


def _padded_embeddings():
    """A (1, C, 16, 16) embedding that is 1.0 in the content region and 0.0 in the padded region."""
    content_rows = int(round(EMBEDDING_SIZE * IMAGE_SHAPE[0] / max(IMAGE_SHAPE)))
    features = np.zeros((1, N_CHANNELS, EMBEDDING_SIZE, EMBEDDING_SIZE), dtype="float32")
    features[:, :, :content_rows] = 1.0
    return {
        "features": features,
        "high_res_feats": [],  # Marks the embeddings as SAM2, which pads exactly like SAM1.
        "input_size": 1024,
        "original_size": IMAGE_SHAPE,
    }


def test_pixel_features_exclude_padding():
    features, grid_shape = compute_pixel_features(_padded_embeddings(), IMAGE_SHAPE, verbose=False)

    # The grid keeps the image aspect ratio and every feature comes from the content region.
    assert np.isclose(grid_shape[0] / grid_shape[1], IMAGE_SHAPE[0] / IMAGE_SHAPE[1], atol=0.01)
    assert features.shape == (grid_shape[0] * grid_shape[1], N_CHANNELS)
    assert np.allclose(features, 1.0), f"padded region leaked into the features: min {features.min()}"


def test_object_features_exclude_padding():
    # An object low in the image, which lands in the padded region if the geometry is wrong.
    # It is inset from the last rows so that interpolation at the content/padding border does not
    # bleed into it, which would blur the distinction this test draws.
    segmentation = np.zeros(IMAGE_SHAPE, dtype="uint32")
    segmentation[-16:-6, -12:] = 1

    seg_ids, features = compute_object_features(_padded_embeddings(), segmentation, verbose=False)

    assert seg_ids.tolist() == [1]
    # Feature layout is [area, per-channel mean], so the embedding means follow the area column.
    assert np.allclose(features[0, 1:], 1.0), f"padded region leaked into the object features: {features[0, 1:]}"
