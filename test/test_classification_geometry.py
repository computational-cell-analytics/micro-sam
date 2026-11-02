"""Geometry tests for the classification feature extraction.

The image encoders (SAM1, SAM2 and the VFM encoders) all resize the longest side and zero-pad the
bottom/right, so a non-square image has its content in the top-left sub-rectangle of the square
embedding. These tests pin that the padded region is discarded instead of being treated as image
content, which silently misplaces every feature for non-square inputs. The area test pins that the
area feature is the pixel count of the object, with or without tiling.
"""

import pytest
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


@pytest.mark.parametrize("shape, objects", [
    ((64, 64), [np.s_[8:20, 8:20], np.s_[30:40, 40:56]]),  # upsampling of the embedding, with torch
    ((8, 8), [np.s_[2:4, 2:4], np.s_[4:6, 3:6]]),  # downsampling of the embedding, with bioimage-cpp
])
def test_object_features_match_skimage_interpolation(shape, objects):
    # The objects stay off the 2 px edge, where torch repeats the edge pixel and skimage mirrors it.
    from skimage.transform import resize

    features = np.random.default_rng(0).normal(size=(1, N_CHANNELS, EMBEDDING_SIZE, EMBEDDING_SIZE))
    embeddings = {**_padded_embeddings(), "features": features.astype("float32"), "original_size": shape}
    segmentation = np.zeros(shape, dtype="uint32")
    for seg_id, bb in enumerate(objects, start=1):
        segmentation[bb] = seg_id

    _, object_features = compute_object_features(embeddings, segmentation, verbose=False)

    resized = resize(features[0].transpose(1, 2, 0), shape + (N_CHANNELS,), preserve_range=True)
    expected = [resized[segmentation == seg_id].mean(axis=0) for seg_id in range(1, len(objects) + 1)]
    assert np.allclose(object_features[:, 1:], expected, atol=1e-5)


class _TileGroup(dict):
    """A stand-in for the zarr group of tiled embeddings: one dataset per tile, plus the tiling attrs."""


def _tiled_embeddings(shape, tile_shape, halo):
    from bioimage_cpp.utils import Blocking

    n_tiles = Blocking([0, 0], list(shape), list(tile_shape)).number_of_blocks
    embedding = np.ones((1, N_CHANNELS, EMBEDDING_SIZE, EMBEDDING_SIZE), dtype="float32")
    features = _TileGroup({str(tile_id): embedding for tile_id in range(n_tiles)})
    features.attrs = {"shape": shape, "tile_shape": tile_shape, "halo": halo}
    return {"features": features, "input_size": None, "original_size": None}


@pytest.mark.parametrize("mode", ["2d", "2d_tiled", "3d"])
def test_object_area_is_pixel_count(mode):
    # Identical objects at all positions relative to the tiles. The image and the 288-320 px tiles are larger
    # than the 256 px grid that the features are computed on, and the halos overlap.
    shape = (600, 600)
    segmentation = np.zeros(shape, dtype="uint32")
    corners = range(6, shape[0] - 12, 24)
    for seg_id, (y, x) in enumerate(((y, x) for y in corners for x in corners), start=1):
        segmentation[y:y + 12, x:x + 12] = seg_id
    expected_size = 12 * 12

    if mode == "2d":
        embeddings = {**_padded_embeddings(), "original_size": shape}
    elif mode == "2d_tiled":
        embeddings = _tiled_embeddings(shape, tile_shape=(256, 256), halo=(32, 32))
    else:  # Each object spans both slices.
        segmentation = np.stack([segmentation, segmentation])
        features = np.ones((2, 1, N_CHANNELS, EMBEDDING_SIZE, EMBEDDING_SIZE), dtype="float32")
        embeddings = {"features": features, "input_size": 1024, "original_size": shape}
        expected_size *= 2
    seg_ids, features = compute_object_features(embeddings, segmentation, verbose=False)

    assert seg_ids.tolist() == list(range(1, segmentation.max() + 1))
    assert np.array_equal(features[:, 0], np.full(len(seg_ids), expected_size))
