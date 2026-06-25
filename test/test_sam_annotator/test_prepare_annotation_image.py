"""Regression tests for multi-channel / dimensionality handling in `prepare_annotation_image`."""

import warnings

import numpy as np
import pytest

from micro_sam.sam_annotator.util import prepare_annotation_image


def _make(shape):
    return np.zeros(shape, dtype="uint8")


# Auto-detect (ndim=None).
@pytest.mark.parametrize(
    "shape, exp_shape, exp_ndim, exp_rgb",
    [
        ((256, 256), (256, 256), 2, False),            # plain 2d grayscale
        ((256, 256, 1), (256, 256), 2, False),         # trailing singleton squeezed
        ((1, 256, 256), (256, 256), 2, False),         # leading singleton squeezed
        ((1, 256, 256, 1), (256, 256), 2, False),      # multiple singletons squeezed
        ((256, 256, 3), (256, 256, 3), 2, True),       # RGB
        ((10, 256, 256), (10, 256, 256), 3, False),    # grayscale volume
        ((2, 256, 256), (2, 256, 256), 3, False),      # channels-first NOT auto-detected -> volume
        ((3, 256, 256), (3, 256, 256), 3, False),      # channels-first -> volume
        ((4, 256, 256), (4, 256, 256), 3, False),      # channels-first -> volume
        ((256, 256, 5), (256, 256, 5), 3, False),      # 5 trailing -> volume
    ],
)
def test_auto_detect(shape, exp_shape, exp_ndim, exp_rgb):
    data, ndim, rgb = prepare_annotation_image(_make(shape))
    assert data.shape == exp_shape
    assert ndim == exp_ndim
    assert rgb is exp_rgb


def test_auto_two_channel_is_padded():
    img = np.ones((32, 48, 2), dtype="uint8")
    data, ndim, rgb = prepare_annotation_image(img)
    assert data.shape == (32, 48, 3) and ndim == 2 and rgb is True
    assert np.all(data[..., :2] == 1) and np.all(data[..., 2] == 0)


def test_auto_four_channel_drops_with_warning():
    with pytest.warns(UserWarning, match="4 channels"):
        data, ndim, rgb = prepare_annotation_image(_make((32, 48, 4)))
    assert data.shape == (32, 48, 3) and ndim == 2 and rgb is True


# 3D + channels (3D+C) is intentionally not supported and must raise, as is anything > 3 spatial dims.
@pytest.mark.parametrize(
    "shape",
    [
        (10, 256, 256, 2),       # 3D + 2 channels
        (10, 256, 256, 3),       # 3D + 3 channels
        (10, 256, 256, 4),       # 3D + 4 channels
        (5, 10, 256, 256),       # 4D with large trailing axis (volumetric time series)
        (2, 3, 10, 256, 256),    # 5D
    ],
)
def test_auto_four_dim_rejected(shape):
    with pytest.raises(ValueError):
        prepare_annotation_image(_make(shape))


# Forced 2d (ndim=2): channels-first and channels-last are both read as a 2d multi-channel image.
@pytest.mark.parametrize(
    "shape, exp_shape, exp_rgb",
    [
        ((256, 256), (256, 256), False),       # grayscale stays grayscale
        ((256, 256, 1), (256, 256), False),    # singleton squeezed -> grayscale
        ((256, 256, 2), (256, 256, 3), True),  # channels-last 2
        ((256, 256, 3), (256, 256, 3), True),  # channels-last 3
        ((256, 256, 4), (256, 256, 3), True),  # channels-last 4
        ((2, 256, 256), (256, 256, 3), True),  # channels-first 2
        ((3, 256, 256), (256, 256, 3), True),  # channels-first 3
        ((4, 256, 256), (256, 256, 3), True),  # channels-first 4
        ((256, 256, 5), (256, 256, 3), True),  # channels-last 5 -> first 3
        ((5, 256, 256), (256, 256, 3), True),  # channels-first 5 -> first 3
    ],
)
def test_forced_2d(shape, exp_shape, exp_rgb):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data, ndim, rgb = prepare_annotation_image(_make(shape), ndim=2)
    assert ndim == 2
    assert data.shape == exp_shape
    assert rgb is exp_rgb


def test_forced_2d_channels_first_values_preserved():
    # The leading axis (size 2) is the channel axis: it becomes R, G; B is zero-padded.
    img = np.zeros((2, 4, 5), dtype="uint8")
    img[0] = 1
    img[1] = 2
    data, ndim, rgb = prepare_annotation_image(img, ndim=2)
    assert data.shape == (4, 5, 3) and ndim == 2 and rgb is True
    assert np.all(data[..., 0] == 1)
    assert np.all(data[..., 1] == 2)
    assert np.all(data[..., 2] == 0)


def test_forced_2d_channels_last_values_preserved():
    img = np.zeros((4, 5, 2), dtype="uint8")
    img[..., 0] = 7
    img[..., 1] = 9
    data, ndim, rgb = prepare_annotation_image(img, ndim=2)
    assert data.shape == (4, 5, 3)
    assert np.all(data[..., 0] == 7)
    assert np.all(data[..., 1] == 9)
    assert np.all(data[..., 2] == 0)


# Forced 3d (ndim=3).
def test_forced_3d_volume():
    data, ndim, rgb = prepare_annotation_image(_make((10, 256, 256)), ndim=3)
    assert data.shape == (10, 256, 256) and ndim == 3 and rgb is False


def test_forced_3d_reads_small_leading_axis_as_volume():
    # A channels-first-looking array forced to 3d is a (Z, H, W) volume, not channels.
    data, ndim, rgb = prepare_annotation_image(_make((4, 256, 256)), ndim=3)
    assert data.shape == (4, 256, 256) and ndim == 3 and rgb is False


def test_forced_3d_on_2d_raises():
    with pytest.raises(ValueError, match="3D volume"):
        prepare_annotation_image(_make((256, 256)), ndim=3)


def test_forced_3d_on_4d_raises():
    # 3D + channels is intentionally blocked: forcing '3d' on a 4D array raises.
    with pytest.raises(ValueError, match="3D volume"):
        prepare_annotation_image(_make((10, 256, 256, 3)), ndim=3)


@pytest.mark.parametrize("bad", [0, 1, 4, "2d"])
def test_invalid_ndim_override(bad):
    with pytest.raises(ValueError, match="Invalid ndim override"):
        prepare_annotation_image(_make((256, 256)), ndim=bad)
