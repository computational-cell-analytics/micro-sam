"""Common normalization for SAM2 raw microscopy inputs."""

from typing import Optional, Tuple, Union

import numpy as np
from torch_em.transform.raw import normalize_percentile

# Persist the preprocessing (normalization + resize) policy in embedding caches so incompatible
# features are not silently reused. The 2d image path uses per-channel min-max (via `to_image`); the
# 3d / video path uses percentile normalization with the tensor resize used in training. The video
# resize suffix invalidates embeddings created by the former skimage path, which did not match it.
IMAGE_PREPROCESSING = "minmax_per_channel"
VIDEO_PREPROCESSING = "percentile_2_98_per_channel_torch_resize_v1"


def normalize_raw(
    raw: np.ndarray,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    output_dtype: Union[str, np.dtype] = "float32",
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> np.ndarray:
    """Percentile-normalize raw image data to the value range of the output dtype.

    The lower and upper percentiles are mapped to the bounds of the output range and values outside of
    them are clipped. The output range is derived from the output dtype: [0, 1] for floating dtypes and
    the full representable range for integer dtypes. The data is normalized in float32 regardless of its
    input dtype and is only cast to the output dtype afterwards.

    Args:
        raw: The raw image data. May be of any floating or integer dtype.
        axis: The axis or axes to compute the percentiles over. By default they are computed over the
            full data. Pass the spatial axes to normalize each channel independently.
        output_dtype: The dtype of the returned data. Must be a floating or an 8- or 16-bit integer dtype.
            Floating dtypes are normalized to [0, 1], integer dtypes to their full representable range.
        lower_percentile: The percentile that is mapped to the lower bound of the output range.
        upper_percentile: The percentile that is mapped to the upper bound of the output range.

    Returns:
        The normalized image data in the output dtype.

    Raises:
        ValueError: If the output dtype is neither a floating nor an 8- or 16-bit integer dtype.
    """
    output_dtype = np.dtype(output_dtype)
    is_small_int = np.issubdtype(output_dtype, np.integer) and output_dtype.itemsize <= 2  # 8- or 16-bit integers
    if not (np.issubdtype(output_dtype, np.floating) or is_small_int):
        raise ValueError(
            f"Invalid output dtype '{output_dtype}'. Expect a floating dtype or an 8- or 16-bit integer dtype."
        )

    raw = np.asarray(raw)
    if raw.size == 0:
        return raw.astype(output_dtype, copy=False)

    normalized = normalize_percentile(raw.astype("float32"), lower=lower_percentile, upper=upper_percentile, axis=axis)
    normalized = np.clip(np.asarray(normalized), 0.0, 1.0)

    # Integer dtypes are mapped to their full range. Round so that the cast does not bias values downwards.
    if is_small_int:
        info = np.iinfo(output_dtype)
        normalized = np.round(normalized * (float(info.max) - float(info.min)) + float(info.min))

    return normalized.astype(output_dtype, copy=False)


def to_image(image: np.ndarray) -> np.ndarray:
    """Map a 2D or channel-last image to min-max-normalized, channel-last uint8 RGB.

    Args:
        image: The input image. Either 2D or channel-last with up to three channels.

    Returns:
        The channel-last uint8 RGB image, with each channel normalized independently.
    """
    from micro_sam.util import _to_image
    return _to_image(image)
