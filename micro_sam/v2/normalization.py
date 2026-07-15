"""Common normalization for SAM2 raw microscopy inputs."""

from typing import Optional, Tuple, Union

import numpy as np
from torch_em.transform.raw import normalize_percentile

RAW_NORMALIZATION = "percentile_2_98"
UINT8_RANGE = (0.0, 255.0)
SAM2_INPUT_RANGE = (0.0, 1.0)


def _full_range(dtype: np.dtype) -> Tuple[float, float]:
    """Return the value range that an output dtype is normalized to by default.

    Args:
        dtype: The output dtype.

    Returns:
        The [0, 1] range for floating dtypes and the full representable range for integer dtypes.
    """
    if np.issubdtype(dtype, np.floating):
        return SAM2_INPUT_RANGE
    info = np.iinfo(dtype)
    return (float(info.min), float(info.max))


def normalize_raw(
    raw: np.ndarray,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    output_range: Optional[Tuple[float, float]] = None,
    output_dtype: Union[str, np.dtype] = "float32",
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> np.ndarray:
    """Percentile-normalize raw image data to the requested output range.

    The lower and upper percentiles are mapped to the bounds of the output range and values outside
    of it are clipped. The data is normalized in float32 regardless of its input dtype and is only
    cast to the output dtype afterwards.

    Args:
        raw: The raw image data.
        axis: The axis or axes to compute the percentiles over. By default they are computed over the
            full data. Pass the spatial axes to normalize each channel independently.
        output_range: The value range the percentiles are mapped to. By default [0, 1] is used for
            floating output dtypes and the full representable range for integer output dtypes.
        output_dtype: The dtype of the returned data. Must be a floating or an 8- or 16-bit integer dtype.
            Integer dtypes only support their full range as output range, so that the mapping does not
            waste or exceed the representable values.
        lower_percentile: The percentile that is mapped to the lower bound of the output range.
        upper_percentile: The percentile that is mapped to the upper bound of the output range.

    Returns:
        The normalized image data in the output dtype.
    """
    output_dtype = np.dtype(output_dtype)
    is_small_int = np.issubdtype(output_dtype, np.integer) and output_dtype.itemsize <= 2  # 8- or 16-bit integers
    if not (np.issubdtype(output_dtype, np.floating) or is_small_int):
        raise ValueError(
            f"Invalid output dtype '{output_dtype}'. Expect a floating dtype or an 8- or 16-bit integer dtype."
        )

    full_range = _full_range(output_dtype)
    if output_range is None:
        output_range = full_range
    elif is_small_int and tuple(float(bound) for bound in output_range) != full_range:
        raise ValueError(
            f"Invalid output range {output_range} for the integer output dtype '{output_dtype}'. "
            f"Integer output dtypes only support their full range {full_range}."
        )

    output_min, output_max = output_range
    if output_max <= output_min:
        raise ValueError(f"Invalid output range {output_range}. The upper bound must exceed the lower bound.")

    raw = np.asarray(raw)
    if raw.size == 0:
        return raw.astype(output_dtype, copy=False)

    normalized = normalize_percentile(
        raw.astype("float32"), lower=lower_percentile, upper=upper_percentile, axis=axis
    )
    normalized = np.clip(np.asarray(normalized), 0.0, 1.0)
    normalized = normalized * (output_max - output_min) + output_min
    return normalized.astype(output_dtype, copy=False)


def to_image(image: np.ndarray) -> np.ndarray:
    """Map a 2D or channel-last image to percentile-normalized, channel-last uint8 RGB.

    Args:
        image: The input image. Either 2D or channel-last with up to three channels.

    Returns:
        The channel-last uint8 RGB image, with each channel normalized independently.
    """
    from micro_sam.util import _ensure_rgb
    return normalize_raw(_ensure_rgb(image), axis=(0, 1), output_dtype="uint8")
