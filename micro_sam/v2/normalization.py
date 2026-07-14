"""Common normalization for SAM2 raw microscopy inputs."""

from typing import Optional, Tuple, Union

import numpy as np
from torch_em.transform.raw import normalize_percentile

RAW_NORMALIZATION = "percentile_2_98"
UINT8_RANGE = (0.0, 255.0)
SAM2_INPUT_RANGE = (0.0, 1.0)


def normalize_raw(
    raw: np.ndarray,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    output_range: Tuple[float, float] = SAM2_INPUT_RANGE,
    dtype: Union[str, np.dtype] = "float32",
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> np.ndarray:
    """Percentile-normalize raw image data to the requested input range.

    The lower and upper percentiles (2nd and 98th by default) are mapped to zero
    and one and values outside this range are clipped.
    """
    dtype = np.dtype(dtype)
    is_small_int = np.issubdtype(dtype, np.integer) and dtype.itemsize <= 2  # 8- or 16-bit integers
    if not (np.issubdtype(dtype, np.floating) or is_small_int):
        raise ValueError(
            f"Invalid normalization dtype '{dtype}'. Expect a floating dtype or an 8- or 16-bit integer dtype."
        )
    output_min, output_max = output_range
    if output_max <= output_min:
        raise ValueError(f"Invalid output range {output_range}. The upper bound must exceed the lower bound.")

    raw = np.asarray(raw)
    if raw.size == 0:
        return raw.astype(dtype, copy=False)

    normalized = normalize_percentile(
        raw.astype("float32"), lower=lower_percentile, upper=upper_percentile, axis=axis
    )
    normalized = np.clip(np.asarray(normalized), 0.0, 1.0)
    normalized = normalized * (output_max - output_min) + output_min
    return normalized.astype(dtype, copy=False)


def to_image(image: np.ndarray) -> np.ndarray:
    """Map a 2D or channel-last image to percentile-normalized, channel-last uint8 RGB."""
    from micro_sam.util import _ensure_rgb
    return normalize_raw(_ensure_rgb(image), axis=(0, 1), output_range=UINT8_RANGE, dtype="uint8")
