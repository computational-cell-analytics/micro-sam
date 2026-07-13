"""Common normalization for SAM2 raw microscopy inputs."""

import warnings
from typing import Optional, Tuple, Union

import numpy as np
from torch_em.transform.raw import normalize_percentile

RAW_NORMALIZATION = "percentile_1_99"
UINT8_RANGE = (0.0, 255.0)
SAM2_INPUT_RANGE = (0.0, 1.0)


def normalize_raw(
    raw: np.ndarray,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    output_range: Tuple[float, float] = SAM2_INPUT_RANGE,
    dtype: Union[str, np.dtype] = "float32",
) -> np.ndarray:
    """Percentile-normalize raw image data to the requested input range.

    The 1st and 99th percentiles are mapped to zero and one and values outside
    this range are clipped.
    """
    dtype = np.dtype(dtype)
    if dtype != np.dtype("uint8") and not np.issubdtype(dtype, np.floating):
        raise ValueError(
            f"Invalid normalization dtype '{dtype}'. Expect a floating dtype or uint8."
        )
    output_min, output_max = output_range
    if output_max <= output_min:
        raise ValueError(f"Invalid output range {output_range}. The upper bound must exceed the lower bound.")

    normalized = normalize_percentile(
        np.asarray(raw).astype("float32"), lower=1.0, upper=99.0, axis=axis
    )
    normalized = np.clip(np.asarray(normalized), 0.0, 1.0)
    normalized = normalized * (output_max - output_min) + output_min
    return normalized.astype(dtype, copy=False)


def to_image(image: np.ndarray) -> np.ndarray:
    """Map a 2D image to percentile-normalized, channel-last uint8 RGB."""
    input_ = image
    ndim = input_.ndim
    n_channels = 1 if ndim == 2 else input_.shape[-1]

    if ndim == 2:
        input_ = np.concatenate([input_[..., None]] * 3, axis=-1)
    elif ndim == 3 and n_channels == 1:
        input_ = np.concatenate([input_] * 3, axis=-1)
    elif ndim == 3 and n_channels == 2:
        zero_channel = np.zeros(input_.shape[:2] + (1,), dtype=input_.dtype)
        input_ = np.concatenate([input_, zero_channel], axis=-1)
    elif ndim == 3 and n_channels == 3:
        pass
    elif ndim == 3 and n_channels > 3:
        warnings.warn(f"You provided an input with {n_channels} channels. Only the first three will be used.")
        input_ = input_[..., :3]
    else:
        raise ValueError(
            f"Invalid input dimensionality {ndim}. Expect either a 2D input (=grayscale image) "
            "or a 3D input (= image with channels)."
        )

    return normalize_raw(input_, axis=(0, 1), output_range=UINT8_RANGE, dtype="uint8")
