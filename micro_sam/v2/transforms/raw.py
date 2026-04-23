import numpy as np

from torch_em.transform.raw import normalize, normalize_percentile

from .labels import _em_cell_label_trafo  # noqa: F401
from .labels import _axondeepseg_pre_label_transform  # noqa: F401
from .labels import _plantseg_label_trafo  # noqa: F401


# NOTE: This is a legacy function: we will keep this for now for unpickling checkpoints saved before the refactor.
def _axondeepseg_label_transform(y):  # noqa: F811
    from skimage.measure import label as connected_components
    return connected_components(y == 2).astype("uint32")


def to_rgb(image):
    if image.ndim == 2:  # Simple triplication with channels first.
        image = np.concatenate([image[None]] * 3, axis=0)

    if image.ndim == 3 and image.shape[-1] == 3:  # Make channels first for RGB images.
        image = image.transpose(2, 0, 1)

    assert image.ndim == 3
    return image


def _to_8bit(raw):
    "Ensures three channels for inputs and rescales them to [0, 1]."
    if raw.ndim == 3 and raw.shape[0] == 1:  # If the inputs have 1 channel, we triplicate it.
        raw = np.concatenate([raw] * 3, axis=0)

    raw = to_rgb(raw)  # Ensure all images are in 3-channels: triplicate one channel to three channels.
    raw = normalize(raw)  # [0, 1] — SAM2 model applies ImageNet normalization internally
    return raw


def _identity(x):
    "Ensures three channels for inputs and normalizes to [0, 1]."
    x = to_rgb(x)
    x = normalize(x)  # [0, 1] — SAM2 model applies ImageNet normalization internally
    return x


def _cellpose_raw_trafo(x):
    """Transforms input images to desired format.

    NOTE: The input channel logic is arranged a bit strangely in `cyto` dataset.
    This function takes care of it here.
    """
    r, g, b = x

    assert g.max() != 0
    if r.max() == 0:
        # The image is 1 channel and exists in green channel only.
        assert b.max() == 0
        x = np.concatenate([g[None]] * 3, axis=0)

    elif r.max() != 0 and g.max() != 0:
        # The image is 2 channels and we sort the channels such that - 0: cell, 1: nucleus
        x = np.stack([g, r, np.zeros_like(b)], axis=0)

    x = to_rgb(x)  # Ensures three channels for inputs and avoids rescaling inputs.
    x = normalize(x)  # [0, 1] — SAM2 model applies ImageNet normalization internally
    return x


def _normalize_percentile(x, axis=None):
    """Transforms input images with percentile normalization.

    NOTE: For example, this is a specific input transformation for
    'rgb' format of TissueNet image data for the expected axes.
    """
    x = normalize_percentile(x, axis=None)  # Use 1st and 99th percentile values for min-max normalization.
    x = np.clip(x, 0, 1)  # [0, 1] — SAM2 model applies ImageNet normalization internally
    return x
