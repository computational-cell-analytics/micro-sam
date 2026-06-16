"""Deprecated alias for :mod:`micro_sam.v1.prompt_based_segmentation`."""
import warnings

from micro_sam.v1 import prompt_based_segmentation as _v1_module

warnings.warn(
    "'micro_sam.prompt_based_segmentation' has moved to 'micro_sam.v1.prompt_based_segmentation'. "
    "Update your imports; this alias will be removed in a future release.",
    DeprecationWarning, stacklevel=2,
)


def __getattr__(name):
    return getattr(_v1_module, name)
