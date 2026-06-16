"""Deprecated alias for :mod:`micro_sam.v1.inference`."""
import warnings

from micro_sam.v1 import inference as _v1_module

warnings.warn(
    "'micro_sam.inference' has moved to 'micro_sam.v1.inference'. "
    "Update your imports; this alias will be removed in a future release.",
    DeprecationWarning, stacklevel=2,
)


def __getattr__(name):
    return getattr(_v1_module, name)
