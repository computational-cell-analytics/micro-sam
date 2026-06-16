"""Deprecated alias for :mod:`micro_sam.v1.training`."""
import warnings
import importlib

warnings.warn(
    "'micro_sam.training' has moved to 'micro_sam.v1.training'. "
    "Update your imports; this alias will be removed in a future release.",
    DeprecationWarning, stacklevel=2,
)


def __getattr__(name):
    return getattr(importlib.import_module("micro_sam.v1.training"), name)
