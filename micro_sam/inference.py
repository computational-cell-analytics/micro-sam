"""Deprecated alias for :mod:`micro_sam.v1.inference`."""
import warnings
import importlib

warnings.warn(
    "'micro_sam.inference' has moved to 'micro_sam.v1.inference'. "
    "Update your imports; this alias will be removed in a future release.",
    DeprecationWarning, stacklevel=2,
)


def __getattr__(name):
    return getattr(importlib.import_module("micro_sam.v1.inference"), name)
