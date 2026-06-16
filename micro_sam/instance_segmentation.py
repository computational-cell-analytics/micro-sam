"""Deprecated alias for :mod:`micro_sam.v1.instance_segmentation`."""
import warnings
import importlib

warnings.warn(
    "'micro_sam.instance_segmentation' has moved to 'micro_sam.v1.instance_segmentation'. "
    "Update your imports; this alias will be removed in a future release.",
    DeprecationWarning, stacklevel=2,
)


def __getattr__(name):
    return getattr(importlib.import_module("micro_sam.v1.instance_segmentation"), name)
