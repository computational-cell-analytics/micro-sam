"""Deprecated alias for :mod:`micro_sam.v1.models.simple_sam_3d_wrapper`."""
import importlib


def __getattr__(name):
    return getattr(importlib.import_module("micro_sam.v1.models.simple_sam_3d_wrapper"), name)
