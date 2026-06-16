"""Deprecated alias for :mod:`micro_sam.v1.models.build_sam`."""
import importlib


def __getattr__(name):
    return getattr(importlib.import_module("micro_sam.v1.models.build_sam"), name)
