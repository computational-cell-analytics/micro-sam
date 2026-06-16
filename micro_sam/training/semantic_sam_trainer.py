"""Deprecated alias for :mod:`micro_sam.v1.training.semantic_sam_trainer`."""
import importlib


def __getattr__(name):
    return getattr(importlib.import_module("micro_sam.v1.training.semantic_sam_trainer"), name)
