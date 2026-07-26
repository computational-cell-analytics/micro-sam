"""Clearing the annotations must not leave the previous prompt driving the next segmentation."""

import platform

import numpy as np
import torch
import pytest

from micro_sam.v2.util import DEFAULT_MODEL
from micro_sam.sam_annotator.annotator import annotator
from micro_sam.sam_annotator._state import AnnotatorState

# Two well-separated squares at known positions. A dense random texture (e.g. 'binary_blobs') is not
# usable here: SAM2 then segments some nearby component, which need not contain the clicked pixel.
FIRST_OBJECT = (48, 88)
SECOND_OBJECT = (168, 208)


def two_objects_2d(size=256):
    image = np.zeros((size, size), dtype="float32")
    image[FIRST_OBJECT[0]:FIRST_OBJECT[1], FIRST_OBJECT[0]:FIRST_OBJECT[1]] = 1.0
    image[SECOND_OBJECT[0]:SECOND_OBJECT[1], SECOND_OBJECT[0]:SECOND_OBJECT[1]] = 1.0
    return image


def two_objects_3d(width=256, depth=4, height=256):
    volume = np.zeros((depth, height, width), dtype="float32")
    for lo, hi in (FIRST_OBJECT, SECOND_OBJECT):
        volume[:, lo:hi, lo:hi] = 1.0
    return volume


def center(bounds):
    return (bounds[0] + bounds[1]) // 2


def interactive_widget():
    return AnnotatorState().widgets["interactive"]


def add_positive_point(viewer, y, x):
    layer = viewer.layers["point_prompts"]
    points = np.concatenate([np.asarray(layer.data).reshape(-1, 2), [[float(y), float(x)]]])
    labels = list(layer.properties.get("label", [])) + ["positive"]
    layer.data = points
    layer.properties = {"label": np.array(labels, dtype=object)}
    layer.refresh()


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.skipif(platform.system() == "Windows", reason="GUI test does not work on Windows.")
def test_clear_annotations_drops_the_previous_prompt(make_napari_viewer_proxy):
    image = two_objects_2d()
    viewer = annotator(
        image, model_type=DEFAULT_MODEL, viewer=make_napari_viewer_proxy(),
        device="cpu", return_viewer=True,
    )
    widget = interactive_widget()
    current_object = viewer.layers["current_object"]

    # Two points far apart, so the objects they segment cannot be confused.
    first = (center(FIRST_OBJECT),) * 2
    second = (center(SECOND_OBJECT),) * 2

    add_positive_point(viewer, *first)
    widget.segment()
    first_seg = current_object.data.copy()
    assert first_seg.max() > 0, "the first prompt produced no segmentation"

    widget.clear()
    assert len(viewer.layers["point_prompts"].data) == 0
    assert current_object.data.max() == 0

    add_positive_point(viewer, *second)
    widget.segment()
    second_seg = current_object.data.copy()
    assert second_seg.max() > 0, "the second prompt produced no segmentation"

    # The new segmentation must follow the new point, not the cleared one.
    assert second_seg[second] > 0, "the segmentation does not cover the new point"
    assert not np.array_equal(first_seg, second_seg), "the cleared prompt still drives the result"

    viewer.close()


def add_positive_point_3d(viewer, z, y, x):
    layer = viewer.layers["point_prompts"]
    points = np.concatenate([np.asarray(layer.data).reshape(-1, 3), [[float(z), float(y), float(x)]]])
    labels = list(layer.properties.get("label", [])) + ["positive"]
    layer.data = points
    layer.properties = {"label": np.array(labels, dtype=object)}
    layer.refresh()


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.skipif(platform.system() == "Windows", reason="GUI test does not work on Windows.")
def test_clear_annotations_drops_the_previous_prompt_3d(make_napari_viewer_proxy):
    """The 3d per-slice path keeps a persistent SAM2 state, so a cleared prompt must not survive."""
    volume = two_objects_3d()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    viewer = annotator(
        volume, model_type=DEFAULT_MODEL, viewer=make_napari_viewer_proxy(),
        device=device, ndim=3, return_viewer=True,
    )
    widget = interactive_widget()
    widget._segment_widget.apply_to_volume = False
    current_object = viewer.layers["current_object"]

    z = 1
    viewer.dims.set_point(0, z)
    first = (center(FIRST_OBJECT),) * 2
    second = (center(SECOND_OBJECT),) * 2

    add_positive_point_3d(viewer, z, *first)
    widget.segment()
    first_seg = current_object.data[z].copy()
    assert first_seg.max() > 0, "the first prompt produced no segmentation"
    assert first_seg[first] > 0, "the first segmentation does not cover the first point"

    widget.clear()
    assert len(viewer.layers["point_prompts"].data) == 0
    assert current_object.data[z].max() == 0

    add_positive_point_3d(viewer, z, *second)
    widget.segment()
    second_seg = current_object.data[z].copy()

    assert second_seg.max() > 0, "the second prompt produced no segmentation"
    assert second_seg[second] > 0, "the segmentation does not cover the new point"
    assert not np.array_equal(first_seg, second_seg), "the cleared prompt still drives the result"

    viewer.close()


@pytest.mark.gui
@pytest.mark.slow
@pytest.mark.skipif(platform.system() == "Windows", reason="GUI test does not work on Windows.")
def test_clear_annotations_drops_the_previous_prompt_volume(make_napari_viewer_proxy):
    """Volume propagation reuses the persistent SAM2 state, so a cleared prompt must not survive."""
    volume = two_objects_3d()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    viewer = annotator(
        volume, model_type=DEFAULT_MODEL, viewer=make_napari_viewer_proxy(),
        device=device, ndim=3, return_viewer=True,
    )
    widget = interactive_widget()
    widget._segment_widget.apply_to_volume = True
    current_object = viewer.layers["current_object"]

    z = 1
    viewer.dims.set_point(0, z)
    first = (center(FIRST_OBJECT),) * 2
    second = (center(SECOND_OBJECT),) * 2

    add_positive_point_3d(viewer, z, *first)
    widget.segment()
    first_seg = current_object.data.copy()
    assert first_seg.max() > 0, "the first prompt produced no segmentation"

    widget.clear()
    assert len(viewer.layers["point_prompts"].data) == 0

    add_positive_point_3d(viewer, z, *second)
    widget.segment()
    second_seg = current_object.data.copy()

    assert second_seg.max() > 0, "the second prompt produced no segmentation"
    assert second_seg[(z,) + second] > 0, "the segmentation does not cover the new point"
    assert not np.array_equal(first_seg, second_seg), "the cleared prompt still drives the result"

    viewer.close()
