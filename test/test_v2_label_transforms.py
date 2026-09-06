"""Tests for the label transforms of the automatic branch, in particular the contact channel."""

import numpy as np
import pytest

from micro_sam.v2.transforms.labels import (
    DirectedPerObjectBoundaryDistanceTransform, GeodesicHybridDistanceTransform, _JointGeodesicLabelTransform,
    touching_boundaries,
)


def _two_squares(gap: int) -> np.ndarray:
    """Two squares side by side, touching for gap=0 or separated by 'gap' background columns."""
    labels = np.zeros((40, 60), dtype="uint16")
    labels[10:30, 10:30] = 1
    labels[10:30, 30 + gap:50 + gap] = 2
    return labels


def test_touching_boundaries_marks_both_sides_of_a_direct_contact():
    contact = touching_boundaries(_two_squares(gap=0), dilation=0)
    rows, cols = np.nonzero(contact)
    assert set(cols.tolist()) == {29, 30}
    # The background pixel just beyond either end of the line sees both objects as well.
    assert rows.min() == 9 and rows.max() == 30
    assert contact[10:30, 29].all() and contact[10:30, 30].all()


def test_touching_boundaries_marks_a_one_pixel_gap():
    contact = touching_boundaries(_two_squares(gap=1), dilation=0)
    assert set(np.nonzero(contact)[1].tolist()) == {30}
    # A two pixel gap is out of reach of the default radius.
    assert not touching_boundaries(_two_squares(gap=2), dilation=0).any()


def test_touching_boundaries_ignores_isolated_objects_and_dilates():
    labels = _two_squares(gap=0)
    labels[2:8, 52:58] = 3
    contact = touching_boundaries(labels)
    assert not contact[2:8, 52:58].any()
    # One dilation pass widens the two pixel line to four pixels.
    assert set(np.nonzero(contact[20])[0].tolist()) == {28, 29, 30, 31}
    assert not touching_boundaries(np.zeros((8, 8), dtype="uint8")).any()


def test_touching_boundaries_handles_3d_and_large_ids():
    labels = np.zeros((3, 20, 20), dtype="uint32")
    labels[:, 5:10, 5:15] = 70000
    labels[:, 10:15, 5:15] = 3
    contact = touching_boundaries(labels, dilation=0)
    assert contact.shape == labels.shape
    assert set(np.nonzero(contact[1])[0].tolist()) == {9, 10}


@pytest.mark.parametrize(
    "transform_class", [DirectedPerObjectBoundaryDistanceTransform, GeodesicHybridDistanceTransform],
)
def test_contact_channel_is_appended_last(transform_class):
    labels = _two_squares(gap=0)
    # Ellipsoidal ends so that the objects do not fill their bounding boxes.
    labels[10:12, 10:12] = 0
    labels[28:30, 48:50] = 0
    plain = transform_class()(labels)
    with_contact = transform_class(contact=True)(labels)
    assert plain.shape == (4, 40, 60)
    assert with_contact.shape == (5, 40, 60)
    assert with_contact.dtype == np.float32
    np.testing.assert_array_equal(with_contact[:4], plain)
    np.testing.assert_array_equal(with_contact[4] > 0, touching_boundaries(labels))
    assert set(np.unique(with_contact[4]).tolist()) == {0.0, 1.0}


def test_contact_channel_follows_the_instance_channel_layout_and_3d_input():
    labels = _two_squares(gap=0)
    joint = _JointGeodesicLabelTransform(contact=True)(labels)
    assert joint.shape == (6, 40, 60)
    np.testing.assert_array_equal(joint[0] > 0, labels > 0)
    np.testing.assert_array_equal(joint[5] > 0, touching_boundaries(labels))

    volume = np.stack([labels, labels])
    target = GeodesicHybridDistanceTransform(contact=True)(volume)
    assert target.shape == (5, 2, 40, 60)
    np.testing.assert_array_equal(target[4] > 0, touching_boundaries(volume))
