from collections import OrderedDict

import numpy as np
import pytest

from micro_sam.sam_annotator.util import (
    collect_mask_inputs,
    iter_mask_input_masks,
    merge_labels_for_direct_commit,
    merge_labels_for_refined_commit,
    merge_refined_mask_candidates,
)


class Layer:
    """Small layer-like object for testing the pure mask-input helpers."""

    def __init__(self, data, name, scale=None, translate=None, rgb=False):
        self.data = np.asarray(data)
        self.name = name
        self.rgb = rgb
        ndim = self.data.ndim - int(rgb)
        self.scale = np.ones(ndim) if scale is None else np.asarray(scale)
        self.translate = np.zeros(ndim) if translate is None else np.asarray(translate)

    def data_to_world(self, coordinate):
        return np.asarray(coordinate) * self.scale + self.translate


class AffineLayer(Layer):
    def __init__(self, data, name, affine, rgb=False):
        super().__init__(data, name, rgb=rgb)
        self.affine = np.asarray(affine)

    def data_to_world(self, coordinate):
        homogeneous = np.concatenate([np.asarray(coordinate), [1.0]])
        return (self.affine @ homogeneous)[:-1]


def test_collect_2d_inputs_unions_equal_ids_and_preserves_sparse_ids():
    image = Layer(np.zeros((5, 6)), "image")
    first = np.zeros((5, 6), dtype="uint16")
    second = np.zeros((5, 6), dtype="uint64")
    first[1:3, 1:3] = 3
    second[2:4, 2:4] = 3
    second[0, 5] = 91

    inputs = collect_mask_inputs([Layer(first, "first"), Layer(second, "second")], image)

    assert inputs.labels.dtype == np.uint32
    assert inputs.object_ids == (3, 91)
    assert inputs.source_names == ("first", "second")
    assert inputs.cropped_source_names == ()
    assert [layer.name for layer in inputs.source_layers] == ["first", "second"]
    assert inputs.occupied_slices == {}
    assert np.array_equal(inputs.labels == 3, (first == 3) | (second == 3))

    masks = list(iter_mask_input_masks(inputs))
    assert [(object_id, z, int(mask.sum())) for object_id, z, mask in masks] == [
        (3, None, 7),
        (91, None, 1),
    ]


def test_collect_boolean_input_maps_foreground_to_id_one():
    image = Layer(np.zeros((3, 4)), "image")
    mask = np.zeros((3, 4), dtype=bool)
    mask[1, 2] = True

    inputs = collect_mask_inputs([Layer(mask, "boolean")], image)

    assert inputs.object_ids == (1,)
    assert inputs.labels.dtype == np.uint32
    assert inputs.labels[1, 2] == 1


def test_collect_3d_inputs_and_occupied_slices():
    image = Layer(np.zeros((4, 5, 6)), "volume", scale=(2, 1, 1), translate=(4, 0, 0))
    labels = np.zeros((4, 5, 6), dtype="uint8")
    labels[0, 1, 1] = 7
    labels[2, 1:3, 1:3] = 7
    labels[1, 4, 5] = 42
    layer = Layer(labels, "cells", scale=(2, 1, 1), translate=(4, 0, 0))

    inputs = collect_mask_inputs([layer], image)

    assert inputs.object_ids == (7, 42)
    assert inputs.occupied_slices == {7: (0, 2), 42: (1,)}
    yielded = list(iter_mask_input_masks(inputs))
    assert [(object_id, z, int(mask.sum())) for object_id, z, mask in yielded] == [
        (7, 0, 1),
        (7, 2, 4),
        (42, 1, 1),
    ]


def test_empty_layers_allowed_but_global_foreground_required():
    image = Layer(np.zeros((3, 3)), "image")
    empty = Layer(np.zeros((3, 3), dtype="uint8"), "empty")
    foreground = np.zeros((3, 3), dtype="uint8")
    foreground[1, 1] = 8

    inputs = collect_mask_inputs([empty, Layer(foreground, "foreground")], image)
    assert inputs.object_ids == (8,)

    with pytest.raises(ValueError, match="do not contain any nonzero"):
        collect_mask_inputs([empty], image)


def test_collect_requires_at_least_one_layer():
    with pytest.raises(ValueError, match="at least one"):
        collect_mask_inputs([], Layer(np.zeros((3, 3)), "image"))


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (np.zeros((2, 3), dtype="float32"), "boolean or integer"),
        (np.full((2, 3), -1, dtype="int16"), "negative label ID"),
        (np.full((2, 3), 2**32, dtype="uint64"), "exceeds the uint32 limit"),
    ],
)
def test_collect_rejects_invalid_label_dtype_or_range(data, message):
    with pytest.raises(ValueError, match=message):
        collect_mask_inputs([Layer(data, "bad")], Layer(np.zeros((2, 3)), "image"))


def test_collect_rejects_shape_and_transform_mismatches():
    image = Layer(np.zeros((3, 4)), "image", scale=(2, 3), translate=(4, 5))

    with pytest.raises(ValueError, match="has shape.*not resampled"):
        collect_mask_inputs([Layer(np.ones((4, 3), dtype="uint8"), "wrong shape")], image)

    shifted = Layer(np.ones((3, 4), dtype="uint8"), "shifted", scale=(2, 3), translate=(4, 6))
    with pytest.raises(ValueError, match="data-to-world transforms differ"):
        collect_mask_inputs([shifted], image)


@pytest.mark.parametrize(
    ("image_shape", "labels_shape"),
    [
        ((3, 4), (4, 5)),
        ((2, 3, 4), (3, 4, 5)),
        ((2, 3, 4), (2, 4, 5)),
    ],
)
def test_collect_crops_one_trailing_pixel_per_oversized_axis(image_shape, labels_shape):
    image = Layer(np.zeros(image_shape), "image")
    labels = np.zeros(labels_shape, dtype="uint16")
    inside = tuple(slice(0, size) for size in image_shape)
    labels[inside] = 17

    # Foreground in the one-pixel trailing border is outside the selected image and must not
    # become part of the prompt.
    trailing_border_coordinate = tuple(
        image_size if label_size == image_size + 1 else 0
        for image_size, label_size in zip(image_shape, labels_shape)
    )
    labels[trailing_border_coordinate] = 91
    inputs = collect_mask_inputs([Layer(labels, "napari labels")], image)

    assert inputs.labels.shape == image_shape
    assert inputs.object_ids == (17,)
    assert inputs.cropped_source_names == ("napari labels",)
    assert np.all(inputs.labels == 17)


@pytest.mark.parametrize("labels_shape", [(2, 4), (3, 6), (5, 4)])
def test_collect_rejects_smaller_or_more_than_one_pixel_larger_shapes(labels_shape):
    image = Layer(np.zeros((3, 4)), "image")
    labels = Layer(np.ones(labels_shape, dtype="uint8"), "wrong shape")

    with pytest.raises(ValueError, match="has shape"):
        collect_mask_inputs([labels], image)


def test_collect_compares_full_affine_transform_and_rgb_spatial_shape():
    affine = np.array([[2.0, 0.3, 4.0], [0.2, 3.0, 5.0], [0.0, 0.0, 1.0]])
    image = AffineLayer(np.zeros((3, 4, 3)), "rgb", affine, rgb=True)
    labels = AffineLayer(np.ones((3, 4), dtype="uint8"), "labels", affine)
    assert collect_mask_inputs([labels], image).labels.shape == (3, 4)

    different = affine.copy()
    different[0, 1] += 0.1
    with pytest.raises(ValueError, match="data-to-world transforms differ"):
        collect_mask_inputs(
            [AffineLayer(np.ones((3, 4), dtype="uint8"), "sheared", different)],
            image,
        )


def test_collect_conflict_error_has_layers_ids_count_and_coordinate():
    image = Layer(np.zeros((4, 4)), "image")
    first = np.zeros((4, 4), dtype="uint8")
    second = np.zeros((4, 4), dtype="uint8")
    first[1:3, 1:3] = 7
    second[2:4, 2:4] = 19

    with pytest.raises(ValueError) as error:
        collect_mask_inputs([Layer(first, "cellpose"), Layer(second, "manual")], image)

    message = str(error.value)
    assert "'cellpose' (ID 7)" in message
    assert "'manual' (ID 19)" in message
    assert "1 pixels" in message
    assert "(2, 2)" in message


def test_deterministic_candidate_merge_prefers_owner_then_smallest_id():
    original = np.zeros((3, 4), dtype="uint32")
    original[1, 1] = 9
    original[1, 2] = 4
    candidate_4 = np.zeros_like(original, dtype=bool)
    candidate_9 = np.zeros_like(original, dtype=bool)
    candidate_4[1, 0:4] = True
    candidate_9[1, 0:4] = True

    forward = merge_refined_mask_candidates(
        original, OrderedDict([(4, candidate_4), (9, candidate_9)])
    )
    reverse = merge_refined_mask_candidates(
        original, OrderedDict([(9, candidate_9), (4, candidate_4)])
    )

    assert np.array_equal(forward, reverse)
    # No source owner at the outer pixels: smallest candidate ID wins.
    assert np.array_equal(forward[1], [4, 9, 4, 4])
    assert set(np.unique(forward)) == {0, 4, 9}


def test_candidate_merge_rejects_invalid_id_and_shape():
    original = np.zeros((2, 3), dtype="uint8")
    with pytest.raises(ValueError, match="valid uint32 foreground"):
        merge_refined_mask_candidates(original, {0: np.ones_like(original)})
    with pytest.raises(ValueError, match="has shape"):
        merge_refined_mask_candidates(original, {1: np.ones((3, 2), dtype=bool)})


def test_direct_commit_merges_background_and_matching_ids_without_mutation():
    destination = np.zeros((4, 5), dtype="uint32")
    destination[1:3, 1:3] = 7
    incoming = np.zeros_like(destination)
    incoming[2:4, 2:4] = 7
    incoming[0, 4] = 31
    destination_before = destination.copy()
    incoming_before = incoming.copy()

    result = merge_labels_for_direct_commit(destination, incoming)

    assert np.array_equal(destination, destination_before)
    assert np.array_equal(incoming, incoming_before)
    assert np.array_equal(result == 7, (destination == 7) | (incoming == 7))
    assert result[0, 4] == 31
    assert result.dtype == np.uint32


def test_direct_commit_conflict_is_atomic_and_actionable():
    destination = np.zeros((3, 3), dtype="uint16")
    incoming = np.zeros((3, 3), dtype="uint16")
    destination[1, 1] = 7
    incoming[1, 1] = 11
    destination_before = destination.copy()
    incoming_before = incoming.copy()

    with pytest.raises(ValueError, match=r"Incoming ID 11.*committed ID 7.*\(1, 1\)"):
        merge_labels_for_direct_commit(destination, incoming)

    assert np.array_equal(destination, destination_before)
    assert np.array_equal(incoming, incoming_before)


def test_refined_commit_replaces_objects_inward_and_outward():
    destination = np.zeros((6, 7), dtype="uint32")
    destination[1:4, 1:4] = 7
    destination[0, 6] = 99
    refined = np.zeros_like(destination)
    # Shrink ID 7 on one edge and expand it on another.
    refined[2:5, 2:5] = 7

    result = merge_labels_for_refined_commit(destination, refined, [7])

    assert result[1, 1] == 0
    assert result[4, 4] == 7
    assert result[0, 6] == 99
    assert np.array_equal(destination[1:4, 1:4], np.full((3, 3), 7))


def test_refined_commit_supports_sparse_ids_and_rejects_id_mismatch():
    destination = np.zeros((3, 4), dtype="uint32")
    refined = np.zeros_like(destination)
    refined[0, 0] = 4
    refined[2, 3] = 81

    result = merge_labels_for_refined_commit(destination, refined, [81, 4])
    assert set(np.unique(result)) == {0, 4, 81}

    refined_missing = refined.copy()
    refined_missing[2, 3] = 0
    with pytest.raises(ValueError, match=r"Missing IDs: \[81\]"):
        merge_labels_for_refined_commit(destination, refined_missing, [4, 81])

    refined_extra = refined.copy()
    refined_extra[1, 1] = 12
    with pytest.raises(ValueError, match=r"unexpected IDs: \[12\]"):
        merge_labels_for_refined_commit(destination, refined_extra, [4, 81])


def test_refined_commit_conflict_is_atomic():
    destination = np.zeros((4, 4), dtype="uint32")
    destination[0:2, 0:2] = 7
    destination[2, 2] = 19
    refined = np.zeros_like(destination)
    refined[1:3, 1:3] = 7
    destination_before = destination.copy()
    refined_before = refined.copy()

    with pytest.raises(ValueError, match=r"Refined ID 7.*committed ID 19.*\(2, 2\)"):
        merge_labels_for_refined_commit(destination, refined, [7])

    assert np.array_equal(destination, destination_before)
    assert np.array_equal(refined, refined_before)
