"""The data signature of an embedding cache must be computed without materializing the input."""

import hashlib

import numpy as np
import pytest

from micro_sam import util
from micro_sam.util import _compute_data_signature


class LazyArray:
    """Stand-in for a dask / zarr / h5py array: slicing works, materializing the whole array does not."""

    def __init__(self, array):
        self._array = array
        self.shape = array.shape
        self.dtype = array.dtype
        self.reads = []

    def __array__(self, *args, **kwargs):
        raise AssertionError("The whole array was materialized.")

    def __getitem__(self, index):
        self.reads.append(index)
        return self._array[index]


@pytest.mark.parametrize("shape", [(64,), (16, 24), (6, 32, 48), (3, 4, 8, 8)])
def test_signature_matches_the_hash_of_the_whole_array(shape):
    """Hashing in blocks must not change the digest, so existing embedding caches stay valid."""
    array = np.random.rand(*shape).astype("float32")

    assert _compute_data_signature(array) == hashlib.sha1(array.tobytes()).hexdigest()


def test_signature_of_a_non_contiguous_array_is_unchanged():
    array = np.random.rand(8, 16, 16).astype("float32")[:, ::2]

    assert _compute_data_signature(array) == hashlib.sha1(np.asarray(array).tobytes()).hexdigest()


def test_a_lazy_volume_is_not_materialized():
    array = np.random.rand(6, 32, 48).astype("float32")

    assert _compute_data_signature(LazyArray(array)) == _compute_data_signature(array)


def test_a_volume_larger_than_the_block_size_is_read_in_parts(monkeypatch):
    array = np.random.rand(6, 8, 8).astype("float32")
    slice_size = array[0].nbytes
    monkeypatch.setattr(util, "DATA_SIGNATURE_BLOCK_SIZE", 2 * slice_size)
    volume = LazyArray(array)

    assert _compute_data_signature(volume) == _compute_data_signature(array)
    assert volume.reads == [slice(0, 2), slice(2, 4), slice(4, 6)]


def test_a_small_volume_is_read_in_one_block():
    volume = LazyArray(np.random.rand(4, 8, 8).astype("float32"))

    _compute_data_signature(volume)

    assert len(volume.reads) == 1
