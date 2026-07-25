import numpy as np
import pytest

from micro_sam.v2.models._video_predictor import _load_video_frames_from_images


class LazyVolume:
    """Stand-in for a dask / zarr / h5py volume: slicing works, materializing the whole array does not."""

    def __init__(self, array):
        self._array = array
        self.shape = array.shape
        self.reads = []

    def __array__(self, *args, **kwargs):
        raise AssertionError("The whole volume was materialized.")

    def __getitem__(self, index):
        self.reads.append(index)
        return self._array[index]


def load_lazy_frames(volume, image_size=64):
    return _load_video_frames_from_images(
        video_path=None, volume=volume, image_size=image_size, offload_video_to_cpu=False,
    )


def test_lazy_volume_is_not_materialized_on_init():
    volume = LazyVolume(np.random.rand(6, 32, 48).astype("float32"))

    images, video_height, video_width = load_lazy_frames(volume)

    assert len(images) == 6
    assert video_height == video_width == 48
    assert volume.reads == []  # setting up the frame sequence must not read any data


def test_only_the_requested_slice_is_read():
    volume = LazyVolume(np.random.rand(6, 32, 48).astype("float32"))
    images, _, _ = load_lazy_frames(volume)

    frame = images[3]

    assert volume.reads == [3]
    assert tuple(frame.shape) == (3, 64, 64)


def test_lazy_volume_rejects_a_non_3d_shape():
    volume = LazyVolume(np.zeros((4, 4), dtype="float32"))

    with pytest.raises(ValueError, match="3D volume"):
        load_lazy_frames(volume)


def test_numpy_and_lazy_volumes_give_the_same_frame():
    array = np.random.rand(4, 16, 16).astype("float32")

    eager, _, _ = load_lazy_frames(array)
    lazy, _, _ = load_lazy_frames(LazyVolume(array))

    np.testing.assert_allclose(eager[2].numpy(), lazy[2].numpy())
