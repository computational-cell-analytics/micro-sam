from types import SimpleNamespace

import numpy as np
import pytest
import torch

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


def run_single_frame_inference(monkeypatch, inference_state):
    """Call the override with the wrapped inference and both CUDA waits recorded instead of run."""
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    from micro_sam.v2.models._video_predictor import CustomVideoPredictor

    events = []

    class Memory:
        """Stands in for the offloaded 'maskmem_features', recording when it is read on the host."""

        def to(self, dtype):
            events.append(f"cast to {dtype}")
            return self

    monkeypatch.setattr(
        SAM2VideoPredictor, "_run_single_frame_inference",
        lambda self, state, **kwargs: ({"maskmem_features": Memory()}, "masks"),
    )
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: SimpleNamespace(synchronize=lambda: events.append("stream"))
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: events.append("device"))

    predictor = CustomVideoPredictor.__new__(CustomVideoPredictor)
    predictor.parameters = lambda: iter([torch.zeros(1, dtype=torch.float32)])
    out = predictor._run_single_frame_inference(inference_state)
    assert out[1] == "masks"
    return events


def test_offloaded_state_is_awaited_before_it_is_read(monkeypatch):
    """The non-blocking copy to the CPU records no event, so the host readers need an explicit wait."""
    events = run_single_frame_inference(monkeypatch, {"offload_state_to_cpu": True, "device": "cuda:0"})

    # The wait covers the current stream only, so a replica on another stream keeps running. Autocast
    # keeps the memory in the dtype SAM2 stores it in, so nothing is cast back on this path.
    assert events == ["stream"]


def test_state_kept_on_the_device_is_not_awaited(monkeypatch):
    events = run_single_frame_inference(monkeypatch, {"offload_state_to_cpu": False, "device": "cuda:0"})

    assert events == []


def test_without_autocast_the_memory_dtype_is_restored_after_the_wait(monkeypatch):
    """The CPU runs in fp32, where SAM2's bfloat16 mask memory has to be cast back before it is used."""
    events = run_single_frame_inference(monkeypatch, {"offload_state_to_cpu": True, "device": "cpu"})

    # No CUDA wait on the CPU, and the restore reads the memory on the host.
    assert events == [f"cast to {torch.float32}"]


def test_autocast_is_used_on_cuda_only(monkeypatch):
    from micro_sam.v2.models._video_predictor import CustomVideoPredictor

    predictor = CustomVideoPredictor.__new__(CustomVideoPredictor)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(major=8))
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert predictor._autocasts({"device": "cuda:0"})
    assert not predictor._autocasts({"device": "cpu"})
    assert not predictor._autocasts({"device": "mps"})  # no MPS hardware here


def test_a_pre_ampere_gpu_keeps_fp32(monkeypatch):
    """Emulated bfloat16 is slower than the fp32 those GPUs run today, so they keep fp32 - on the GPU."""
    from micro_sam.v2.models._video_predictor import CustomVideoPredictor

    predictor = CustomVideoPredictor.__new__(CustomVideoPredictor)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(major=7))

    assert not predictor._autocasts({"device": "cuda:0"})


def test_a_cuda_request_without_cuda_keeps_fp32(monkeypatch):
    from micro_sam.v2.models._video_predictor import CustomVideoPredictor

    predictor = CustomVideoPredictor.__new__(CustomVideoPredictor)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert not predictor._autocasts({"device": "cuda:0"})


@pytest.mark.parametrize("macos_14, expected", [(True, True), (False, False)])
def test_mps_follows_the_macos_version_torch_gates_on(monkeypatch, macos_14, expected):
    """Below macOS 14 torch disables the autocast itself, so claiming one would skip the dtype restore."""
    from micro_sam.v2.models._video_predictor import CustomVideoPredictor

    predictor = CustomVideoPredictor.__new__(CustomVideoPredictor)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_macos_or_newer", lambda major, minor: macos_14)

    assert predictor._autocasts({"device": "mps"}) is expected


def test_memory_encoder_restores_the_model_dtype_without_autocast(monkeypatch):
    """Off CUDA the model runs in fp32, so SAM2's bfloat16 downcast has to be undone."""
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    from micro_sam.v2.models._video_predictor import CustomVideoPredictor

    features = torch.zeros(4, dtype=torch.bfloat16)
    monkeypatch.setattr(
        SAM2VideoPredictor, "_run_memory_encoder", lambda self, state, *args, **kwargs: (features, "pos_enc")
    )
    predictor = CustomVideoPredictor.__new__(CustomVideoPredictor)
    predictor.parameters = lambda: iter([torch.zeros(1, dtype=torch.float32)])

    restored, pos_enc = predictor._run_memory_encoder({"offload_state_to_cpu": False, "device": "cpu"})

    assert restored.dtype == torch.float32 and pos_enc == "pos_enc"

    # On CUDA autocast makes SAM2's own bfloat16 storage self-consistent, so it is left alone.
    kept, _ = predictor._run_memory_encoder({"offload_state_to_cpu": False, "device": "cuda:0"})
    assert kept.dtype == torch.bfloat16


def test_missing_memory_features_are_passed_through(monkeypatch):
    from micro_sam.v2.models._video_predictor import CustomVideoPredictor

    predictor = CustomVideoPredictor.__new__(CustomVideoPredictor)

    assert predictor._restore_memory_dtype(None) is None
