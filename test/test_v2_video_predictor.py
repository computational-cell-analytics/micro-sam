import contextlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from micro_sam.v2.models._video_predictor import _volume_geometry


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


def test_lazy_volume_is_not_materialized():
    volume = LazyVolume(np.random.rand(6, 32, 48).astype("float32"))

    num_frames, video_size = _volume_geometry(volume)

    assert num_frames == 6
    assert video_size == 48
    assert volume.reads == []  # only the shape is needed, never the data


def test_volume_geometry_rejects_a_non_3d_shape():
    volume = LazyVolume(np.zeros((4, 4), dtype="float32"))

    with pytest.raises(ValueError, match="3D volume"):
        _volume_geometry(volume)


def run_single_frame_inference(monkeypatch, inference_state, autocasts, calls=1):
    """Call the override with the wrapped inference and both CUDA waits recorded instead of run.

    The precision is forced rather than read from the host, so the wait and the dtype restore are
    tested on a runner without a GPU too. Which hardware selects which precision is covered separately
    by the '_autocasts' tests below.
    """
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    from micro_sam.v2.models._video_predictor import CustomVideoPredictor

    monkeypatch.setattr(CustomVideoPredictor, "_autocasts", lambda self, state: autocasts)
    monkeypatch.setattr(CustomVideoPredictor, "_autocast", lambda self, state: contextlib.nullcontext())
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
    for _ in range(calls):
        out = predictor._run_single_frame_inference(inference_state)
        assert out[1] == "masks"
    return events


def test_single_frame_inference_does_not_wait_after_each_offload(monkeypatch):
    """Returning on-device masks does not require waiting for the offloaded state.

    SAM2 calls '_run_single_frame_inference' once per object per frame. A host wait there stops the
    next object from being queued while the previous transfer finishes.
    """
    events = run_single_frame_inference(
        monkeypatch, {"offload_state_to_cpu": True, "device": "cuda:0"}, autocasts=True, calls=4
    )

    assert events == []


def test_propagation_waits_before_batching_offloaded_memory(monkeypatch):
    """The host concatenation must not read an unfinished device-to-host copy."""
    from micro_sam.v2.models._video_predictor import CustomVideoPredictor

    events = []
    predictor = CustomVideoPredictor.__new__(CustomVideoPredictor)
    predictor.clear_non_cond_mem_around_input = False
    monkeypatch.setattr(predictor, "propagate_in_video_preflight", lambda state: None)
    monkeypatch.setattr(predictor, "_get_obj_num", lambda state: 1)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: SimpleNamespace(synchronize=lambda: events.append("wait"))
    )
    monkeypatch.setattr(
        predictor, "_track_frame_batch",
        lambda state, group, frame_idx, reverse: events.append("batch") or torch.zeros(1, 1, 2, 2),
    )
    monkeypatch.setattr(predictor, "_get_orig_video_res_output", lambda state, masks: (masks, masks))
    inference_state = {
        "num_frames": 2,
        "obj_ids": [1],
        "device": "cuda:0",
        "offload_state_to_cpu": True,
        "output_dict_per_obj": {
            0: {"cond_frame_outputs": {0: frame_output(1.0)}, "non_cond_frame_outputs": {}}
        },
        "frames_tracked_per_obj": {0: {}},
    }

    outputs = list(predictor.propagate_in_video(inference_state, start_frame_idx=1, max_frame_num_to_track=1))

    assert len(outputs) == 1
    assert events == ["wait", "batch"]


def test_state_kept_on_the_device_is_not_awaited(monkeypatch):
    events = run_single_frame_inference(
        monkeypatch, {"offload_state_to_cpu": False, "device": "cuda:0"}, autocasts=True
    )

    assert events == []


def test_without_autocast_the_memory_dtype_is_restored_after_the_wait(monkeypatch):
    """The CPU runs in fp32, where SAM2's bfloat16 mask memory has to be cast back before it is used."""
    events = run_single_frame_inference(
        monkeypatch, {"offload_state_to_cpu": True, "device": "cpu"}, autocasts=False
    )

    # No CUDA wait on the CPU, and the restore reads the memory on the host.
    assert events == [f"cast to {torch.float32}"]


def test_the_dtype_restore_waits_for_the_offloaded_memory(monkeypatch):
    """A pre-Ampere GPU offloading its state is the one path that reads the memory on the host."""
    events = run_single_frame_inference(
        monkeypatch, {"offload_state_to_cpu": True, "device": "cuda:0"}, autocasts=False
    )

    # The wait covers the current stream only, so a replica on another stream keeps running.
    assert events == ["stream", f"cast to {torch.float32}"]


def test_consolidation_waits_for_the_offloaded_masks(monkeypatch):
    """Consolidation copies every object's offloaded 'pred_masks' into one host buffer."""
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    from micro_sam.v2.models._video_predictor import CustomVideoPredictor

    events = []
    monkeypatch.setattr(
        SAM2VideoPredictor, "_consolidate_temp_output_across_obj",
        lambda self, state, *args, **kwargs: events.append("consolidate") or "consolidated",
    )
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: SimpleNamespace(synchronize=lambda: events.append("stream"))
    )

    predictor = CustomVideoPredictor.__new__(CustomVideoPredictor)
    out = predictor._consolidate_temp_output_across_obj({"offload_state_to_cpu": True, "device": "cuda:0"}, 0)

    assert out == "consolidated"
    assert events == ["stream", "consolidate"]


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
    monkeypatch.setattr(
        CustomVideoPredictor, "_autocasts", lambda self, state: state["device"] != "cpu"
    )
    monkeypatch.setattr(CustomVideoPredictor, "_autocast", lambda self, state: contextlib.nullcontext())
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


def frame_output(value, n_objects=1, mem_dim=2, size=2):
    """A frame's stored output, shaped as SAM2 stores it."""
    return {
        "maskmem_features": torch.full((n_objects, mem_dim, size, size), float(value)),
        "maskmem_pos_enc": [torch.full((n_objects, mem_dim, size, size), 0.5)],
        "pred_masks": torch.full((n_objects, 1, size, size), float(value)),
        "obj_ptr": torch.full((n_objects, mem_dim), float(value)),
        "object_score_logits": torch.full((n_objects, 1), float(value)),
    }


def test_batched_memory_concatenates_only_the_frames_that_are_read():
    from micro_sam.v2.models._video_predictor import _BatchedMemory

    per_object = [{0: frame_output(1.0), 3: frame_output(2.0)}, {0: frame_output(3.0), 3: frame_output(4.0)}]
    memory = _BatchedMemory(per_object)

    assert len(memory) == 2
    assert set(memory) == {0, 3}
    assert memory.get(7) is None

    batched = memory[0]
    assert batched["maskmem_features"].shape[0] == 2
    assert batched["maskmem_features"][0].flatten()[0] == 1.0
    assert batched["maskmem_features"][1].flatten()[0] == 3.0
    # Shared across objects, so it only carries the batch axis of the group.
    assert batched["maskmem_pos_enc"][0].shape[0] == 2
    # The unread frame was never concatenated.
    assert list(memory._batched) == [0]


def test_a_batched_frame_output_splits_back_into_its_objects():
    from micro_sam.v2.models._video_predictor import _batch_frame_outputs, _slice_frame_output

    entries = [frame_output(1.0), frame_output(2.0), frame_output(3.0)]
    batched = _batch_frame_outputs(entries)

    for index, entry in enumerate(entries):
        restored = _slice_frame_output(batched, index)
        for key in ("maskmem_features", "pred_masks", "obj_ptr", "object_score_logits"):
            assert torch.equal(restored[key], entry[key])
        assert torch.equal(restored["maskmem_pos_enc"][0], entry["maskmem_pos_enc"][0])


def test_absent_memory_features_survive_the_batching():
    from micro_sam.v2.models._video_predictor import _batch_frame_outputs, _slice_frame_output

    entries = [frame_output(1.0), frame_output(2.0)]
    for entry in entries:
        entry["maskmem_features"] = None
        entry["maskmem_pos_enc"] = None

    batched = _batch_frame_outputs(entries)
    assert batched["maskmem_features"] is None
    assert batched["maskmem_pos_enc"] is None
    assert _slice_frame_output(batched, 1)["maskmem_features"] is None


def test_objects_share_a_batch_only_when_they_read_the_same_frames():
    from micro_sam.v2.models._video_predictor import CustomVideoPredictor

    predictor = CustomVideoPredictor.__new__(CustomVideoPredictor)
    # The first two are prompted on slice 0 and tracked equally far; the third was prompted on slice 5.
    inference_state = {"output_dict_per_obj": {
        0: {"cond_frame_outputs": {0: None}, "non_cond_frame_outputs": {1: None}},
        1: {"cond_frame_outputs": {0: None}, "non_cond_frame_outputs": {1: None}},
        2: {"cond_frame_outputs": {5: None}, "non_cond_frame_outputs": {1: None}},
    }}

    groups = predictor._memory_groups(inference_state, [0, 1, 2])

    assert sorted(sorted(group) for group in groups) == [[0, 1], [2]]


def test_skipping_the_prompt_output_consolidates_nothing(monkeypatch):
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    from micro_sam.v2.models._video_predictor import CustomVideoPredictor

    events = []
    monkeypatch.setattr(
        SAM2VideoPredictor, "_consolidate_temp_output_across_obj",
        lambda self, state, *args, **kwargs: events.append("consolidate") or "consolidated",
    )
    predictor = CustomVideoPredictor.__new__(CustomVideoPredictor)
    state = {"offload_state_to_cpu": False, "device": "cpu"}

    with predictor.skip_prompt_output():
        out = predictor._consolidate_temp_output_across_obj(state, 0)
        assert out["pred_masks_video_res"] is None
        # Nothing to resize, so the masks pass straight through.
        assert predictor._get_orig_video_res_output(state, out["pred_masks_video_res"]) == (None, None)
    assert events == []

    # The flag is per use, so the next caller gets its masks.
    assert predictor._consolidate_temp_output_across_obj(state, 0) == "consolidated"
    assert events == ["consolidate"]


def test_the_cache_is_sized_to_the_volume_when_it_fits(monkeypatch):
    from micro_sam.v2.models._video_predictor import _cache_capacity

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (8 * 10**9, 16 * 10**9))
    # A quarter of 8 GB holds 20 slices of 100 MB, so a 12 slice volume is covered entirely.
    assert _cache_capacity("cuda:0", 100 * 10**6, 12) == 12
    # A 40 slice volume is not, and the fixed cap is kept rather than a useless part of it.
    assert _cache_capacity("cuda:0", 100 * 10**6, 40) == 20


def test_the_cache_falls_back_to_the_fixed_cap(monkeypatch):
    from micro_sam.v2.models._video_predictor import _cache_capacity, MAX_CACHED_FRAMES

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1 * 10**9, 16 * 10**9))
    # Nothing like a slice fits, so it never goes below what it cached before.
    assert _cache_capacity("cuda:0", 10 * 10**9, 40) == MAX_CACHED_FRAMES
    # Off the accelerator there is nothing to size against.
    assert _cache_capacity("cpu", 100 * 10**6, 40) == MAX_CACHED_FRAMES
    assert _cache_capacity("cuda:0", None, 40) == MAX_CACHED_FRAMES
