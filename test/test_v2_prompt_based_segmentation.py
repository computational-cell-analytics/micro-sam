import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from bioimage_cpp.utils import Blocking

from micro_sam.v2.prompt_based_segmentation import (
    PromptableSegmentation3D,
    ReplicatedPromptableSegmentation3D,
    TiledPromptableSegmentation3D,
    promptable_segmentation_2d,
)


class FakeTileSegmenter:
    def add_point_prompts(self, **kwargs):
        pass

    def add_box_prompts(self, **kwargs):
        pass

    def get_progress_total(self, z_range=None):
        return 8 if z_range is None else z_range[1] - z_range[0] + 1


def test_promptable_segmentation_2d_normalizes_raw(monkeypatch):
    from micro_sam.v2.normalization import to_image

    class RecordingImagePredictor:
        device = "cpu"  # 'encode_image' reads it to pick the encoder precision.
        _features = None

        def set_image(self, image):
            self.image = image
            self._orig_hw = [image.shape[:2]]

        def predict(self, **kwargs):
            masks = np.ones((1, *self.image.shape[:2]), dtype=bool)
            return masks, np.ones(1), None

    monkeypatch.setattr("micro_sam.v2.util.configure_image_predictor", lambda predictor: predictor)
    predictor = RecordingImagePredictor()
    raw = np.arange(24, dtype="uint16").reshape(3, 8) * 1000

    promptable_segmentation_2d(
        predictor, image=raw, points=np.array([[2, 2]]), labels=np.array([1]),
    )

    assert np.array_equal(predictor.image, to_image(raw))
    assert predictor.image.dtype == np.uint8
    assert predictor.image.shape == (3, 8, 3)


class FakeImagePredictor:
    """A stand-in SAM2 image predictor that records every 'predict' call and mimics its shapes."""

    def __init__(self, shape=(16, 16)):
        self._orig_hw = [shape]
        self.model = SimpleNamespace(image_size=64)
        self.mask_threshold = 0.0
        self._transforms = None
        self.calls = []

    def predict(self, point_coords=None, point_labels=None, box=None, mask_input=None, multimask_output=False):
        self.calls.append({
            "point_coords": point_coords, "point_labels": point_labels,
            "box": box, "mask_input": mask_input,
        })
        if box is not None:
            n_objects = len(np.asarray(box))
        elif point_coords is not None:
            coords = np.asarray(point_coords)
            n_objects = coords.shape[0] if coords.ndim == 3 else 1
        else:
            n_objects = 1

        height, width = self._orig_hw[0]
        masks = np.ones((n_objects, 1, height, width), dtype="float32")
        if n_objects == 1:  # SAM2 squeezes the batch axis, so one object comes back as (C, H, W).
            masks = masks[0]
        return masks, np.ones(n_objects), np.ones((n_objects, 1, 256, 256), dtype="float32")


def box_calls_of(predictor):
    return [call for call in predictor.calls if call["box"] is not None]


@pytest.mark.parametrize("n_points", (1, 2, 3))
def test_batched_2d_rectangle_is_not_conditioned_on_the_point_prediction(n_points):
    """A rectangle has no mask prompt, so combining it with points must not feed it one."""
    predictor = FakeImagePredictor()
    points = np.array([[2 + i, 3] for i in range(n_points)])

    seg = promptable_segmentation_2d(
        predictor, points=points, labels=np.ones(n_points, dtype=int),
        boxes=[np.array([1, 1, 8, 8])], masks=[None], batched=True,
    )

    assert seg is not None
    assert seg.max() == n_points + 1  # one object per positive point, plus the box.
    box_calls = box_calls_of(predictor)
    assert len(box_calls) == 1
    assert box_calls[0]["mask_input"] is None


def test_batched_2d_points_with_multiple_rectangles():
    predictor = FakeImagePredictor()

    seg = promptable_segmentation_2d(
        predictor, points=np.array([[2, 3]]), labels=np.array([1]),
        boxes=[np.array([1, 1, 8, 8]), np.array([9, 9, 14, 14])], masks=[None, None], batched=True,
    )

    assert seg is not None
    assert seg.max() == 3  # one object for the point and one per box.
    assert all(call["mask_input"] is None for call in box_calls_of(predictor))


def test_batched_2d_mask_prompt_survives_point_prompts():
    """A polygon / ellipse cue must still reach its own box when positive points are also present."""
    predictor = FakeImagePredictor()
    mask = np.zeros((16, 16), dtype="uint8")
    mask[2:6, 2:6] = 1

    promptable_segmentation_2d(
        predictor, points=np.array([[10, 10]]), labels=np.array([1]),
        boxes=[np.array([2, 2, 6, 6])], masks=[mask], batched=True,
    )

    box_calls = box_calls_of(predictor)
    assert len(box_calls) == 1
    # The cue comes from the drawn mask (mostly background, so negative logits), not from the
    # all-foreground point prediction.
    assert box_calls[0]["mask_input"].min() < 0


def test_batched_2d_negative_points_are_shared_with_every_object():
    predictor = FakeImagePredictor()

    promptable_segmentation_2d(
        predictor, points=np.array([[2, 3], [4, 5], [12, 12]]), labels=np.array([1, 1, 0]),
        boxes=[np.array([1, 1, 8, 8])], masks=[None], batched=True,
    )

    point_call = predictor.calls[0]
    # Two objects, each the positive point followed by the shared negative one.
    assert np.asarray(point_call["point_labels"]).tolist() == [[1, 0], [1, 0]]
    assert np.asarray(box_calls_of(predictor)[0]["point_labels"]).tolist() == [[0]]


def test_points_with_multiple_boxes_are_rejected():
    """SAM2 cannot broadcast one point batch over several boxes, so this is skipped, not crashed."""
    predictor = FakeImagePredictor()

    seg = promptable_segmentation_2d(
        predictor, points=np.array([[2, 3]]), labels=np.array([1]),
        boxes=[np.array([1, 1, 8, 8]), np.array([9, 9, 14, 14])], masks=[None, None], batched=False,
    )

    assert seg is None
    assert predictor.calls == []


def make_tiled_segmenter():
    segmenter_cls = TiledPromptableSegmentation3D
    segmenter = segmenter_cls.__new__(segmenter_cls)
    segmenter.shape = (8, 16, 16)
    segmenter.halo = (0, 0)
    segmenter.tiling = Blocking([0, 0], [16, 16], [8, 8])
    segmenter._segmenters = {}

    def get_segmenter(tile_id):
        return segmenter._segmenters.setdefault(tile_id, FakeTileSegmenter())

    segmenter._get_segmenter = get_segmenter
    return segmenter


@pytest.mark.parametrize("device", ("mps", torch.device("mps"), torch.device("mps", 0)))
def test_promptable_segmentation_3d_disables_offloading_on_mps(device, monkeypatch):
    # Offloading to CPU on MPS races and gives garbage masks, so it must stay off for every way the
    # device can be spelled - in particular the indexed 'mps:0' the tiled variant resolves.
    monkeypatch.setattr("micro_sam.v2.util._get_device", lambda device=None: device)
    monkeypatch.setattr(PromptableSegmentation3D, "init_predictor", lambda self: None)
    segmenter = PromptableSegmentation3D(
        predictor=None, volume=np.zeros((4, 8, 8), dtype="uint8"), volume_embeddings=None, device=device,
    )

    assert not segmenter.offload_state_to_cpu


def test_promptable_segmentation_3d_keeps_offloading_on_cuda(monkeypatch):
    monkeypatch.setattr("micro_sam.v2.util._get_device", lambda device=None: device)
    monkeypatch.setattr(PromptableSegmentation3D, "init_predictor", lambda self: None)
    segmenter = PromptableSegmentation3D(
        predictor=None, volume=np.zeros((4, 8, 8), dtype="uint8"), volume_embeddings=None,
        device=torch.device("cuda", 0),
    )

    assert segmenter.offload_state_to_cpu


def test_promptable_segmentation_3d_progress_total():
    segmenter = PromptableSegmentation3D.__new__(PromptableSegmentation3D)
    segmenter.volume = np.zeros((8, 16, 16), dtype="uint8")
    segmenter._clear_pushed_prompts()  # normally done in '__init__', which '__new__' skips

    assert segmenter.get_progress_total() == 8
    assert segmenter.get_progress_total((2, 5)) == 4


def test_tiled_promptable_segmentation_3d_progress_total():
    segmenter = make_tiled_segmenter()

    assert segmenter.get_progress_total() == 0

    segmenter.add_point_prompts(frame_ids=0, points=[[1, 1]], point_labels=[1])
    assert segmenter.get_progress_total() == 8
    assert segmenter.get_progress_total((2, 5)) == 4

    segmenter.add_point_prompts(frame_ids=0, points=[[1, 9]], point_labels=[1])
    assert segmenter.get_progress_total() == 16
    assert segmenter.get_progress_total((2, 5)) == 8


def test_tiled_promptable_segmentation_3d_box_progress_total():
    segmenter = make_tiled_segmenter()

    segmenter.add_box_prompts(frame_ids=0, boxes=[np.array([1, 1, 7, 9])])

    assert len(segmenter._segmenters) == 2
    assert segmenter.get_progress_total() == 16


class RecordingPredictor:
    """A stand-in SAM2 predictor that records every 'add_new_points_or_box' call."""

    def __init__(self):
        self.image_size = 32
        self.calls = []
        self.mask_calls = []
        self.active_objects = set()

    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, clear_old_points=False,
                              points=None, labels=None, box=None):
        self.active_objects.add(obj_id)
        self.calls.append({
            "frame_idx": frame_idx, "obj_id": obj_id, "clear_old_points": clear_old_points,
            "points": None if points is None else np.asarray(points).tolist(),
            "labels": None if labels is None else np.asarray(labels).tolist(),
            "box": None if box is None else np.asarray(box).tolist(),
        })
        return None, [obj_id], torch.zeros((1, 1, 32, 32))

    def add_new_mask(self, inference_state, frame_idx, obj_id, mask):
        self.active_objects.add(obj_id)
        self.mask_calls.append({"frame_idx": frame_idx, "obj_id": obj_id, "mask": mask})

    def reset_state(self, inference_state):
        self.active_objects.clear()


def make_recording_segmenter():
    segmenter = PromptableSegmentation3D.__new__(PromptableSegmentation3D)
    segmenter.predictor = RecordingPredictor()
    segmenter.volume = np.zeros((8, 32, 32), dtype="uint8")
    segmenter.inference_state = {}
    segmenter._pushed_points = {}
    segmenter._pushed_boxes = {}
    segmenter._pushed_masks = {}
    segmenter._prompt_history = []
    segmenter._prompt_signatures = set()
    return segmenter


def test_add_multiple_point_prompts_in_one_call():
    segmenter = make_recording_segmenter()
    segmenter.add_point_prompts(frame_ids=3, points=np.array([[10, 12], [20, 22]]), point_labels=np.array([1, 0]))

    calls = segmenter.predictor.calls
    assert len(calls) == 2  # both points added, not truncated to one
    assert all(c["frame_idx"] == 3 and not c["clear_old_points"] for c in calls)
    # points are pushed to SAM2 in (x, y) order.
    assert calls[0]["points"] == [[12, 10]] and calls[0]["labels"] == [1]
    assert calls[1]["points"] == [[22, 20]] and calls[1]["labels"] == [0]


def test_point_prompts_are_deduped_across_runs():
    segmenter = make_recording_segmenter()
    segmenter.add_point_prompts(frame_ids=3, points=np.array([[10, 12]]), point_labels=np.array([1]))
    # re-run with the same point plus a new one: only the new point is pushed.
    segmenter.add_point_prompts(frame_ids=3, points=np.array([[10, 12], [20, 22]]), point_labels=np.array([1, 1]))

    assert len(segmenter.predictor.calls) == 2
    assert segmenter.predictor.calls[1]["points"] == [[22, 20]]


def test_point_prompts_batched_object_ids():
    segmenter = make_recording_segmenter()
    segmenter.add_point_prompts(
        frame_ids=1, points=np.array([[1, 2], [3, 4]]), point_labels=np.array([1, 1]), object_id=[5, 6],
    )
    assert [c["obj_id"] for c in segmenter.predictor.calls] == [5, 6]


def test_box_prompt_clears_and_is_deduped():
    segmenter = make_recording_segmenter()
    box = np.array([2, 2, 20, 20])
    segmenter.add_box_prompts(frame_ids=3, boxes=[box])
    segmenter.add_box_prompts(frame_ids=3, boxes=[box])  # identical box: no-op

    calls = segmenter.predictor.calls
    assert len(calls) == 1
    assert calls[0]["box"] is not None and calls[0]["clear_old_points"] is True


def test_box_before_points_combine():
    segmenter = make_recording_segmenter()
    segmenter.add_box_prompts(frame_ids=3, boxes=[np.array([2, 2, 20, 20])])
    segmenter.add_point_prompts(frame_ids=3, points=np.array([[10, 12]]), point_labels=np.array([0]))

    calls = segmenter.predictor.calls
    assert len(calls) == 2
    assert calls[0]["box"] is not None and calls[0]["clear_old_points"] is True
    assert calls[1]["box"] is None and calls[1]["clear_old_points"] is False


def test_box_after_points_readds_points():
    segmenter = make_recording_segmenter()
    segmenter.add_point_prompts(frame_ids=3, points=np.array([[10, 12]]), point_labels=np.array([0]))
    segmenter.add_box_prompts(frame_ids=3, boxes=[np.array([2, 2, 20, 20])])

    calls = segmenter.predictor.calls
    # point, then box (clears points), then the point re-added so box and point combine.
    assert len(calls) == 3
    assert calls[1]["box"] is not None and calls[1]["clear_old_points"] is True
    assert calls[2]["box"] is None and calls[2]["clear_old_points"] is False
    assert calls[2]["points"] == [[12, 10]]


def test_reset_predictor_clears_pushed_prompts():
    segmenter = make_recording_segmenter()
    segmenter.add_point_prompts(frame_ids=3, points=np.array([[10, 12]]), point_labels=np.array([1]))
    segmenter.reset_predictor()

    assert segmenter._pushed_points == {} and segmenter._pushed_boxes == {}
    # the same point is pushed again after a reset.
    segmenter.add_point_prompts(frame_ids=3, points=np.array([[10, 12]]), point_labels=np.array([1]))
    assert len(segmenter.predictor.calls) == 2


def test_sync_prompt_state_keeps_additive_refinement_incremental():
    segmenter = make_recording_segmenter()
    first = {("point", 3, 10, 12, 1)}

    segmenter.sync_prompt_state(first)
    segmenter.add_point_prompts(frame_ids=3, points=np.array([[10, 12]]), point_labels=np.array([1]))
    segmenter.sync_prompt_state(first | {("point", 3, 20, 22, 1)})
    segmenter.add_point_prompts(
        frame_ids=3, points=np.array([[10, 12], [20, 22]]), point_labels=np.array([1, 1])
    )

    assert len(segmenter.predictor.calls) == 2  # only the added point was pushed


def test_sync_prompt_state_replays_after_a_prompt_is_deleted():
    segmenter = make_recording_segmenter()
    kept, deleted = ("point", 3, 10, 12, 1), ("point", 3, 20, 22, 1)

    segmenter.sync_prompt_state({kept, deleted})
    segmenter.add_point_prompts(
        frame_ids=3, points=np.array([[10, 12], [20, 22]]), point_labels=np.array([1, 1])
    )
    assert len(segmenter.predictor.calls) == 2

    # One point is gone, so the state is stale: it is rebuilt and the remaining point replayed.
    segmenter.sync_prompt_state({kept})
    segmenter.add_point_prompts(frame_ids=3, points=np.array([[10, 12]]), point_labels=np.array([1]))

    assert len(segmenter.predictor.calls) == 3
    assert segmenter.predictor.calls[-1]["points"] == [[12, 10]]


def test_sync_prompt_state_replays_after_a_point_is_relabelled():
    segmenter = make_recording_segmenter()

    segmenter.sync_prompt_state({("point", 3, 10, 12, 1)})
    segmenter.add_point_prompts(frame_ids=3, points=np.array([[10, 12]]), point_labels=np.array([1]))
    segmenter.sync_prompt_state({("point", 3, 10, 12, 0)})
    segmenter.add_point_prompts(frame_ids=3, points=np.array([[10, 12]]), point_labels=np.array([0]))

    assert len(segmenter.predictor.calls) == 2
    assert segmenter.predictor.calls[-1]["labels"] == [0]


def test_grouped_propagation_restores_all_prompts_before_a_deletion(monkeypatch):
    segmenter = make_recording_segmenter()
    kept = ("point", 1, 10, 12, 1)
    deleted = ("point", 5, 20, 22, 1)
    segmenter.sync_prompt_state({kept, deleted})
    segmenter.add_point_prompts(
        frame_ids=[1, 5], points=np.array([[10, 12], [20, 22]]), point_labels=np.array([1, 1]),
        object_id=[1, 2],
    )

    propagated_objects = []

    def propagate_group(*args, **kwargs):
        propagated_objects.append(set(segmenter.predictor.active_objects))
        return {}

    monkeypatch.setattr(segmenter, "_propagate_both_directions", propagate_group)
    segmenter.propagate_prompts()

    assert propagated_objects == [{1}, {2}]
    assert segmenter.predictor.active_objects == {1, 2}
    assert set(segmenter._pushed_points) == {(1, 1), (2, 5)}
    assert segmenter._prompt_signatures == {kept, deleted}

    segmenter.sync_prompt_state({kept})
    segmenter.add_point_prompts(
        frame_ids=1, points=np.array([[10, 12]]), point_labels=np.array([1]), object_id=1
    )

    assert segmenter.predictor.active_objects == {1}


def test_replaying_masks_restores_their_bookkeeping():
    segmenter = make_recording_segmenter()
    mask = np.ones((32, 32), dtype=bool)
    segmenter.add_mask_prompts(frame_ids=5, masks=[mask], object_id=2, refine=False)
    snapshot = tuple(operation.copied() for operation in segmenter._prompt_history)
    signature = next(iter(segmenter._pushed_masks[(2, 5)]))
    segmenter.predictor.mask_calls.clear()

    segmenter._replay_prompts(snapshot, {2})

    assert set(segmenter._pushed_masks) == {(2, 5)}
    assert np.array_equal(segmenter._pushed_masks[(2, 5)][signature], mask)
    assert segmenter.predictor.active_objects == {2}
    assert len(segmenter.predictor.mask_calls) == 1
    assert segmenter.predictor.mask_calls[0]["frame_idx"] == 5
    assert segmenter.predictor.mask_calls[0]["obj_id"] == 2
    assert np.array_equal(segmenter.predictor.mask_calls[0]["mask"], mask)


def test_segment_slice_clears_the_pushed_prompt_bookkeeping():
    """'segment_slice' discards the SAM2 state, so the dedup bookkeeping must not outlive it."""
    segmenter = make_recording_segmenter()
    segmenter._image_style_trafo = None
    segmenter.add_point_prompts(frame_ids=3, points=np.array([[10, 12]]), point_labels=np.array([1]))
    assert segmenter._pushed_points

    segmenter.segment_slice(frame_idx=3, points=np.array([[12, 10]]), labels=np.array([1]))
    assert segmenter._pushed_points == {}

    # The prompt is pushed again instead of being deduped against a state that no longer exists.
    segmenter.add_point_prompts(frame_ids=3, points=np.array([[10, 12]]), point_labels=np.array([1]))
    assert segmenter.predictor.calls[-1]["points"] == [[12, 10]]


def test_tracking_pattern_pushes_points_per_frame():
    # Tracking annotates one object across several frames, adding points one at a time to each frame
    # (mirrors 'propagate_track'). The (object, frame) dedup keys per frame, so the same (y, x) on
    # different frames is pushed to each frame rather than collapsed into one.
    segmenter = make_recording_segmenter()
    prompts_per_frame = {0: [[8, 12]], 2: [[8, 12]], 5: [[30, 40]]}
    for t, points in prompts_per_frame.items():
        for point in np.array(points):
            segmenter.add_point_prompts(frame_ids=int(t), points=np.array([point]), point_labels=np.array([1]))

    calls = segmenter.predictor.calls
    assert [c["frame_idx"] for c in calls] == [0, 2, 5]
    assert calls[0]["points"] == [[12, 8]] and calls[1]["points"] == [[12, 8]]  # same point, distinct frames
    assert calls[2]["points"] == [[40, 30]]


def test_tracking_reset_between_tracks_repushes_prompts():
    # 'propagate_track' resets the predictor before each track, so an identical prompt on the next
    # track is pushed again instead of being silently deduped against the previous track. The point
    # and box sit on different frames so the box does not re-add the point (see the same-frame case).
    segmenter = make_recording_segmenter()
    segmenter.add_point_prompts(frame_ids=0, points=np.array([[8, 12]]), point_labels=np.array([1]))
    segmenter.add_box_prompts(frame_ids=3, boxes=[np.array([2, 2, 20, 20])])
    assert len(segmenter.predictor.calls) == 2

    segmenter.reset_predictor()  # start of the next track
    assert segmenter._pushed_points == {} and segmenter._pushed_boxes == {}

    segmenter.add_point_prompts(frame_ids=0, points=np.array([[8, 12]]), point_labels=np.array([1]))
    segmenter.add_box_prompts(frame_ids=3, boxes=[np.array([2, 2, 20, 20])])
    assert len(segmenter.predictor.calls) == 4


@pytest.mark.slow
def test_video_predictor_correction_flags_and_propagation():
    # Guards the fork fix shared with the tracking predictor: the interactive-correction flags are
    # enabled and the per-object memory-clear helper (which the fork calls but does not define) is
    # restored, so 'clear_non_cond_mem_around_input' runs during propagation instead of raising.
    from micro_sam.v2.util import get_sam2_model, precompute_image_embeddings, DEFAULT_MODEL

    predictor = get_sam2_model(model_type=DEFAULT_MODEL, input_type="videos", device="cpu")
    assert predictor.add_all_frames_to_correct_as_cond is True
    assert predictor.clear_non_cond_mem_around_input is True
    assert callable(getattr(predictor, "_clear_obj_non_cond_mem_around_input", None))

    volume = np.zeros((5, 128, 128), dtype="float32")
    yy, xx = np.mgrid[0:128, 0:128]
    volume[:, ((yy - 64) ** 2 + (xx - 64) ** 2) < 22 ** 2] = 220.0

    embeddings = precompute_image_embeddings(predictor, volume, ndim=3, verbose=False)
    segmenter = PromptableSegmentation3D(predictor, volume, embeddings, device="cpu")
    segmenter.add_point_prompts(frame_ids=2, points=np.array([[64, 64]]), point_labels=np.array([1]))
    seg = segmenter.predict()

    # Forward+reverse propagation with the flags on still segments the object across frames.
    assert seg.shape == volume.shape
    assert seg[2].sum() > 0  # the prompted frame
    assert int((seg.reshape(seg.shape[0], -1).sum(axis=1) > 0).sum()) >= 2  # propagated to neighbors


class TrackingPropagationPredictor:
    """A stand-in predictor whose propagation yields a scripted sequence of per-frame masks.

    'occupancy' gives one entry per frame: True for a frame whose object mask is non-empty. The
    predictor records how many frames were actually pulled, which is what early stopping is meant to
    reduce - the frames it never yields are the network evaluations that were skipped.
    """

    def __init__(self, occupancy):
        self.occupancy = occupancy
        self.frames_yielded = 0

    def propagate_in_video(self, inference_state, reverse=False):
        for frame_idx, occupied in enumerate(self.occupancy):
            self.frames_yielded += 1
            logits = torch.full((1, 1, 4, 4), 1.0 if occupied else -1.0)
            yield frame_idx, [1], logits


def _propagate_with_patience(occupancy, patience):
    segmenter = PromptableSegmentation3D.__new__(PromptableSegmentation3D)
    segmenter.predictor = TrackingPropagationPredictor(occupancy)
    segmenter.inference_state = {}
    segments = segmenter._propagate_in_direction(reverse=False, early_stop_patience=patience)
    return segments, segmenter.predictor.frames_yielded


def test_early_stopping_skips_frames_past_the_end_of_every_object():
    # Object present on frames 0-3, gone from 4 onwards.
    occupancy = [True] * 4 + [False] * 6

    full, full_frames = _propagate_with_patience(occupancy, None)
    stopped, stopped_frames = _propagate_with_patience(occupancy, 2)

    # Without a patience the predictor is run on every frame of the volume.
    assert full_frames == 10
    assert sorted(full) == list(range(10))

    # With patience 2 it stops on the second consecutive empty frame, frame 5.
    assert stopped_frames == 6
    assert sorted(stopped) == [0, 1, 2, 3, 4, 5]


def test_early_stopping_is_output_preserving():
    # The reason early stopping is on by default: the frames it skips hold nothing but empty masks,
    # so every non-empty mask of the full propagation survives unchanged.
    occupancy = [True] * 4 + [False] * 6

    full, _ = _propagate_with_patience(occupancy, None)
    stopped, _ = _propagate_with_patience(occupancy, 2)

    def non_empty(segments):
        return {
            frame: {obj: mask.tolist() for obj, mask in per_object.items()}
            for frame, per_object in segments.items()
            if any(mask.any() for mask in per_object.values())
        }

    assert non_empty(stopped) == non_empty(full)
    assert set(full) - set(stopped) == {6, 7, 8, 9}


def test_early_stopping_tolerates_a_single_dropped_mask():
    # SAM2 can drop a mask for one frame and recover it, so a patience of 2 must not stop on a
    # single empty frame and truncate the rest of the object.
    occupancy = [True, True, False, True, True, False, False, True]

    segments, frames = _propagate_with_patience(occupancy, 2)

    # Frames 5 and 6 are the first consecutive pair, so frame 7 is never reached.
    assert frames == 7
    assert sorted(segments) == [0, 1, 2, 3, 4, 5, 6]
    # The object that reappeared after the isolated gap on frame 2 was kept.
    assert segments[3][1].any() and segments[4][1].any()


def test_volume_early_stop_patience_defaults_to_two():
    # Adopted in evaluation/optimization/notes/APG_3D_OPTIMIZATION.md experiment 5; the annotator already used it.
    from micro_sam.v2.automatic_prompt_generation import DEFAULT_PROMPT_GENERATION

    assert DEFAULT_PROMPT_GENERATION["early_stop_patience"] == 2


def test_add_prompt_set_pushes_a_box_and_its_points_in_one_call():
    segmenter = make_recording_segmenter()
    segmenter.add_prompt_set(
        frame_id=3, points=np.array([[10, 12], [20, 22]]), point_labels=np.array([1, 0]),
        box=np.array([4, 5, 12, 13]), object_id=2,
    )

    # One call, where the incremental methods would have made three: every push re-runs the mask
    # decoder on the conditioning frame, which is what this exists to avoid.
    calls = segmenter.predictor.calls
    assert len(calls) == 1
    assert calls[0]["frame_idx"] == 3 and calls[0]["obj_id"] == 2
    # SAM2 requires 'clear_old_points' whenever a box is given.
    assert calls[0]["clear_old_points"] is True
    # Points go to SAM2 in (x, y), the box in (x0, y0, x1, y1).
    assert calls[0]["points"] == [[12.0, 10.0], [22.0, 20.0]]
    assert calls[0]["labels"] == [1, 0]
    assert calls[0]["box"] == [[5, 4, 13, 12]]


def test_add_prompt_set_records_the_anchor_it_conditioned():
    segmenter = make_recording_segmenter()
    segmenter.add_prompt_set(
        frame_id=5, points=np.array([[10, 12]]), point_labels=np.array([1]),
        box=np.array([4, 5, 12, 13]), object_id=7,
    )
    segmenter.add_prompt_set(frame_id=2, points=np.array([[1, 1]]), point_labels=np.array([1]), object_id=7)

    # The bookkeeping the incremental methods keep, so the anchor and a replay still see these.
    assert segmenter._anchor_per_object() == {7: 2}
    assert (7, 5) in segmenter._pushed_boxes
    assert (10, 12, 1) in segmenter._pushed_points[(7, 5)]


def test_add_prompt_set_replacement_updates_active_bookkeeping():
    segmenter = make_recording_segmenter()
    first_point = np.array([[4, 6]])
    segmenter.add_prompt_set(
        frame_id=3, points=first_point, point_labels=np.array([1]),
        box=np.array([1, 2, 10, 11]), object_id=2,
    )
    segmenter.add_prompt_set(
        frame_id=3, points=np.array([[14, 16]]), point_labels=np.array([1]),
        box=np.array([8, 9, 20, 21]), object_id=2, clear_old_points=True,
    )

    key = (2, 3)
    assert segmenter._pushed_points[key] == {(14, 16, 1)}
    assert segmenter._pushed_boxes[key] == {(8, 9, 20, 21)}

    # The first point was cleared in SAM2, so it must not be deduped when it is explicitly re-added.
    segmenter.add_point_prompts(frame_ids=3, points=first_point, point_labels=np.array([1]), object_id=2)
    assert len(segmenter.predictor.calls) == 3
    assert segmenter.predictor.calls[-1]["points"] == [[6.0, 4.0]]
    assert segmenter._pushed_points[key] == {(4, 6, 1), (14, 16, 1)}
    assert len(segmenter._prompt_history) == 3


def test_add_prompt_set_rejects_box_append_without_mutation():
    segmenter = make_recording_segmenter()

    with pytest.raises(ValueError, match="clear_old_points=True"):
        segmenter.add_prompt_set(
            frame_id=3, points=np.array([[4, 6]]), point_labels=np.array([1]),
            box=np.array([1, 2, 10, 11]), object_id=2, clear_old_points=False,
        )

    assert segmenter.predictor.calls == []
    assert segmenter._pushed_points == {}
    assert segmenter._pushed_boxes == {}
    assert segmenter._prompt_history == []


def test_replay_preserves_joint_prompt_batch_and_order():
    segmenter = make_recording_segmenter()
    points = np.array([[4, 6], [8, 10], [12, 14]])
    labels = np.array([1, 0, 0])
    segmenter.add_prompt_set(
        frame_id=3, points=points, point_labels=labels,
        box=np.array([1, 2, 20, 21]), object_id=2,
    )
    snapshot = tuple(operation.copied() for operation in segmenter._prompt_history)
    original_call = dict(segmenter.predictor.calls[-1])
    segmenter.predictor.calls.clear()

    segmenter._replay_prompts(snapshot, {2})

    assert segmenter.predictor.calls == [original_call]
    assert len(segmenter._prompt_history) == 1


def test_multi_anchor_replay_uses_each_logical_call_once(monkeypatch):
    segmenter = make_recording_segmenter()
    segmenter.add_prompt_set(
        frame_id=1, points=np.array([[4, 6], [8, 10]]), point_labels=np.array([1, 0]),
        box=np.array([1, 2, 20, 21]), object_id=1,
    )
    segmenter.add_prompt_set(
        frame_id=5, points=np.array([[12, 14], [16, 18]]), point_labels=np.array([1, 0]),
        box=np.array([3, 4, 22, 23]), object_id=2,
    )
    monkeypatch.setattr(segmenter, "_propagate_both_directions", lambda *args, **kwargs: {})

    segmenter.propagate_prompts()

    # Two initial calls, one call across the grouped replays for each object, then two calls to
    # restore the combined state. Every replay keeps both points batched with its box.
    assert len(segmenter.predictor.calls) == 6
    assert [len(call["points"]) for call in segmenter.predictor.calls] == [2] * 6
    assert [call["obj_id"] for call in segmenter.predictor.calls] == [1, 2, 1, 2, 1, 2]
    assert segmenter.predictor.active_objects == {1, 2}


def test_mixed_replay_preserves_mask_exclusivity_and_final_active_state():
    segmenter = make_recording_segmenter()
    key = (2, 3)
    segmenter.add_prompt_set(
        frame_id=3, points=np.array([[4, 6]]), point_labels=np.array([1]),
        box=np.array([1, 2, 10, 11]), object_id=2,
    )
    mask = np.ones((32, 32), dtype=bool)
    segmenter.add_mask_prompts(frame_ids=3, masks=[mask], object_id=2, refine=False)
    assert key not in segmenter._pushed_points and key not in segmenter._pushed_boxes
    assert key in segmenter._pushed_masks

    segmenter.add_point_prompts(
        frame_ids=3, points=np.array([[14, 16]]), point_labels=np.array([0]), object_id=2,
    )
    assert key not in segmenter._pushed_masks
    assert segmenter._pushed_points[key] == {(14, 16, 0)}

    snapshot = tuple(operation.copied() for operation in segmenter._prompt_history)
    segmenter.predictor.calls.clear()
    segmenter.predictor.mask_calls.clear()
    segmenter._replay_prompts(snapshot, {2})

    assert len(segmenter.predictor.calls) == 2
    assert len(segmenter.predictor.mask_calls) == 1
    assert key not in segmenter._pushed_masks
    assert segmenter._pushed_points[key] == {(14, 16, 0)}
    assert len(segmenter._prompt_history) == 3


def test_reset_clears_prompt_operation_history():
    segmenter = make_recording_segmenter()
    segmenter.add_prompt_set(
        frame_id=3, points=np.array([[4, 6]]), point_labels=np.array([1]), object_id=2,
    )
    assert segmenter._prompt_history

    segmenter.reset_tracking()

    assert segmenter._prompt_history == []


def test_add_prompt_set_conditions_on_a_box_alone():
    segmenter = make_recording_segmenter()
    segmenter.add_prompt_set(frame_id=1, box=np.array([0, 0, 8, 8]), object_id=1)
    assert len(segmenter.predictor.calls) == 1
    assert segmenter.predictor.calls[0]["points"] is None
    assert segmenter.predictor.calls[0]["box"] == [[0, 0, 8, 8]]
    # Nothing to push is not an error, and pushes nothing.
    segmenter.add_prompt_set(frame_id=1, object_id=1)
    assert len(segmenter.predictor.calls) == 1


@pytest.mark.parametrize("n_points", [1, 2, 7])
def test_add_prompt_set_never_pushes_a_negative_stride(n_points):
    """A single point is the case that bites.

    The caller hands (y, x) and this reverses to the (x, y) SAM2 wants, so the array it builds is a
    reversed view. For one point that view is flagged C-contiguous - numpy ignores the stride of a
    size-1 axis - while still carrying a negative stride, and torch refuses those. Nothing about the
    reversal is specific to one point, so all three counts are pinned.
    """
    segmenter = make_recording_segmenter()
    points_yx = np.arange(2 * n_points, dtype="float32").reshape(n_points, 2)[:, ::-1]
    segmenter.add_prompt_set(
        frame_id=0, points=points_yx, point_labels=np.ones(n_points, dtype="int32"),
        box=np.array([1, 2, 9, 9]), object_id=1,
    )

    call = segmenter.predictor.calls[0]
    for name in ("points", "labels"):
        pushed = np.asarray(call[name])
        assert all(stride > 0 for stride in pushed.strides), f"{name} has a negative stride"
        # The check torch itself makes, which is what the strides are about.
        torch.tensor(pushed)
    assert np.asarray(call["points"]).shape == (n_points, 2)


def _replicated_segmenter(n_devices):
    """A propagation pool whose per-device states are just their worker index."""
    pool = ReplicatedPromptableSegmentation3D.__new__(ReplicatedPromptableSegmentation3D)
    pool._predictor_devices = [(None, torch.device("cpu"))] * n_devices
    pool._segmenters = {}
    pool._get_segmenter = lambda worker_id: pool._segmenters.setdefault(worker_id, f"state-{worker_id}")
    return pool


def test_replicated_propagation_gives_every_worker_its_own_state():
    # A pass conditions its own objects, so two passes must never share one video-predictor state.
    pool = _replicated_segmenter(3)
    barrier = threading.Barrier(3, timeout=30)
    seen = []

    def run(segmenter, job):
        barrier.wait()  # Only returns if all three workers really run at the same time.
        seen.append(segmenter)
        return job

    assert pool.map_passes([0, 1, 2], run) == [0, 1, 2]
    assert sorted(seen) == ["state-0", "state-1", "state-2"]


def test_replicated_propagation_builds_no_state_it_cannot_use():
    # A run with one pass must not pay for a full model replica per device.
    pool = _replicated_segmenter(4)

    assert pool.map_passes(["only"], lambda segmenter, job: (segmenter, job)) == [("state-0", "only")]
    assert list(pool._segmenters) == [0]


def test_replicated_propagation_returns_the_jobs_in_order():
    pool = _replicated_segmenter(2)
    jobs = list(range(6))

    assert pool.map_passes(jobs, lambda segmenter, job: job * 2) == [job * 2 for job in jobs]
