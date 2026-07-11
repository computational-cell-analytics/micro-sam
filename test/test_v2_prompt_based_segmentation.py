import numpy as np
import pytest
from bioimage_cpp.utils import Blocking

from micro_sam.v2.prompt_based_segmentation import (
    PromptableSegmentation3D,
    TiledPromptableSegmentation3D,
)


class FakeTileSegmenter:
    def add_point_prompts(self, **kwargs):
        pass

    def add_box_prompts(self, **kwargs):
        pass


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


def test_promptable_segmentation_3d_progress_total():
    segmenter = PromptableSegmentation3D.__new__(PromptableSegmentation3D)
    segmenter.volume = np.zeros((8, 16, 16), dtype="uint8")

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
        self.calls = []

    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, clear_old_points,
                              points=None, labels=None, box=None):
        self.calls.append({
            "frame_idx": frame_idx, "obj_id": obj_id, "clear_old_points": clear_old_points,
            "points": None if points is None else np.asarray(points).tolist(),
            "labels": None if labels is None else np.asarray(labels).tolist(),
            "box": None if box is None else np.asarray(box).tolist(),
        })

    def reset_state(self, inference_state):
        pass


def make_recording_segmenter():
    segmenter = PromptableSegmentation3D.__new__(PromptableSegmentation3D)
    segmenter.predictor = RecordingPredictor()
    segmenter.volume = np.zeros((8, 32, 32), dtype="uint8")
    segmenter.inference_state = {}
    segmenter._pushed_points = {}
    segmenter._pushed_boxes = {}
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
