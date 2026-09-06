import json
import sys
from pathlib import Path

import numpy as np
import pytest


OPTIMIZATION_ROOT = Path(__file__).parents[1] / "finetuning/v2/evaluation/optimization"
sys.path.insert(0, str(OPTIMIZATION_ROOT))

extractor = pytest.importorskip("extract_apg_3d_tracks")
replay = pytest.importorskip("screen_apg_3d_filter")


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(0)
    masks = [rng.random((3, 5, 7)) > 0.5, rng.random((2, 8, 8)) > 0.2, np.zeros((1, 2, 2), dtype=bool)]
    payload, offsets, shapes = extractor.pack_masks(masks)
    assert offsets[0] == 0 and len(offsets) == len(masks) + 1
    for index, mask in enumerate(masks):
        np.testing.assert_array_equal(extractor.unpack_mask(payload, offsets, shapes, index), mask)


def test_union_prompts_keeps_each_anchor_once_with_its_first_ladders_metadata():
    ladder_a = ({"points": np.array([[[4.0, 5.0]], [[10.0, 11.0]]], dtype="float32"),
                 "frames": np.array([0, 2])}, {"features": np.array([[1.0, 1.0], [2.0, 2.0]], dtype="float32")})
    ladder_b = ({"points": np.array([[[4.0, 5.0]], [[20.0, 21.0]]], dtype="float32"),
                 "frames": np.array([0, 1])}, {"features": np.array([[9.0, 9.0], [3.0, 3.0]], dtype="float32")})
    prompts, membership, features, origin = extractor.union_prompts([ladder_a, ladder_b])
    assert prompts["points"].shape == (3, 1, 2) and prompts["frames"].tolist() == [0, 2, 1]
    assert membership.tolist() == [[True, True], [True, False], [False, True]]
    assert features[0].tolist() == [1.0, 1.0] and origin.tolist() == [0, 0, 1]


def _fake_cache(tmp_path):
    """Two candidates on frame 0 (one weak, one strong), one on frame 1; tracks for all three."""
    crop = tmp_path / "crop"
    crop.mkdir()
    anchor_masks = [np.ones((8, 8), dtype=bool), np.ones((8, 8), dtype=bool), np.ones((8, 8), dtype=bool)]
    payload, offsets, shapes = extractor.pack_masks(anchor_masks)
    np.savez(
        crop / "candidates.npz",
        prompt_index=np.array([0, 1, 2]), frame=np.array([0, 0, 1]),
        point_xy=np.array([[4.0, 4.0], [20.0, 4.0], [4.0, 4.0]], dtype="float32"),
        anchor_predicted_iou=np.array([0.9, 0.5, 0.8], dtype="float32"),
        anchor_stability=np.ones(3, dtype="float32"),
        alternative_features=np.zeros((3, 3, 4), dtype="float32"),
        alternative_scores=np.zeros((3, 3), dtype="float32"), alternative_stability=np.ones((3, 3), dtype="float32"),
        anchor_mask_payload=payload, anchor_mask_offsets=offsets, anchor_mask_shapes=shapes,
        anchor_box_start=np.array([[0, 0], [0, 16], [0, 0]]),
        prompt_frame=np.array([0, 0, 1]), prompt_point_xy=np.array([[4.0, 4.0], [20.0, 4.0], [4.0, 4.0]]),
        ladder_membership=np.array([[True, True], [False, True], [True, True]]),
        component_features=np.zeros((3, 2), dtype="float32"), component_origin_ladder=np.array([0, 1, 0]),
        component_feature_names=np.array(["a", "b"]),
        ladders=np.array([json.dumps([1.5, 10.0]), json.dumps([1.0, 3.0])]),
        feature_schema=np.array("token_lowres_v1"),
    )
    tracks = [np.ones((2, 8, 8), dtype=bool), np.ones((2, 8, 8), dtype=bool), np.ones((1, 8, 8), dtype=bool)]
    payload, offsets, shapes = extractor.pack_masks(tracks)
    np.savez(
        crop / "tracks.npz", prompt_index=np.array([0, 1, 2]),
        box_start=np.array([[0, 0, 0], [0, 0, 16], [1, 0, 0]]), box_stop=np.array([[2, 8, 8], [2, 8, 24], [2, 8, 8]]),
        mask_payload=payload, mask_offsets=offsets, mask_shapes=shapes,
        track_iou=np.array([0.9, 0.2, 0.7], dtype="float32"), track_gt_id=np.array([1, 2, 1]),
        volume_shape=np.array([2, 8, 32]),
    )
    (crop / "complete.json").write_text("{}")
    return replay.CropCache(crop)


def test_anchor_survivors_apply_threshold_and_ladder_membership(tmp_path):
    cache = _fake_cache(tmp_path)
    # Ladder 0: candidates 0 and 2 belong; both pass 0.6.
    assert replay.anchor_survivors(cache, 0).tolist() == [0, 2]
    # Ladder 1: all three belong, but candidate 1 (0.5) fails the anchor threshold.
    assert replay.anchor_survivors(cache, 1).tolist() == [0, 2]
    assert replay.anchor_survivors(cache, 1, score_threshold=0.4).tolist() == [0, 1, 2]


def test_passes_count_per_anchor_frame(tmp_path):
    cache = _fake_cache(tmp_path)
    assert replay.passes_for(cache, np.array([0, 1, 2])) == 2
    assert replay.passes_for(cache, np.array([0, 1])) == 1
    assert replay.passes_for(cache, np.array([], dtype="int64")) == 0


def test_replay_merges_cached_tracks_into_a_segmentation(tmp_path):
    cache = _fake_cache(tmp_path)
    labels = np.zeros((2, 8, 32), dtype="uint32")
    labels[:, :, :8] = 1
    result = replay.replay(cache, np.array([0, 1, 2]), labels, None, "sparse")
    assert result["tracks"] == 3 and result["candidates"] == 3 and result["propagation_passes"] == 2
    # Candidate 2's track duplicates candidate 0's on the second slice, so at most two objects survive.
    assert 1 <= result["predicted_objects"] <= 2
    assert 0.0 <= result["msa"] <= 1.0


def test_fold_thresholds_exclude_the_test_fold():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.95], dtype="float32")
    folds = np.array([0, 0, 1, 1, 2, 2])
    eligible = np.ones(6, dtype=bool)
    thresholds = replay.fold_thresholds(scores, folds, eligible, retention=0.5)
    # Fold 2's threshold comes from folds 0 and 1 only (0.1 .. 0.4), so it cannot see its own 0.9s.
    assert thresholds[2] == pytest.approx(0.25)
    assert thresholds[0] > thresholds[2]
