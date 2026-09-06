import json
import sys
from pathlib import Path

import numpy as np
import pytest


OPTIMIZATION_ROOT = Path(__file__).parents[1] / "finetuning/v2/evaluation/optimization"
sys.path.insert(0, str(OPTIMIZATION_ROOT))

extractor = pytest.importorskip("extract_apg_3d_tracks")
trainer = pytest.importorskip("train_apg_3d_filter")


def _crop(cache_root, sample_id, n, seed):
    from micro_sam.v2.multimask_selection import SELECTOR_FEATURE_SCHEMAS
    from micro_sam.v2.automatic_prompt_generation import VOLUME_CANDIDATE_FEATURE_NAMES
    rng = np.random.default_rng(seed)
    crop = cache_root / sample_id.replace(":", "_")
    crop.mkdir(parents=True)
    n_features = len(SELECTOR_FEATURE_SCHEMAS["token_lowres_v1"])
    features = rng.normal(size=(n, 3, n_features)).astype("float32")
    features[0, 1] = np.nan  # one empty alternative
    payload, offsets, shapes = extractor.pack_masks([np.ones((8, 8), dtype=bool)] * n)
    np.savez(
        crop / "candidates.npz", prompt_index=np.arange(n), frame=np.zeros(n, dtype="int64"),
        point_xy=np.zeros((n, 2), dtype="float32"), anchor_predicted_iou=rng.uniform(0.5, 1, n).astype("float32"),
        anchor_stability=np.ones(n, dtype="float32"), alternative_features=features,
        alternative_scores=rng.uniform(size=(n, 3)).astype("float32"), alternative_stability=np.ones((n, 3), "float32"),
        anchor_mask_payload=payload, anchor_mask_offsets=offsets, anchor_mask_shapes=shapes,
        anchor_box_start=np.zeros((n, 2), dtype="int64"), prompt_frame=np.zeros(n, dtype="int64"),
        prompt_point_xy=np.zeros((n, 2), dtype="float32"), ladder_membership=np.ones((n, 2), dtype=bool),
        component_features=rng.normal(size=(n, len(VOLUME_CANDIDATE_FEATURE_NAMES))).astype("float32"),
        component_origin_ladder=np.zeros(n, dtype="int64"),
        component_feature_names=np.asarray(VOLUME_CANDIDATE_FEATURE_NAMES),
        ladders=np.array([json.dumps([1.5, 10.0]), json.dumps([1.0, 3.0])]),
        feature_schema=np.asarray("token_lowres_v1"),
    )
    tracks = [np.ones((2, 8, 8), dtype=bool)] * n
    payload, offsets, shapes = extractor.pack_masks(tracks)
    np.savez(
        crop / "tracks.npz", prompt_index=np.arange(n), box_start=np.zeros((n, 3), dtype="int64"),
        box_stop=np.tile([2, 8, 8], (n, 1)), mask_payload=payload, mask_offsets=offsets, mask_shapes=shapes,
        track_iou=rng.uniform(size=n).astype("float32"), track_gt_id=np.ones(n, dtype="int64"),
        volume_shape=np.array([2, 8, 8]),
    )
    (crop / "complete.json").write_text("{}")


def test_aggregate_and_train_on_a_synthetic_cache(tmp_path):
    cache = tmp_path / "cache"
    samples = []
    for index, (dataset, fold) in enumerate([("a", 0), ("a", 1), ("b", 2), ("b", 3), ("c", 4), ("c", 0)]):
        sample_id = f"{dataset}:{index:012d}"
        _crop(cache, sample_id, 30, index)
        samples.append({"sample_id": sample_id, "dataset": dataset, "family": dataset, "source_id": f"{dataset}{index}",
                        "fold": fold, "seen_in_training": dataset == "c"})
    manifest = {"manifest_checksum": "m", "samples": samples}
    dataset = trainer.aggregate(cache, manifest, tmp_path / "training")
    data = np.load(dataset, allow_pickle=False)
    assert data["features"].shape == (180, 3, 275) and np.isfinite(data["features"]).all()
    assert data["missing_alternative"].sum() == 6
    # Every dataset gets the same total weight.
    weights = data["weight"]
    totals = {name: float(weights[data["dataset"] == name].sum()) for name in ("a", "b", "c")}
    assert totals["a"] == pytest.approx(totals["b"]) == pytest.approx(totals["c"])
    artifact = trainer.train(dataset, tmp_path / "models", "token_v1", ["persistence", "log_z_extent"], 8, 0.1, "cpu",
                             lodo=True)
    assert artifact.exists()
    oof = np.load(artifact.with_name(artifact.stem + "_oof.npz"))
    assert oof["oof"].shape == (180,) and oof["lodo"].shape == (180,)
    scorer = trainer.load_volume_candidate_scorer(artifact, device="cpu")
    import torch
    scores = scorer.predict_candidates(torch.zeros(4, 3, 258), torch.zeros(4, 2))
    assert scores.shape == (4,) and torch.isfinite(scores).all()
    assert scorer.input_schema == "token_v1" and scorer.component_feature_names == ("persistence", "log_z_extent")
