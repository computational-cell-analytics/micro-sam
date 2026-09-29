import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EVALUATION_ROOT = Path(__file__).parents[1] / "finetuning/v2/evaluation"
sys.path.insert(0, str(EVALUATION_ROOT))
sys.path.insert(0, str(EVALUATION_ROOT / "optimization"))

import benchmark_ais_optimization as ais  # noqa
from benchmark_apg_3d import object_counts as reference_object_counts  # noqa
from common import unmatched_objects  # noqa


def _blobs(shape, centers, radii):
    labels = np.zeros(shape, dtype="uint32")
    grid = np.indices(shape)
    for index, (center, radius) in enumerate(zip(centers, radii), start=1):
        distance = sum(((g - c) / r) ** 2 for g, c, r in zip(grid, center, radius))
        labels[(distance <= 1) & (labels == 0)] = index
    return labels


@pytest.fixture(scope="module")
def geodesic_prediction():
    """A noisy geodesic hybrid field of three touching-ish 2d objects, as the v4 decoder would predict it."""
    from micro_sam.v2.transforms.labels import GeodesicHybridDistanceTransform

    labels = _blobs((128, 160), [(40, 50), (40, 95), (95, 110)], [(25, 22), (25, 24), (20, 30)])
    target = GeodesicHybridDistanceTransform(foreground=True)(labels).astype("float32")
    rng = np.random.default_rng(0)
    prediction = target + rng.normal(0, 0.02, target.shape).astype("float32")
    prediction[0] = np.clip(prediction[0], 0, 1)
    return prediction, labels


def test_resolve_postprocessing_fills_library_defaults():
    from micro_sam.v2.postprocessing import default_postprocessing

    resolved = ais.resolve_postprocessing({}, "hvit_t")
    assert resolved["sparse"] == default_postprocessing("hvit_t", "sparse")
    assert resolved["dense"] == default_postprocessing("hvit_t", "dense")
    volumes = ais.resolve_postprocessing({}, "hvit_t", ndim=3)
    assert volumes["sparse"] == default_postprocessing("hvit_t", "sparse", ndim=3)

    flat = ais.resolve_postprocessing({"n_iter": 200, "dt": 1.0}, "hvit_t")
    assert flat["sparse"]["n_iter"] == 200 and flat["sparse"]["dt"] == 1.0
    assert flat["dense"] == resolved["dense"]

    nested = ais.resolve_postprocessing({"sparse": {"sigma": 1.0}, "dense": {"beta": 0.7}}, "hvit_t")
    assert nested["sparse"]["sigma"] == 1.0 and nested["dense"]["beta"] == 0.7

    with pytest.raises(ValueError, match="Unknown AIS parameters"):
        ais.resolve_postprocessing({"candidate_threshold": 1.0}, "hvit_t")
    with pytest.raises(ValueError, match="only contain 'sparse' and 'dense'"):
        ais.resolve_postprocessing({"sparse": {}, "n_iter": 50}, "hvit_t")


def test_resolve_postprocessing_null_uses_default_and_off_disables_filter():
    defaults = ais.resolve_postprocessing({}, "hvit_t")["sparse"]
    resolved = ais.resolve_postprocessing({"boundary_magnitude_max": None}, "hvit_t")["sparse"]
    assert resolved["boundary_magnitude_max"] == defaults["boundary_magnitude_max"]
    assert np.isinf(
        ais.resolve_postprocessing({"boundary_magnitude_max": "off"}, "hvit_t")["sparse"][
            "boundary_magnitude_max"
        ]
    )
    with pytest.raises(ValueError, match="only valid for boundary_magnitude_max"):
        ais.resolve_postprocessing({"seed_floor": "off"}, "hvit_t")


def test_load_config_defaults_and_file(tmp_path):
    name, mode, params_2d, params_3d = ais.load_config(None, "hvit_t")
    assert (name, mode) == ("current-defaults", "auto")
    assert params_2d == ais.resolve_postprocessing({}, "hvit_t", ndim=2)
    assert params_3d == ais.resolve_postprocessing({}, "hvit_t", ndim=3)
    assert params_3d["sparse"]["min_size"] == 100 and params_2d["sparse"]["min_size"] == 50

    path = tmp_path / "candidate.json"
    path.write_text(json.dumps({"name": "travel", "params_2d": {"n_iter": 400}, "params_3d": {"n_iter": 100}}))
    name, mode, params_2d, params_3d = ais.load_config(path, "hvit_t")
    assert name == "travel" and params_2d["sparse"]["n_iter"] == 400 and params_3d["sparse"]["n_iter"] == 100

    # A volume takes the image overrides when it has none of its own.
    path.write_text(json.dumps({"name": "shared", "mode": "sparse", "params_2d": {"sigma": 2.0}}))
    _, mode, params_2d, params_3d = ais.load_config(path, "hvit_t")
    assert mode == "sparse" and params_3d["sparse"]["sigma"] == 2.0

    path.write_text(json.dumps({"name": "bad", "mode": "flow"}))
    with pytest.raises(ValueError, match="Unknown mode"):
        ais.load_config(path, "hvit_t")


def test_prediction_cache_validates_checkpoint_sample_and_shapes(tmp_path):
    cache = ais.PredictionCache(tmp_path, "checkpoint-a", "manifest-a")
    sample = {"sample_id": "toy:0"}
    prediction = np.zeros((4, 8, 9), dtype="float32")
    labels = np.zeros((8, 9), dtype="uint32")
    record = {
        "checkpoint_checksum": "checkpoint-a", "sample_id": "toy:0", "shape": list(prediction.shape),
    }
    cache.store(sample, prediction, labels, None, record)
    loaded, loaded_labels, valid, loaded_record = cache.load(sample)
    assert np.array_equal(loaded, prediction) and np.array_equal(loaded_labels, labels)
    assert valid is None and loaded_record == record

    _, record_path = cache.paths(sample)
    bad = dict(record, checkpoint_checksum="checkpoint-b")
    record_path.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="different checkpoint"):
        cache.load(sample)


def test_sparse_pipeline_matches_library(geodesic_prediction):
    from micro_sam.v2.postprocessing import flow_instance_segmentation

    prediction, labels = geodesic_prediction
    params = ais.resolve_postprocessing(
        {"min_size": 20, "n_iter": 200, "dt": 0.5, "density_threshold": 5.0}, "hvit_t",
    )["sparse"]
    expected = flow_instance_segmentation(prediction[0], prediction[1:], model_type="hvit_t", n_threads=2, **params)
    intermediates = ais.sparse_pipeline(prediction, params, None, 2)
    assert np.array_equal(intermediates["segmentation"], expected)
    assert intermediates["seeds"].max() == 3
    assert set(intermediates) >= {"before_min_size", "fg_mask", "density", "heightmap"}
    assert len(np.unique(expected)) - 1 == 3


def test_segment_prediction_matches_postprocess_unisam2(geodesic_prediction):
    from common import postprocess_unisam2

    prediction, _ = geodesic_prediction
    params = ais.resolve_postprocessing({"min_size": 20}, "hvit_t")["sparse"]
    mine = ais.segment_prediction(prediction, params, dense=False, spacing=None, model_type="hvit_t", n_threads=2)
    reference = postprocess_unisam2(prediction, "livecell", "hvit_t", params={"min_size": 20})
    assert np.array_equal(mine, reference)


def test_matched_ids_agrees_with_unmatched_objects(geodesic_prediction):
    prediction, labels = geodesic_prediction
    params = ais.resolve_postprocessing({"min_size": 20}, "hvit_t")["sparse"]
    segmentation = ais.sparse_pipeline(prediction, params, None, 2)["segmentation"]
    # Delete one instance and shave another so that a match fails on IoU rather than on absence.
    segmentation[segmentation == 1] = 0
    rows = np.where(segmentation == 2)[0]
    segmentation[rows.min():rows.min() + 30][segmentation[rows.min():rows.min() + 30] == 2] = 0
    matched = set(ais.matched_ids(labels, segmentation).tolist())
    unmatched = set(np.unique(unmatched_objects(labels, segmentation)).tolist()) - {0}
    assert matched | unmatched == {1, 2, 3} and not (matched & unmatched)
    assert 1 in unmatched


def test_object_counts_agree_with_reference_for_volumes():
    labels = _blobs((12, 64, 64), [(6, 20, 20), (6, 40, 44), (1, 50, 12), (10, 12, 50)],
                    [(4, 10, 10), (5, 12, 9), (1, 8, 8), (0.5, 6, 6)])
    assert labels.max() == 4
    segmentation = labels.copy()
    segmentation[segmentation == 2] = 0  # a miss
    segmentation[labels == 3] = 7  # matched under another id
    segmentation[:, 30:34, :] = 0  # shave everything
    mine = ais.object_counts(labels, segmentation)
    reference = reference_object_counts(labels, segmentation)
    assert mine["gt_objects"] == reference["gt_objects"] == 4
    assert mine["matched"] == reference["merged"]
    assert mine["unmatched"] == reference["unmatched"]
    assert mine["severed_objects"] == reference["severed_objects"] >= 1
    assert mine["genuine_misses"] == reference["genuine_misses"]
    assert mine["predicted_objects"] == 3


def test_object_counts_for_images_report_no_severed_objects():
    labels = _blobs((64, 64), [(20, 20), (44, 44)], [(10, 10), (12, 9)])
    counts = ais.object_counts(labels, labels)
    assert counts == {
        "gt_objects": 2, "severed_objects": 0, "matched": 2, "unmatched": 0, "genuine_misses": 0,
        "predicted_objects": 2,
    }
    empty = ais.object_counts(labels, np.zeros_like(labels))
    assert empty["matched"] == 0 and empty["unmatched"] == 2 and empty["genuine_misses"] == 2


def test_seed_diagnostics_count_misses_splits_and_background_seeds():
    labels = _blobs((64, 96), [(20, 20), (20, 60), (48, 40)], [(10, 10), (10, 12), (9, 20)])
    seeds = np.zeros_like(labels, dtype="uint64")
    seeds[20, 20] = 1  # object 1: one seed
    seeds[18, 58] = 2
    seeds[22, 64] = 3  # object 2: split
    seeds[5, 90] = 4  # background
    seeds[60, 5] = 5  # background
    segmentation = labels.copy()
    segmentation[labels == 3] = 0  # object 3 (no seed) is missing from the result
    intermediates = {"seeds": seeds, "fg_mask": labels != 0, "before_min_size": labels}
    diagnostics = ais.seed_diagnostics(intermediates, labels, segmentation)
    assert diagnostics["n_seeds"] == 5
    assert diagnostics["gt_with_0_seeds"] == 1
    assert diagnostics["gt_with_1_seed"] == 1
    assert diagnostics["gt_with_2plus_seeds"] == 1
    assert diagnostics["background_seeds"] == 2
    assert diagnostics["seeded_unmatched"] == 0
    assert diagnostics["unseeded_missing"] == 1 and diagnostics["unseeded_absorbed"] == 0
    assert diagnostics["matched_before_min_size"] == 3
    assert diagnostics["fg_iou"] == 1.0
    assert diagnostics["matched_iou"] == 1.0

    # A seeded object the watershed then loses is lost at the assignment; here object 1 is undersized
    # (only a quarter of it survives) and object 2, with two seeds, is a split.
    segmentation = labels.copy()
    segmentation[labels == 1] = 0
    segmentation[16:24, 16:24][labels[16:24, 16:24] == 1] = 1
    columns = np.indices(labels.shape)[1]
    segmentation[(labels == 2) & (columns >= 56) & (columns < 64)] = 9  # three parts, none above IoU 0.5
    segmentation[(labels == 2) & (columns >= 64)] = 10
    diagnostics = ais.seed_diagnostics(intermediates, labels, segmentation)
    assert diagnostics["seeded_unmatched"] == 2
    assert diagnostics["seeded_undersized"] == 1 and diagnostics["seeded_split"] == 1
    assert diagnostics["seeded_merged"] == 0 and diagnostics["seeded_oversized"] == 0

    # One instance covering all three objects: object 1 (one seed) is merged, object 2 (two seeds) is a
    # split, and the unseeded object 3 is absorbed (less than half of the instance is its own).
    segmentation = np.where(labels != 0, 1, 0).astype("uint32")
    diagnostics = ais.seed_diagnostics(intermediates, labels, segmentation)
    assert diagnostics["seeded_merged"] == 1 and diagnostics["seeded_split"] == 1
    assert diagnostics["unseeded_absorbed"] == 1 and diagnostics["unseeded_missing"] == 0


def test_object_fates_reports_iou_and_flags():
    labels = _blobs((64, 96), [(20, 20), (20, 60), (48, 40)], [(10, 10), (10, 12), (9, 20)])
    fates = ais.object_fates(labels, labels)
    assert fates["ids"].tolist() == [1, 2, 3]
    assert np.allclose(fates["iou"], 1.0) and fates["absorbed"].all() and not fates["merged"].any()
    assert not fates["undersized"].any()
    fates = ais.object_fates(labels, np.zeros_like(labels))
    assert np.allclose(fates["iou"], 0.0) and not fates["absorbed"].any()


def _sample_rows(datasets, msa_by_dataset, family=None, seen=""):
    rows = []
    for dataset in datasets:
        for index, msa in enumerate(msa_by_dataset[dataset]):
            rows.append({
                "sample_id": f"{dataset}:{index}", "dataset": dataset, "ndim": 2,
                "family": family.get(dataset, dataset) if family else dataset, "seen_in_training": seen,
                "metric_mode": "sparse", "postprocessing_mode": "sparse", "initialization_seconds": 1.0,
                "generation_seconds": 0.5, "total_seconds": 1.5, "peak_cuda_memory_bytes": 10 + index,
                "msa": msa, "gt_objects": 4, "predicted_objects": 3, "matched": 3, "unmatched": 1,
                "severed_objects": 0, "genuine_misses": 1, "matched_before_min_size": 3, "n_seeds": 3,
                "gt_with_0_seeds": 1, "gt_with_1_seed": 3, "gt_with_2plus_seeds": 0, "background_seeds": 0,
                "seeded_unmatched": 0, "fg_iou": 0.9, "pipeline_mismatch": 0,
            })
    return pd.DataFrame(rows)


def test_summarize_reports_means_sums_and_balanced_row():
    samples = _sample_rows(["a", "b"], {"a": [0.2, 0.4], "b": [0.8, 0.8, 0.8]})
    summary = ais.summarize(samples).set_index("dataset")
    assert summary.loc["a", "msa_mean"] == pytest.approx(0.3)
    assert summary.loc["b", "n_samples"] == 3 and summary.loc["b", "matched"] == 9
    assert summary.loc[ais.BALANCED_ROW, "msa_mean"] == pytest.approx(0.55)
    assert summary.loc[ais.BALANCED_ROW, "total_seconds"] == pytest.approx(7.5)
    assert summary.loc["b", "peak_cuda_memory_bytes"] == 12
    assert "__family_macro__" not in summary.index


def test_summarize_adds_family_macros_for_crop_manifests():
    samples = _sample_rows(
        ["cremi", "cremi_seen", "gonuclear"], {"cremi": [0.1], "cremi_seen": [0.3], "gonuclear": [0.6]},
        family={"cremi": "cremi", "cremi_seen": "cremi", "gonuclear": "gonuclear"},
    )
    samples.loc[samples["dataset"] == "cremi_seen", "seen_in_training"] = "True"
    samples.loc[samples["dataset"] != "cremi_seen", "seen_in_training"] = "False"
    summary = ais.summarize(samples).set_index("dataset")
    assert summary.loc["__dataset_balanced__", "msa_mean"] == pytest.approx((0.1 + 0.3 + 0.6) / 3)
    assert summary.loc["__family_macro__", "msa_mean"] == pytest.approx((0.2 + 0.6) / 2)
    assert summary.loc["__unseen_macro__", "msa_mean"] == pytest.approx((0.1 + 0.6) / 2)


def test_gate_table_applies_the_generalization_rule():
    baseline = pd.Series({"a": 0.5, "b": 0.4, "c": 0.3, "d": 0.2})
    verdict = ais.gate_table(baseline, pd.Series({"a": 0.53, "b": 0.42, "c": 0.31, "d": 0.21}))
    assert verdict["passed"] and verdict["n_up"] == 4
    # One dataset below both loss limits fails, however large the balanced gain.
    verdict = ais.gate_table(baseline, pd.Series({"a": 0.9, "b": 0.9, "c": 0.9, "d": 0.18}))
    assert not verdict["checks"]["no_dataset_below_loss_limits"]
    # A tiny absolute loss on a near-zero score is tolerated by the absolute limit.
    verdict = ais.gate_table(pd.Series({"a": 0.5, "b": 0.01}), pd.Series({"a": 0.6, "b": 0.008}))
    assert verdict["checks"]["no_dataset_below_loss_limits"]
    # Too many datasets down fails.
    verdict = ais.gate_table(baseline, pd.Series({"a": 0.9, "b": 0.39, "c": 0.29, "d": 0.19}))
    assert not verdict["checks"]["up_on_all_but_two"]
    # Below the balanced gain fails.
    verdict = ais.gate_table(baseline, pd.Series({"a": 0.501, "b": 0.401, "c": 0.301, "d": 0.201}))
    assert not verdict["checks"]["balanced_gain_at_least_2_percent"]


def test_dataset_scores_use_negated_cremi_on_dense_data():
    samples = _sample_rows(["a"], {"a": [0.2, 0.4]})
    dense = samples.copy()
    dense["dataset"], dense["metric_mode"], dense["cremi"] = "snemi", "dense", [0.9, 0.7]
    scores = ais.dataset_scores(pd.concat([samples, dense], ignore_index=True))
    assert scores["a"] == pytest.approx(0.3) and scores["snemi"] == pytest.approx(-0.8)


def test_report_joins_subsets_and_flags_the_gate(tmp_path):
    def write_run(name, subset, msa_by_dataset):
        run_dir = tmp_path / f"{name}-{subset}"
        run_dir.mkdir()
        samples = _sample_rows(sorted(msa_by_dataset), msa_by_dataset)
        samples.to_csv(run_dir / "samples.csv", index=False)
        (run_dir / "metadata.json").write_text(json.dumps({"status": "complete", "config_name": name}))
        return run_dir

    runs = {
        "current-defaults": [
            write_run("current-defaults", "primary", {"a": [0.4], "b": [0.5]}),
            write_run("current-defaults", "extra", {"c": [0.6]}),
        ],
        "candidate": [
            write_run("candidate", "primary", {"a": [0.44], "b": [0.55]}),
            write_run("candidate", "extra", {"c": [0.63]}),
        ],
    }
    table, details = ais.report(runs, "current-defaults")
    table = table.set_index("config")
    assert table.loc["candidate", "passed"] and table.loc["candidate", "n_datasets"] == 3
    assert not table.loc["current-defaults", "passed"]
    assert details.query("config == 'candidate' and dataset == 'a'")["relative"].iloc[0] == pytest.approx(0.1)


def test_grid_combinations_deduplicate_flow_travel():
    grid = {"n_iter": [50, 100], "dt": [0.5, 1.0], "sigma": [0.5]}
    combinations = ais.grid_combinations(grid, "sparse")
    travels = sorted(round(c["n_iter"] * c["dt"], 6) for c in combinations)
    assert travels == [25.0, 50.0, 100.0]
    with pytest.raises(ValueError, match="Unknown sparse grid parameters"):
        ais.grid_combinations({"beta": [0.5]}, "sparse")

    explicit = ais.grid_combinations({"combinations": [{"n_iter": 100}, {"n_iter": 100}, {"n_iter": 200}]}, "sparse")
    assert explicit == [{"n_iter": 100}, {"n_iter": 200}]
    with pytest.raises(ValueError, match="at least one"):
        ais.grid_combinations({"combinations": []}, "sparse")
    with pytest.raises(ValueError, match="Unknown sparse grid parameters"):
        ais.grid_combinations({"combinations": [{"beta": 0.5}]}, "sparse")

    families = ais.grid_combinations({
        "shared": {"n_iter": [400], "dt": [0.5]},
        "families": {"base": {}, "ridge": {"contact_weight": [0.5, 1.0]}},
    }, "sparse")
    assert len(families) == 3
    assert [combo["mechanism_family"] for combo in families] == ["base", "ridge", "ridge"]
    with pytest.raises(ValueError, match="redefines shared"):
        ais.grid_combinations({
            "shared": {"n_iter": [400]}, "families": {"bad": {"n_iter": [800]}},
        }, "sparse")


def test_sweep_shards_keep_expensive_cache_groups_together():
    grid = {
        "foreground_threshold": [0.4, 0.5], "sigma": [0.5], "n_iter": [400, 800], "dt": [0.5],
        "density_threshold": [5.0, 10.0], "min_size": [25, 50], "foreground_weight": [0.5, 1.0],
    }
    combinations = [
        ais.resolve_postprocessing({"sparse": combo}, "hvit_t")["sparse"]
        for combo in ais.grid_combinations(grid, "sparse")
    ]
    shards = [ais.shard_combinations(combinations, "sparse", index, 3) for index in range(3)]
    assert sum(map(len, shards)) == len(combinations)
    assert {json.dumps(combo, sort_keys=True) for shard in shards for combo in shard} == {
        json.dumps(combo, sort_keys=True) for combo in combinations
    }
    flow_keys = ais.SWEEP_CACHE_KEYS["sparse"]
    groups = [{tuple(combo[key] for key in flow_keys) for combo in shard} for shard in shards]
    assert all(not (first & second) for index, first in enumerate(groups) for second in groups[index + 1:])
    work = [sum(group[2] for group in shard_groups) for shard_groups in groups]
    assert max(work) - min(work) <= max(group[2] for shard_groups in groups for group in shard_groups)
    with pytest.raises(ValueError, match="Invalid shard"):
        ais.shard_combinations(combinations, "sparse", 3, 3)
    with pytest.raises(ValueError, match="only 4 distinct"):
        ais.shard_combinations(combinations, "sparse", 0, 5)


def test_shared_configuration_ranks_by_mean_relative_optimum(tmp_path):
    grid = pd.DataFrame({"sigma": [0.5, 1.0, 2.0], "n_iter": [50, 50, 50]})
    for dataset, scores in {"a": [0.5, 0.4, 0.2], "b": [0.3, 0.6, 0.3]}.items():
        table = grid.copy()
        table["n_images"], table["msa_mean"], table["msa_std"] = 3, scores, 0.0
        table.to_csv(tmp_path / f"{dataset}.csv", index=False)
    shared = ais.shared_configuration(tmp_path, ["a", "b"])
    assert list(shared.columns[:2]) == ["sigma", "n_iter"]
    assert shared.iloc[0]["sigma"] == 1.0  # 0.8 + 1.0 over 1.0 + 0.5
    assert shared.iloc[0]["mean_relative"] == pytest.approx(0.9)
    assert shared.iloc[0]["balanced"] == pytest.approx(0.5)
    assert (tmp_path / "shared_config.csv").exists()


def test_gt_seed_markers_and_ridge_heightmap():
    labels = _blobs((64, 96), [(20, 20), (20, 60), (48, 40)], [(10, 10), (10, 12), (9, 20)])
    markers = ais.gt_seed_markers(labels)
    assert markers.dtype == np.uint64
    ids, counts = np.unique(markers[markers != 0], return_counts=True)
    assert ids.tolist() == [1, 2, 3] and counts.tolist() == [9, 9, 9]
    # Every marker sits inside its own object.
    for index in ids:
        assert set(labels[markers == index].tolist()) == {index}
    # A marker never leaks into a neighbouring object or the background, even for a one-pixel object.
    tiny = np.zeros((8, 8), dtype="uint32")
    tiny[2:6, 2:6] = 1
    tiny[3, 3] = 2
    markers = ais.gt_seed_markers(tiny)
    assert (tiny[markers == 2] == 2).all() and (markers == 2).sum() == 1
    ridge = ais.gt_ridge_heightmap(labels)
    assert ridge.dtype == np.float32 and ridge.flags["C_CONTIGUOUS"]
    assert set(np.unique(ridge).tolist()) == {0.0, 1.0}
    assert (ridge[labels == 0] == 0).all()


def test_oracle_sample_recovers_ground_truth_with_gt_seeds_and_foreground(geodesic_prediction):
    prediction, labels = geodesic_prediction
    sample = {"sample_id": "toy:0", "dataset": "toy", "ndim": 2}
    context = {"ndim": 2, "metric_mode": "sparse", "postprocessing_mode": "sparse", "spacing": None,
               "border_min_size": 0}
    params = ais.resolve_postprocessing({"min_size": 20}, "hvit_t")
    row = ais.oracle_sample(sample, context, prediction, labels, None, params, n_threads=2)
    assert set(f"msa_{name}" for name in ais.ORACLES) <= set(row)
    assert row["msa_gt_seeds_gt_fg"] >= row["msa_baseline"]
    assert row["msa_gt_seeds_gt_fg"] > 0.95 and row["matched_gt_seeds_gt_fg"] == 3
    summary = ais.summarize_oracles(pd.DataFrame([row, {**row, "sample_id": "toy:1"}])).set_index("dataset")
    assert summary.loc[ais.BALANCED_ROW, "msa_baseline"] == pytest.approx(row["msa_baseline"])
    assert summary.loc["toy", "gain_gt_seeds_gt_fg"] == pytest.approx(
        row["msa_gt_seeds_gt_fg"] / row["msa_baseline"] - 1.0
    )


def test_rank_shared_flags_gate_against_the_reference():
    import report_ais_sweep as rs

    grid = pd.DataFrame({"sigma": [0.5, 1.0, 0.5, 1.0], "boundary_magnitude_max": [np.nan, np.nan, 0.4, 0.4]})
    tables = {}
    msa = {"a": [0.50, 0.48, 0.55, 0.54], "b": [0.30, 0.31, 0.33, 0.30], "c": [0.20, 0.22, 0.22, 0.10]}
    for dataset, scores in msa.items():
        table = grid.copy()
        table["n_images"], table["msa_mean"], table["msa_std"] = 5, scores, 0.0
        tables[dataset] = table
    ranked = rs.rank_shared(tables, reference={"sigma": 0.5, "boundary_magnitude_max": None})
    ranked = ranked.set_index(["sigma", "boundary_magnitude_max"])
    # The reference row: no change, not passing.
    assert ranked.loc[(0.5, "none"), "balanced_gain"] == pytest.approx(0.0)
    assert not ranked.loc[(0.5, "none"), "passed"]
    # sigma 0.5 with the filter improves every dataset by at least 10 %: passes.
    assert ranked.loc[(0.5, 0.4), "passed"] and ranked.loc[(0.5, 0.4), "n_up"] == 3
    assert ranked.loc[(0.5, 0.4), "rel_c"] == pytest.approx(0.10)
    # sigma 1.0 with the filter halves dataset c: fails the loss limit despite the balanced gain.
    assert not ranked.loc[(1.0, 0.4), "passed"]
    assert ranked["mean_relative_optimum"].max() <= 1.0
    with pytest.raises(ValueError, match="matches 0 rows"):
        rs.rank_shared(tables, reference={"sigma": 2.0, "boundary_magnitude_max": None})


def test_sweep_tables_union_keeps_mechanism_families(monkeypatch, tmp_path):
    import report_ais_sweep as rs

    def fake_load(grid_path, *_args, **_kwargs):
        table = pd.DataFrame({"n_iter": [800], "n_images": [2], "msa_mean": [0.5], "msa_std": [0.1]})
        if grid_path.stem == "ridge":
            table["contact_weight"] = 1.0
        return {"a": table, "b": table.copy()}

    monkeypatch.setattr(rs, "load_sweep_tables", fake_load)
    tables = rs.load_sweep_tables_many(
        [Path("base.json"), Path("ridge.json")], ["primary"], tmp_path, tmp_path, tmp_path,
        "hvit_t", "boundary",
    )
    assert set(tables["a"]["mechanism_family"]) == {"base", "ridge"}
    assert set(tables["a"]["contact_weight"].astype(str)) == {"none", "1.0"}
    ranked = rs.rank_shared(tables)
    assert len(ranked) == 2 and set(ranked["mechanism_family"]) == {"base", "ridge"}


def test_sweep_table_keeps_embedded_mechanism_family(monkeypatch, tmp_path):
    import report_ais_sweep as rs

    table = pd.DataFrame({
        "n_iter": [800, 800], "mechanism_family": ["base", "ridge"],
        "n_images": [2, 2], "msa_mean": [0.5, 0.6], "msa_std": [0.1, 0.1],
    })
    monkeypatch.setattr(rs, "load_sweep_tables", lambda *_args, **_kwargs: {"a": table})
    tables = rs.load_sweep_tables_many(
        [Path("boundary.json")], ["primary"], tmp_path, tmp_path, tmp_path, "hvit_t", "boundary",
    )
    assert set(tables["a"]["mechanism_family"]) == {"base", "ridge"}


def test_sweep_plateau_selection_prefers_robust_cheaper_candidate():
    import report_ais_sweep as rs

    ranked = pd.DataFrame([
        {"balanced": 0.6000, "min_relative_optimum": 0.96, "n_iter": 1600,
         "contact_weight": 2.0, "contact_mask_threshold": 0.5, "foreground_threshold": 0.4},
        {"balanced": 0.5995, "min_relative_optimum": 0.98, "n_iter": 800,
         "contact_weight": "none", "contact_mask_threshold": "none", "foreground_threshold": 0.45},
        {"balanced": 0.5900, "min_relative_optimum": 1.00, "n_iter": 400,
         "contact_weight": "none", "contact_mask_threshold": "none", "foreground_threshold": 0.5},
    ])
    selected = rs.select_plateau(ranked, tolerance=0.001)
    assert selected["foreground_threshold"] == 0.45
    config = rs.selected_config(selected, "baseline-dice-optimum")
    assert config["params_2d"]["sparse"] == {"foreground_threshold": 0.45, "n_iter": 800}
    assert rs.select_plateau(ranked.drop(columns=["contact_weight"]), tolerance=0.001)[
        "foreground_threshold"
    ] == 0.45
    selected["boundary_magnitude_max"] = np.inf
    assert rs.selected_config(selected, "off")["params_2d"]["sparse"]["boundary_magnitude_max"] == "off"


def test_polish_grid_refines_boundary_coordinates_and_edges():
    import prepare_ais_reoptimization_polish as polish

    ranking = pd.DataFrame([
        {"mechanism_family": "ridge", "balanced": 0.50, "foreground_threshold": 0.4,
         "foreground_weight": 0.75, "min_size": 50, "boundary_magnitude_max": 0.4,
         "n_iter": 1200, "contact_weight": 1.0},
        {"mechanism_family": "ridge", "balanced": 0.502, "foreground_threshold": 0.4,
         "foreground_weight": 0.75, "min_size": 50, "boundary_magnitude_max": 0.4,
         "n_iter": 1600, "contact_weight": 1.0},
    ])
    combinations = polish.polish_combinations(ranking, top_per_family=1)
    assert any(combo.get("contact_weight") == 3.0 for combo in combinations)
    assert any(combo.get("n_iter") == 2400 for combo in combinations)
    assert any(combo.get("boundary_magnitude_max") == "off" for combo in combinations)


def test_polish_grid_restores_numeric_optional_parameters_from_csv_strings():
    import prepare_ais_reoptimization_polish as polish

    ranking = pd.DataFrame([{
        "mechanism_family": "combined", "balanced": 0.5, "foreground_threshold": 0.45,
        "foreground_weight": 0.5, "min_size": 50, "boundary_magnitude_max": 0.4,
        "density_threshold": 20.0, "n_iter": 1200, "sigma": 0.5, "dt": 0.5,
        "contact_weight": "1.0", "contact_mask_threshold": "0.5",
    }])
    combinations = polish.polish_combinations(ranking, top_per_family=1)
    assert combinations
    assert all(
        not isinstance(combo.get(key), str)
        for combo in combinations
        for key in ("contact_weight", "contact_mask_threshold")
        if key in combo
    )


def test_polish_cli_reports_safe_shard_count(tmp_path, capsys):
    import prepare_ais_reoptimization_polish as polish

    ranking = pd.DataFrame([{
        "mechanism_family": "base", "balanced": 0.5, "foreground_threshold": 0.4,
        "foreground_weight": 0.75, "min_size": 50, "boundary_magnitude_max": 0.4,
        "n_iter": 800, "sigma": 0.5, "dt": 0.5,
    }])
    ranking_path, output_path = tmp_path / "ranking.csv", tmp_path / "polish.json"
    ranking.to_csv(ranking_path, index=False)
    assert polish.main(["--ranking", str(ranking_path), "--output", str(output_path)]) == 0
    output = capsys.readouterr().out
    combinations = json.loads(output_path.read_text())["combinations"]
    resolved = [ais.resolve_postprocessing({"sparse": combo}, "hvit_t")["sparse"] for combo in combinations]
    expected = len({tuple(combo[key] for key in ais.SWEEP_CACHE_KEYS["sparse"]) for combo in resolved})
    assert f"{expected} flow-cache groups" in output
    assert f"no more than {expected} sweep shards" in output


@pytest.fixture(scope="module")
def contact_prediction(geodesic_prediction):
    """The fixture's field plus a fifth channel with the ground-truth contact lines."""
    from micro_sam.v2.transforms.labels import touching_boundaries

    prediction, labels = geodesic_prediction
    contact = touching_boundaries(labels).astype("float32")[None]
    return np.concatenate([prediction, contact], axis=0), labels


def test_resolve_postprocessing_accepts_the_contact_keywords():
    params = ais.resolve_postprocessing({"contact_weight": 1.0, "contact_mask_threshold": 0.5}, "hvit_t")["sparse"]
    assert params["contact_weight"] == 1.0 and params["contact_mask_threshold"] == 0.5
    assert "contact_weight" not in ais.resolve_postprocessing({}, "hvit_t")["sparse"]


@pytest.mark.parametrize("overrides", [{}, {"contact_weight": 1.0}, {"contact_mask_threshold": 0.5}])
def test_sparse_pipeline_matches_library_with_a_contact_channel(contact_prediction, overrides):
    from micro_sam.v2.postprocessing import flow_instance_segmentation

    prediction, labels = contact_prediction
    params = ais.resolve_postprocessing({"min_size": 20, "foreground_weight": 1.0, **overrides}, "hvit_t")["sparse"]
    expected = flow_instance_segmentation(
        prediction[0], prediction[1:4], model_type="hvit_t", n_threads=2, contact=prediction[4], **params,
    )
    intermediates = ais.sparse_pipeline(prediction, params, None, 2)
    assert np.array_equal(intermediates["segmentation"], expected)
    mine = ais.segment_prediction(prediction, params, dense=False, spacing=None, model_type="hvit_t", n_threads=2)
    assert np.array_equal(mine, expected)
    diagnostics = ais.seed_diagnostics(intermediates, labels, expected)
    assert 0.5 < diagnostics["fg_area_ratio"] < 2.0
    assert "fg_area_ratio" in ais.METRIC_COLUMNS


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"contact_weight": 1.0},
        {"contact_mask_threshold": 0.5},
        {"contact_weight": 1.0, "contact_mask_threshold": 0.5},
    ],
)
def test_cached_sparse_scorer_matches_library_with_boundary_channel(contact_prediction, overrides, monkeypatch):
    import parameter_search
    from micro_sam.v2.postprocessing import flow_instance_segmentation

    prediction, labels = contact_prediction
    params = ais.resolve_postprocessing(
        {
            "min_size": 20,
            "n_iter": 200,
            "density_threshold": 5.0,
            "foreground_weight": 1.0,
            **overrides,
        },
        "hvit_t",
    )["sparse"]
    expected = flow_instance_segmentation(
        prediction[0], prediction[1:4], contact=prediction[4], model_type="hvit_t", n_threads=2, **params,
    )
    monkeypatch.setattr(
        parameter_search,
        "compute_metrics",
        lambda segmentation, *_args, **_kwargs: {"segmentation": segmentation.copy()},
    )
    result = parameter_search.score_image_sparse_cached(prediction, labels, [params], n_threads=2)[0]
    assert np.array_equal(result["segmentation"], expected)


def test_cached_sparse_scorer_rejects_boundary_parameters_without_channel(geodesic_prediction):
    import parameter_search

    prediction, labels = geodesic_prediction
    params = ais.resolve_postprocessing({"contact_weight": 1.0}, "hvit_t")["sparse"]
    with pytest.raises(ValueError, match="need prediction channel 4"):
        parameter_search.score_image_sparse_cached(prediction, labels, [params], n_threads=2)
