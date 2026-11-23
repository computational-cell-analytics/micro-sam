import sys
from pathlib import Path

import pytest
import pandas as pd


EVALUATION_ROOT = Path(__file__).parents[1] / "finetuning/v2/evaluation"
sys.path.insert(0, str(EVALUATION_ROOT))

import parameter_search  # noqa


def test_parameter_search_recomputes_a_cache_that_lacks_a_grid_column(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "hvit_t"
    output_dir.mkdir()
    pd.DataFrame([{
        "candidate_threshold": 1.0,
        "msa_mean": 0.8,
        "msa_std": 0.0,
    }]).to_csv(output_dir / "sample.csv", index=False)
    config = {
        "criterion": "msa",
        "crop": None,
        "grid": {"candidate_threshold": [1.0], "propagation_waves": [1, 4]},
        "metric_mode": "msa",
        "mode": "apg",
        "ndim": 3,
        "spacing": None,
        "z_range": None,
    }
    monkeypatch.setattr(parameter_search, "has_val_split", lambda dataset_name: True)
    monkeypatch.setattr(parameter_search, "tuning_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(
        parameter_search, "n_samples", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("recompute")),
    )

    with pytest.raises(RuntimeError, match="recompute"):
        parameter_search.tune_parameters(
            model=None, mode="apg", dataset_name="sample", data_root="data", model_type="hvit_t",
            output_root=str(tmp_path), device="cpu",
        )

    assert "lacks parameter columns: ['propagation_waves']" in capsys.readouterr().out


def test_parameter_search_accepts_a_cache_with_the_current_grid(tmp_path):
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame([{
        "candidate_threshold": 1.0,
        "propagation_waves": 1,
        "msa_mean": 0.8,
        "msa_std": 0.0,
    }]).to_csv(csv_path, index=False)
    config = {"grid": {"candidate_threshold": [1.0], "propagation_waves": [1, 4]}}

    assert parameter_search.sweep_cache_is_current(csv_path, config)


def _volume_with_unannotated_object():
    import numpy as np

    labels = np.zeros((4, 32, 32), dtype="uint32")
    labels[:, 2:10, 2:10] = 1
    labels[:, 2:10, 14:22] = 2
    valid_roi = np.ones(labels.shape, dtype="bool")
    valid_roi[:, 24:] = False  # Tissue that is not annotated.
    seg = labels.copy()
    seg[:, 26:30, 26:30] = 3  # An object the annotation does not cover.
    return seg, labels, valid_roi


def test_compute_metrics_scores_inside_the_annotated_region_only():
    seg, labels, valid_roi = _volume_with_unannotated_object()
    unmasked = parameter_search.compute_metrics(seg, labels, "sparse")["msa"]
    masked = parameter_search.compute_metrics(seg, labels, "sparse", valid_roi=valid_roi)["msa"]
    assert unmasked < 1.0
    assert masked == pytest.approx(1.0)
    # The same restriction as the test evaluation, `common.run_evaluation`.
    seg_eval = seg.copy()
    seg_eval[~valid_roi] = 0
    assert parameter_search.compute_metrics(seg_eval, labels, "sparse")["msa"] == pytest.approx(masked)


def test_compute_metrics_keeps_labelled_objects_only():
    seg, labels, _ = _volume_with_unannotated_object()
    assert parameter_search.compute_metrics(seg, labels, "sparse", labelled_only=True)["msa"] == pytest.approx(1.0)


def test_score_image_forwards_the_scored_region(monkeypatch):
    import numpy as np

    seg, labels, valid_roi = _volume_with_unannotated_object()
    seen = []

    def fake_metrics(seg, labels, metric_mode, border_min_size=0, valid_roi=None, labelled_only=False, slivers=None):
        seen.append((valid_roi, labelled_only, slivers))
        return {"msa": 1.0}

    monkeypatch.setattr(parameter_search, "compute_metrics", fake_metrics)
    prediction = np.zeros((5,) + labels.shape, dtype="float32")
    prediction[0] = labels > 0
    params = [{
        "foreground_threshold": 0.5, "sigma": 0.5, "n_iter": 2, "dt": 1.0, "density_threshold": 0.5,
        "foreground_weight": 0.9, "min_size": 0,
    }]
    parameter_search.score_image(prediction, labels, "sparse", params, n_threads=1, valid_roi=valid_roi)
    assert len(seen) == 1 and seen[0][0] is valid_roi and seen[0][1] is False and seen[0][2] is None
    slivers = (10.0, labels == 0, 0)
    parameter_search.score_image(prediction, labels, "sparse", params, n_threads=1, slivers=slivers)
    assert seen[1][2] is slivers


def test_tile_starts_adds_edge_tiles_only_for_half_a_tile_of_new_data():
    region = (slice(0, 40), slice(0, 978), slice(10, 658))
    assert parameter_search.tile_starts(region, (32, 512, 512)) == [(0, 0, 10)]
    starts = parameter_search.tile_starts(region, (32, 512, 512), edge_aligned=True)
    # z: 8 slices rest (< 16), y: 466 px rest (>= 256), x: 136 px rest (< 256).
    assert starts == [(0, 0, 10), (0, 466, 10)]


def test_centered_tile_starts_reproduce_the_test_crops():
    import common

    # The annotated boxes of two plantseg_root test volumes. No crop of them is below the object count floor.
    _, starts = common.EVAL_CROPS["plantseg_root"]
    shapes = {"Movie1_t00045_crop_gt.h5": (200, 450, 1000), "Movie2_T00020_crop_gt.h5": (353, 450, 1590)}
    for volume, shape in shapes.items():
        region = tuple(slice(0, size) for size in shape)
        tiles = parameter_search.tile_starts(region, (32, 512, 512), centered=True)
        assert tiles == [start for start, _ in starts[volume]]
    with pytest.raises(ValueError):
        parameter_search.tile_starts(region, (32, 512, 512), edge_aligned=True, centered=True)


def test_spread_tiles_samples_every_volume_over_its_object_counts():
    # (object count, volume, start): 'a' has 8 tiles with 10 to 80 objects and one near-empty tile, 'b' 2 tiles.
    tiles = [(10 * (i + 1), "a", (32 * i, 0, 0)) for i in range(8)] + [(1, "a", (256, 0, 0))]
    tiles += [(30, "b", (0, 0, 0)), (60, "b", (32, 0, 0))]
    chosen = parameter_search.spread_tiles(tiles, max_tiles=5)
    # 'b' gets one tile, 'a' the other four: the ones at evenly spaced ranks of its 8 tiles above the floor.
    assert chosen == [("a", (32, 0, 0)), ("a", (96, 0, 0)), ("a", (160, 0, 0)), ("a", (224, 0, 0)), ("b", (32, 0, 0))]
    # All tiles above the floor stay if they fit.
    assert len(parameter_search.spread_tiles(tiles, max_tiles=20)) == 10


def test_plantseg_root_and_nucverse3d_tune_on_tiles_placed_like_the_test_crops():
    import common

    for dataset, n_volumes in (("plantseg_root", 2), ("nucverse3d", 3)):
        _, starts = common.TUNING_CROPS[dataset]
        assert len(starts) == n_volumes and all(starts.values())
        assert sum(len(crops) for crops in starts.values()) == 10
    # The glia slab is one of the ten nucverse3d crops instead of one of three.
    assert len(common.TUNING_CROPS["nucverse3d"][1]["C2_M01.h5"]) == 1


def _volume_with_slivers():
    import numpy as np

    # Three objects of 6 x 10 x 10 voxels inside the volume, and the sliver of a fourth one on the first slice.
    labels = np.zeros((16, 40, 40), dtype="uint32")
    labels[4:10, 2:12, 2:12] = 1
    labels[4:10, 2:12, 14:24] = 2
    labels[4:10, 14:24, 2:12] = 3
    labels[:1, 26:36, 26:36] = 4  # 100 voxels; the median object has 600.
    return labels


def test_find_slivers_needs_a_small_object_on_a_face(monkeypatch):
    import common

    labels = _volume_with_slivers()
    labels[6, 30, 30] = 5  # A speck inside the volume is no sliver.
    labels[10:16, 2:12, 26:36] = 6  # Neither is a large object that a face cuts.
    min_size, mask, n_slivers = common.find_slivers(labels, 60)
    assert n_slivers == 0 and not mask.any()  # The cut piece of 100 voxels is above the bound.

    labels[:1, 26:36, 31:36] = 0  # 50 voxels are left of it.
    min_size, mask, n_slivers = common.find_slivers(labels, 60)
    assert min_size == 60 and n_slivers == 1 and (mask == (labels == 4)).all()
    assert common.has_scored_objects(labels, (min_size, mask, n_slivers))
    assert not common.has_scored_objects(labels * (labels == 4), common.find_slivers(labels * (labels == 4), 60))

    # The bound of a dataset is the one measured for its split (sliver_sizes.json).
    monkeypatch.setattr(common, "SLIVER_SIZES", {"plantseg_root": {"test": {"min_size": 60}, "val": {"min_size": 40}}})
    assert common.dataset_slivers("plantseg_root", labels)[2] == 1
    assert common.dataset_slivers("plantseg_root", labels, split="val")[2] == 0
    assert common.dataset_slivers("lucchi", labels) is None  # The EM organelle volumes keep their scoring.
    with pytest.raises(RuntimeError, match="prepare --sliver_sizes"):
        common.dataset_slivers("nucverse3d", labels)


def test_every_dataset_scored_without_slivers_has_a_measured_bound():
    import common

    for dataset in common.SLIVER_DATASETS:
        assert common.sliver_min_size(dataset) > 0
    for dataset in parameter_search.IN_DOMAIN_CELLS["3d"]:
        entry = common.SLIVER_SIZES[dataset]["val"]
        assert entry["min_size"] == pytest.approx(common.SLIVER_FRACTION * entry["median_size"], abs=0.1)
        assert entry["n_crops"] == common.n_samples(dataset, common.DATA_ROOT, split="val")


def test_exclude_slivers_scores_neither_the_sliver_nor_what_is_predicted_in_its_place():
    import numpy as np
    import common

    labels = _volume_with_slivers()
    labels[:1, 26:36, 31:36] = 0
    seg = labels.copy()
    seg[seg == 4] = 0
    seg[:1, 26:36, 26:33] = 7   # A prediction that covers the sliver and 20 voxels next to it.
    seg[15:, 2:8, 2:8] = 8      # A predicted sliver on the opposite face.
    seg[4:10, 14:24, 14:24] = 9  # A false positive inside the volume stays.

    without = parameter_search.compute_metrics(seg.copy(), labels, "sparse")
    scored = parameter_search.compute_metrics(seg.copy(), labels, "sparse", slivers=common.find_slivers(labels, 60))
    assert "gt_slivers" not in without and scored["gt_slivers"] == 1
    # Three matches and the false positive inside the volume: 3 / 4. With the slivers the predicted sliver on the
    # opposite face is a second false positive, and the sliver of the ground truth a fourth match.
    assert scored["sa50"] == pytest.approx(3 / 4)
    assert without["sa50"] == pytest.approx(4 / 6)

    gt, out, n_slivers = common.exclude_slivers(labels, seg, common.find_slivers(labels, 60))
    assert n_slivers == 1 and set(np.unique(gt)) == {0, 1, 2, 3} and set(np.unique(out)) == {0, 1, 2, 3, 9}
    # The search finds the slivers of the tuning volumes that are scored without them, and only of those.
    assert parameter_search.tuning_config("plantseg_root", "ais")["slivers"]
    assert not parameter_search.tuning_config("cremi", "ais")["slivers"]
    assert not parameter_search.tuning_config("lucchi", "ais")["slivers"]


def test_results_scored_with_slivers_are_not_current(tmp_path):
    import common

    old, new, skipped = tmp_path / "old.csv", tmp_path / "new.csv", tmp_path / "skipped.csv"
    pd.DataFrame([{"mSA": 0.3, "SA50": 0.5}]).to_csv(old, index=False)
    pd.DataFrame([{"mSA": 0.4, "SA50": 0.6, "gt_slivers": 3}]).to_csv(new, index=False)
    pd.DataFrame([{"skipped": True}]).to_csv(skipped, index=False)
    assert not common.result_is_current(str(old), "nucverse3d")
    assert common.result_is_current(str(new), "nucverse3d") and common.result_is_current(str(skipped), "nucverse3d")
    assert common.result_is_current(str(old), "livecell") and common.result_is_current(str(old), "lucchi")
    assert not common.result_is_current(str(tmp_path / "missing.csv"), "livecell")

    # The same for a stored sweep of the search.
    sweep, config = tmp_path / "sweep.csv", {"grid": {"min_size": [50]}, "criterion": "msa", "slivers": True}
    pd.DataFrame([{"min_size": 50, "msa_mean": 0.3}]).to_csv(sweep, index=False)
    assert not parameter_search.sweep_cache_is_current(sweep, config)
    assert parameter_search.sweep_cache_is_current(sweep, {**config, "slivers": False})
    pd.DataFrame([{"min_size": 50, "msa_mean": 0.3, "gt_slivers_mean": 2.0}]).to_csv(sweep, index=False)
    assert parameter_search.sweep_cache_is_current(sweep, config)


def test_evaluate_samples_scores_the_volumes_without_slivers(tmp_path, monkeypatch):
    import common

    labels = _volume_with_slivers()
    labels[:1, 26:36, 31:36] = 0
    seg = labels.copy()
    seg[seg == 4] = 0  # The sliver is not segmented.
    monkeypatch.setattr(common, "SLIVER_SIZES", {"nucverse3d": {"test": {"min_size": 60}}})
    monkeypatch.setattr(common, "n_samples", lambda *args, **kwargs: 1)
    monkeypatch.setattr(common, "load_data", lambda *args, **kwargs: iter([(labels.astype("float32"), labels, None)]))
    results = {}
    for dataset in ("nucverse3d", "lucchi"):
        save_path = str(tmp_path / f"{dataset}.csv")
        results[dataset] = common.evaluate_samples(lambda raw: seg.copy(), dataset, "data", save_path, "test")
        assert common.result_is_current(save_path, dataset)
    assert results["nucverse3d"]["mSA"].iloc[0] == pytest.approx(1.0)
    assert results["nucverse3d"]["gt_slivers"].iloc[0] == 1
    assert results["nucverse3d"]["mSA_with_slivers"].iloc[0] == pytest.approx(0.75)
    assert results["lucchi"]["mSA"].iloc[0] == pytest.approx(0.75) and "gt_slivers" not in results["lucchi"]


def test_compute_metrics_ignores_label_zero_in_the_cremi_score():
    import numpy as np

    labels = np.zeros((4, 8, 8), dtype="uint32")
    labels[:, :4, :4], labels[:, :4, 4:] = 1, 2  # The lower half is not annotated.
    seg = np.zeros_like(labels)
    seg[:, :4] = 1  # Merges the two objects.
    scored = parameter_search.compute_metrics(seg, labels, "dense", valid_roi=labels > 0)
    ignored = parameter_search.compute_metrics(seg, labels, "dense", valid_roi=labels > 0, ignore_zero=True)
    # The unannotated half matches trivially, which hides a part of the merge unless it is ignored.
    assert ignored["vi_merge"] == pytest.approx(1.0)
    assert scored["vi_merge"] == pytest.approx(0.5)
    assert ignored["cremi"] > scored["cremi"]


@pytest.mark.parametrize("dataset", ["platynereis_cells", "synapsenet_compartments"])
def test_volume_filling_em_cells_are_tuned_and_scored_as_dense_datasets(dataset):
    import common

    assert dataset in common.DATASETS_DENSE
    assert dataset in common.CREMI_IGNORE_ZERO_LABEL
    assert dataset in parameter_search.IN_DOMAIN_CELLS["dense"]
    assert dataset not in parameter_search.IN_DOMAIN_CELLS["3d"]
    assert parameter_search.tuning_config(dataset, "ais")["metric_mode"] == "dense"
    assert dataset in common.TUNING_CROPS


@pytest.mark.parametrize("dataset", ["liconn", "xpress"])
def test_liconn_and_xpress_are_evaluated_but_not_tuned_on(dataset):
    import common

    assert dataset in common.DATASETS_DENSE
    assert dataset in common.DATASETS_3D_EM_NEURITE_ID
    assert dataset not in parameter_search.search_datasets(None, ["dense"], "cells")["dense"]


def test_orgline_is_in_the_2d_tuning_pool():
    import common

    assert common.VAL_SPLITS["orgline"] == "val"
    assert "orgline" in parameter_search.search_datasets(None, ["2d"], "cells")["2d"]


def test_platynereis_cells_tuning_crops_stay_inside_the_labelled_boxes():
    import common

    crop_shape, starts = common.TUNING_CROPS["platynereis_cells"]
    for volume, roi in common.PLATYNEREIS_CELLS_VAL_ROIS.items():
        assert len(starts[volume]) == 3
        # The crops start inside the labelled box, on its lateral corner (the loader clips them at its end).
        for start, _ in starts[volume]:
            assert all(sl.start <= s < sl.stop for s, sl in zip(start, roi))
            assert tuple(start[1:]) == (roi[1].start, roi[2].start)


def test_cartocell_is_ranked_by_sa50_on_the_test_like_volumes(tmp_path):
    import common

    assert "cartocell" in common.DATASETS_SA50
    config = parameter_search.tuning_config("cartocell", "ais")
    assert config["criterion"] == "sa50"
    assert parameter_search.CRITERION_ASCENDING["sa50"] is False
    assert parameter_search.tuning_config("gonuclear", "ais")["criterion"] == "msa"
    assert parameter_search.tuning_config("cremi", "ais")["criterion"] == "cremi"
    # Only the test volumes that moved into tuning hold crops; the validation folder is another acquisition.
    _, starts = common.TUNING_CROPS["cartocell"]
    with_crops = {volume for volume, crops in starts.items() if crops}
    assert with_crops == set(common.CARTOCELL_TUNING_TEST_VOLUMES)
    assert sum(len(crops) for crops in starts.values()) == 9
    # A sweep that was ranked by mSA holds no SA50 column and is recomputed.
    row = {key: values[0] for key, values in config["grid"].items()}
    old, new = tmp_path / "old.csv", tmp_path / "new.csv"
    # cartocell is scored without slivers (common.SLIVER_DATASETS), which a current sweep records.
    row["gt_slivers_mean"] = 0.0
    pd.DataFrame([{**row, "msa_mean": 0.1, "msa_std": 0.0}]).to_csv(old, index=False)
    pd.DataFrame([{**row, "msa_mean": 0.1, "sa50_mean": 0.3}]).to_csv(new, index=False)
    assert not parameter_search.sweep_cache_is_current(old, config)
    assert parameter_search.sweep_cache_is_current(new, config)


def test_compute_metrics_reports_sa50():
    import numpy as np

    labels = np.zeros((1, 8, 8), dtype="uint32")
    labels[:, :4], labels[:, 4:] = 1, 2
    seg = labels.copy()
    seg[:, 4:5] = 0  # The second object keeps three quarters of its pixels: an IoU of 0.75.
    metrics = parameter_search.compute_metrics(seg, labels, "sparse")
    assert metrics["sa50"] == pytest.approx(1.0)  # Both objects match at an IoU of 0.5 ...
    assert metrics["msa"] < 1.0  # ... but the second one not at the higher thresholds.
    seg[:, 4:7] = 0  # A quarter of its pixels: no match, so one hit, one miss and one false positive.
    assert parameter_search.compute_metrics(seg, labels, "sparse")["sa50"] == pytest.approx(1 / 3)


def _synthetic_prediction(shape, n_per_axis=2):
    """A five-channel prediction of a grid of touching boxes: foreground, distances to the centers, boundaries."""
    import numpy as np

    ndim = len(shape)
    labels = np.zeros(shape, dtype="uint32")
    prediction = np.zeros((5,) + shape, dtype="float32")
    size_y, size_x = shape[-2] // n_per_axis, shape[-1] // n_per_axis
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")).astype("float32")
    for i in range(n_per_axis):
        for j in range(n_per_axis):
            y, x = slice(i * size_y, (i + 1) * size_y), slice(j * size_x, (j + 1) * size_x)
            box = (slice(None),) * (ndim - 2) + (y, x)
            labels[box] = i * n_per_axis + j + 1
            center = (i * size_y + size_y / 2, j * size_x + size_x / 2)
            for axis, c in zip((-2, -1), center):
                prediction[(4 + axis,) + box] = (grid[(axis,) + box] - c) / (size_y / 2)
            boundary = np.zeros(labels[box].shape, dtype="float32")
            boundary[..., 0, :] = boundary[..., -1, :] = boundary[..., :, 0] = boundary[..., :, -1] = 1
            prediction[(4,) + box] = boundary
    prediction[0] = 1 - 0.6 * prediction[4]
    return prediction, labels


def _scored_segmentations(monkeypatch, prediction, labels, mode, params):
    segmentations = []

    def record(seg, *args, **kwargs):
        segmentations.append(seg.copy())
        return {"msa": 1.0, "cremi": 0.0}

    monkeypatch.setattr(parameter_search, "compute_metrics", record)
    scores = parameter_search.score_image(prediction, labels, mode, params, n_threads=1)
    assert all(score is not None for score in scores)
    return segmentations


def _same_partition(a, b):
    import numpy as np

    pairs = np.unique(np.stack([a.ravel(), b.ravel()]), axis=1)
    return len(pairs[0]) == len(np.unique(a)) == len(np.unique(b))


def test_sparse_scoring_with_boundary_parameters_matches_the_library(monkeypatch):
    import numpy as np
    from micro_sam.v2.postprocessing import flow_instance_segmentation

    prediction, labels = _synthetic_prediction((48, 48))
    base = {
        "foreground_threshold": 0.3, "sigma": 0.5, "n_iter": 20, "dt": 1.0, "density_threshold": 2.0,
        "foreground_weight": 0.5, "min_size": 5,
    }
    params = [
        base, {**base, "boundary_weight": 0, "boundary_mask_threshold": None},
        {**base, "boundary_weight": 0.5, "boundary_mask_threshold": None},
        {**base, "boundary_weight": 0, "boundary_mask_threshold": 0.5},
        {**base, "boundary_weight": 1.0, "boundary_mask_threshold": 0.5},
    ]
    segmentations = _scored_segmentations(monkeypatch, prediction, labels, "sparse", params)
    assert len(segmentations) == len(params)
    for combo, seg in zip(params, segmentations):
        expected = flow_instance_segmentation(
            prediction[0], prediction[1:4], model_type="hvit_t", boundary=prediction[4], n_threads=1, **combo
        )
        assert np.array_equal(seg, expected), combo
    assert segmentations[0].max() > 1
    assert np.array_equal(segmentations[0], segmentations[1])


def test_dense_scoring_with_a_boundary_weight_matches_the_evaluation(monkeypatch):
    import numpy as np
    import common
    from micro_sam.v2.instance_segmentation import _segment_from_predictions
    from micro_sam.v2.postprocessing import multicut_boundary_map

    prediction, labels = _synthetic_prediction((4, 48, 48))
    foreground, boundary = prediction[0], prediction[4]
    inverted = (foreground.max() - foreground) / (foreground.max() - foreground).max()
    assert np.array_equal(multicut_boundary_map(foreground), inverted)
    assert np.array_equal(multicut_boundary_map(foreground, boundary, 0), inverted)
    assert np.allclose(multicut_boundary_map(foreground, boundary, 1.0), boundary)
    with pytest.raises(ValueError):
        multicut_boundary_map(foreground, None, 0.5)

    base = {"beta": 0.5, "density_threshold": 2.0, "sigma": 0.5, "n_iter": 20, "dt": 1.0}
    params = [base, {**base, "boundary_weight": 0.5}, {**base, "beta": 0.6, "boundary_weight": 0.5}]
    segmentations = _scored_segmentations(monkeypatch, prediction, labels, "dense", params)
    for combo, seg in zip(params, segmentations):
        expected = common.postprocess_unisam2(prediction.copy(), "cremi", "hvit_t", params=combo)
        assert _same_partition(seg, expected), combo
        # The library segments the same way.
        library = _segment_from_predictions(prediction.copy(), mode="dense", model_type="hvit_t", **combo)
        assert _same_partition(library, expected), combo
    # The combos that share an oversegmentation are grouped by all of its parameters, also for the shards.
    assert parameter_search.n_shard_groups(params, "dense", 3) == 2


def test_boundary_combos_start_from_the_best_combos():
    import json

    sparse = [{"density_threshold": value, "min_size": 50} for value in (10.0, 20.0, 30.0)]
    ranked = [json.dumps(params, sort_keys=True) for params in sparse]
    combos = parameter_search.boundary_combos(ranked, "3d", top=2)
    pairs = 4 * 4 - 1  # Every pair of the boundary grid but the one without a boundary term.
    assert len(combos) == 2 * pairs
    assert {combo["density_threshold"] for combo in combos} == {10.0, 20.0}
    assert not any(c["boundary_weight"] == 0 and c["boundary_mask_threshold"] is None for c in combos)
    # A combo of an earlier boundary stage is no start for the stage.
    again = parameter_search.boundary_combos([json.dumps(combos[0], sort_keys=True)] + ranked, "3d", top=2)
    assert again == combos

    dense = [
        {"beta": beta, "density_threshold": threshold, "sigma": 0.5, "n_iter": 25, "dt": 0.5}
        for threshold, beta in ((3.0, 0.5), (3.0, 0.4), (5.0, 0.5), (1.0, 0.6))
    ]
    combos = parameter_search.boundary_combos([json.dumps(params, sort_keys=True) for params in dense], "dense", top=2)
    # The two best oversegmentation settings, with every mix weight and every beta swept so far.
    assert len(combos) == 2 * 4 * 3
    assert {combo["density_threshold"] for combo in combos} == {3.0, 5.0}
    assert {combo["beta"] for combo in combos} == {0.4, 0.5, 0.6}


def test_per_dataset_min_size_keeps_the_winner_unless_another_value_scores_higher():
    import json

    winner = {"density_threshold": 10.0, "min_size": 50, "boundary_mask_threshold": None}
    rows = {
        json.dumps({**winner, "min_size": 25}, sort_keys=True): {"a": 0.5, "b": 0.7, "c": 0.4},
        json.dumps(winner, sort_keys=True): {"a": 0.6, "b": 0.6, "c": 0.4},
        json.dumps({**winner, "min_size": 100}, sort_keys=True): {"a": 0.4, "b": 0.6, "c": 0.4},
        # Another combination does not count, whatever its score.
        json.dumps({**winner, "density_threshold": 20.0, "min_size": 200}, sort_keys=True): dict(a=0.9, b=0.9, c=0.9),
    }
    scores = pd.DataFrame(rows).T
    best, compared = parameter_search.per_dataset_min_size(scores, winner)
    assert best == {"a": 50, "b": 25, "c": 50}
    assert compared["b"] == {"global": 0.6, "per_dataset": 0.7}
