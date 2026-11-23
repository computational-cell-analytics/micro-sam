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

    def fake_metrics(seg, labels, metric_mode, border_min_size=0, valid_roi=None, labelled_only=False):
        seen.append((valid_roi, labelled_only))
        return {"msa": 1.0}

    monkeypatch.setattr(parameter_search, "compute_metrics", fake_metrics)
    prediction = np.zeros((5,) + labels.shape, dtype="float32")
    prediction[0] = labels > 0
    params = [{
        "foreground_threshold": 0.5, "sigma": 0.5, "n_iter": 2, "dt": 1.0, "density_threshold": 0.5,
        "foreground_weight": 0.9, "min_size": 0,
    }]
    parameter_search.score_image(prediction, labels, "sparse", params, n_threads=1, valid_roi=valid_roi)
    assert len(seen) == 1 and seen[0][0] is valid_roi and seen[0][1] is False


def test_tile_starts_adds_edge_tiles_only_for_half_a_tile_of_new_data():
    region = (slice(0, 40), slice(0, 978), slice(10, 658))
    assert parameter_search.tile_starts(region, (32, 512, 512)) == [(0, 0, 10)]
    starts = parameter_search.tile_starts(region, (32, 512, 512), edge_aligned=True)
    # z: 8 slices rest (< 16), y: 466 px rest (>= 256), x: 136 px rest (< 256).
    assert starts == [(0, 0, 10), (0, 466, 10)]


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
