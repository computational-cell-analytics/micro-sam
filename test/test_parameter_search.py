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
