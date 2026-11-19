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
