import sys
from pathlib import Path

import pandas as pd
import pytest


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
