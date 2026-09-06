import json
import sys
from pathlib import Path

import pytest


OPTIMIZATION_ROOT = Path(__file__).parents[1] / "finetuning/v2/evaluation/optimization"
sys.path.insert(0, str(OPTIMIZATION_ROOT))

runner = pytest.importorskip("benchmark_apg_3d")


def test_volume_params_start_from_the_volume_defaults():
    from micro_sam.v2.automatic_prompt_generation import default_prompt_generation
    params = runner.resolve_volume_params({})
    volume = default_prompt_generation("hvit_t", is_volume=True)
    for key in ("sigma", "min_candidate_size", "max_overlap", "min_size", "score_threshold"):
        assert params[key] == volume[key]
    assert tuple(params["candidate_threshold"]) == tuple(volume["candidate_threshold"])
    assert params["foreground_threshold"] == 0.7 and params["n_iter"] == 50
    assert params["early_stop_patience"] == 2 and params["refinement"] is None


def test_volume_params_apply_overrides_and_reject_unknown_keys(tmp_path):
    params = runner.resolve_volume_params({"candidate_threshold_3d": [1.0, 3.0, 10.0], "refinement": "points+boxes"})
    assert params["candidate_threshold"] == [1.0, 3.0, 10.0] and params["refinement"] == "points+boxes"
    with pytest.raises(ValueError, match="Unknown volume parameters"):
        runner.resolve_volume_params({"multimask_scorer": "microscopy"})
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"name": "x", "params_2d": {"score_threshold": 0.1}, "params_3d": {"sigma": 0.5}}))
    name, params = runner.load_volume_config(config)
    assert name == "x" and params["sigma"] == 0.5 and params["score_threshold"] != 0.1


def test_ladder_keys_and_run_identity_are_stable():
    assert runner._ladder_key((1.5, 10.0)) == "seeded_1p5_10"
    assert runner._ladder_key((0.5, 2.0, 10.0)) == "seeded_0p5_2_10"
    first = runner.run_identity("cfg", {"a": 1}, {})
    assert first == runner.run_identity("cfg", {"a": 1}, {})
    assert first != runner.run_identity("cfg", {"a": 2}, {})
