import json
import sys
from pathlib import Path

import pytest
import numpy as np


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
    with pytest.raises(ValueError, match="Unknown volume parameters"):
        runner.resolve_volume_params({"candidate_budget": 8})
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"name": "x", "params_2d": {"score_threshold": 0.1}, "params_3d": {"sigma": 0.5}}))
    name, params = runner.load_volume_config(config)
    assert name == "x" and params["sigma"] == 0.5 and params["score_threshold"] != 0.1


def test_run_identity_is_stable():
    first = runner.run_identity("cfg", {"a": 1}, "checkpoint", "manifest", "trial-1")
    assert first == runner.run_identity("cfg", {"a": 1}, "checkpoint", "manifest", "trial-1")
    assert first != runner.run_identity("cfg", {"a": 2}, "checkpoint", "manifest", "trial-1")


@pytest.mark.parametrize("field", ["checkpoint_id", "manifest_checksum", "trial_id"])
def test_run_identity_separates_experiments(tmp_path, field):
    identity = {"checkpoint_id": "checkpoint", "manifest_checksum": "manifest", "trial_id": "trial-1"}
    first = runner.run_dir(tmp_path, "primary", "cfg", {}, **identity)
    first.mkdir(parents=True)
    identity[field] = "different"
    second = runner.run_dir(tmp_path, "primary", "cfg", {}, **identity)
    assert first != second
    assert runner.sibling_run_dirs(second) == []


@pytest.mark.parametrize("missed_id", [0, 1, 2])
def test_object_counts_include_matched_severed_objects(missed_id):
    labels = np.zeros((8, 8, 8), dtype="uint32")
    labels[2:5, 2:4, 2:4] = 1
    labels[5:7, 5:7, 6:] = 2
    segmentation = labels.copy()
    segmentation[segmentation == missed_id] = 0
    counts = runner.object_counts(labels, segmentation)
    assert counts == {
        "gt_objects": 2, "severed_objects": 1, "merged": 2 - int(missed_id != 0),
        "non_severed_matches": int(missed_id != 1), "unmatched": int(missed_id != 0),
        "genuine_misses": int(missed_id == 1),
    }
