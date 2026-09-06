import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


OPTIMIZATION_ROOT = Path(__file__).parents[1] / "finetuning/v2/evaluation/optimization"
sys.path.insert(0, str(OPTIMIZATION_ROOT))

structural = pytest.importorskip("screen_apg_structural")


def test_variant_grid_is_fixed_and_starts_with_the_registry_control():
    grid = structural.variant_grid()
    assert next(iter(grid)) == "registry" and grid["registry"] == {"prompt_type": "point", "select": {}}
    assert {variant["prompt_type"] for variant in grid.values()} <= set(structural.PROMPT_TYPES)
    for name, variant in grid.items():
        assert set(variant["select"]) <= {
            "fusion", "arbitration", "max_overlap", "recover_residual", "score_threshold",
        }, name
    assert "adaptive-fg-agreement" in grid and grid["adaptive-fg-agreement"]["adaptive"] == [0.4, 0.5, 0.6, 0.7]


def test_object_recall_counts_seeded_and_proposed_objects():
    labels = np.zeros((32, 32), dtype="uint32")
    labels[2:10, 2:10] = 1
    labels[20:30, 20:30] = 2
    good = np.ones((8, 8), dtype=bool)
    poor = np.zeros((10, 10), dtype=bool)
    poor[:3, :3] = True
    records = [
        {"point": (5.0, 5.0), "bounding_box": (slice(2, 10), slice(2, 10)), "segmentation": good},
        {"point": (25.0, 25.0), "bounding_box": (slice(20, 30), slice(20, 30)), "segmentation": poor},
        {"point": (15.0, 15.0), "bounding_box": (slice(14, 16), slice(14, 16)), "segmentation": np.ones((2, 2), bool)},
    ]
    seeded, proposed = structural.object_recall_counts(records, labels)
    assert (seeded, proposed) == (2, 1)


def test_foreground_agreement_is_the_dice_with_the_predicted_foreground():
    segmentation = np.zeros((8, 8), dtype="uint32")
    segmentation[:4] = 1
    foreground = np.zeros((8, 8), dtype="float32")
    foreground[:, :4] = 0.9
    assert structural.foreground_agreement(segmentation, foreground) == pytest.approx(0.5)
    assert structural.foreground_agreement(np.zeros((8, 8), "uint32"), np.zeros((8, 8), "float32")) == 1.0


def _summary(values):
    rows = []
    for variant, per_dataset in values.items():
        for dataset, msa in per_dataset.items():
            rows.append({
                "variant": variant, "dataset": dataset, "msa_mean": msa, "predicted_objects": 10, "gt_objects": 10,
            })
    return pd.DataFrame(rows)


def test_gate_table_requires_most_datasets_up_no_regression_and_a_balanced_gain():
    datasets = [f"d{i}" for i in range(11)]
    registry = {dataset: 0.3 for dataset in datasets}
    winner = {dataset: 0.3 * 1.03 for dataset in datasets}
    winner["d0"] = 0.3 * 0.99  # one minor loss
    winner["d1"] = 0.3
    loser = dict(winner)
    loser["d2"] = 0.3 * 0.9  # a real regression
    flat = {dataset: 0.3 * 1.005 for dataset in datasets}
    gates = structural.gate_table(_summary({"registry": registry, "winner": winner, "loser": loser, "flat": flat}))
    gates = gates.set_index("variant")
    assert bool(gates.loc["winner", "gate"]) is True
    assert gates.loc["winner", "datasets_up"] == 9 and gates.loc["winner", "regressions"] == ""
    assert bool(gates.loc["loser", "gate"]) is False and gates.loc["loser", "regressions"] == "d2"
    assert bool(gates.loc["flat", "gate"]) is False
    assert bool(gates.loc["registry", "gate"]) is False


def test_identity_check_compares_the_registry_replay_per_image():
    replay = pd.DataFrame([
        {"sample_id": "a", "variant": "registry", "msa": 0.5, "predicted_objects": 3},
        {"sample_id": "b", "variant": "registry", "msa": 0.25, "predicted_objects": 2},
        {"sample_id": "a", "variant": "fusion-both", "msa": 0.9, "predicted_objects": 4},
    ])
    reference = pd.DataFrame([
        {"sample_id": "a", "msa": 0.5, "predicted_objects": 3}, {"sample_id": "b", "msa": 0.25, "predicted_objects": 2},
    ])
    check = structural.identity_check(replay, reference)
    assert check["identical"] is True and check["n_compared"] == 2
    reference.loc[1, "msa"] = 0.26
    assert structural.identity_check(replay, reference)["identical"] is False
