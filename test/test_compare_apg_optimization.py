import importlib.util
from pathlib import Path

import pandas as pd
import pytest


_MODULE_PATH = Path(__file__).parents[1] / "finetuning/v2/evaluation/optimization/compare_apg_optimization.py"
_SPEC = importlib.util.spec_from_file_location("compare_apg_optimization", _MODULE_PATH)
compare_apg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compare_apg)


def _comparison_input(config_name, peak_memory=None):
    datasets = sorted(compare_apg.EXPECTED_DATASETS[2])
    data = {
        "msa_mean": [0.8] * len(datasets),
        "total_seconds": [10.0] * len(datasets),
    }
    if peak_memory is not None:
        data["peak_cuda_memory_bytes"] = peak_memory
    return {"config_name": config_name}, pd.DataFrame(data, index=datasets)


def test_compare_preserves_all_null_peak_memory():
    baseline = _comparison_input("baseline")
    candidate = _comparison_input("candidate", [float("nan")] * len(compare_apg.EXPECTED_DATASETS[2]))

    _, rows = compare_apg._compare(baseline, candidate, target="quality", ndim=2)

    assert [row["candidate_peak_cuda_memory_bytes"] for row in rows] == [None] * len(rows)


def test_compare_serializes_measured_peak_memory_as_integers():
    baseline = _comparison_input("baseline")
    peaks = [1000, 2000, 3000, 4000, 5000]
    candidate = _comparison_input("candidate", peaks)

    _, rows = compare_apg._compare(baseline, candidate, target="quality", ndim=2)

    assert [row["candidate_peak_cuda_memory_bytes"] for row in rows] == peaks
    assert all(isinstance(row["candidate_peak_cuda_memory_bytes"], int) for row in rows)


def test_replacement_gate_allows_small_absolute_loss_for_near_zero_baseline():
    baseline = _comparison_input("baseline", [1000] * len(compare_apg.EXPECTED_DATASETS[2]))
    candidate = _comparison_input("candidate", [1000] * len(compare_apg.EXPECTED_DATASETS[2]))
    dataset = candidate[1].index[0]
    baseline[1].loc[dataset, "msa_mean"] = 0.044
    candidate[1].loc[dataset, "msa_mean"] = 0.040
    candidate[1]["total_seconds"] = 9.0

    decision, rows = compare_apg._compare(baseline, candidate, target="replacement", ndim=2)

    assert decision["checks"]["every_dataset_quality_loss_within_relative_or_absolute_limit"]
    assert next(row for row in rows if row["dataset"] == dataset)["msa_delta"] == pytest.approx(-0.004)


def test_refinement_gate_accepts_bounded_runtime_for_an_improving_candidate():
    baseline = _comparison_input("baseline", [1000] * len(compare_apg.EXPECTED_DATASETS[2]))
    candidate = _comparison_input("candidate", [1050] * len(compare_apg.EXPECTED_DATASETS[2]))
    candidate[1]["msa_mean"] = 0.81
    candidate[1]["total_seconds"] = [10.5, 10.6, 10.7, 10.8, 11.4]

    decision, _ = compare_apg._compare(baseline, candidate, target="refinement", ndim=2)

    assert decision["accepted"]
    assert all(decision["checks"].values())


def test_refinement_gate_rejects_a_single_dataset_runtime_above_15_percent():
    baseline = _comparison_input("baseline", [1000] * len(compare_apg.EXPECTED_DATASETS[2]))
    candidate = _comparison_input("candidate", [1000] * len(compare_apg.EXPECTED_DATASETS[2]))
    candidate[1]["msa_mean"] = 0.81
    candidate[1]["total_seconds"] = [10.0, 10.0, 10.0, 10.0, 11.6]

    decision, _ = compare_apg._compare(baseline, candidate, target="refinement", ndim=2)

    assert not decision["accepted"]
    assert not decision["checks"]["every_dataset_runtime_regression_at_most_15_percent"]
