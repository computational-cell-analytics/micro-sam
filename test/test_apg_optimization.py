import sys
from pathlib import Path

import pytest
import pandas as pd


OPTIMIZATION_ROOT = Path(__file__).parents[1] / "finetuning/v2/evaluation/optimization"
sys.path.insert(0, str(OPTIMIZATION_ROOT))

from compare_apg_optimization import _compare, _validate_compatible, EXPECTED_DATASETS  # noqa


def _metadata(device="cuda:0", accelerator="NVIDIA A100"):
    return {
        "manifest_checksum": "manifest",
        "checkpoint_checksum": "checkpoint",
        "checkpoint_name": "best",
        "model_type": "hvit_t",
        "device": device,
        "hardware": {"accelerator": accelerator},
    }


def test_runtime_comparison_requires_matching_device_and_hardware():
    reference = _metadata()
    _validate_compatible([(reference, None), (_metadata(), None)])

    with pytest.raises(ValueError, match="Benchmark identities differ"):
        _validate_compatible([(reference, None), (_metadata(device="cpu"), None)])
    with pytest.raises(ValueError, match="Benchmark identities differ"):
        _validate_compatible([(reference, None), (_metadata(accelerator="NVIDIA H100"), None)])


def test_runtime_comparison_requires_hardware_metadata():
    incomplete = _metadata()
    del incomplete["hardware"]

    with pytest.raises(ValueError, match="missing required identity fields: hardware"):
        _validate_compatible([(incomplete, None), (_metadata(), None)])


def _comparison_table(peaks):
    datasets = sorted(("livecell", "tissuenet", "dynamicnuclearnet", "deepbacs", "dic_hepg2"))
    return pd.DataFrame({
        "msa_mean": [1.0] * len(datasets),
        "total_seconds": [1.0] * len(datasets),
        "peak_cuda_memory_bytes": peaks,
    }, index=datasets)


@pytest.mark.parametrize(
    "baseline_peaks,candidate_peaks,expected",
    [
        ([float("nan")] * 5, [float("nan")] * 5, True),
        ([100.0] + [float("nan")] * 4, [109.0] + [float("nan")] * 4, True),
        ([100.0] + [float("nan")] * 4, [111.0] + [float("nan")] * 4, False),
    ],
)
def test_memory_gate_uses_only_finite_paired_measurements(baseline_peaks, candidate_peaks, expected):
    baseline = ({"config_name": "baseline"}, _comparison_table(baseline_peaks))
    candidate = ({"config_name": "candidate"}, _comparison_table(candidate_peaks))

    decision, _ = _compare(baseline, candidate, target="refinement", ndim=2)

    assert decision["checks"]["peak_cuda_memory_increase_at_most_10_percent"] is expected


def _comparison_input(config_name, peak_memory=None):
    datasets = sorted(EXPECTED_DATASETS[2])
    data = {
        "msa_mean": [0.8] * len(datasets),
        "total_seconds": [10.0] * len(datasets),
    }
    if peak_memory is not None:
        data["peak_cuda_memory_bytes"] = peak_memory
    return {"config_name": config_name}, pd.DataFrame(data, index=datasets)


def test_compare_preserves_all_null_peak_memory():
    baseline = _comparison_input("baseline")
    candidate = _comparison_input("candidate", [float("nan")] * len(EXPECTED_DATASETS[2]))

    _, rows = _compare(baseline, candidate, target="quality", ndim=2)

    assert [row["candidate_peak_cuda_memory_bytes"] for row in rows] == [None] * len(rows)


def test_compare_serializes_measured_peak_memory_as_integers():
    baseline = _comparison_input("baseline")
    peaks = [1000, 2000, 3000, 4000, 5000]
    candidate = _comparison_input("candidate", peaks)

    _, rows = _compare(baseline, candidate, target="quality", ndim=2)

    assert [row["candidate_peak_cuda_memory_bytes"] for row in rows] == peaks
    assert all(isinstance(row["candidate_peak_cuda_memory_bytes"], int) for row in rows)


def test_replacement_gate_allows_small_absolute_loss_for_near_zero_baseline():
    baseline = _comparison_input("baseline", [1000] * len(EXPECTED_DATASETS[2]))
    candidate = _comparison_input("candidate", [1000] * len(EXPECTED_DATASETS[2]))
    dataset = candidate[1].index[0]
    baseline[1].loc[dataset, "msa_mean"] = 0.044
    candidate[1].loc[dataset, "msa_mean"] = 0.040
    candidate[1]["total_seconds"] = 9.0

    decision, rows = _compare(baseline, candidate, target="replacement", ndim=2)

    assert decision["checks"]["every_dataset_quality_loss_within_relative_or_absolute_limit"]
    assert next(row for row in rows if row["dataset"] == dataset)["msa_delta"] == pytest.approx(-0.004)


def test_refinement_gate_accepts_bounded_runtime_for_an_improving_candidate():
    baseline = _comparison_input("baseline", [1000] * len(EXPECTED_DATASETS[2]))
    candidate = _comparison_input("candidate", [1050] * len(EXPECTED_DATASETS[2]))
    candidate[1]["msa_mean"] = 0.81
    candidate[1]["total_seconds"] = [10.5, 10.6, 10.7, 10.8, 11.4]

    decision, _ = _compare(baseline, candidate, target="refinement", ndim=2)

    assert decision["accepted"]
    assert all(decision["checks"].values())


def test_refinement_gate_rejects_a_single_dataset_runtime_above_15_percent():
    baseline = _comparison_input("baseline", [1000] * len(EXPECTED_DATASETS[2]))
    candidate = _comparison_input("candidate", [1000] * len(EXPECTED_DATASETS[2]))
    candidate[1]["msa_mean"] = 0.81
    candidate[1]["total_seconds"] = [10.0, 10.0, 10.0, 10.0, 11.6]

    decision, _ = _compare(baseline, candidate, target="refinement", ndim=2)

    assert not decision["accepted"]
    assert not decision["checks"]["every_dataset_runtime_regression_at_most_15_percent"]
