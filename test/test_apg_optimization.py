import sys
from pathlib import Path

import pytest
import pandas as pd


OPTIMIZATION_ROOT = Path(__file__).parents[1] / "finetuning/v2/evaluation/optimization"
sys.path.insert(0, str(OPTIMIZATION_ROOT))

from compare_apg_optimization import _compare, _validate_compatible  # noqa


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
