import sys
from pathlib import Path

import pytest


OPTIMIZATION_ROOT = Path(__file__).parents[1] / "finetuning/v2/evaluation/optimization"
sys.path.insert(0, str(OPTIMIZATION_ROOT))

from compare_apg_optimization import _validate_compatible  # noqa


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
