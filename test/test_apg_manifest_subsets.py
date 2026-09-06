import sys
from pathlib import Path

import pytest


OPTIMIZATION_ROOT = Path(__file__).parents[1] / "finetuning/v2/evaluation/optimization"
sys.path.insert(0, str(OPTIMIZATION_ROOT))

benchmark = pytest.importorskip("benchmark_apg_optimization")


def _fake_candidates(dataset, n):
    return [
        {"dataset": dataset, "ndim": 2, "raw_path": f"{dataset}/img{i}.tif", "label_path": f"{dataset}/lab{i}.tif",
         "raw_key": None, "label_key": None, "roi": [[0, 512], [0, 512]], "object_count": 5 + i,
         "foreground_fraction": 0.1 + 0.01 * i}
        for i in range(n)
    ]


def test_training_extra_selection_caps_small_pools(monkeypatch, tmp_path):
    pools = {"yeaz": 50, "puma": 26}
    monkeypatch.setattr(benchmark, "_scan_2d_dataset", lambda dataset, root: _fake_candidates(dataset, pools[dataset]))
    counts = {"yeaz": 40, "puma": 40}
    with pytest.raises(RuntimeError, match="only 26"):
        benchmark._select_2d_samples(tmp_path, counts=counts, datasets=("yeaz", "puma"))
    samples = benchmark._select_2d_samples(tmp_path, counts=counts, datasets=("yeaz", "puma"), allow_fewer=True)
    by_dataset = {}
    for sample in samples:
        by_dataset[sample["dataset"]] = by_dataset.get(sample["dataset"], 0) + 1
    assert by_dataset == {"yeaz": 40, "puma": 26}
    assert len({sample["sample_id"] for sample in samples}) == 66


def test_sample_counts_and_subsets():
    assert benchmark._sample_counts_2d("training_extra") == benchmark.SAMPLE_COUNTS_2D_TRAINING_EXTRA
    assert set(benchmark.TRAINING_EXTRA_DATASETS).isdisjoint(benchmark.DATASETS_2D)
    with pytest.raises(ValueError):
        benchmark._sample_counts_2d("unknown")
