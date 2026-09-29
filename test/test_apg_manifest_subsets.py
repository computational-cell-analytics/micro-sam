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


def test_ood_extended_selection_is_stratified_and_uses_test_data(monkeypatch, tmp_path):
    def candidates(dataset, _root, split="val", validate_raw=False, skip_read_errors=False):
        assert split == "test" and validate_raw and skip_read_errors
        if dataset == "bitdepth_nucseg":
            strata = benchmark.OOD_EXTENDED_STRATUM_COUNTS[dataset]
            template = f"{dataset}/data/{{stratum}}/images/img{{index}}.tif"
        elif dataset == "cellbindb":
            strata = {key: value + 2 for key, value in benchmark.OOD_EXTENDED_STRATUM_COUNTS[dataset].items()}
            template = f"{dataset}/Other/{{stratum}}/sample{{index}}/img.tif"
        elif dataset == "vicar":
            strata = {key: value + 2 for key, value in benchmark.OOD_EXTENDED_STRATUM_COUNTS[dataset].items()}
            template = f"{dataset}/labelled/{{stratum}}/img{{index}}.tif"
        else:
            return _fake_candidates(dataset, benchmark.SAMPLE_COUNTS_2D_OOD_EXTENDED[dataset])
        result = []
        for stratum, count in strata.items():
            for index in range(count):
                sample = _fake_candidates(dataset, 1)[0]
                sample["raw_path"] = template.format(stratum=stratum, index=index)
                sample["label_path"] = sample["raw_path"].replace("img", "lab")
                sample["object_count"] = index + 1
                result.append(sample)
        return result

    monkeypatch.setattr(benchmark, "_scan_2d_dataset", candidates)
    samples = benchmark._select_ood_extended_samples(tmp_path)
    counts = {}
    strata = {}
    for sample in samples:
        counts[sample["dataset"]] = counts.get(sample["dataset"], 0) + 1
        if "stratum" in sample:
            key = (sample["dataset"], sample["stratum"])
            strata[key] = strata.get(key, 0) + 1
    assert counts == benchmark.SAMPLE_COUNTS_2D_OOD_EXTENDED
    assert strata == {
        (dataset, stratum): count
        for dataset, expected in benchmark.OOD_EXTENDED_STRATUM_COUNTS.items()
        for stratum, count in expected.items()
    }
    assert len({sample["sample_id"] for sample in samples}) == sum(counts.values()) == 180
