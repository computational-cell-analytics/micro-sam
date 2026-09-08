import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


OPTIMIZATION_ROOT = Path(__file__).parents[1] / "finetuning/v2/evaluation/optimization"
sys.path.insert(0, str(OPTIMIZATION_ROOT))

import report_ais_checkpoint_comparison as comparison  # noqa


def _paired_rows(delta=0.02):
    rows = []
    for dataset, strata in {"plain": [""], "stratified": ["x", "y"]}.items():
        for stratum in strata:
            for index in range(4):
                row = {
                    "sample_id": f"{dataset}:{stratum}:{index}", "dataset": dataset, "stratum": stratum,
                    "msa_baseline": 0.5 + 0.01 * index, "msa_boundary": 0.5 + 0.01 * index + delta,
                    "gt_objects_baseline": 10, "gt_objects_boundary": 10,
                }
                for column in comparison.EXTENT_COLUMNS:
                    row[f"{column}_baseline"] = 0.7
                    row[f"{column}_boundary"] = 0.72
                for column in comparison.TIME_COLUMNS:
                    row[f"{column}_baseline"] = 1.0
                    row[f"{column}_boundary"] = 1.1
                for column in (*comparison.FATE_COLUMNS, *comparison.OTHER_COUNT_COLUMNS):
                    row[f"{column}_baseline"] = 1
                    row[f"{column}_boundary"] = 1
                rows.append(row)
    return pd.DataFrame(rows)


def test_balanced_scores_equal_weight_strata():
    paired = _paired_rows()
    group = paired[paired["dataset"] == "stratified"].copy()
    group.loc[group["stratum"] == "x", "msa_baseline"] = 0.1
    group.loc[group["stratum"] == "y", "msa_baseline"] = 0.9
    assert comparison.balanced_scores(group)[0] == pytest.approx(0.5)


def test_hierarchical_bootstrap_and_domain_table_are_paired():
    paired = _paired_rows(delta=0.03)
    overall, intervals = comparison.bootstrap(paired, n_bootstrap=500, seed=7)
    assert overall["absolute_ci_low"] == pytest.approx(0.03)
    assert overall["absolute_ci_high"] == pytest.approx(0.03)
    assert overall["probability_boundary_better"] == 1.0
    domains = comparison.dataset_table(paired, intervals)
    assert domains["improved"].all() and not domains["material_loss"].any()
    assert np.allclose(domains["absolute_delta"], 0.03)


def test_manifest_coverage_rejects_partial_or_wrong_stratum():
    paired = _paired_rows()
    manifest = {"samples": [
        {"sample_id": row.sample_id, "dataset": row.dataset, "stratum": row.stratum}
        for row in paired.itertuples()
    ]}
    comparison.validate_manifest_coverage(paired, manifest)
    with pytest.raises(ValueError, match="1 missing"):
        comparison.validate_manifest_coverage(paired.iloc[:-1], manifest)
    wrong = paired.copy()
    wrong.loc[0, "stratum"] = "wrong"
    with pytest.raises(ValueError, match="1 missing and 1 unexpected"):
        comparison.validate_manifest_coverage(wrong, manifest)


def test_training_disjointness_audit_detects_dataset_alias(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    raw = data_root / "ood/raw.tif"
    label = data_root / "ood/label.tif"
    raw.parent.mkdir()
    raw.touch()
    label.touch()
    manifest = {"samples": [{"dataset": "vicar", "raw_path": "ood/raw.tif", "label_path": "ood/label.tif"}]}
    training = tmp_path / "training.json"
    boundary_training = tmp_path / "boundary_training.json"
    training.write_text(json.dumps({"variant": "baseline", "datasets": {"train": {"train": []}}}))
    boundary_training.write_text(json.dumps({"variant": "boundary", "datasets": {"train": {"train": []}}}))
    audit = comparison.audit_training_disjointness(manifest, data_root, [training, boundary_training])
    assert audit["passed"] and not audit["dataset_overlap"]

    training.write_text(json.dumps({"variant": "baseline", "datasets": {"vicar_cells": {"train": []}}}))
    with pytest.raises(RuntimeError, match="overlaps decoder training"):
        comparison.audit_training_disjointness(manifest, data_root, [training, boundary_training])
