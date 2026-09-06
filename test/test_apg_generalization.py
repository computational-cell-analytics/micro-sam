import sys
from pathlib import Path

import pandas as pd
import pytest


OPTIMIZATION_ROOT = Path(__file__).parents[1] / "finetuning/v2/evaluation/optimization"
sys.path.insert(0, str(OPTIMIZATION_ROOT))

generalization = pytest.importorskip("evaluate_apg_generalization")


def _results(rows):
    return pd.DataFrame([{"config": c, "dataset": d, "seen": d in generalization.SEEN, "msa": m} for c, d, m in rows])


def test_tasks_cover_every_dataset_and_config(tmp_path):
    tasks = generalization.build_tasks(
        tmp_path, configs=["registry-defaults", "selector-only"], datasets=["livecell", "yeaz"],
    )
    tags = [tag for tag, _ in tasks]
    assert len(tags) == len(set(tags)) == 4
    commands = dict(tasks)
    assert "--skip_tuning" in commands["e1_registry-defaults_livecell"]
    assert "--apg_params" not in commands["e1_registry-defaults_livecell"]
    assert "--multimask_scorer_artifact" in commands["e1_selector-only_yeaz"]
    assert "--result_tag selector-only" in commands["e1_selector-only_yeaz"]


def test_compare_groups_seen_and_unseen_and_guards_near_zero_baselines():
    unseen = [d for d in generalization.unseen_datasets()][:2]
    rows = [
        ("registry-defaults", "livecell", 0.30), ("selector-only", "livecell", 0.36),
        ("registry-defaults", unseen[0], 0.50), ("selector-only", unseen[0], 0.56),
        # A near-zero baseline losing 40% relative but only 0.004 absolute is not a regression.
        ("registry-defaults", unseen[1], 0.010), ("selector-only", unseen[1], 0.006),
    ]
    decision = generalization.compare_production_results(_results(rows))
    entry = decision["candidates"]["selector-only"]
    assert entry["macros"]["seen"]["n_datasets"] == 1 and entry["macros"]["unseen"]["n_datasets"] == 2
    assert entry["regressions"] == []
    control, candidate = (0.50 + 0.010) / 2, (0.56 + 0.006) / 2
    assert entry["macros"]["unseen"]["relative_change"] == pytest.approx((candidate - control) / control, rel=1e-3)
    assert entry["accepted"] is True
    # A real unseen regression blocks acceptance.
    rows[-1] = ("selector-only", unseen[1], 0.001)
    rows[-2] = ("registry-defaults", unseen[1], 0.100)
    decision = generalization.compare_production_results(_results(rows))
    entry = decision["candidates"]["selector-only"]
    assert entry["regressions"] == [unseen[1]] and entry["accepted"] is False
