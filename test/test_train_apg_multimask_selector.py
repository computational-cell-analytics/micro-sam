import sys
from pathlib import Path

import numpy as np
import pytest


OPTIMIZATION_ROOT = Path(__file__).parents[1] / "finetuning/v2/evaluation/optimization"
sys.path.insert(0, str(OPTIMIZATION_ROOT))

trainer = pytest.importorskip("train_apg_multimask_selector")


def _dataset(path, n_groups, seed, datasets=("a", "b"), setting=None):
    from micro_sam.v2.multimask_selection import MULTIMASK_FEATURE_VERSION, SELECTOR_FEATURE_SCHEMAS
    rng = np.random.default_rng(seed)
    names = SELECTOR_FEATURE_SCHEMAS["lowres_v1"]
    features = rng.normal(size=(n_groups * 3, len(names))).astype("float32")
    targets = np.clip(features[:, 0] * 0.2 + 0.5 + rng.normal(scale=0.05, size=n_groups * 3), 0, 1).astype("float32")
    sample_ids = np.repeat([f"img{i}" for i in range(n_groups)], 3)
    groups = np.repeat([f"img{i}:{i}" for i in range(n_groups)], 3)
    alternatives = np.tile([0, 1, 2], n_groups).astype("int8")
    folds = np.repeat(np.arange(n_groups) % 5, 3).astype("int8")
    dataset_names = np.repeat([datasets[i % len(datasets)] for i in range(n_groups)], 3)
    np.savez(
        path, features=features, targets=targets, sample_ids=sample_ids, datasets=dataset_names, groups=groups,
        folds=folds, alternatives=alternatives, weights=np.ones(n_groups * 3, dtype="float32"),
        feature_version=np.asarray(MULTIMASK_FEATURE_VERSION), feature_names=np.asarray(names),
        input_schema=np.asarray("lowres_v1"), manifest_checksum=np.asarray("m"), n_alternatives=np.asarray(3),
        proposal_setting=np.asarray("{}" if setting is None else setting),
    )
    return path


def test_pooled_datasets_balance_weights_and_keep_dataset_names(tmp_path):
    first = _dataset(tmp_path / "one.npz", 40, 0)
    second = _dataset(tmp_path / "two.npz", 20, 1)
    pooled = trainer._load_pooled_datasets([first, second], None)
    assert pooled["features"].shape == (60, 3, 19)
    assert pooled["weights"][:40].sum() == pytest.approx(pooled["weights"][40:].sum())
    assert set(pooled["datasets"]) == {"a", "b"} and pooled["group_offsets"].tolist() == [0, 40, 60]


def test_train_selector_writes_per_input_oof_and_lodo(tmp_path):
    first = _dataset(tmp_path / "one.npz", 60, 0)
    second = _dataset(tmp_path / "two.npz", 30, 1)
    out = tmp_path / "models"
    artifact = trainer.train_selector([first, second], out, "cpu", hidden_size=8, lodo=True)
    assert artifact.name.endswith("-pooled2.pt")
    oof = np.load(out / f"{artifact.stem}_oof.npy")
    assert oof.shape == (270,)
    assert np.load(out / f"{artifact.stem}_oof_one.npy").shape == (180,)
    assert np.load(out / f"{artifact.stem}_oof_two.npy").shape == (90,)
    lodo = np.load(out / f"{artifact.stem}_lodo.npy")
    assert lodo.shape == (270,) and np.isfinite(lodo).all()
    assert np.load(out / f"{artifact.stem}_lodo_one.npy").shape == (180,)
    np.testing.assert_array_equal(np.load(out / f"{artifact.stem}_lodo_two.npy"), lodo[180:])
    import torch
    state = torch.load(artifact, map_location="cpu", weights_only=False)
    assert set(state["metadata"]["oof_metrics"]["lodo"]) == {"a", "b"}
    assert len(state["metadata"]["training_datasets"]) == 2


def test_incomplete_groups_are_padded_without_flat_rows(tmp_path):
    path = _dataset(tmp_path / "one.npz", 30, 0)
    data = dict(np.load(path, allow_pickle=False))
    # Drop the second alternative of the first prompt, as an empty mask would.
    keep = np.ones(len(data["targets"]), dtype=bool)
    keep[1] = False
    for key in ("features", "targets", "sample_ids", "datasets", "groups", "folds", "alternatives", "weights"):
        data[key] = data[key][keep]
    np.savez(tmp_path / "gappy.npz", **data)
    grouped = trainer._load_grouped_dataset(tmp_path / "gappy.npz")
    assert grouped["n_incomplete_groups"] == 1 and grouped["features"].shape == (30, 3, 19)
    assert grouped["rows"][0].tolist()[1] == -1 and (grouped["rows"][1:] >= 0).all()
    np.testing.assert_allclose(grouped["features"][0, 1], grouped["features"][0, [0, 2]].mean(axis=0), rtol=1e-5)
    assert grouped["targets"][0, 1] == 0.0
    artifact = trainer.train_selector([tmp_path / "gappy.npz"], tmp_path / "models", "cpu", hidden_size=8)
    oof = np.load(artifact.with_name(artifact.stem + "_oof.npy"))
    assert oof.shape == (89,) and np.isfinite(oof).all()


def test_generic_feature_subsets_standardization_and_linear_matched_variants(tmp_path):
    first = _dataset(tmp_path / "first.npz", 40, 3, datasets=("a", "b"))
    second = _dataset(tmp_path / "second.npz", 30, 4, datasets=("c",))
    out = tmp_path / "models"
    artifact = trainer.train_selector(
        [first, second], out, "cpu", hidden_size=8, lodo=True, feature_set="sam_scores",
        per_image="append", model_kind="linear", target_kind="matched",
    )
    assert artifact.name == "lowres_v1-groupwise-linear-matched-fs_sam_scores-z_append-pooled2.pt"
    import torch
    state = torch.load(artifact, weights_only=False)
    names = state["feature_names"]
    assert names[:7] == list(trainer.GENERIC_FEATURE_SETS["sam_scores"])
    assert names[7:] == [f"{name}_z" for name in names[:7]]
    assert state["kind"] == "groupwise_linear" and state["target"] == "matched"
    oof = np.load(out / f"{artifact.stem}_oof.npy")
    assert oof.shape == (70 * 3,) and np.all((oof >= 0) & (oof <= 1))
    results = trainer.json.load(open(out / f"{artifact.stem}_training_results.json"))
    lodo = results["metrics"]["lodo"]
    assert set(lodo) == {"a", "b", "c"}
    for entry in lodo.values():
        assert 0.0 <= entry["lodo_matched_auc"] <= 1.0
        assert entry["predicted_iou_matched_auc"] is not None
        assert entry["predicted_iou_selected_iou"] is not None


def test_per_image_standardization_is_zero_mean_per_image():
    features = np.asarray([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0], [2.0, 0.0], [4.0, 0.0]], dtype="float32")
    sample_ids = np.asarray(["x", "x", "x", "y", "y"])
    standardized = trainer._per_image_standardize(features, sample_ids)
    np.testing.assert_allclose(standardized[:3].mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(standardized[3:, 0], [-1.0, 1.0])
    np.testing.assert_allclose(standardized[3:, 1], 0.0)  # constant column keeps scale 1


def test_feature_set_missing_from_schema_raises(tmp_path):
    path = _dataset(tmp_path / "d.npz", 10, 5)
    grouped = trainer._load_grouped_dataset(path, feature_set="scale_free")
    assert grouped["features"].shape[-1] == len(trainer.GENERIC_FEATURE_SETS["scale_free"])
    assert "log_area" not in grouped["feature_names"]
