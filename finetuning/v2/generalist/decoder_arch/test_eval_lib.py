"""Tests of the evaluation building blocks: python -m pytest -o addopts="" test_eval_lib.py"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_lib as ev  # noqa: E402


def _disks(shape, centers, radius):
    labels = np.zeros(shape, dtype="uint32")
    grid = np.indices(shape)
    for i, center in enumerate(centers, 1):
        distance = sum((g - c) ** 2 for g, c in zip(grid, center))
        labels[(distance <= radius ** 2) & (labels == 0)] = i
    return labels


def _prediction(labels, seed=0):
    """A noisy version of the training target of 'labels': foreground, three directed distances, boundary."""
    from micro_sam.v2.transforms.labels import GeodesicHybridDistanceTransform
    target = np.asarray(GeodesicHybridDistanceTransform(with_boundaries=True)(labels), dtype="float32")
    if labels.ndim == 2:
        target = np.concatenate([target[:1], np.zeros_like(target[:1]), target[1:]], axis=0)[:5]
    rng = np.random.default_rng(seed)
    noisy = target + rng.normal(0, 0.15, target.shape).astype("float32")
    noisy[0], noisy[4] = np.clip(noisy[0], 0, 1), np.clip(noisy[4], 0, 1)
    noisy[1:4] = np.clip(noisy[1:4], -1, 1)
    return noisy


@pytest.fixture(scope="module")
def sample_2d():
    labels = _disks((128, 128), [(30, 30), (30, 52), (90, 90), (64, 20), (100, 40)], 12)
    return _prediction(labels), labels


def test_prediction_layout(sample_2d):
    prediction, labels = sample_2d
    assert prediction.shape == (5, *labels.shape)


@pytest.mark.parametrize("valid", [False, True])
def test_score_grid_matches_library(sample_2d, valid):
    prediction, labels = sample_2d
    valid_roi = None
    if valid:
        valid_roi = np.ones(labels.shape, bool)
        valid_roi[:, :20] = False
    params_list = [
        {"foreground_threshold": ft, "sigma": sigma, "n_iter": n_iter, "dt": 1.0, "density_threshold": dens,
         "foreground_weight": fw, "min_size": min_size}
        for ft in (0.4, 0.6) for sigma in (0.5, 1.0) for n_iter in (25, 100) for dens in (5.0, 20.0)
        for fw in (0.5, 0.9) for min_size in (10, 100)
    ]
    scores = ev.score_grid(prediction, labels, params_list, valid_roi=valid_roi, border_min_size=10)
    assert scores.shape == (len(params_list),) and np.isfinite(scores).all()
    for i in range(0, len(params_list), 7):
        seg = ev.mask_segmentation(ev.flow_segmentation(prediction, params_list[i]), valid_roi, 10)
        assert scores[i] == pytest.approx(ev.msa_scores(seg, labels)["msa"], abs=1e-12)
    assert scores.max() > 0.5  # a clean target segments well with some combination


def test_score_grid_3d():
    labels = _disks((12, 64, 64), [(6, 20, 20), (6, 44, 44), (6, 20, 46)], 9)
    prediction = _prediction(labels)
    params_list = [
        {"foreground_threshold": 0.5, "sigma": 0.5, "n_iter": n, "dt": 1.0, "density_threshold": d,
         "foreground_weight": 0.9, "min_size": 50}
        for n in (25, 50) for d in (20.0, 80.0)
    ]
    scores = ev.score_grid(prediction, labels, params_list)
    for i, params in enumerate(params_list):
        seg = ev.flow_segmentation(prediction, params)
        assert scores[i] == pytest.approx(ev.msa_scores(seg, labels)["msa"], abs=1e-12)


def test_object_errors():
    labels = np.zeros((40, 40), "uint32")
    labels[0:10, 0:10], labels[0:10, 10:20], labels[20:30, 0:10], labels[30:40, 30:40] = 1, 2, 3, 4
    seg = np.zeros_like(labels)
    seg[0:10, 0:20] = 1  # merges 1 and 2
    seg[20:25, 0:10], seg[25:30, 0:10] = 2, 3  # splits 3; 4 is missed
    errors = ev.object_errors(seg, labels)
    assert errors == {"n_gt": 4, "merged": 0.5, "split": 0.25, "missed": 0.25}
    assert ev.object_errors(labels, labels) == {"n_gt": 4, "merged": 0.0, "split": 0.0, "missed": 0.0}


def test_field_diagnostics(sample_2d):
    prediction, labels = sample_2d
    clean = np.zeros_like(prediction)
    clean[0] = labels > 0
    from skimage.segmentation import find_boundaries
    clean[4] = find_boundaries(labels, mode="inner")
    diagnostics = ev.field_diagnostics(clean, labels)
    assert diagnostics == {"fg_iou": 1.0, "boundary_recall": 1.0}


def test_grid_contains_defaults():
    for ndim in (2, 3):
        combos = ev.grid_params(ndim)
        keys = {ev.params_key(c) for c in combos}
        assert ev.params_key(ev.default_params(ndim)) in keys
        assert len(keys) == len(combos)


def test_edge_extension():
    swept = {k: list(v) for k, v in ev.GRIDS[2].items()}
    inner = {k: v[1] if len(v) > 1 else v[0] for k, v in swept.items()}
    assert ev.edge_extension(ev.normalize_params(inner), swept) == []
    edge = dict(inner, density_threshold=50.0, foreground_threshold=0.3)
    extension = ev.edge_extension(ev.normalize_params(edge), swept)
    changed = sorted((k, e[k]) for e in extension for k in ev.PARAM_KEYS if e[k] != ev.normalize_params(edge)[k])
    assert changed == [("density_threshold", 100.0), ("foreground_threshold", 0.2)]


def test_select_params():
    scores = {
        "a": np.array([[0.5, 0.5], [0.6, 0.6], [0.2, 0.2]]),  # best: combo 1
        "b": np.array([[0.30, 0.30], [0.10, 0.10], [0.29, np.nan]]),  # best: combo 0
    }
    winner, relative, means, _ = ev.select_params(scores)
    assert winner == 0  # relative (0.5 / 0.6 + 1) / 2 beats (1 + 1 / 3) / 2
    assert relative[1] == pytest.approx(1 / 2 + 1 / 6)
    assert means[1, 2] == pytest.approx(0.29)
    # Grouped: 'a' and 'b' form one group, 'c' another; c's best is combo 2.
    scores["c"] = np.array([[0.1], [0.1], [0.4]])
    winner, relative, _, group_means = ev.select_params(scores, {"ab": ("a", "b"), "c": ("c",)})
    assert group_means[0, 0] == pytest.approx(0.4) and group_means[1, 2] == pytest.approx(0.4)
    assert relative[2] == pytest.approx((0.245 / 0.4 + 1) / 2) and winner == 2


def test_paired_bootstrap():
    rng = np.random.default_rng(1)
    reference = rng.uniform(0.3, 0.7, 200)
    same = ev.paired_bootstrap([reference], [reference])
    assert same["diff"] == 0 and same["diff_lo"] == 0 and same["diff_hi"] == 0
    better = ev.paired_bootstrap([reference + 0.05], [reference])
    assert better["diff"] == pytest.approx(0.05) and better["diff_lo"] == pytest.approx(0.05)
    noisy = ev.paired_bootstrap([reference + rng.normal(0.02, 0.05, 200)], [reference])
    assert noisy["diff_lo"] < noisy["diff"] < noisy["diff_hi"]
    two = ev.paired_bootstrap([reference[:50] + 0.1, reference[50:]], [reference[:50], reference[50:]])
    assert two["diff"] == pytest.approx(0.05)
    weighted = ev.paired_bootstrap([reference[:50] + 0.1, reference[50:]], [reference[:50], reference[50:]],
                                   weights=[1, 3])
    assert weighted["diff"] == pytest.approx(0.025)
    assert weighted["rel"] == pytest.approx(0.025 / weighted["reference"])


def test_tile_starts():
    assert ev._tile_starts(0, 300, 512) == [(0, 300)]
    assert ev._tile_starts(0, 1330, 512) == [(0, 512), (512, 512), (818, 512)]
    assert ev._tile_starts(160, 200, 32) == [(160, 32), (168, 32)]


def test_dataset_lists():
    assert len(ev.REPORTED_DATASETS) == 12
    assert len(ev.SCORED_TEST_DATASETS) == 15
    assert {ev.ndim_of(d) for d in ev.TEST_GROUPS["3d_external"]} == {3}
    assert {ev.ndim_of(d) for d in ev.TEST_GROUPS["2d_external"]} == {2}


def _fake_run(root, run, offset, rng, loss_dataset=None):
    """Test results of a run whose per-sample mSA is a shared base plus 'offset'."""
    import csv
    import json
    out = os.path.join(root, run, "test")
    os.makedirs(out)
    for k, dataset in enumerate(ev.SCORED_TEST_DATASETS):
        base = np.random.default_rng(k).uniform(0.3, 0.7, 20)
        shift = offset - (0.2 if dataset == loss_dataset else 0.0)
        with open(os.path.join(out, f"{dataset}.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample", "n_gt", "fg_iou", "boundary_recall", "msa_default", "msa_tuned",
                             "merged_tuned", "split_tuned", "missed_tuned", "n_pred_tuned"])
            for i, b in enumerate(base):
                writer.writerow([i, 10, 0.8, 0.5, b, b + shift + rng.normal(0, 0.001), 0.1, 0.05, 0.02, 11])
        with open(os.path.join(out, f"{dataset}.json"), "w") as f:
            json.dump({"limit": None}, f)


def test_report(tmp_path):
    import subprocess
    rng = np.random.default_rng(0)
    _fake_run(tmp_path, "armA_current_seed0", 0.0, rng)
    _fake_run(tmp_path, "armA_current_seed1", 0.005, rng)
    _fake_run(tmp_path, "armB_current_skips_seed0", 0.03, rng)
    _fake_run(tmp_path, "armC_hiera_u_seed0", 0.03, rng, loss_dataset="cellpose")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_decoder_arch.py")
    subprocess.run([sys.executable, script, "--eval-root", str(tmp_path), "--n-boot", "200"], check=True)
    import csv
    with open(tmp_path / "report" / "scores.csv") as f:
        scores = {(r["run"], r["params"]): r for r in csv.DictReader(f)}
    a, b = scores[("armA_current_seed0", "tuned")], scores[("armB_current_skips_seed0", "tuned")]
    assert float(b["headline"]) - float(a["headline"]) == pytest.approx(0.03, abs=1e-3)
    assert float(a["embedseg"]) == pytest.approx(np.mean([float(a[c]) for c in ev.EMBEDSEG_COLLECTIONS]))
    report = (tmp_path / "report" / "report.md").read_text()
    assert "`armB_current_skips_seed0`: **beats the reference**" in report
    assert "`armC_hiera_u_seed0`: **does not beat the reference**" in report and "losing > 5 %: cellpose" in report
    assert "headline 0.005" in report
    assert "`armA_current_seed1`: **seed replicate" in report
