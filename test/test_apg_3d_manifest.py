import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


OPTIMIZATION_ROOT = Path(__file__).parents[1] / "finetuning/v2/evaluation/optimization"
sys.path.insert(0, str(OPTIMIZATION_ROOT))

manifest_module = pytest.importorskip("apg3d_manifest")


def _write_volume(path, labels):
    with h5py.File(path, "w") as f:
        f.create_dataset("raw", data=np.random.default_rng(0).integers(0, 255, labels.shape, dtype="uint8"))
        f.create_dataset("labels", data=labels)


def _labels(depth=16, size=32, annotated=slice(None), with_invalid=False):
    labels = np.zeros((depth, size, size), dtype="int64")
    labels[annotated, 2:10, 2:10] = 1
    labels[annotated, 2:10, 20:28] = 2
    labels[annotated, 20:28, 2:10] = 3
    if with_invalid:
        labels[:, 20:28, 20:28] = -1
    return labels


def _spec(data_root, names, crop=(8, 16, 16), holdout=(), mask_invalid=False, dataset="fake"):
    def paths(root):
        return [(str(root / f"{name}.h5"), str(root / f"{name}.h5")) for name in names]

    return manifest_module.SourceSpec(
        dataset, dataset, paths, "raw", "labels", manifest_module._full, lambda shape: crop,
        lambda path: Path(path).stem in holdout, seen_in_training=False, mask_invalid_labels=mask_invalid, target=6,
    )


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    _write_volume(root / "vol_a.h5", _labels())
    _write_volume(root / "vol_b.h5", _labels())
    # Annotated on slices 2..13 only: both 8-slice crops span 6 annotated slices, below the rule's
    # min(24, 8) = 8 for a crop this shallow.
    _write_volume(root / "vol_sparse.h5", _labels(annotated=slice(2, 14)))
    _write_volume(root / "vol_invalid.h5", _labels(with_invalid=True))
    return root


def test_grid_scan_yields_non_overlapping_crops_that_meet_the_depth_rule(data_root):
    spec = _spec(data_root, ["vol_a"])
    candidates = manifest_module._scan_source(spec, str(data_root / "vol_a.h5"), str(data_root / "vol_a.h5"), data_root)
    # 16 slices / 8 x (32 / 16)^2 = 2 x 4 grid positions; the lower-right quadrant holds no object.
    assert len(candidates) == 6
    rois = [manifest_module._roi_from_json(c["roi"]) for c in candidates]
    for index, first in enumerate(rois):
        for second in rois[index + 1:]:
            assert not all(a.start < b.stop and b.start < a.stop for a, b in zip(first, second))
    assert all(c["realized_depth"] == 8 and c["depth_flag"] == "shallow-8" for c in candidates)
    assert sorted(c["object_count"] for c in candidates) == [1, 1, 1, 1, 1, 1]


def test_depth_rule_rejects_crops_the_loader_would_trim_too_far(data_root):
    spec = _spec(data_root, ["vol_sparse"])
    candidates = manifest_module._scan_source(
        spec, str(data_root / "vol_sparse.h5"), str(data_root / "vol_sparse.h5"), data_root,
    )
    # Slices 2..13 are annotated: the z=0 crop spans 6 of 8 and the z=8 crop 6 of 8; both fall short.
    assert candidates == []


def test_invalid_labels_are_masked_before_counting(data_root):
    spec = _spec(data_root, ["vol_invalid"], mask_invalid=True)
    candidates = manifest_module._scan_source(
        spec, str(data_root / "vol_invalid.h5"), str(data_root / "vol_invalid.h5"), data_root,
    )
    assert candidates and all(c["mask_invalid_labels"] for c in candidates)
    lower_right = [c for c in candidates if manifest_module._roi_from_json(c["roi"])[1].start == 16
                   and manifest_module._roi_from_json(c["roi"])[2].start == 16]
    # The -1 block is not an object.
    assert all(c["object_count"] == 0 for c in lower_right) or not lower_right


def test_holdout_split_and_folds_are_grouped_by_source(data_root, monkeypatch):
    spec = _spec(data_root, ["vol_a", "vol_b"], holdout=("vol_b",))
    monkeypatch.setattr(manifest_module, "tuning_source_specs", lambda: [spec])
    monkeypatch.setattr(manifest_module, "LEGACY_MANIFESTS", ())
    primary = manifest_module.build_manifest(data_root, "primary")
    holdout = manifest_module.build_manifest(data_root, "holdout")
    assert {s["source_id"] for s in primary["samples"]} == {"vol_a.h5"}
    assert {s["source_id"] for s in holdout["samples"]} == {"vol_b.h5"}
    assert len({s["fold"] for s in primary["samples"]}) == 1
    manifest_module.validate_manifest(holdout, data_root, primary=primary)
    # Re-selecting from the same pool is deterministic.
    assert manifest_module.build_manifest(data_root, "primary")["manifest_checksum"] == primary["manifest_checksum"]


def test_validation_catches_edits_snemi_leaks_and_shared_sources(data_root, monkeypatch):
    spec = _spec(data_root, ["vol_a", "vol_b"], holdout=("vol_b",))
    monkeypatch.setattr(manifest_module, "tuning_source_specs", lambda: [spec])
    monkeypatch.setattr(manifest_module, "LEGACY_MANIFESTS", ())
    primary = manifest_module.build_manifest(data_root, "primary")
    edited = json.loads(json.dumps(primary))
    edited["samples"][0]["object_count"] = 99
    with pytest.raises(RuntimeError, match="checksum"):
        manifest_module.validate_manifest(edited, data_root)
    leaky = json.loads(json.dumps(primary))
    leaky["samples"][0]["dataset"] = "snemi"
    leaky["samples"][0]["roi"] = [[80, 88], [0, 16], [0, 16]]
    leaky["manifest_checksum"] = manifest_module._content_checksum(manifest_module._identity(leaky))
    with pytest.raises(RuntimeError, match="evaluated slab"):
        manifest_module.validate_manifest(leaky, data_root)
    shared = json.loads(json.dumps(primary))
    with pytest.raises(RuntimeError, match="shares a source"):
        manifest_module.validate_manifest(shared, data_root, primary=primary)


def test_load_sample_trims_to_the_annotated_span_and_masks_invalid_voxels(data_root):
    sample = {
        "sample_id": "fake:x", "raw_path": "vol_invalid.h5", "label_path": "vol_invalid.h5", "raw_key": "raw",
        "label_key": "labels", "roi": [[0, 8], [0, 32], [0, 32]], "normalization_z_range": [0, 16],
        "mask_invalid_labels": True,
    }
    source = manifest_module.load_normalized_source(sample, data_root)
    assert source.shape == (16, 32, 32) and source.dtype == np.float32
    raw, labels, valid = manifest_module.load_sample(sample, data_root, source)
    assert raw.shape == labels.shape == (8, 32, 32)
    assert valid is not None and not valid[:, 20:28, 20:28].any() and valid[:, 2:10, 2:10].all()
    assert labels.max() == 3 and (labels[:, 20:28, 20:28] == 0).all()


def test_load_labels_matches_load_sample(data_root):
    sample = {
        "sample_id": "fake:y", "raw_path": "vol_invalid.h5", "label_path": "vol_invalid.h5", "raw_key": "raw",
        "label_key": "labels", "roi": [[4, 12], [0, 32], [0, 32]], "normalization_z_range": [0, 16],
        "mask_invalid_labels": True,
    }
    source = manifest_module.load_normalized_source(sample, data_root)
    _, labels, valid = manifest_module.load_sample(sample, data_root, source)
    expected = labels.copy()
    expected[~valid] = 0
    np.testing.assert_array_equal(manifest_module.load_labels(sample, data_root), expected)
