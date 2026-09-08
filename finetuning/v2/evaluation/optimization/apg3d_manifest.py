"""Build, validate and load the deep 3d crop manifests of the APG 3d campaign.

The stopped 3d campaign tuned on 32 crops, 22 of them from one dataset, and its SNEMI crops
overlapped the slab the production evaluation scores. This manifest draws deep crops from every 3d
source that `common.VAL_SPLITS` allows tuning on, balances the per-dataset counts, keeps the holdout
source-disjoint, and refuses any SNEMI crop that touches the evaluated slab. Datasets without a
validation split (see `TEST_ONLY_DATASETS`) form a `test` subset that is opened once at the end.

Every crop records whether its source was joint-training data (`seen_in_training`), so every
learned result can be reported for all sources and for unseen sources alone.

Usage examples:
    python apg3d_manifest.py build --subset primary
    python apg3d_manifest.py build --subset holdout
    python apg3d_manifest.py build --subset test
    python apg3d_manifest.py show --subset primary
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from skimage.measure import label as connected_components
from tqdm import tqdm

EVALUATION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVALUATION_ROOT))

import common  # noqa
from common import DATASET_SPACING, PLATYNEREIS_NUCLEI_VAL_SAMPLES, get_data_paths  # noqa
from optimization.benchmark_apg_optimization import (  # noqa
    DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, _add_complexity, _array_shape, _atomic_write_json, _content_checksum,
    _object_statistics, _quantile_targets, _read_array, _relative_data_path, _roi_from_json, _roi_to_json,
    _select_nearest, _source_path,
)

SCHEMA = "apg3d-v1"
SUBSETS = ("primary", "holdout", "test")
CAMPAIGN_ROOT = DEFAULT_OUTPUT_ROOT / "3d_v2"
DEEP_DEPTH = 32
# A crop counts as deep once the annotated span the loader keeps reaches this many slices.
MIN_REALIZED_DEPTH = 24
OBJECTS_PER_PASS = 16
# The production evaluation scores SNEMI on original slices 81:89 (`common.load_volume`: drop z < 70,
# center-crop 8 of the remaining 30). Tuning crops must stay clear of it.
SNEMI_TEST_SLAB = (81, 89)
SEED = 17
TEST_ONLY_DATASETS = ("blastospim", "cartocell", "cellseg_3d", "mouse_embryo", "nis3d", "plantseg", "pnas_arabidopsis")
LEGACY_MANIFESTS = (
    DEFAULT_OUTPUT_ROOT / "3d_campaign" / "manifest_primary_v1.json",
    DEFAULT_OUTPUT_ROOT / "3d_campaign" / "manifest_holdout_v1.json",
)


@dataclasses.dataclass
class SourceSpec:
    """One source of crops: a dataset (or sub-dataset) with its legal slab, crop shape and split rule."""

    dataset: str
    family: str
    paths: Callable[[Path], List[Tuple[str, str]]]
    raw_key: Optional[str]
    label_key: Optional[str]
    legal_z: Callable[[Tuple[int, ...]], List[Tuple[int, int]]]
    crop_shape: Callable[[Tuple[int, ...]], Tuple[int, int, int]]
    holdout: Callable[[str], bool]
    seen_in_training: Any = False
    metric_mode: str = "sparse"
    mask_invalid_labels: bool = False
    target: int = 8
    z_step: Optional[int] = None
    subset: str = "primary"
    # Cap on the source volumes scanned (deterministic by hash), for datasets with hundreds of volumes.
    max_sources: Optional[int] = None


def _pairs(images: Sequence[str], labels: Sequence[str]) -> List[Tuple[str, str]]:
    return list(zip(sorted(images), sorted(labels)))


def _same(paths: Sequence[str]) -> List[Tuple[str, str]]:
    return [(path, path) for path in sorted(paths)]


def _full(shape: Tuple[int, ...]) -> List[Tuple[int, int]]:
    return [(0, int(shape[0]))]


def _deep_xy(shape: Tuple[int, ...], depth: int = DEEP_DEPTH) -> Tuple[int, int, int]:
    return (min(depth, int(shape[0])), min(512, int(shape[1])), min(512, int(shape[2])))


def _embedseg(name: str, data_root: Path) -> List[Tuple[str, str]]:
    from torch_em.data.datasets import embedseg_data
    images, labels = embedseg_data.get_embedseg_paths(str(data_root / "embedseg"), name=name, split="train")
    return _pairs(images, labels)


def _celegans(data_root: Path) -> List[Tuple[str, str]]:
    images, labels, _, _ = get_data_paths("celegans_atlas", str(data_root), split="val")
    return _pairs(images, labels)


def _gonuclear(data_root: Path) -> List[Tuple[str, str]]:
    paths, _, _, _ = get_data_paths("gonuclear", str(data_root), split="val")
    return _same(paths)


def _cremi(samples: Tuple[str, ...]) -> Callable[[Path], List[Tuple[str, str]]]:
    def paths(data_root: Path) -> List[Tuple[str, str]]:
        from torch_em.data.datasets import cremi
        return _same(cremi.get_cremi_paths(str(data_root / "cremi"), samples=samples))
    return paths


def _snemi(data_root: Path) -> List[Tuple[str, str]]:
    paths, _, _, _ = get_data_paths("snemi", str(data_root), split="val")
    return _same(paths)


def _humanneurons(data_root: Path) -> List[Tuple[str, str]]:
    return _same(sorted(glob(str(data_root / "humanneurons" / "*.h5"))))


def _platynereis_nuclei(data_root: Path) -> List[Tuple[str, str]]:
    paths, _, _, _ = get_data_paths("platynereis_nuclei", str(data_root), split="val")
    return _same(paths)


def _snemi_legal_z(shape: Tuple[int, ...]) -> List[Tuple[int, int]]:
    # Original slices 70:81 and 89:100: what is held out from training and from the evaluated slab.
    return [(70, SNEMI_TEST_SLAB[0]), (SNEMI_TEST_SLAB[1], int(shape[0]))]


def _platynereis_legal_z(raw_path: str) -> Callable[[Tuple[int, ...]], List[Tuple[int, int]]]:
    def legal(shape: Tuple[int, ...]) -> List[Tuple[int, int]]:
        return [common.platynereis_nuclei_val_z_range(raw_path)]
    return legal


def _basename_is(*names: str) -> Callable[[str], bool]:
    return lambda path: Path(path).stem in names


def _basename_contains(*parts: str) -> Callable[[str], bool]:
    return lambda path: any(part in Path(path).name for part in parts)


def tuning_source_specs() -> List[SourceSpec]:
    """The sources tuning may use, with the rule that decides which of their volumes are holdout."""
    return [
        SourceSpec(
            "celegans_atlas", "celegans", _celegans, None, None, _full,
            lambda shape: (DEEP_DEPTH, 140, 512), _basename_is("pha4I2L_0408071"), seen_in_training=False,
            target=10,
        ),
        SourceSpec(
            "embedseg_skull", "embedseg", lambda root: _embedseg("Mouse-Skull-Nuclei-CBG", root), None, None, _full,
            lambda shape: _deep_xy(shape), _basename_is("X1"), seen_in_training=True,
        ),
        # Mouse-Organoid-Cells-CBG is left out: its label volumes annotate only four cells each over a
        # 19-slice span, so a dense evaluation would count every unannotated cell as a false positive.
        SourceSpec(
            "embedseg_platy_nuclei", "embedseg", lambda root: _embedseg("Platynereis-Nuclei-CBG", root), None, None,
            _full, lambda shape: _deep_xy(shape), _basename_contains("dataset_hdf5_000"), seen_in_training=True,
        ),
        SourceSpec(
            "embedseg_platy_ish", "embedseg", lambda root: _embedseg("Platynereis-ISH-Nuclei-CBG", root), None, None,
            _full, lambda shape: _deep_xy(shape), _basename_contains("X02"), seen_in_training=True,
        ),
        SourceSpec(
            "gonuclear", "gonuclear", _gonuclear, "raw/nuclei", "labels/nuclei",
            lambda shape: [common.VAL_Z_RANGE["gonuclear"]], lambda shape: _deep_xy(shape),
            _basename_is("1170"), seen_in_training=False, target=10,
        ),
        SourceSpec(
            "cremi", "cremi", _cremi(("C",)), "volumes/raw", "volumes/labels/neuron_ids",
            lambda shape: [common.VAL_Z_RANGE["cremi"]], lambda shape: _deep_xy(shape),
            # Sample C is the only unseen CREMI volume, so its holdout is one quadrant; see _holdout_rois.
            lambda path: False, seen_in_training=False, metric_mode="dense", target=4,
        ),
        SourceSpec(
            "cremi_seen", "cremi", _cremi(("A", "B")), "volumes/raw", "volumes/labels/neuron_ids",
            lambda shape: [common.VAL_Z_RANGE["cremi"]], lambda shape: _deep_xy(shape),
            _basename_is("sampleB"), seen_in_training=True, metric_mode="dense", target=4,
        ),
        SourceSpec(
            "snemi", "snemi", _snemi, "volumes/raw", "volumes/labels/neuron_ids", _snemi_legal_z,
            lambda shape: (11, 512, 512), lambda path: False, seen_in_training=False, metric_mode="dense", target=6,
        ),
        SourceSpec(
            "humanneurons", "humanneurons", _humanneurons, "raw", "labels",
            lambda shape: [common.VAL_Z_RANGE["humanneurons"]], lambda shape: (16, 512, 512),
            lambda path: False, seen_in_training=False, metric_mode="dense", target=8,
        ),
        SourceSpec(
            "platynereis_nuclei", "platynereis_nuclei", _platynereis_nuclei, "volumes/raw",
            "volumes/labels/nucleus_instance_labels", None, lambda shape: _deep_xy(shape, depth=16),
            _basename_contains("nuclei_08"), seen_in_training="maybe", mask_invalid_labels=True, target=4,
        ),
    ]


# Single-source datasets hold out one spatial quadrant instead of a volume. (y start, x start) of the
# quadrant that becomes holdout; every crop overlapping it is excluded from primary.
QUADRANT_HOLDOUT = {
    "cremi": (738, 738),
    "snemi": (512, 512),
    "humanneurons": (1536, 1536),
}


def _test_source_specs() -> List[SourceSpec]:
    """The production-evaluation datasets without a validation split, used once at the very end."""
    specs = []
    for dataset in TEST_ONLY_DATASETS:
        def paths(data_root: Path, dataset=dataset) -> List[Tuple[str, str]]:
            images, labels, _, _ = get_data_paths(dataset, str(data_root), split="test")
            return _pairs(images, labels)

        specs.append(SourceSpec(
            dataset, dataset, paths, None, None, _full, lambda shape: _deep_xy(shape),
            lambda path: False, seen_in_training=False, target=8, subset="test", max_sources=12,
        ))
    return specs


def _keys_for_test_dataset(dataset: str, data_root: Path) -> Tuple[Optional[str], Optional[str]]:
    _, _, raw_key, label_key = get_data_paths(dataset, str(data_root), split="test")
    return raw_key, label_key


# ----------------------------------------------------------------------------------------------
# scanning


def _grid_starts(start: int, stop: int, size: int, step: Optional[int] = None) -> List[int]:
    step = step or size
    if stop - start < size:
        return []
    starts = list(range(start, stop - size + 1, step))
    return starts or [start]


def _read_labels(path: Path, key: Optional[str], roi: Tuple[slice, ...], mask_invalid: bool) -> np.ndarray:
    labels = _read_array(path, key, roi=roi).astype("int64")
    if mask_invalid:
        labels[labels == -1] = 0
    return labels


def _capped_sources(spec: SourceSpec, pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    if spec.max_sources is None or len(pairs) <= spec.max_sources:
        return pairs
    ranked = sorted(pairs, key=lambda pair: hashlib.sha256(Path(pair[0]).name.encode()).hexdigest())
    return sorted(ranked[:spec.max_sources])


def _scan_source(spec: SourceSpec, raw_path: str, label_path: str, data_root: Path) -> List[Dict[str, Any]]:
    """Every non-overlapping deep crop of one source volume that the deep rule accepts."""
    label_source = Path(label_path)
    shape = _array_shape(label_source, spec.label_key)
    if len(shape) != 3:
        raise RuntimeError(f"Expected a 3d label volume for '{spec.dataset}', got {shape} at '{label_path}'.")
    crop = spec.crop_shape(shape)
    legal_z = spec.legal_z(shape) if spec.legal_z is not None else [_platynereis_legal_z(raw_path)(shape)[0]]
    candidates = []
    for z_start, z_stop in legal_z:
        z_stop = min(z_stop, shape[0])
        depth = min(crop[0], z_stop - z_start)
        if depth <= 0:
            continue
        for z0 in _grid_starts(z_start, z_stop, depth, spec.z_step):
            for y0 in _grid_starts(0, shape[1], crop[1]):
                for x0 in _grid_starts(0, shape[2], crop[2]):
                    roi = (slice(z0, z0 + depth), slice(y0, y0 + crop[1]), slice(x0, x0 + crop[2]))
                    labels = _read_labels(label_source, spec.label_key, roi, spec.mask_invalid_labels)
                    instances = connected_components(labels).astype("uint32")
                    object_count, foreground_fraction = _object_statistics(instances)
                    if object_count == 0:
                        continue
                    # The loader trims unannotated end slices, so the depth a crop propagates through
                    # is its annotated span. A deep crop has to keep most of its declared depth.
                    annotated = np.any(instances != 0, axis=(1, 2))
                    span = int(len(annotated) - int(np.argmax(annotated[::-1])) - int(np.argmax(annotated)))
                    if span < min(MIN_REALIZED_DEPTH, depth):
                        continue
                    candidates.append({
                        "dataset": spec.dataset,
                        "family": spec.family,
                        "ndim": 3,
                        "raw_path": _relative_data_path(raw_path, data_root),
                        "label_path": _relative_data_path(label_path, data_root),
                        "raw_key": spec.raw_key,
                        "label_key": spec.label_key,
                        "roi": _roi_to_json(roi),
                        "normalization_z_range": [int(z_start), int(z_stop)],
                        "declared_depth": int(depth),
                        "realized_depth": span,
                        "depth_flag": "deep" if span >= MIN_REALIZED_DEPTH else f"shallow-{span}",
                        "seen_in_training": spec.seen_in_training,
                        "metric_mode": spec.metric_mode,
                        "spacing": list(DATASET_SPACING.get(spec.family, (1, 1, 1))),
                        "mask_invalid_labels": spec.mask_invalid_labels,
                        "object_count": object_count,
                        "foreground_fraction": foreground_fraction,
                        "source_id": _relative_data_path(raw_path, data_root),
                    })
    return candidates


def _overlaps_quadrant(candidate: Dict[str, Any], quadrant: Tuple[int, int]) -> bool:
    roi = _roi_from_json(candidate["roi"])
    return roi[1].stop > quadrant[0] and roi[2].stop > quadrant[1]


def _split_holdout(spec: SourceSpec, candidates: List[Dict[str, Any]]) -> Tuple[List[dict], List[dict]]:
    """Split one source's candidates into primary and holdout pools."""
    holdout = [c for c in candidates if spec.holdout(c["raw_path"])]
    primary = [c for c in candidates if not spec.holdout(c["raw_path"])]
    if spec.dataset in QUADRANT_HOLDOUT:
        quadrant = QUADRANT_HOLDOUT[spec.dataset]
        holdout = [c for c in candidates if _overlaps_quadrant(c, quadrant)]
        primary = [c for c in candidates if not _overlaps_quadrant(c, quadrant)]
    return primary, holdout


def _stable_fold(source_id: str, n_folds: int = 5) -> int:
    return int(hashlib.sha256(source_id.encode()).hexdigest(), 16) % n_folds


def _assign_folds(samples: List[Dict[str, Any]]) -> None:
    """Folds grouped by source volume; a dataset with one source keeps all its crops in one fold."""
    by_dataset = defaultdict(list)
    for sample in samples:
        by_dataset[sample["dataset"]].append(sample)
    for dataset, group in by_dataset.items():
        sources = sorted({sample["source_id"] for sample in group})
        if len(sources) == 1:
            fold_of = {sources[0]: _stable_fold(sources[0])}
        else:
            # Spread the sources over the folds as evenly as the count allows, deterministically.
            ordered = sorted(sources, key=_stable_fold)
            fold_of = {source: index % 5 for index, source in enumerate(ordered)}
        for sample in group:
            sample["fold"] = int(fold_of[sample["source_id"]])
            sample["fold_group"] = sample["source_id"]


def _balanced_select(candidates: List[Dict[str, Any]], n_target: int) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    _add_complexity(candidates)
    n_take = min(n_target, len(candidates))
    return _select_nearest(candidates, _quantile_targets(n_take), prefer_distinct_sources=True)


def _legacy_lookup() -> Dict[Tuple[str, str, str], str]:
    lookup = {}
    for path in LEGACY_MANIFESTS:
        if not path.exists():
            continue
        with open(path) as f:
            manifest = json.load(f)
        for sample in manifest["samples"]:
            key = (sample["raw_path"], json.dumps(sample["roi"]), json.dumps(sample.get("normalization_z_range")))
            lookup[key] = sample["sample_id"]
    return lookup


def _sample_identity(sample: Dict[str, Any]) -> str:
    identity = {key: sample[key] for key in ("dataset", "raw_path", "label_path", "roi")}
    return f"{sample['dataset']}:{_content_checksum(identity)[:12]}"


def build_manifest(data_root: Path, subset: str, include_seen: bool = True) -> Dict[str, Any]:
    """Select the crops of one subset from every eligible source."""
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}; expected one of {SUBSETS}.")
    samples: List[Dict[str, Any]] = []
    under_target = {}
    if subset == "test":
        for spec in _test_source_specs():
            spec.raw_key, spec.label_key = _keys_for_test_dataset(spec.dataset, data_root)
            candidates = []
            pairs = _capped_sources(spec, spec.paths(data_root))
            for raw_path, label_path in tqdm(pairs, desc=f"scan-{spec.dataset}", leave=False):
                candidates.extend(_scan_source(spec, raw_path, label_path, data_root))
            chosen = _balanced_select(candidates, spec.target)
            if len(chosen) < spec.target:
                under_target[spec.dataset] = len(chosen)
            samples.extend(chosen)
    else:
        for spec in tuning_source_specs():
            if spec.seen_in_training is True and not include_seen:
                continue
            candidates = []
            for raw_path, label_path in tqdm(spec.paths(data_root), desc=f"scan-{spec.dataset}", leave=False):
                candidates.extend(_scan_source(spec, raw_path, label_path, data_root))
            primary, holdout = _split_holdout(spec, candidates)
            pool = primary if subset == "primary" else holdout
            target = spec.target if subset == "primary" else max(2, spec.target // 3)
            chosen = _balanced_select(pool, target)
            if len(chosen) < target:
                under_target[spec.dataset] = len(chosen)
            samples.extend(chosen)
    legacy = _legacy_lookup()
    for sample in samples:
        sample["subset"] = subset
        sample["sample_id"] = _sample_identity(sample)
        key = (sample["raw_path"], json.dumps(sample["roi"]), json.dumps(sample["normalization_z_range"]))
        sample["legacy_sample_id"] = legacy.get(key)
    _assign_folds(samples)
    samples.sort(key=lambda sample: (sample["dataset"], sample["sample_id"]))
    manifest = {
        "schema": SCHEMA,
        "subset": subset,
        "data_root": str(data_root),
        "selection_policy": {
            "deep_depth": DEEP_DEPTH,
            "deep_rule": "both end slices annotated; crops non-overlapping on a grid",
            "snemi_test_slab": list(SNEMI_TEST_SLAB),
            "quadrant_holdout": QUADRANT_HOLDOUT,
            "include_seen_sources": include_seen,
            "targets": {spec.dataset: spec.target for spec in tuning_source_specs()},
            "under_target": under_target,
            "seed": SEED,
        },
        "families": sorted({sample["family"] for sample in samples}),
        "samples": samples,
    }
    manifest["manifest_checksum"] = _content_checksum(_identity(manifest))
    validate_manifest(manifest, data_root)
    return manifest


def _identity(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return {key: manifest[key] for key in ("schema", "subset", "selection_policy", "samples")}


def validate_manifest(manifest: Dict[str, Any], data_root: Path, primary: Optional[Dict[str, Any]] = None) -> None:
    if manifest.get("schema") != SCHEMA:
        raise RuntimeError(f"Unsupported manifest schema {manifest.get('schema')!r}.")
    if manifest.get("manifest_checksum") != _content_checksum(_identity(manifest)):
        raise RuntimeError("The manifest content does not match its checksum.")
    seen_ids = set()
    by_source = defaultdict(list)
    for sample in manifest["samples"]:
        if sample["sample_id"] in seen_ids:
            raise RuntimeError(f"Duplicate sample id {sample['sample_id']}.")
        seen_ids.add(sample["sample_id"])
        if sample["object_count"] <= 0:
            raise RuntimeError(f"Sample {sample['sample_id']} has no objects.")
        _source_path(sample["raw_path"], data_root)
        _source_path(sample["label_path"], data_root)
        roi = _roi_from_json(sample["roi"])
        if sample["dataset"] == "snemi" and roi[0].start < SNEMI_TEST_SLAB[1] and roi[0].stop > SNEMI_TEST_SLAB[0]:
            raise RuntimeError(f"SNEMI sample {sample['sample_id']} overlaps the evaluated slab {SNEMI_TEST_SLAB}.")
        by_source[sample["source_id"]].append(roi)
    for source, rois in by_source.items():
        for index, first in enumerate(rois):
            for second in rois[index + 1:]:
                if all(a.start < b.stop and b.start < a.stop for a, b in zip(first, second)):
                    raise RuntimeError(f"Overlapping crops within source {source}.")
    if primary is not None:
        primary_sources = {sample["source_id"] for sample in primary["samples"]}
        for sample in manifest["samples"]:
            if sample["source_id"] in primary_sources and sample["dataset"] not in QUADRANT_HOLDOUT:
                raise RuntimeError(f"Holdout sample {sample['sample_id']} shares a source with the primary subset.")
            if sample["dataset"] in QUADRANT_HOLDOUT:
                roi = _roi_from_json(sample["roi"])
                for other in primary["samples"]:
                    if other["source_id"] != sample["source_id"]:
                        continue
                    other_roi = _roi_from_json(other["roi"])
                    if all(a.start < b.stop and b.start < a.stop for a, b in zip(roi, other_roi)):
                        raise RuntimeError(f"Holdout sample {sample['sample_id']} overlaps a primary crop.")


def manifest_path(subset: str, campaign_root: Path = CAMPAIGN_ROOT) -> Path:
    return campaign_root / f"manifest_{subset}_{SCHEMA}.json"


def load_manifest(subset: str, campaign_root: Path = CAMPAIGN_ROOT, data_root: Path = DEFAULT_DATA_ROOT) -> Dict:
    path = manifest_path(subset, campaign_root)
    with open(path) as f:
        manifest = json.load(f)
    validate_manifest(manifest, data_root)
    return manifest


# ----------------------------------------------------------------------------------------------
# loading samples


def load_normalized_source(sample: Dict[str, Any], data_root: Path) -> np.ndarray:
    """The raw volume of the legal slab, normalized like the production evaluation normalizes it."""
    z_start, z_stop = sample["normalization_z_range"]
    raw = _read_array(_source_path(sample["raw_path"], data_root), sample["raw_key"],
                      roi=(slice(z_start, z_stop), slice(None), slice(None)))
    return (common.normalize_raw(raw) * 255.0).astype("float32")


def load_sample(
    sample: Dict[str, Any], data_root: Path, normalized_source: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """The crop's raw, its connected-component labels and the valid mask (None unless partially annotated)."""
    roi = _roi_from_json(sample["roi"])
    z_offset = int(sample["normalization_z_range"][0])
    raw = np.asarray(normalized_source[roi[0].start - z_offset:roi[0].stop - z_offset, roi[1], roi[2]])
    labels = _read_array(_source_path(sample["label_path"], data_root), sample["label_key"], roi=roi).astype("int64")
    valid = None
    if sample.get("mask_invalid_labels"):
        valid = labels != -1
        labels[labels == -1] = 0
    annotated = np.any(labels != 0, axis=(1, 2))
    if not annotated.any():
        raise RuntimeError(f"Crop {sample['sample_id']} has no annotated voxels.")
    z0 = int(np.argmax(annotated))
    z1 = len(annotated) - int(np.argmax(annotated[::-1]))
    raw, labels = raw[z0:z1], labels[z0:z1]
    if valid is not None:
        valid = valid[z0:z1]
    labels = connected_components(labels).astype("uint32")
    if raw.shape != labels.shape:
        raise RuntimeError(f"Shape mismatch for {sample['sample_id']}: {raw.shape} vs {labels.shape}.")
    return raw.astype("float32", copy=False), labels, valid


def load_labels(sample: Dict[str, Any], data_root: Path) -> np.ndarray:
    """The crop's labels exactly as `load_sample` returns them, without reading the raw volume."""
    roi = _roi_from_json(sample["roi"])
    labels = _read_array(_source_path(sample["label_path"], data_root), sample["label_key"], roi=roi).astype("int64")
    valid = None
    if sample.get("mask_invalid_labels"):
        valid = labels != -1
        labels[labels == -1] = 0
    annotated = np.any(labels != 0, axis=(1, 2))
    z0 = int(np.argmax(annotated))
    z1 = len(annotated) - int(np.argmax(annotated[::-1]))
    labels = labels[z0:z1]
    labels = connected_components(labels).astype("uint32")
    if valid is not None:
        labels[~valid[z0:z1]] = 0
    return labels


def realized_depth(sample: Dict[str, Any], data_root: Path) -> int:
    roi = _roi_from_json(sample["roi"])
    labels = _read_labels(_source_path(sample["label_path"], data_root), sample["label_key"], roi,
                          bool(sample.get("mask_invalid_labels")))
    annotated = np.any(labels != 0, axis=(1, 2))
    return int(annotated.sum()) if not annotated.any() else int(
        len(annotated) - int(np.argmax(annotated[::-1])) - int(np.argmax(annotated))
    )


# ----------------------------------------------------------------------------------------------
# CLI


def _show(manifest: Dict[str, Any]) -> None:
    counts = defaultdict(lambda: {"n": 0, "objects": 0, "deep": 0, "seen": set(), "sources": set()})
    for sample in manifest["samples"]:
        entry = counts[sample["dataset"]]
        entry["n"] += 1
        entry["objects"] += sample["object_count"]
        entry["deep"] += sample["depth_flag"] == "deep"
        entry["seen"].add(str(sample["seen_in_training"]))
        entry["sources"].add(sample["source_id"])
    print(f"{manifest['subset']} ({manifest['manifest_checksum']}): {len(manifest['samples'])} crops")
    for dataset, entry in sorted(counts.items()):
        print(f"  {dataset:24s} crops {entry['n']:3d}  deep {entry['deep']:3d}  objects {entry['objects']:6d}  "
              f"sources {len(entry['sources']):2d}  seen {','.join(sorted(entry['seen']))}")
    under = manifest["selection_policy"].get("under_target")
    if under:
        print(f"  under target: {under}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("build", "show", "depth"))
    parser.add_argument("--subset", choices=SUBSETS, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--campaign-root", type=Path, default=CAMPAIGN_ROOT)
    parser.add_argument("--exclude-seen-sources", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rebuild even if the manifest exists.")
    args = parser.parse_args(argv)
    path = manifest_path(args.subset, args.campaign_root)
    if args.command == "build":
        if path.exists() and not args.force:
            print(f"Manifest exists: {path}")
            manifest = load_manifest(args.subset, args.campaign_root, args.data_root)
        else:
            manifest = build_manifest(args.data_root, args.subset, include_seen=not args.exclude_seen_sources)
            if args.subset == "holdout":
                primary = load_manifest("primary", args.campaign_root, args.data_root)
                validate_manifest(manifest, args.data_root, primary=primary)
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(path, manifest)
            print(f"Wrote {path}")
        _show(manifest)
        return 0
    manifest = load_manifest(args.subset, args.campaign_root, args.data_root)
    if args.command == "show":
        _show(manifest)
        return 0
    for sample in manifest["samples"]:
        depth = realized_depth(sample, args.data_root)
        expected = sample["realized_depth"]
        flag = "" if depth == expected else f"  <- differs from the manifest's {expected}"
        print(f"  {sample['sample_id']:32s} depth {depth:3d} objects {sample['object_count']:5d}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
