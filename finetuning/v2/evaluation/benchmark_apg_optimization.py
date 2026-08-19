"""Run a small, deterministic APG optimization benchmark on ten validation datasets.

The benchmark is deliberately bounded: one invocation evaluates one parameter configuration on
240 representative 2d images and two representative crops from each of five 3d datasets. The
default configuration is the current library default for APG with the best joint hvit_t checkpoint.

No data is downloaded and the data root is treated as read-only. All manifests, checkpoint exports
and results are written below the output root.

Examples:
    # Build or inspect the deterministic subset without loading a model.
    python benchmark_apg_optimization.py --prepare-only

    # Run the current APG defaults.
    python benchmark_apg_optimization.py

    # Run one alternative configuration.
    python benchmark_apg_optimization.py --config apg_candidate_thresholds.json

The optional JSON configuration has this shape:
    {
      "name": "candidate-threshold-experiment",
      "params_2d": {"candidate_threshold": 1.0},
      "params_3d": {"candidate_threshold_3d": [1.0, 5.0]}
    }
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import xxhash
from elf.io import open_file
from skimage.measure import label as connected_components
from tqdm import tqdm

import common
from common import (
    CROP_SHAPE_2D,
    DATASET_SPACING,
    DATASETS_3D_EM,
    GENERATE_PARAM_KEYS,
    GT_MIN_SIZE_2D,
    VAL_Z_RANGE,
    VOLUME_SPEED_OPTIONS,
    build_apg_segmenter,
    checkpoint_checksum,
    drop_severed_objects,
    ensure_8bit_range,
    get_data_paths,
    get_joint_checkpoint,
    genuine_misses,
    read_2d,
    resolve_params,
    sorted_path_pairs,
)
from evaluate_automatic_segmentation import compute_metrics
from micro_sam.v2.normalization import normalize_raw


DATASETS_2D = ("livecell", "tissuenet", "dynamicnuclearnet", "deepbacs", "dic_hepg2")
DATASETS_3D = ("celegans_atlas", "embedseg", "gonuclear", "cremi", "snemi")
LIVECELL_TYPES = ("A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SKOV3", "SkBr3")

DEFAULT_DATA_ROOT = Path("/mnt/vast-nhr/projects/cidas/cca/data")
DEFAULT_OUTPUT_ROOT = Path("/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/apg_optimization")

MANIFEST_SCHEMA_VERSION = 3
# DeepBacs only has 30 validation images. Keep all of them and use the ten remaining slots for DIC,
# whose initially surprising score benefits most from additional coverage.
SAMPLE_COUNTS_2D = {
    "livecell": 80,
    "tissuenet": 40,
    "dynamicnuclearnet": 40,
    "deepbacs": 30,
    "dic_hepg2": 50,
}
TARGETS_3D = (0.5, 0.8)
CROP_SHAPE_3D = (8, 256, 256)
# An 8-slice crop is thinner than a typical C. elegans nucleus (about 11-13 slices in this
# validation data), so it mainly measures objects severed by the benchmark itself.
CROP_SHAPE_3D_OVERRIDES = {"celegans_atlas": (32, 256, 256)}
CANDIDATE_GRID_3D = (4, 3, 3)

# Conservative first-run estimates from the current A100. Observed times replace them as samples run.
INITIAL_SAMPLE_SECONDS = {2: 5.0, 3: 35.0}

IMPLEMENTATION_FILES = (
    Path(__file__),
    Path(common.__file__),
    Path(__file__).with_name("evaluate_automatic_segmentation.py"),
    Path(common.__file__).parents[3] / "micro_sam/v2/automatic_prompt_generation.py",
    Path(common.__file__).parents[3] / "micro_sam/v2/instance_segmentation.py",
    Path(common.__file__).parents[3] / "micro_sam/v2/postprocessing.py",
    Path(common.__file__).parents[3] / "micro_sam/v2/prompt_based_segmentation.py",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")


def _content_checksum(value: Any) -> str:
    return xxhash.xxh128(_json_bytes(value)).hexdigest()


def _implementation_checksum() -> str:
    """Hash the code that determines loading, prompting, propagation, merging and scoring."""
    checksum = xxhash.xxh128()
    for path in IMPLEMENTATION_FILES:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                checksum.update(block)
        checksum.update(b"\0")
    return checksum.hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp_path, "w") as f:
        json.dump(value, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")
    os.replace(tmp_path, path)


def _atomic_write_csv(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    table.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def _resolved(path: Path, strict: bool = False) -> Path:
    return path.expanduser().resolve(strict=strict)


def _assert_beneath(path: Path, root: Path, what: str) -> Path:
    resolved = _resolved(path, strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"{what} is outside the data root: '{resolved}'.") from error
    return resolved


def _validate_roots(data_root: Path, output_root: Path, manifest_path: Path) -> Tuple[Path, Path, Path]:
    data_root = _resolved(data_root, strict=True)
    output_root = _resolved(output_root)
    manifest_path = _resolved(manifest_path)
    if output_root == data_root or data_root in output_root.parents:
        raise ValueError(f"The output root must not be inside the read-only data root: '{data_root}'.")
    if manifest_path == data_root or data_root in manifest_path.parents:
        raise ValueError(f"The manifest must not be written inside the read-only data root: '{data_root}'.")
    if manifest_path != output_root and output_root not in manifest_path.parents:
        raise ValueError(f"The manifest must be stored below the output root: '{output_root}'.")
    return data_root, output_root, manifest_path


def _relative_data_path(path: str, data_root: Path) -> str:
    return str(_assert_beneath(Path(path), data_root, "Dataset path").relative_to(data_root))


def _source_path(relative_path: str, data_root: Path) -> Path:
    return _assert_beneath(data_root / relative_path, data_root, "Manifest path")


def _roi_to_json(roi: Sequence[slice]) -> List[List[int]]:
    return [[int(axis.start), int(axis.stop)] for axis in roi]


def _roi_from_json(roi: Sequence[Sequence[int]]) -> Tuple[slice, ...]:
    return tuple(slice(int(start), int(stop)) for start, stop in roi)


def _object_statistics(labels: np.ndarray) -> Tuple[int, float]:
    count = int(len(np.unique(labels)) - 1)
    foreground_fraction = float(np.count_nonzero(labels) / labels.size) if labels.size else 0.0
    return count, foreground_fraction


def _percentile_ranks(values: Sequence[float]) -> np.ndarray:
    """Return deterministic average ranks in [0, 1], including tied values."""
    values = np.asarray(values)
    if len(values) <= 1:
        return np.full(len(values), 0.5, dtype="float64")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype="float64")
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2
        start = stop
    return ranks / (len(values) - 1)


def _add_complexity(candidates: List[Dict[str, Any]]) -> None:
    object_ranks = _percentile_ranks([entry["object_count"] for entry in candidates])
    foreground_ranks = _percentile_ranks([entry["foreground_fraction"] for entry in candidates])
    for entry, object_rank, foreground_rank in zip(candidates, object_ranks, foreground_ranks):
        entry["complexity"] = float((object_rank + foreground_rank) / 2)


def _sample_identity(entry: Dict[str, Any]) -> str:
    identity = {
        "dataset": entry["dataset"],
        "raw_path": entry["raw_path"],
        "label_path": entry["label_path"],
        "roi": entry["roi"],
    }
    return f"{entry['dataset']}:{_content_checksum(identity)[:12]}"


def _select_nearest(
    candidates: List[Dict[str, Any]], targets: Sequence[float], prefer_distinct_sources: bool = False,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    # Track candidate indices explicitly. Selected entries gain `target_quantile`, so comparing the
    # copied dictionaries would no longer exclude their source candidate on the next iteration.
    remaining = list(enumerate(candidates))
    for target in targets:
        pool = remaining
        if prefer_distinct_sources and selected:
            used_sources = {entry["label_path"] for entry in selected}
            distinct = [item for item in pool if item[1]["label_path"] not in used_sources]
            if distinct:
                pool = distinct
            else:
                non_overlapping = [
                    item for item in pool
                    if all(_roi_overlap_fraction(item[1]["roi"], other["roi"]) == 0 for other in selected)
                ]
                if non_overlapping:
                    pool = non_overlapping
        if not pool:
            raise RuntimeError(f"Could not select {len(targets)} distinct samples from {len(candidates)} candidates.")
        chosen_index, source = min(
            pool,
            key=lambda item: (
                abs(item[1]["complexity"] - target),
                abs(item[1]["foreground_fraction"] - 0.5),
                item[1]["label_path"],
                item[1]["roi"],
            ),
        )
        remaining = [item for item in remaining if item[0] != chosen_index]
        chosen = source.copy()
        chosen["target_quantile"] = float(target)
        selected.append(chosen)
    return selected


def _quantile_targets(n_samples: int) -> Tuple[float, ...]:
    """Return evenly spaced quantile midpoints without selecting the distribution endpoints."""
    return tuple((index + 0.5) / n_samples for index in range(n_samples))


def _roi_overlap_fraction(first: Sequence[Sequence[int]], second: Sequence[Sequence[int]]) -> float:
    intersection = 1
    first_size = 1
    second_size = 1
    for (first_start, first_stop), (second_start, second_stop) in zip(first, second):
        intersection *= max(0, min(first_stop, second_stop) - max(first_start, second_start))
        first_size *= first_stop - first_start
        second_size *= second_stop - second_start
    return float(intersection / min(first_size, second_size))


def _center_crop_roi(shape: Sequence[int], crop_shape: Sequence[int]) -> Tuple[slice, ...]:
    roi = []
    for size, crop_size in zip(shape, crop_shape):
        crop_size = min(size, crop_size)
        start = (size - crop_size) // 2
        roi.append(slice(start, start + crop_size))
    return tuple(roi)


def _scan_2d_dataset(dataset: str, data_root: Path) -> List[Dict[str, Any]]:
    raw_paths, label_paths, raw_key, label_key = get_data_paths(
        dataset, str(data_root), download=False, split="val"
    )
    candidates = []
    pairs = sorted_path_pairs(raw_paths, label_paths)
    for raw_path, label_path in tqdm(pairs, desc=f"select-{dataset}", leave=False):
        raw_relative = _relative_data_path(raw_path, data_root)
        label_relative = _relative_data_path(label_path, data_root)
        labels = read_2d(str(_source_path(label_relative, data_root)), label_key)
        roi = _center_crop_roi(labels.shape[:2], CROP_SHAPE_2D)
        labels = connected_components(labels[roi]).astype("uint32")
        labels = drop_severed_objects(labels, GT_MIN_SIZE_2D.get(dataset, 0))
        object_count, foreground_fraction = _object_statistics(labels)
        if object_count == 0:
            continue
        candidates.append({
            "dataset": dataset,
            "ndim": 2,
            "raw_path": raw_relative,
            "label_path": label_relative,
            "raw_key": raw_key,
            "label_key": label_key,
            "roi": _roi_to_json(roi),
            "object_count": object_count,
            "foreground_fraction": foreground_fraction,
        })
    if not candidates:
        raise RuntimeError(f"No non-empty validation images found for '{dataset}'.")
    return candidates


def _select_2d_samples(data_root: Path) -> List[Dict[str, Any]]:
    samples = []
    for dataset in DATASETS_2D:
        candidates = _scan_2d_dataset(dataset, data_root)
        if dataset == "livecell":
            by_type = defaultdict(list)
            for candidate in candidates:
                cell_type = Path(candidate["raw_path"]).name.split("_", 1)[0]
                by_type[cell_type].append(candidate)
            missing = sorted(set(LIVECELL_TYPES) - set(by_type))
            if missing:
                raise RuntimeError(f"LIVECell validation data is missing cell types: {missing}.")
            selected = []
            n_per_type, remainder = divmod(SAMPLE_COUNTS_2D[dataset], len(LIVECELL_TYPES))
            if remainder:
                raise RuntimeError("The LIVECell sample count must be divisible by its cell-type count.")
            for cell_type in LIVECELL_TYPES:
                group = by_type[cell_type]
                _add_complexity(group)
                choices = _select_nearest(group, _quantile_targets(n_per_type))
                for choice in choices:
                    choice["stratum"] = cell_type
                    selected.append(choice)
        else:
            _add_complexity(candidates)
            selected = _select_nearest(candidates, _quantile_targets(SAMPLE_COUNTS_2D[dataset]))
        for sample in selected:
            sample["sample_id"] = _sample_identity(sample)
            samples.append(sample)
    return samples


def _read_array(path: Path, key: Optional[str], roi: Optional[Tuple[slice, ...]] = None) -> np.ndarray:
    if key is None:
        array = np.asarray(common.load_image(str(path)))
        return array if roi is None else np.asarray(array[roi])
    with open_file(str(path), mode="r") as data:
        dataset = data[key]
        return np.asarray(dataset[:] if roi is None else dataset[roi])


def _array_shape(path: Path, key: Optional[str]) -> Tuple[int, ...]:
    if key is None:
        return tuple(np.asarray(common.load_image(str(path))).shape)
    with open_file(str(path), mode="r") as data:
        return tuple(data[key].shape)


def _even_starts(start: int, stop: int, crop_size: int, n_positions: int) -> List[int]:
    last_start = stop - crop_size
    if last_start <= start:
        return [start]
    return sorted({int(round(value)) for value in np.linspace(start, last_start, n_positions)})


def _validation_z_range(dataset: str, depth: int) -> Tuple[int, int]:
    base_start = 70 if dataset == "snemi" else 0
    base_stop = depth
    z_range = VAL_Z_RANGE.get(dataset)
    if z_range is None:
        return base_start, base_stop
    return base_start + z_range[0], min(base_start + z_range[1], base_stop)


def _scan_3d_source(
    dataset: str,
    raw_path: str,
    label_path: str,
    raw_key: Optional[str],
    label_key: Optional[str],
    data_root: Path,
) -> List[Dict[str, Any]]:
    raw_relative = _relative_data_path(raw_path, data_root)
    label_relative = _relative_data_path(label_path, data_root)
    label_source = _source_path(label_relative, data_root)
    shape = _array_shape(label_source, label_key)
    if len(shape) != 3:
        raise RuntimeError(f"Expected a 3d label volume for '{dataset}', got shape {shape} at '{label_source}'.")
    z_start, z_stop = _validation_z_range(dataset, shape[0])
    valid_shape = (z_stop - z_start, shape[1], shape[2])
    requested_crop_shape = CROP_SHAPE_3D_OVERRIDES.get(dataset, CROP_SHAPE_3D)
    crop_shape = tuple(min(size, crop) for size, crop in zip(valid_shape, requested_crop_shape))
    starts = (
        _even_starts(z_start, z_stop, crop_shape[0], CANDIDATE_GRID_3D[0]),
        _even_starts(0, shape[1], crop_shape[1], CANDIDATE_GRID_3D[1]),
        _even_starts(0, shape[2], crop_shape[2], CANDIDATE_GRID_3D[2]),
    )

    def collect(read: Callable[[Tuple[slice, ...]], np.ndarray]) -> List[Dict[str, Any]]:
        candidates = []
        for z0 in starts[0]:
            for y0 in starts[1]:
                for x0 in starts[2]:
                    roi = (
                        slice(z0, z0 + crop_shape[0]),
                        slice(y0, y0 + crop_shape[1]),
                        slice(x0, x0 + crop_shape[2]),
                    )
                    labels = connected_components(read(roi)).astype("uint32")
                    object_count, foreground_fraction = _object_statistics(labels)
                    if object_count == 0:
                        continue
                    candidates.append({
                        "dataset": dataset,
                        "ndim": 3,
                        "raw_path": raw_relative,
                        "label_path": label_relative,
                        "raw_key": raw_key,
                        "label_key": label_key,
                        "roi": _roi_to_json(roi),
                        "normalization_z_range": [z_start, z_stop],
                        "object_count": object_count,
                        "foreground_fraction": foreground_fraction,
                    })
        return candidates

    if label_key is None:
        labels = np.asarray(common.load_image(str(label_source)))
        return collect(lambda roi: np.asarray(labels[roi]))
    with open_file(str(label_source), mode="r") as data:
        labels = data[label_key]
        return collect(lambda roi: np.asarray(labels[roi]))


def _select_3d_samples(data_root: Path) -> List[Dict[str, Any]]:
    samples = []
    for dataset in DATASETS_3D:
        raw_paths, label_paths, raw_key, label_key = get_data_paths(
            dataset, str(data_root), download=False, split="val"
        )
        candidates = []
        for raw_path, label_path in tqdm(
            sorted_path_pairs(raw_paths, label_paths), desc=f"select-{dataset}", leave=False
        ):
            candidates.extend(
                _scan_3d_source(dataset, raw_path, label_path, raw_key, label_key, data_root)
            )
        if len(candidates) < len(TARGETS_3D):
            raise RuntimeError(f"Only {len(candidates)} occupied 3d candidates found for '{dataset}'.")
        _add_complexity(candidates)
        selected = _select_nearest(candidates, TARGETS_3D, prefer_distinct_sources=True)
        for sample in selected:
            sample["sample_id"] = _sample_identity(sample)
            samples.append(sample)
    return samples


def _manifest_identity(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": manifest["schema_version"],
        "selection_policy": manifest["selection_policy"],
        "samples": manifest["samples"],
    }


def _validate_manifest(manifest: Dict[str, Any], data_root: Path) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported manifest schema {manifest.get('schema_version')}; expected {MANIFEST_SCHEMA_VERSION}."
        )
    expected_checksum = _content_checksum(_manifest_identity(manifest))
    if manifest.get("manifest_checksum") != expected_checksum:
        raise RuntimeError("The subset manifest content does not match its checksum.")
    samples = manifest.get("samples", [])
    counts = defaultdict(int)
    identifiers = set()
    for sample in samples:
        counts[(sample["dataset"], sample["ndim"])] += 1
        if sample["sample_id"] in identifiers:
            raise RuntimeError(f"Duplicate manifest sample id: {sample['sample_id']}.")
        identifiers.add(sample["sample_id"])
        _source_path(sample["raw_path"], data_root)
        _source_path(sample["label_path"], data_root)
        if sample["object_count"] <= 0:
            raise RuntimeError(f"Manifest sample '{sample['sample_id']}' has empty ground truth.")
    expected = {(dataset, 2): SAMPLE_COUNTS_2D[dataset] for dataset in DATASETS_2D}
    expected.update({(dataset, 3): 2 for dataset in DATASETS_3D})
    if dict(counts) != expected:
        raise RuntimeError(f"Unexpected manifest sample counts: got {dict(counts)}, expected {expected}.")


def prepare_manifest(data_root: Path, manifest_path: Path) -> Dict[str, Any]:
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        _validate_manifest(manifest, data_root)
        return manifest

    samples = _select_2d_samples(data_root) + _select_3d_samples(data_root)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "data_root": str(data_root),
        "selection_policy": {
            "2d_crop_shape": list(CROP_SHAPE_2D),
            "2d_sample_counts": SAMPLE_COUNTS_2D,
            "2d_complexity_targets": "even quantile midpoints within each dataset and LIVECell cell type",
            "3d_crop_shapes": {
                dataset: list(CROP_SHAPE_3D_OVERRIDES.get(dataset, CROP_SHAPE_3D))
                for dataset in DATASETS_3D
            },
            "3d_candidate_grid": list(CANDIDATE_GRID_3D),
            "3d_complexity_targets": list(TARGETS_3D),
            "complexity": "mean percentile rank of object count and foreground fraction",
        },
        "samples": samples,
    }
    manifest["manifest_checksum"] = _content_checksum(_manifest_identity(manifest))
    _validate_manifest(manifest, data_root)
    _atomic_write_json(manifest_path, manifest)
    return manifest


def _load_config(path: Optional[Path]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    if path is None:
        config = {"name": "current-defaults", "params_2d": {}, "params_3d": {}}
    else:
        with open(path) as f:
            config = json.load(f)
    allowed_top_level = {"name", "params_2d", "params_3d"}
    unknown_top_level = set(config) - allowed_top_level
    if unknown_top_level:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown_top_level)}.")
    name = config.get("name", "unnamed")
    overrides_2d = config.get("params_2d", {})
    overrides_3d = config.get("params_3d", {})
    if not isinstance(name, str) or not isinstance(overrides_2d, dict) or not isinstance(overrides_3d, dict):
        raise TypeError("Configuration 'name' must be a string and parameter overrides must be objects.")
    allowed_2d = set(GENERATE_PARAM_KEYS)
    allowed_3d = (allowed_2d - {"candidate_threshold"}) | {"candidate_threshold_3d"}
    unknown_2d = set(overrides_2d) - allowed_2d
    unknown_3d = set(overrides_3d) - allowed_3d
    if unknown_2d or unknown_3d:
        raise ValueError(f"Unknown APG parameters: 2d={sorted(unknown_2d)}, 3d={sorted(unknown_3d)}.")
    return name, resolve_params(overrides_2d, ndim=2), resolve_params(overrides_3d, ndim=3)


def _load_2d_sample(sample: Dict[str, Any], data_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    raw = ensure_8bit_range(read_2d(str(_source_path(sample["raw_path"], data_root)), sample["raw_key"]))
    labels = read_2d(str(_source_path(sample["label_path"], data_root)), sample["label_key"])
    roi = _roi_from_json(sample["roi"])
    labels = connected_components(labels[roi]).astype("uint32")
    labels = drop_severed_objects(labels, GT_MIN_SIZE_2D.get(sample["dataset"], 0))
    raw = raw[roi]
    if raw.shape[:2] != labels.shape:
        raise RuntimeError(f"Shape mismatch for '{sample['sample_id']}': raw {raw.shape}, labels {labels.shape}.")
    return raw, labels


def _load_normalized_3d_source(sample: Dict[str, Any], data_root: Path) -> np.ndarray:
    raw_path = _source_path(sample["raw_path"], data_root)
    z_start, z_stop = sample["normalization_z_range"]
    roi = (slice(z_start, z_stop), slice(None), slice(None))
    raw = _read_array(raw_path, sample["raw_key"], roi=roi)
    return (normalize_raw(raw) * 255.0).astype("float32")


def _load_3d_sample(
    sample: Dict[str, Any], data_root: Path, normalized_source: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    source_roi = _roi_from_json(sample["roi"])
    norm_z_start = int(sample["normalization_z_range"][0])
    raw_roi = (
        slice(source_roi[0].start - norm_z_start, source_roi[0].stop - norm_z_start),
        source_roi[1],
        source_roi[2],
    )
    raw = np.asarray(normalized_source[raw_roi])
    labels = _read_array(
        _source_path(sample["label_path"], data_root), sample["label_key"], roi=source_roi
    )
    annotated = np.any(labels != 0, axis=(1, 2))
    if not annotated.any():
        raise RuntimeError(f"Selected crop '{sample['sample_id']}' no longer has annotated voxels.")
    z_start = int(np.argmax(annotated))
    z_stop = len(annotated) - int(np.argmax(annotated[::-1]))
    raw, labels = raw[z_start:z_stop], labels[z_start:z_stop]
    labels = connected_components(labels).astype("uint32")
    if raw.shape != labels.shape:
        raise RuntimeError(f"Shape mismatch for '{sample['sample_id']}': raw {raw.shape}, labels {labels.shape}.")
    return raw.astype("float32", copy=False), labels


def _git_revision() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _summarize(samples: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column for column in ("msa", "cremi", "vi_split", "vi_merge", "adapted_rand")
        if column in samples.columns
    ]
    rows = []
    for dataset, group in samples.groupby("dataset", sort=False):
        row = {
            "dataset": dataset,
            "n_samples": len(group),
            "total_seconds": float(group["total_seconds"].sum()),
            "initialization_seconds": float(group["initialization_seconds"].sum()),
            "generation_seconds": float(group["generation_seconds"].sum()),
        }
        for metric in metric_columns:
            values = group[metric].dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=0)) if len(values) else np.nan
        for diagnostic in ("unmatched", "genuine_misses"):
            if diagnostic in group:
                values = group[diagnostic].dropna()
                row[diagnostic] = int(values.sum()) if len(values) else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    overall = {
        "dataset": "__dataset_balanced__",
        "n_samples": int(samples.shape[0]),
        "total_seconds": float(samples["total_seconds"].sum()),
        "initialization_seconds": float(samples["initialization_seconds"].sum()),
        "generation_seconds": float(samples["generation_seconds"].sum()),
    }
    for metric in metric_columns:
        values = summary[f"{metric}_mean"].dropna()
        overall[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
        overall[f"{metric}_std"] = float(values.std(ddof=0)) if len(values) else np.nan
    for diagnostic in ("unmatched", "genuine_misses"):
        if diagnostic in summary:
            values = summary[diagnostic].dropna()
            overall[diagnostic] = int(values.sum()) if len(values) else np.nan
    return pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)


def _runtime_projection(
    started: float, pending: Iterable[Dict[str, Any]], completed: pd.DataFrame,
) -> float:
    estimates = dict(INITIAL_SAMPLE_SECONDS)
    if not completed.empty:
        for ndim in (2, 3):
            observed = completed.loc[completed["ndim"] == ndim, "total_seconds"]
            if len(observed):
                estimates[ndim] = max(estimates[ndim], float(observed.max()))
    return time.perf_counter() - started + sum(estimates[sample["ndim"]] for sample in pending)


def _sample_row(
    sample: Dict[str, Any], segmentation: np.ndarray, labels: np.ndarray,
    initialization_seconds: float, generation_seconds: float,
) -> Dict[str, Any]:
    metric_mode = "dense" if sample["dataset"] in DATASETS_3D_EM else "sparse"
    border_min_size = GT_MIN_SIZE_2D.get(sample["dataset"], 0) if sample["ndim"] == 2 else 0
    metrics = compute_metrics(segmentation, labels, metric_mode, border_min_size=border_min_size)
    row = {
        "sample_id": sample["sample_id"],
        "dataset": sample["dataset"],
        "ndim": sample["ndim"],
        "raw_path": sample["raw_path"],
        "roi": json.dumps(sample["roi"], separators=(",", ":")),
        "gt_objects": int(len(np.unique(labels)) - 1),
        "predicted_objects": int(len(np.unique(segmentation)) - 1),
        "initialization_seconds": initialization_seconds,
        "generation_seconds": generation_seconds,
        "total_seconds": initialization_seconds + generation_seconds,
        **metrics,
    }
    if sample["ndim"] == 3:
        row["unmatched"], row["genuine_misses"] = genuine_misses(labels, segmentation)
    return row


def _run_sample(
    segmenter: Any, sample: Dict[str, Any], raw: np.ndarray, labels: np.ndarray,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    segmenter.clear_state()
    start = time.perf_counter()
    segmenter.initialize(raw, ndim=sample["ndim"], **(VOLUME_SPEED_OPTIONS if sample["ndim"] == 3 else {}))
    initialized = time.perf_counter()
    generation_kwargs = dict(params)
    if sample["ndim"] == 3:
        generation_kwargs["spacing"] = DATASET_SPACING.get(sample["dataset"])
    segmentation = segmenter.generate(**generation_kwargs).astype("uint32")
    generated = time.perf_counter()
    return _sample_row(
        sample, segmentation, labels,
        initialization_seconds=initialized - start,
        generation_seconds=generated - initialized,
    )


def _run_dimension(
    ndim: int,
    samples: List[Dict[str, Any]],
    completed: pd.DataFrame,
    samples_path: Path,
    data_root: Path,
    model_type: str,
    joint_checkpoint: str,
    checkpoint_id: str,
    export_root: Path,
    params: Dict[str, Any],
    device: str,
    started: float,
    budget_seconds: float,
) -> pd.DataFrame:
    completed_ids = set(completed["sample_id"]) if not completed.empty else set()
    pending = [sample for sample in samples if sample["ndim"] == ndim and sample["sample_id"] not in completed_ids]
    if not pending:
        return completed
    projected = _runtime_projection(started, pending, completed)
    if projected > budget_seconds:
        raise RuntimeError(
            f"Projected runtime is {projected / 60:.1f} minutes, above the {budget_seconds / 60:.1f}-minute budget."
        )
    segmenter = build_apg_segmenter(
        model_type, ndim, device, joint_checkpoint=joint_checkpoint,
        joint_checksum=checkpoint_id, export_root=str(export_root),
    )
    current_source = None
    normalized_source = None
    try:
        for index, sample in enumerate(tqdm(pending, desc=f"apg-{ndim}d")):
            remaining = pending[index:]
            projected = _runtime_projection(started, remaining, completed)
            if projected > budget_seconds:
                raise RuntimeError(
                    f"Stopping before '{sample['sample_id']}': projected runtime is {projected / 60:.1f} "
                    f"minutes, above the {budget_seconds / 60:.1f}-minute budget."
                )
            if ndim == 2:
                raw, labels = _load_2d_sample(sample, data_root)
            else:
                source = (sample["raw_path"], tuple(sample["normalization_z_range"]))
                if source != current_source:
                    normalized_source = _load_normalized_3d_source(sample, data_root)
                    current_source = source
                raw, labels = _load_3d_sample(sample, data_root, normalized_source)
            row = _run_sample(segmenter, sample, raw, labels, params)
            completed = pd.concat([completed, pd.DataFrame([row])], ignore_index=True)
            _atomic_write_csv(samples_path, completed)
    finally:
        segmenter.clear_state()
        del segmenter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return completed


def run_benchmark(
    manifest: Dict[str, Any], data_root: Path, output_root: Path, model_type: str,
    joint_checkpoint: str, config_name: str, params_2d: Dict[str, Any], params_3d: Dict[str, Any],
    device: str, time_budget_minutes: float, started: Optional[float] = None,
) -> Tuple[Path, pd.DataFrame, Dict[str, Any]]:
    started = time.perf_counter() if started is None else started
    checkpoint_path = get_joint_checkpoint(model_type, joint_checkpoint)
    checkpoint_id = checkpoint_checksum(checkpoint_path)
    implementation_checksum = _implementation_checksum()
    config_identity = {"params_2d": params_2d, "params_3d": params_3d}
    config_checksum = _content_checksum(config_identity)
    manifest_checksum = manifest["manifest_checksum"]
    run_dir = output_root / model_type / checkpoint_id / (
        f"{manifest_checksum}-{config_checksum}-{implementation_checksum}"
    )
    samples_path = run_dir / "samples.csv"
    summary_path = run_dir / "summary.csv"
    metadata_path = run_dir / "metadata.json"
    export_root = output_root / "model_exports"
    run_dir.mkdir(parents=True, exist_ok=True)

    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        if metadata.get("status") == "complete" and samples_path.exists() and summary_path.exists():
            print(f"Completed result already exists at '{run_dir}'.")
            return run_dir, pd.read_csv(summary_path), metadata

    completed = pd.read_csv(samples_path) if samples_path.exists() else pd.DataFrame()
    metadata = {
        "status": "running",
        "config_name": config_name,
        "config_checksum": config_checksum,
        "manifest_checksum": manifest_checksum,
        "implementation_checksum": implementation_checksum,
        "checkpoint_checksum": checkpoint_id,
        "checkpoint_name": joint_checkpoint,
        "model_type": model_type,
        "params_2d": params_2d,
        "params_3d": params_3d,
        "device": device,
        "gpu": torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else None,
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "git_revision": _git_revision(),
        "time_budget_minutes": time_budget_minutes,
    }
    _atomic_write_json(metadata_path, metadata)

    try:
        for ndim, params in ((2, params_2d), (3, params_3d)):
            completed = _run_dimension(
                ndim, manifest["samples"], completed, samples_path, data_root, model_type,
                joint_checkpoint, checkpoint_id, export_root, params, device, started,
                time_budget_minutes * 60,
            )
        expected_ids = {sample["sample_id"] for sample in manifest["samples"]}
        completed_ids = set(completed["sample_id"])
        if completed_ids != expected_ids:
            raise RuntimeError(
                f"Run finished with {len(completed_ids)} of {len(expected_ids)} expected samples."
            )
        summary = _summarize(completed)
        _atomic_write_csv(summary_path, summary)
        metadata["status"] = "complete"
        metadata["wall_seconds"] = time.perf_counter() - started
        metadata["n_samples"] = len(completed)
        _atomic_write_json(metadata_path, metadata)
    except Exception as error:
        metadata["status"] = "failed"
        metadata["wall_seconds"] = time.perf_counter() - started
        metadata["error"] = f"{type(error).__name__}: {error}"
        _atomic_write_json(metadata_path, metadata)
        raise
    return run_dir, summary, metadata


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Read-only dataset root.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None, help="Subset manifest; defaults below output-root.")
    parser.add_argument("--config", type=Path, default=None, help="One JSON APG configuration to evaluate.")
    parser.add_argument("--model-type", default="hvit_t", choices=common.MODEL_TYPES)
    parser.add_argument("--joint-checkpoint", default="best", help="Joint checkpoint name without '.pt'.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--time-budget-minutes", type=float, default=30.0)
    parser.add_argument("--prepare-only", action="store_true", help="Create and validate the subset, then stop.")
    args = parser.parse_args()

    manifest_path = args.manifest or (args.output_root / f"subset_manifest_v{MANIFEST_SCHEMA_VERSION}.json")
    data_root, output_root, manifest_path = _validate_roots(args.data_root, args.output_root, manifest_path)
    if args.time_budget_minutes <= 0:
        parser.error("--time-budget-minutes must be positive.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("A CUDA device was requested, but CUDA is not available.")
    if not args.device.startswith("cuda"):
        print("Warning: the 30-minute runtime target was calibrated on an A100, not on CPU.", file=sys.stderr)

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = prepare_manifest(data_root, manifest_path)
    print(
        f"Manifest: {manifest_path}\n"
        f"Checksum: {manifest['manifest_checksum']}\n"
        f"Samples: {sum(sample['ndim'] == 2 for sample in manifest['samples'])} 2d + "
        f"{sum(sample['ndim'] == 3 for sample in manifest['samples'])} 3d"
    )
    if args.prepare_only:
        return

    config_name, params_2d, params_3d = _load_config(args.config)
    run_dir, summary, metadata = run_benchmark(
        manifest, data_root, output_root, args.model_type, args.joint_checkpoint,
        config_name, params_2d, params_3d, args.device, args.time_budget_minutes, started=started,
    )
    print(summary.to_string(index=False))
    print(f"Run directory: {run_dir}")
    print(f"Wall time: {metadata['wall_seconds'] / 60:.2f} minutes")


if __name__ == "__main__":
    main()
