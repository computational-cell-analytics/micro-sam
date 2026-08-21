"""Run a small, deterministic APG optimization benchmark on ten validation datasets.

The benchmark is deliberately bounded: one invocation evaluates one parameter configuration on
240 representative 2d images and one representative crop from each of five 3d datasets. A run may
select either dimension without constructing the other dimension's segmenter. The
default configuration is the current library default for APG with the best joint hvit_t checkpoint.

No data is downloaded and the data root is treated as read-only. All manifests, checkpoint exports
and results are written below the output root.

Examples:
    # Build or inspect the deterministic subset without loading a model.
    python benchmark_apg_optimization.py --prepare-only

    # Run the current APG defaults.
    python benchmark_apg_optimization.py --ndim 2 --trial-id baseline-1

    # Run one alternative configuration.
    python benchmark_apg_optimization.py --config apg_candidate_thresholds.json

    # Opt in to the 32-slice 3d crops, which keep their own manifest beside the standard one.
    python benchmark_apg_optimization.py --ndim 3 --crops-3d deep --trial-id baseline-3d-1

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

MANIFEST_SCHEMA_VERSION = 5
# DeepBacs only has 30 validation images. Keep all of them and use the ten remaining slots for DIC,
# whose initially surprising score benefits most from additional coverage.
SAMPLE_COUNTS_2D = {
    "livecell": 80,
    "tissuenet": 40,
    "dynamicnuclearnet": 40,
    "deepbacs": 30,
    "dic_hepg2": 50,
}
# The held-out subset, image-disjoint from the primary one wherever the validation pool allows it.
# The primary set has been mined by repeated screening, so candidate configurations tuned on it are
# confirmed here. DeepBacs is exhausted (all 30 validation images are primary), so it is reused
# verbatim and its column is not held out. DIC has 93 usable validation images, of which the primary
# set takes 50, leaving 43; the test splits would close both gaps but are the evaluated splits, so
# selecting on them is exactly the leak `VAL_SPLITS` exists to prevent. Unequal counts do not skew
# the primary quality figure, which is an equal-weight mean of the per-dataset means.
SAMPLE_COUNTS_2D_HOLDOUT = {
    "livecell": 80,
    "tissuenet": 40,
    "dynamicnuclearnet": 40,
    "deepbacs": 30,
    "dic_hepg2": 43,
}
HOLDOUT_REUSED_DATASETS = ("deepbacs",)
MANIFEST_SUBSETS = ("primary", "holdout")
TARGETS_3D = (0.5,)
# Match the 512 x 512 training field of view and use enough depth to contain representative 3d
# structure. C. elegans keeps the deeper crop needed to contain its 11-13-slice nuclei; its source
# volumes are only 140 pixels wide in Y.
CROP_SHAPE_3D = (12, 512, 512)
CROP_SHAPE_3D_OVERRIDES = {"celegans_atlas": (32, 140, 512)}
# SNEMI has 30 held-out slices after the training range. The general tuning code historically used
# eight of them; this benchmark needs twelve to meet the minimum representative crop depth.
VALIDATION_Z_RANGE_OVERRIDES = {"snemi": (0, 12)}
CANDIDATE_GRID_3D = (4, 3, 3)

# Opt-in deep 3d crop set. A propagation pass costs one frame step per slice, so a 12-slice crop
# leaves a depth-dependent optimization like early stopping almost nothing to skip and measures the
# crop rather than the mechanism. C. elegans keeps its Y extent of 140, and SNEMI stops at 30 because
# 30 is its entire held-out range: slices before 70 were used for training.
CROP_SHAPE_3D_DEEP = (32, 512, 512)
CROP_SHAPE_3D_DEEP_OVERRIDES = {"celegans_atlas": (32, 140, 512), "snemi": (30, 512, 512)}
VALIDATION_Z_RANGE_DEEP_OVERRIDES = {"snemi": (0, 30)}
# The candidate grid is deliberately shared with the standard set, so the two variants differ only in
# depth and in the annotation rule below. That is what makes a deep result attributable to depth.
CROP_VARIANTS_3D = ("standard", "deep")

# Conservative first-run estimates from the current A100. Observed times replace them as samples run.
INITIAL_SAMPLE_SECONDS = {2: 5.0, 3: 35.0}
VOLUME_DIAGNOSTICS = (
    "proposed_candidates", "scored_candidates", "unique_anchor_slices", "propagation_passes",
    "propagated_frame_steps", "early_stopped_frame_steps",
)

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


def _default_manifest_path(output_root: Path, variant: str, subset: str = "primary") -> Path:
    """Where a crop variant and subset keep their manifest.

    Each variant gets its own file so both crop sets stay reproducible from one revision, and the
    schema version is not bumped: `_validate_manifest` requires an exact match, so a bump would make
    the existing standard manifest unloadable and orphan every result measured against it. The
    holdout subset follows the same rule with its own suffix.
    """
    suffix = "" if variant == "standard" else f"_{variant}3d"
    if subset != "primary":
        suffix += f"_{subset}"
    return output_root / f"subset_manifest_v{MANIFEST_SCHEMA_VERSION}{suffix}.json"


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


def _select_2d_samples(
    data_root: Path,
    counts: Dict[str, int] = SAMPLE_COUNTS_2D,
    exclude_raw_paths: Optional[Dict[str, set]] = None,
    reuse_samples: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Select the 2d samples, per dataset at even complexity quantiles.

    Args:
        data_root: The read-only dataset root.
        counts: Number of samples per dataset.
        exclude_raw_paths: Raw paths (relative to the data root) that must not be selected, per
            dataset. This is how the holdout subset stays image-disjoint from the primary one.
        reuse_samples: Samples to copy verbatim instead of selecting, per dataset. This is how an
            exhausted dataset (all validation images already primary) still appears in the holdout.

    Returns:
        The selected samples, with their 'sample_id' set.
    """
    exclude_raw_paths = exclude_raw_paths or {}
    reuse_samples = reuse_samples or {}
    samples = []
    for dataset in DATASETS_2D:
        if dataset in reuse_samples:
            reused = [dict(sample) for sample in reuse_samples[dataset]]
            if len(reused) != counts[dataset]:
                raise RuntimeError(
                    f"Expected {counts[dataset]} reused samples for '{dataset}', got {len(reused)}."
                )
            samples.extend(reused)
            continue
        excluded = exclude_raw_paths.get(dataset, set())
        candidates = [
            candidate for candidate in _scan_2d_dataset(dataset, data_root)
            if candidate["raw_path"] not in excluded
        ]
        if dataset == "livecell":
            by_type = defaultdict(list)
            for candidate in candidates:
                cell_type = Path(candidate["raw_path"]).name.split("_", 1)[0]
                by_type[cell_type].append(candidate)
            missing = sorted(set(LIVECELL_TYPES) - set(by_type))
            if missing:
                raise RuntimeError(f"LIVECell validation data is missing cell types: {missing}.")
            selected = []
            n_per_type, remainder = divmod(counts[dataset], len(LIVECELL_TYPES))
            if remainder:
                raise RuntimeError("The LIVECell sample count must be divisible by its cell-type count.")
            for cell_type in LIVECELL_TYPES:
                group = by_type[cell_type]
                # Complexity is pool-relative, so it is computed on the (possibly filtered) group.
                _add_complexity(group)
                choices = _select_nearest(group, _quantile_targets(n_per_type))
                for choice in choices:
                    choice["stratum"] = cell_type
                    selected.append(choice)
        else:
            _add_complexity(candidates)
            selected = _select_nearest(candidates, _quantile_targets(counts[dataset]))
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


def _crop_shape_3d(dataset: str, variant: str) -> Tuple[int, ...]:
    """The 3d crop shape a dataset uses in the given crop variant."""
    if variant == "deep":
        return CROP_SHAPE_3D_DEEP_OVERRIDES.get(dataset, CROP_SHAPE_3D_DEEP)
    return CROP_SHAPE_3D_OVERRIDES.get(dataset, CROP_SHAPE_3D)


def _z_range_overrides_3d(variant: str) -> Dict[str, Tuple[int, int]]:
    """The per-dataset validation z-slab overrides of the given crop variant."""
    return VALIDATION_Z_RANGE_DEEP_OVERRIDES if variant == "deep" else VALIDATION_Z_RANGE_OVERRIDES


def _validation_z_range(dataset: str, depth: int, variant: str) -> Tuple[int, int]:
    base_start = 70 if dataset == "snemi" else 0
    base_stop = depth
    z_range = _z_range_overrides_3d(variant).get(dataset, VAL_Z_RANGE.get(dataset))
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
    variant: str,
) -> List[Dict[str, Any]]:
    raw_relative = _relative_data_path(raw_path, data_root)
    label_relative = _relative_data_path(label_path, data_root)
    label_source = _source_path(label_relative, data_root)
    shape = _array_shape(label_source, label_key)
    if len(shape) != 3:
        raise RuntimeError(f"Expected a 3d label volume for '{dataset}', got shape {shape} at '{label_source}'.")
    z_start, z_stop = _validation_z_range(dataset, shape[0], variant)
    valid_shape = (z_stop - z_start, shape[1], shape[2])
    crop_shape = _crop_shape_3d(dataset, variant)
    # Do not silently shrink a crop and distort the encoder scale. A dataset may expose multiple
    # source volumes (as EmbedSeg does), so skip undersized sources and select from those that fit.
    if any(size < crop for size, crop in zip(valid_shape, crop_shape)):
        return []
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
                    # _load_3d_sample trims unannotated end slices, so a crop whose first or last
                    # slice is empty propagates fewer slices than its declared depth. The deep set
                    # exists to test depth, so it only accepts crops the trim leaves alone.
                    if variant == "deep" and not (labels[0].any() and labels[-1].any()):
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


def _select_3d_samples(data_root: Path, variant: str) -> List[Dict[str, Any]]:
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
                _scan_3d_source(dataset, raw_path, label_path, raw_key, label_key, data_root, variant)
            )
        if len(candidates) < len(TARGETS_3D):
            raise RuntimeError(
                f"Only {len(candidates)} eligible 3d candidates found for '{dataset}' "
                f"in the '{variant}' crop variant."
            )
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


def _selection_policy_3d(variant: str) -> Dict[str, Any]:
    """The part of the selection policy the 3d crop variant determines."""
    return {
        "3d_crop_shapes": {dataset: list(_crop_shape_3d(dataset, variant)) for dataset in DATASETS_3D},
        "3d_candidate_grid": list(CANDIDATE_GRID_3D),
        "3d_validation_z_range_overrides": _z_range_overrides_3d(variant),
    }


def _sample_counts_2d(subset: str) -> Dict[str, int]:
    """The expected per-dataset 2d sample counts of a manifest subset."""
    if subset not in MANIFEST_SUBSETS:
        raise ValueError(f"Unknown manifest subset '{subset}'; expected one of {list(MANIFEST_SUBSETS)}.")
    return SAMPLE_COUNTS_2D_HOLDOUT if subset == "holdout" else SAMPLE_COUNTS_2D


def _validate_manifest(manifest: Dict[str, Any], data_root: Path, variant: str, subset: str = "primary") -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported manifest schema {manifest.get('schema_version')}; expected {MANIFEST_SCHEMA_VERSION}."
        )
    expected_checksum = _content_checksum(_manifest_identity(manifest))
    if manifest.get("manifest_checksum") != expected_checksum:
        raise RuntimeError("The subset manifest content does not match its checksum.")
    # prepare_manifest reuses an existing manifest instead of rebuilding it, so a stored subset has
    # to be checked against the active crop variant. Otherwise a deep run pointed at the standard
    # manifest would silently benchmark 12-slice crops. Both sides go through JSON because a manifest
    # is validated twice: once freshly built, still holding tuples, and once loaded back as lists.
    expected_policy = json.loads(_json_bytes(_selection_policy_3d(variant)))
    policy = manifest.get("selection_policy", {})
    stored_policy = json.loads(_json_bytes({key: policy.get(key) for key in expected_policy}))
    if stored_policy != expected_policy:
        raise RuntimeError(
            f"The subset manifest was not built for the '{variant}' 3d crop variant: "
            f"{stored_policy} != {expected_policy}."
        )
    # A pre-holdout manifest carries no 'subset' key, so its absence means the primary subset. This
    # keeps every existing primary manifest, and hence every result keyed on its checksum, valid.
    stored_subset = policy.get("subset", "primary")
    if stored_subset != subset:
        raise RuntimeError(f"The manifest holds the '{stored_subset}' subset, but '{subset}' was requested.")
    if subset == "holdout" and not policy.get("holdout_of"):
        raise RuntimeError("A holdout manifest must record the primary manifest it was built against.")
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
    sample_counts = _sample_counts_2d(subset)
    expected = {(dataset, 2): sample_counts[dataset] for dataset in DATASETS_2D}
    expected.update({(dataset, 3): 1 for dataset in DATASETS_3D})
    if dict(counts) != expected:
        raise RuntimeError(f"Unexpected manifest sample counts: got {dict(counts)}, expected {expected}.")


def _holdout_2d_samples(data_root: Path, primary: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select the holdout 2d samples: image-disjoint from the primary subset, except where exhausted.

    Returns:
        The samples, and the policy entries that record what the holdout was built against.
    """
    primary_raw_paths = defaultdict(set)
    primary_by_dataset = defaultdict(list)
    for sample in primary["samples"]:
        if sample["ndim"] != 2:
            continue
        primary_raw_paths[sample["dataset"]].add(sample["raw_path"])
        primary_by_dataset[sample["dataset"]].append(sample)

    reuse = {dataset: primary_by_dataset[dataset] for dataset in HOLDOUT_REUSED_DATASETS}
    samples = _select_2d_samples(
        data_root, counts=SAMPLE_COUNTS_2D_HOLDOUT, exclude_raw_paths=primary_raw_paths, reuse_samples=reuse,
    )

    # The whole point of the holdout is disjointness, so it is asserted rather than assumed.
    for sample in samples:
        if sample["dataset"] in HOLDOUT_REUSED_DATASETS:
            continue
        if sample["raw_path"] in primary_raw_paths[sample["dataset"]]:
            raise RuntimeError(f"Holdout sample '{sample['sample_id']}' reuses a primary image.")

    policy = {
        "subset": "holdout",
        "holdout_of": primary["manifest_checksum"],
        # Not held out: every validation image of these datasets is already in the primary subset.
        "reused_datasets": list(HOLDOUT_REUSED_DATASETS),
    }
    return samples, policy


def prepare_manifest(
    data_root: Path, manifest_path: Path, variant: str = "standard", subset: str = "primary",
    primary_manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if variant not in CROP_VARIANTS_3D:
        raise ValueError(f"Unknown 3d crop variant '{variant}'; expected one of {list(CROP_VARIANTS_3D)}.")
    sample_counts = _sample_counts_2d(subset)
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        _validate_manifest(manifest, data_root, variant, subset)
        return manifest

    subset_policy = {}
    if subset == "holdout":
        # The holdout is defined by exclusion, so the primary manifest it excludes must exist first.
        primary_path = primary_manifest_path or _default_manifest_path(manifest_path.parent, variant)
        if not primary_path.exists():
            raise RuntimeError(
                f"The holdout subset is built against the primary manifest, which does not exist yet: "
                f"'{primary_path}'. Prepare the primary manifest first."
            )
        with open(primary_path) as f:
            primary = json.load(f)
        _validate_manifest(primary, data_root, variant, "primary")
        samples_2d, subset_policy = _holdout_2d_samples(data_root, primary)
        # The volumes are carried over verbatim: the holdout is a 2d instrument, but the schema and
        # its validator expect one volume per 3d dataset, and a copied volume keeps `--ndim 3` honest.
        samples = samples_2d + [dict(sample) for sample in primary["samples"] if sample["ndim"] == 3]
    else:
        samples = _select_2d_samples(data_root) + _select_3d_samples(data_root, variant)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "data_root": str(data_root),
        "selection_policy": {
            "2d_crop_shape": list(CROP_SHAPE_2D),
            "2d_sample_counts": sample_counts,
            "2d_complexity_targets": "even quantile midpoints within each dataset and LIVECell cell type",
            "3d_complexity_targets": list(TARGETS_3D),
            "complexity": "mean percentile rank of object count and foreground fraction",
            **subset_policy,
            **_selection_policy_3d(variant),
        },
        "samples": samples,
    }
    manifest["manifest_checksum"] = _content_checksum(_manifest_identity(manifest))
    _validate_manifest(manifest, data_root, variant, subset)
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


def _realized_depth_3d(sample: Dict[str, Any], data_root: Path) -> int:
    """The number of slices a 3d sample actually propagates through.

    The manifest roi always has the declared crop depth, but `_load_3d_sample` drops unannotated end
    slices, so the depth a run really sees can be smaller. Reported by --prepare-only, because a crop
    set chosen for its depth is only worth running once the depth is known to survive the trim.
    """
    source_roi = _roi_from_json(sample["roi"])
    labels = _read_array(_source_path(sample["label_path"], data_root), sample["label_key"], roi=source_roi)
    annotated = np.any(labels != 0, axis=(1, 2))
    if not annotated.any():
        return 0
    return int(len(annotated) - int(np.argmax(annotated[::-1])) - int(np.argmax(annotated)))


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
        if "peak_cuda_memory_bytes" in group:
            values = group["peak_cuda_memory_bytes"].dropna()
            row["peak_cuda_memory_bytes"] = int(values.max()) if len(values) else np.nan
        for metric in metric_columns:
            values = group[metric].dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=0)) if len(values) else np.nan
        for diagnostic in ("unmatched", "genuine_misses", *VOLUME_DIAGNOSTICS):
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
    if "peak_cuda_memory_bytes" in samples:
        values = samples["peak_cuda_memory_bytes"].dropna()
        overall["peak_cuda_memory_bytes"] = int(values.max()) if len(values) else np.nan
    for metric in metric_columns:
        values = summary[f"{metric}_mean"].dropna()
        overall[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
        overall[f"{metric}_std"] = float(values.std(ddof=0)) if len(values) else np.nan
    for diagnostic in ("unmatched", "genuine_misses", *VOLUME_DIAGNOSTICS):
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
    initialization_seconds: float, generation_seconds: float, peak_cuda_memory_bytes: Optional[int],
    generation_stats: Optional[Dict[str, int]] = None,
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
        "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
        **metrics,
    }
    if sample["ndim"] == 3:
        row["unmatched"], row["genuine_misses"] = genuine_misses(labels, segmentation)
        generation_stats = generation_stats or {}
        row.update({key: int(generation_stats.get(key, 0)) for key in VOLUME_DIAGNOSTICS})
    return row


def _run_sample(
    segmenter: Any, sample: Dict[str, Any], raw: np.ndarray, labels: np.ndarray,
    params: Dict[str, Any], device: str,
) -> Dict[str, Any]:
    segmenter.clear_state()
    cuda_device = torch.device(device) if device.startswith("cuda") else None
    if cuda_device is not None:
        torch.cuda.reset_peak_memory_stats(cuda_device)
    start = time.perf_counter()
    segmenter.initialize(raw, ndim=sample["ndim"], **(VOLUME_SPEED_OPTIONS if sample["ndim"] == 3 else {}))
    initialized = time.perf_counter()
    generation_kwargs = dict(params)
    if sample["ndim"] == 3:
        generation_kwargs["spacing"] = DATASET_SPACING.get(sample["dataset"])
    segmentation = segmenter.generate(**generation_kwargs).astype("uint32")
    generated = time.perf_counter()
    peak_cuda_memory_bytes = (
        int(torch.cuda.max_memory_allocated(cuda_device)) if cuda_device is not None else None
    )
    return _sample_row(
        sample, segmentation, labels,
        initialization_seconds=initialized - start,
        generation_seconds=generated - initialized,
        peak_cuda_memory_bytes=peak_cuda_memory_bytes,
        generation_stats=getattr(segmenter, "_last_generation_stats", None),
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
            row = _run_sample(segmenter, sample, raw, labels, params, device)
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
    device: str, time_budget_minutes: float, dimensions: Sequence[int] = (2, 3),
    trial_id: str = "trial-1", started: Optional[float] = None, crops_3d: str = "standard",
    subset: str = "primary",
) -> Tuple[Path, pd.DataFrame, Dict[str, Any]]:
    started = time.perf_counter() if started is None else started
    checkpoint_path = get_joint_checkpoint(model_type, joint_checkpoint)
    checkpoint_id = checkpoint_checksum(checkpoint_path)
    implementation_checksum = _implementation_checksum()
    dimensions = tuple(sorted(set(dimensions)))
    if not dimensions or not set(dimensions).issubset({2, 3}):
        raise ValueError(f"Dimensions must be a non-empty subset of (2, 3), got {dimensions}.")
    if not trial_id:
        raise ValueError("The timing trial id must not be empty.")
    config_identity = {
        "params_2d": params_2d,
        "params_3d": params_3d,
        "dimensions": dimensions,
        "trial_id": trial_id,
    }
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
        "dimensions": list(dimensions),
        "trial_id": trial_id,
        # Recorded for attribution only. The crop variant and subset already reach the run directory
        # through the manifest checksum, so they must not enter a checksum of their own.
        "crops_3d": crops_3d,
        "subset": subset,
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
        params_by_dimension = {2: params_2d, 3: params_3d}
        for ndim in dimensions:
            completed = _run_dimension(
                ndim, manifest["samples"], completed, samples_path, data_root, model_type,
                joint_checkpoint, checkpoint_id, export_root, params_by_dimension[ndim], device, started,
                time_budget_minutes * 60,
            )
        expected_ids = {
            sample["sample_id"] for sample in manifest["samples"] if sample["ndim"] in dimensions
        }
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
    parser.add_argument(
        "--ndim", choices=("2", "3", "both"), default="both",
        help="Evaluate only images, only volumes, or both (default).",
    )
    parser.add_argument(
        "--trial-id", default="trial-1",
        help="Identity of this serialized timing trial; changing it creates an independent result.",
    )
    parser.add_argument("--model-type", default="hvit_t", choices=common.MODEL_TYPES)
    parser.add_argument("--joint-checkpoint", default="best", help="Joint checkpoint name without '.pt'.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--time-budget-minutes", type=float, default=30.0)
    parser.add_argument("--prepare-only", action="store_true", help="Create and validate the subset, then stop.")
    parser.add_argument(
        "--crops-3d", choices=CROP_VARIANTS_3D, default="standard",
        help="The 3d crop set. 'deep' opts in to 32-slice crops, the depth a depth-dependent "
             "optimization needs to be measurable, and keeps its own manifest beside the standard one.",
    )
    parser.add_argument(
        "--subset", choices=MANIFEST_SUBSETS, default="primary",
        help="The validation subset. 'holdout' is image-disjoint from the primary subset (except "
             "the exhausted DeepBacs) and confirms configurations that were tuned on it.",
    )
    args = parser.parse_args()

    manifest_path = args.manifest or _default_manifest_path(args.output_root, args.crops_3d, args.subset)
    data_root, output_root, manifest_path = _validate_roots(args.data_root, args.output_root, manifest_path)
    if args.time_budget_minutes <= 0:
        parser.error("--time-budget-minutes must be positive.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("A CUDA device was requested, but CUDA is not available.")
    if not args.device.startswith("cuda"):
        print("Warning: the 30-minute runtime target was calibrated on an A100, not on CPU.", file=sys.stderr)

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = prepare_manifest(data_root, manifest_path, args.crops_3d, subset=args.subset)
    print(
        f"Manifest: {manifest_path}\n"
        f"Checksum: {manifest['manifest_checksum']}\n"
        f"3d crops: {args.crops_3d}\n"
        f"Subset: {args.subset}\n"
        f"Samples: {sum(sample['ndim'] == 2 for sample in manifest['samples'])} 2d + "
        f"{sum(sample['ndim'] == 3 for sample in manifest['samples'])} 3d"
    )
    dimensions = (2, 3) if args.ndim == "both" else (int(args.ndim),)
    if args.prepare_only:
        for sample in manifest["samples"]:
            if sample["ndim"] != 3:
                continue
            declared = _crop_shape_3d(sample["dataset"], args.crops_3d)[0]
            realized = _realized_depth_3d(sample, data_root)
            flag = "" if realized == declared else f"  <- trimmed from the declared depth {declared}"
            print(f"  {sample['dataset']:16s} roi {sample['roi']} depth {realized} "
                  f"objects {sample['object_count']}{flag}")
    if args.prepare_only:
        return

    config_name, params_2d, params_3d = _load_config(args.config)
    run_dir, summary, metadata = run_benchmark(
        manifest, data_root, output_root, args.model_type, args.joint_checkpoint,
        config_name, params_2d, params_3d, args.device, args.time_budget_minutes,
        dimensions=dimensions, trial_id=args.trial_id, started=started, crops_3d=args.crops_3d,
        subset=args.subset,
    )
    print(summary.to_string(index=False))
    print(f"Run directory: {run_dir}")
    print(f"Wall time: {metadata['wall_seconds'] / 60:.2f} minutes")


if __name__ == "__main__":
    main()
