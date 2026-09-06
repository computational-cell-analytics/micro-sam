"""Benchmark AIS (decoder-based automatic instance segmentation) post-processing on cached predictions.

The UniSAM2 decoder prediction of a sample, a (4, *spatial) array of foreground probability and three
directed-distance channels, does not depend on any post-processing choice. This benchmark therefore
predicts every sample of a manifest once (`predict`, GPU), caches the prediction, and runs every
post-processing configuration, diagnostic and parameter sweep on the cache (CPU). A configuration run
still writes a canonical run directory in the layout of `benchmark_apg_optimization.py`, so
`compare_apg_optimization.py` reads it unchanged.

Manifests are reused from the APG campaigns: the 2d subset manifests (`--kind v5`: primary, holdout,
training_extra; 240 / 233 / 157 images plus five standard volumes) and the deep 3d crop manifests
(`--kind apg3d`: primary, holdout, test). Nothing is rebuilt and the data root is read-only.

Usage examples:
    # Cache the predictions of the primary subset on the session GPU.
    python benchmark_ais_optimization.py predict --kind v5 --subset primary

    # Run the library defaults on the cache (the baseline) and a candidate.
    python benchmark_ais_optimization.py run --kind v5 --subset primary
    python benchmark_ais_optimization.py run --kind v5 --subset primary --config configs/ais_travel_200.json

    # Screen several configurations on two subsets, then report them against the baseline.
    python benchmark_ais_optimization.py screen --kind v5 --subset primary training_extra \\
        --configs configs/ais_control_registry_defaults.json configs/ais_s0_*.json --name s0_screen
    python benchmark_ais_optimization.py report --index <root>/ais/screens/<stamp>_s0_screen.json

    # Sweep a parameter grid on the cache, one shard of the grid per task.
    python benchmark_ais_optimization.py sweep --kind v5 --subset primary --grid configs/ais_grid_lm.json \\
        --datasets livecell --shard-index 0 --num-shards 4

The configuration file has this shape (a flat parameter dict is read as sparse overrides):
    {
      "name": "travel-200",
      "mode": "auto",
      "params_2d": {"sparse": {"n_iter": 200, "dt": 1.0}, "dense": {"beta": 0.6}},
      "params_3d": {"n_iter": 200}
    }
"""

from __future__ import annotations

import argparse
import datetime
import glob
import itertools
import json
import platform
import sys
import time
from concurrent import futures
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import xxhash

EVALUATION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVALUATION_ROOT))

import common  # noqa
from common import (  # noqa
    DATASETS_3D_EM, DATASET_SPACING, GT_MIN_SIZE_2D, build_model, checkpoint_checksum, drop_severed_objects,
    export_joint_checkpoint, get_joint_checkpoint, predict_unisam2,
)
from parameter_search import (  # noqa
    compute_metrics, dense_boundary_and_distances, deduplicate_flow_travel, score_image_dense_cached,
    score_image_sparse_cached,
)
from optimization import apg3d_manifest  # noqa
from optimization.benchmark_apg_optimization import (  # noqa
    DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, MANIFEST_SUBSETS, _atomic_write_csv, _atomic_write_json,
    _content_checksum, _default_manifest_path, _git_revision, _hardware_identity, _load_2d_sample,
    _load_3d_sample, _load_normalized_3d_source, prepare_manifest,
)
from optimization.benchmark_apg_3d import _bootstrap_ci  # noqa

from micro_sam.v2.postprocessing import (  # noqa
    _compute_flow_density, default_postprocessing, drop_instances_without_boundary_dip, flow_instance_segmentation,
    run_multicut, watershed_heightmap,
)
from bioimage_cpp.segmentation import label as connected_components, watershed  # noqa

REPOSITORY_ROOT = EVALUATION_ROOT.parents[2]
CAMPAIGN = "ais"
KINDS = ("v5", "apg3d")
MODES = ("auto", "sparse", "dense")
BALANCED_ROW = "__dataset_balanced__"

# The keywords of the two post-processing functions, i.e. what a configuration may override.
SPARSE_KEYS = (
    "foreground_threshold", "n_iter", "dt", "sigma", "density_threshold", "min_size", "foreground_weight",
    "boundary_magnitude_max",
)
DENSE_KEYS = ("beta", "density_threshold", "n_iter", "dt", "sigma")
# Metric columns of a sample row; means and standard deviations are reported per dataset.
METRIC_COLUMNS = ("msa", "cremi", "vi_split", "vi_merge", "adapted_rand", "fg_iou", "matched_iou")
# Count columns; sums are reported per dataset.
COUNT_COLUMNS = (
    "gt_objects", "predicted_objects", "matched", "unmatched", "severed_objects", "genuine_misses",
    "matched_before_min_size", "n_seeds", "gt_with_0_seeds", "gt_with_1_seed", "gt_with_2plus_seeds",
    "background_seeds", "seeded_unmatched", "seeded_split", "seeded_merged", "seeded_undersized",
    "seeded_oversized", "unseeded_absorbed", "unseeded_missing", "pipeline_mismatch",
)
# The generalization gate of the 2026-09 screens (EXPERIMENTAL_SETUP.md, section 9).
GATE = {"max_down": 2, "max_relative_loss": -0.02, "max_absolute_loss": -0.005, "min_balanced_gain": 0.02}

IMPLEMENTATION_FILES = (
    Path(__file__),
    Path(common.__file__),
    EVALUATION_ROOT / "parameter_search.py",
    REPOSITORY_ROOT / "micro_sam/v2/instance_segmentation.py",
    REPOSITORY_ROOT / "micro_sam/v2/postprocessing.py",
)


def implementation_checksum() -> str:
    """Hash the code that determines the prediction, the post-processing and the scoring."""
    checksum = xxhash.xxh128()
    for path in IMPLEMENTATION_FILES:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                checksum.update(block)
        checksum.update(b"\0")
    return checksum.hexdigest()


# ----------------------------------------------------------------------------------------------
# configurations


def resolve_postprocessing(
    overrides: Optional[Dict[str, Any]], model_type: str, ndim: int = 2,
) -> Dict[str, Dict[str, Any]]:
    """The sparse and dense parameters a run uses, with 'overrides' on top of the library defaults.

    A flat dict is read as sparse overrides; the nested form ``{"sparse": {...}, "dense": {...}}`` sets
    both. 'ndim' selects the image or volume defaults. The result is what `flow_instance_segmentation` /
    `run_multicut` receive, so a run without overrides is exactly the library default and shares its run
    directory with an explicit copy of it.
    """
    overrides = dict(overrides or {})
    if set(overrides) & {"sparse", "dense"}:
        unknown = set(overrides) - {"sparse", "dense"}
        if unknown:
            raise ValueError(f"A nested configuration may only contain 'sparse' and 'dense', got {sorted(unknown)}.")
        sparse, dense = dict(overrides.get("sparse", {})), dict(overrides.get("dense", {}))
    else:
        sparse, dense = overrides, {}
    unknown_sparse, unknown_dense = set(sparse) - set(SPARSE_KEYS), set(dense) - set(DENSE_KEYS)
    if unknown_sparse or unknown_dense:
        raise ValueError(f"Unknown AIS parameters: sparse={sorted(unknown_sparse)}, dense={sorted(unknown_dense)}.")
    return {
        "sparse": {**default_postprocessing(model_type, "sparse", ndim=ndim), **sparse},
        "dense": {**default_postprocessing(model_type, "dense", ndim=ndim), **dense},
    }


def load_config(path: Optional[Path], model_type: str) -> Tuple[str, str, Dict[str, Any], Dict[str, Any]]:
    """Read one configuration file: its name, mode and the resolved 2d and 3d parameters."""
    if path is None:
        config: Dict[str, Any] = {"name": "current-defaults"}
    else:
        with open(path) as f:
            config = json.load(f)
    unknown = set(config) - {"name", "mode", "params_2d", "params_3d"}
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}.")
    mode = config.get("mode", "auto")
    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}'; expected one of {MODES}.")
    name = config.get("name", path.stem if path is not None else "current-defaults")
    params_2d = resolve_postprocessing(config.get("params_2d", {}), model_type, ndim=2)
    # Without its own overrides a volume takes the image overrides, over the library's volume defaults.
    params_3d = resolve_postprocessing(config.get("params_3d", config.get("params_2d", {})), model_type, ndim=3)
    return str(name), mode, params_2d, params_3d


# ----------------------------------------------------------------------------------------------
# manifests and samples


def load_campaign_manifest(kind: str, subset: str, output_root: Path, data_root: Path, campaign_root: Path) -> Dict:
    """The 2d subset manifest (`v5`) or the deep 3d crop manifest (`apg3d`) of one subset, validated."""
    if kind == "v5":
        if subset not in MANIFEST_SUBSETS:
            raise ValueError(f"Unknown v5 subset '{subset}'; expected one of {MANIFEST_SUBSETS}.")
        manifest_path = _default_manifest_path(output_root, "standard", subset)
        if not manifest_path.exists():
            raise FileNotFoundError(f"The manifest does not exist and is not rebuilt here: '{manifest_path}'.")
        manifest = prepare_manifest(data_root, manifest_path, "standard", subset=subset)
        manifest["kind"], manifest["subset"] = kind, subset
        return manifest
    if kind == "apg3d":
        manifest = apg3d_manifest.load_manifest(subset, campaign_root, data_root)
        manifest["kind"] = kind
        return manifest
    raise ValueError(f"Unknown manifest kind '{kind}'; expected one of {KINDS}.")


def sample_context(sample: Dict[str, Any], kind: str, mode: str) -> Dict[str, Any]:
    """Metric mode, post-processing mode, spacing and border size floor of one sample."""
    ndim = int(sample["ndim"])
    if kind == "apg3d":
        metric_mode = sample["metric_mode"]
        spacing = tuple(sample["spacing"]) if sample.get("spacing") else None
    else:
        metric_mode = "dense" if sample["dataset"] in DATASETS_3D_EM else "sparse"
        spacing = DATASET_SPACING.get(sample["dataset"]) if ndim == 3 else None
    if spacing is not None and tuple(spacing) == (1, 1, 1):
        spacing = None
    dense = (metric_mode == "dense") if mode == "auto" else (mode == "dense")
    return {
        "ndim": ndim,
        "metric_mode": metric_mode,
        "postprocessing_mode": "dense" if dense else "sparse",
        "spacing": spacing,
        "border_min_size": GT_MIN_SIZE_2D.get(sample["dataset"], 0) if ndim == 2 else 0,
    }


def sample_file_stem(sample: Dict[str, Any]) -> str:
    return sample["sample_id"].replace(":", "_")


class SampleLoader:
    """Loads raw data and labels of manifest samples, caching the normalized 3d source volume."""

    def __init__(self, kind: str, data_root: Path) -> None:
        self.kind, self.data_root = kind, data_root
        self._source_key: Optional[tuple] = None
        self._source: Optional[np.ndarray] = None

    def _normalized_source(self, sample: Dict[str, Any]) -> np.ndarray:
        key = (sample["raw_path"], tuple(sample["normalization_z_range"]))
        if key != self._source_key:
            self._source = None
            if self.kind == "apg3d":
                self._source = apg3d_manifest.load_normalized_source(sample, self.data_root)
            else:
                self._source = _load_normalized_3d_source(sample, self.data_root)
            self._source_key = key
        return self._source

    def load(self, sample: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """The sample's raw data, connected-component labels and valid mask (None unless partially annotated)."""
        if self.kind == "apg3d":
            return apg3d_manifest.load_sample(sample, self.data_root, self._normalized_source(sample))
        if int(sample["ndim"]) == 2:
            raw, labels = _load_2d_sample(sample, self.data_root)
        else:
            raw, labels = _load_3d_sample(sample, self.data_root, self._normalized_source(sample))
        return raw, labels, None


# ----------------------------------------------------------------------------------------------
# prediction cache


class PredictionCache:
    """Decoder predictions of one manifest, one file per sample, below
    '<output root>/ais/predictions/<checkpoint id>/<manifest checksum>/'."""

    def __init__(self, output_root: Path, checkpoint_id: str, manifest_checksum: str) -> None:
        self.root = output_root / CAMPAIGN / "predictions" / checkpoint_id / manifest_checksum
        self.checkpoint_id = checkpoint_id

    def paths(self, sample: Dict[str, Any]) -> Tuple[Path, Path]:
        stem = sample_file_stem(sample)
        return self.root / f"{stem}.npz", self.root / f"{stem}.json"

    def has(self, sample: Dict[str, Any]) -> bool:
        return all(path.exists() for path in self.paths(sample))

    def load(self, sample: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
        """The cached prediction, labels, valid mask and the prediction record of one sample.

        Every array is read once in full; indexing a compressed archive per row decompresses it again.
        """
        array_path, record_path = self.paths(sample)
        with np.load(array_path) as data:
            prediction = np.ascontiguousarray(data["prediction"], dtype="float32")
            labels = np.ascontiguousarray(data["labels"], dtype="uint32")
            valid = np.ascontiguousarray(data["valid"], dtype=bool) if "valid" in data.files else None
        with open(record_path) as f:
            record = json.load(f)
        return prediction, labels, valid, record

    def store(
        self, sample: Dict[str, Any], prediction: np.ndarray, labels: np.ndarray, valid: Optional[np.ndarray],
        record: Dict[str, Any],
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        array_path, record_path = self.paths(sample)
        arrays = {
            "prediction": prediction.astype("float32", copy=False), "labels": labels.astype("uint32", copy=False),
        }
        if valid is not None:
            arrays["valid"] = valid.astype(bool, copy=False)
        tmp = array_path.with_suffix(".tmp.npz")
        # Uncompressed: a screen reads every file many times, and float32 predictions compress poorly anyway.
        np.savez(tmp, **arrays)
        tmp.replace(array_path)
        _atomic_write_json(record_path, record)

    def records(self, samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        records = []
        for sample in samples:
            _, record_path = self.paths(sample)
            if record_path.exists():
                with open(record_path) as f:
                    records.append(json.load(f))
        return records


class Predictor:
    """Builds the UniSAM2 decoder on first use and predicts one sample at a time.

    The decoder half of the joint checkpoint is exported below '<output root>/model_exports', keyed by
    the checkpoint checksum, like the APG benchmark does (the library's default export root is not
    writable for every user).
    """

    def __init__(
        self, model_type: str, joint_checkpoint: str, checkpoint_id: str, device: str, output_root: Path,
    ) -> None:
        self.model_type, self.joint_checkpoint, self.checkpoint_id, self.device = (
            model_type, joint_checkpoint, checkpoint_id, device,
        )
        self.export_root = output_root / "model_exports"
        self._model = None

    @property
    def model(self):
        if self._model is None:
            _, decoder_path = export_joint_checkpoint(
                self.model_type, self.joint_checkpoint, source_checksum=self.checkpoint_id,
                export_root=str(self.export_root),
            )
            self._model = build_model(
                mode="ais", model_type=self.model_type, device=self.device, ndim=2, checkpoint_path=decoder_path,
            )
        return self._model

    def predict(self, raw: np.ndarray, ndim: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        cuda_device = torch.device(self.device) if self.device.startswith("cuda") else None
        if cuda_device is not None:
            torch.cuda.reset_peak_memory_stats(cuda_device)
        started = time.perf_counter()
        prediction = predict_unisam2(self.model, raw, ndim=ndim, device=self.device)
        seconds = time.perf_counter() - started
        record = {
            "predict_seconds": seconds,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(cuda_device)) if cuda_device else None,
            "device": self.device,
            "hardware": _hardware_identity(self.device),
            "checkpoint_checksum": self.checkpoint_id,
            "checkpoint_name": self.joint_checkpoint,
            "model_type": self.model_type,
            "implementation_checksum": implementation_checksum(),
            "git_revision": _git_revision(),
            "torch": torch.__version__,
            "shape": list(prediction.shape),
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        return np.ascontiguousarray(prediction, dtype="float32"), record


def ensure_prediction(
    cache: PredictionCache, sample: Dict[str, Any], loader: SampleLoader, predictor: Optional[Predictor],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Read a sample from the cache, predicting and caching it first when it is missing."""
    if cache.has(sample):
        return cache.load(sample)
    if predictor is None:
        raise FileNotFoundError(
            f"No cached prediction for '{sample['sample_id']}' under '{cache.root}'. Run 'predict' first, or pass "
            "--predict-missing."
        )
    raw, labels, valid = loader.load(sample)
    prediction, record = predictor.predict(raw, int(sample["ndim"]))
    record["sample_id"] = sample["sample_id"]
    cache.store(sample, prediction, labels, valid, record)
    return prediction, labels, valid, record


# ----------------------------------------------------------------------------------------------
# post-processing, mirrored pipeline and diagnostics


def segment_prediction(
    prediction: np.ndarray, params: Dict[str, Any], dense: bool, spacing: Optional[tuple], model_type: str,
    n_threads: int,
) -> np.ndarray:
    """Post-process one prediction exactly like `common.postprocess_unisam2` does in production."""
    if dense:
        boundary_map, distances = dense_boundary_and_distances(prediction)
        if boundary_map.ndim == 2:
            seg = run_multicut(
                boundary_map[None], distances[:, None], model_type=model_type, n_threads=n_threads, **params,
            )[0]
        else:
            seg = run_multicut(boundary_map, distances, model_type=model_type, n_threads=n_threads, **params)
    else:
        seg = flow_instance_segmentation(
            prediction[0], prediction[1:], model_type=model_type, spacing=spacing, n_threads=n_threads, **params,
        )
    return seg.astype("uint32")


def sparse_pipeline(
    prediction: np.ndarray, params: Dict[str, Any], spacing: Optional[tuple], n_threads: int,
) -> Dict[str, np.ndarray]:
    """`flow_instance_segmentation` step by step, keeping the intermediates the diagnostics read.

    'params' must be fully resolved (see `resolve_postprocessing`). The segmentation must equal the
    library's; `score_sample` records a mismatch per sample, which is the bit-identity check of an epoch.
    """
    foreground, directed = prediction[0], prediction[1:]
    ndim = foreground.ndim
    if directed.shape[0] > ndim:
        directed = directed[-ndim:]
    fg_mask = foreground > params["foreground_threshold"]
    density = _compute_flow_density(
        directed, fg_mask, n_iter=int(params["n_iter"]), dt=params["dt"], sigma=params["sigma"], spacing=spacing,
        n_threads=n_threads,
    )
    seeds = connected_components(density > params["density_threshold"])
    hmap = watershed_heightmap(foreground, directed, params["foreground_weight"])
    before = watershed(hmap, markers=seeds, mask=fg_mask)
    seg = before
    min_size = int(params["min_size"])
    if min_size > 0:
        ids, sizes = np.unique(before, return_counts=True)
        discard = ids[(sizes < min_size) & (ids > 0)]
        seg = before.copy()
        seg[np.isin(seg, discard)] = 0
        seg = watershed(hmap, markers=seg, mask=fg_mask)
    max_median = params.get("boundary_magnitude_max")
    if max_median is not None and np.isfinite(max_median):
        seg = drop_instances_without_boundary_dip(seg, directed, max_median)
    return {
        "segmentation": seg.astype("uint32"), "before_min_size": before.astype("uint32"), "seeds": seeds,
        "fg_mask": fg_mask, "density": density, "heightmap": hmap,
    }


def contingency(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Overlap counts of every (a, b) label pair with at least one non-zero label."""
    mask = (a != 0) | (b != 0)
    av, bv = a[mask].astype("int64", copy=False), b[mask].astype("int64", copy=False)
    if av.size == 0:
        empty = np.array([], dtype="int64")
        return empty, empty, empty
    stride = int(bv.max()) + 1
    keys, counts = np.unique(av * stride + bv, return_counts=True)
    return keys // stride, keys % stride, counts


def matched_ids(labels: np.ndarray, segmentation: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """The ground-truth ids that some predicted instance matches at the IoU threshold.

    At a threshold of 0.5 or more at most one instance can match an object, and it is the instance with
    the largest overlap, so this equals the complement of `common.unmatched_objects`.
    """
    gt, seg, inter = contingency(labels, segmentation)
    keep = (gt != 0) & (seg != 0)
    gt, seg, inter = gt[keep], seg[keep], inter[keep]
    if gt.size == 0:
        return np.array([], dtype="int64")
    gt_sizes = np.bincount(labels.ravel().astype("int64"))
    seg_sizes = np.bincount(segmentation.ravel().astype("int64"))
    iou = inter / (gt_sizes[gt] + seg_sizes[seg] - inter)
    return np.unique(gt[iou >= iou_threshold])


def object_counts(labels: np.ndarray, segmentation: np.ndarray, max_span: int = 2) -> Dict[str, int]:
    """Ground-truth object counts of one sample: all, crop-severed (volumes), matched, unmatched, genuine misses.

    The same numbers as `benchmark_apg_3d.object_counts`, computed from one contingency table instead of
    one pass over the volume per object. Severed objects (those spanning at most 'max_span' slices) are
    only defined for a volume; an image reports 0 severed objects and every miss as genuine. 'matched'
    counts matched objects among the unsevered ones (the reference calls it 'merged').
    """
    gt_ids = np.unique(labels)
    gt_ids = gt_ids[gt_ids != 0]
    matched = matched_ids(labels, segmentation)
    if labels.ndim == 3:
        spans = np.zeros(int(labels.max()) + 1, dtype="int64")
        for plane in labels:
            spans[np.unique(plane)] += 1
        severed = gt_ids[spans[gt_ids] <= max_span]
    else:
        severed = np.array([], dtype=gt_ids.dtype)
    unmatched = np.setdiff1d(gt_ids, matched, assume_unique=True)
    # Like the reference, 'matched' counts the objects the crop did not sever, so that
    # gt_objects = severed_objects + matched + genuine_misses.
    return {
        "gt_objects": int(len(gt_ids)),
        "severed_objects": int(len(severed)),
        "matched": int(len(np.setdiff1d(matched, severed, assume_unique=True))),
        "unmatched": int(len(unmatched)),
        "genuine_misses": int((~np.isin(unmatched, severed)).sum()),
        "predicted_objects": int(len(np.unique(segmentation)) - 1),
    }


def object_fates(labels: np.ndarray, segmentation: np.ndarray) -> Dict[str, np.ndarray]:
    """What became of every ground-truth object: its majority instance, the IoU with it, and flags.

    Returns arrays over the ground-truth ids ('ids'): 'iou' (with the instance overlapping most of the
    object, 0 without any), 'absorbed' (that instance covers at least half of the object), 'merged' (that
    instance covers at least half of two or more objects) and 'undersized' (the instance is smaller than
    the object).
    """
    ids = np.unique(labels)
    ids = ids[ids != 0]
    gt, seg, inter = contingency(labels, segmentation)
    keep = (gt != 0) & (seg != 0)
    gt, seg, inter = gt[keep], seg[keep], inter[keep]
    n = int(labels.max()) + 1
    iou = np.zeros(n, dtype="float64")
    absorbed = np.zeros(n, dtype=bool)
    merged = np.zeros(n, dtype=bool)
    undersized = np.zeros(n, dtype=bool)
    if gt.size:
        gt_sizes = np.bincount(labels.ravel().astype("int64"), minlength=n)
        seg_sizes = np.bincount(segmentation.ravel().astype("int64"))
        order = np.lexsort((-inter, gt))
        first = np.ones(len(order), dtype=bool)
        first[1:] = gt[order][1:] != gt[order][:-1]
        major_gt, major_seg, major_inter = gt[order][first], seg[order][first], inter[order][first]
        iou[major_gt] = major_inter / (gt_sizes[major_gt] + seg_sizes[major_seg] - major_inter)
        strong = major_inter >= 0.5 * gt_sizes[major_gt]
        absorbed[major_gt] = strong
        claims = np.bincount(major_seg[strong], minlength=len(seg_sizes))
        merged[major_gt] = strong & (claims[major_seg] >= 2)
        undersized[major_gt] = seg_sizes[major_seg] < gt_sizes[major_gt]
    return {
        "ids": ids, "iou": iou[ids], "absorbed": absorbed[ids], "merged": merged[ids], "undersized": undersized[ids],
    }


def seed_diagnostics(
    intermediates: Dict[str, np.ndarray], labels: np.ndarray, segmentation: np.ndarray,
) -> Dict[str, Any]:
    """Where the sparse pipeline loses objects: the seeds, the size filter, or the assignment.

    Per ground-truth object the number of seed components inside it (0 = a miss before any
    assignment, 2+ = a split), seeds whose majority pixel is background, objects matched before the
    size filter, the IoU of the thresholded foreground with the ground-truth foreground, and the fate of
    the objects the result lost (IoU below 0.5): seeded ones are 'split' (two or more seeds), 'merged'
    (their instance also covers another object), 'undersized' or 'oversized' (an extent error);
    unseeded ones are 'absorbed' (mostly covered by a neighbour's instance) or 'missing'. 'matched_iou'
    is the mean IoU of the matched objects, a boundary-precision figure.
    """
    matched = matched_ids(labels, segmentation)
    fates = object_fates(labels, segmentation)
    seeds = intermediates["seeds"]
    seed_ids, gt_ids, counts = contingency(seeds, labels)
    n_seeds = int(seeds.max())
    seeds_per_object = np.zeros(int(labels.max()) + 1, dtype="int64")
    inside = (seed_ids != 0) & (gt_ids != 0)
    np.add.at(seeds_per_object, gt_ids[inside], 1)
    gt_present = np.unique(labels)
    gt_present = gt_present[gt_present != 0]
    per_object = seeds_per_object[gt_present]
    # A seed belongs to the label most of its pixels fall on; background seeds are false starts.
    background_seeds = 0
    if n_seeds > 0:
        order = np.lexsort((-counts, seed_ids))
        first = np.ones(len(order), dtype=bool)
        first[1:] = seed_ids[order][1:] != seed_ids[order][:-1]
        majority_label = gt_ids[order][first]
        majority_seed = seed_ids[order][first]
        background_seeds = int(((majority_label == 0) & (majority_seed != 0)).sum())
    is_matched = np.isin(gt_present, matched)
    seeded, lost = per_object >= 1, ~is_matched
    seeded_lost = seeded & lost
    split = seeded_lost & (per_object >= 2)
    merged = seeded_lost & ~split & fates["merged"]
    extent = seeded_lost & ~split & ~merged
    fg_mask, gt_fg = intermediates["fg_mask"], labels != 0
    union = int((fg_mask | gt_fg).sum())
    return {
        "n_seeds": n_seeds,
        "gt_with_0_seeds": int((per_object == 0).sum()),
        "gt_with_1_seed": int((per_object == 1).sum()),
        "gt_with_2plus_seeds": int((per_object >= 2).sum()),
        "background_seeds": background_seeds,
        "seeded_unmatched": int(seeded_lost.sum()),
        "seeded_split": int(split.sum()),
        "seeded_merged": int(merged.sum()),
        "seeded_undersized": int((extent & fates["undersized"]).sum()),
        "seeded_oversized": int((extent & ~fates["undersized"]).sum()),
        "unseeded_absorbed": int((~seeded & lost & fates["absorbed"]).sum()),
        "unseeded_missing": int((~seeded & lost & ~fates["absorbed"]).sum()),
        "matched_before_min_size": int(len(matched_ids(labels, intermediates["before_min_size"]))),
        "fg_iou": float((fg_mask & gt_fg).sum() / union) if union else float("nan"),
        "matched_iou": float(fates["iou"][is_matched].mean()) if is_matched.any() else float("nan"),
    }


# ----------------------------------------------------------------------------------------------
# running one configuration


def run_identity(
    params_2d: Dict[str, Any], params_3d: Dict[str, Any], mode: str, dimensions: Sequence[int], trial_id: str,
    device: str, hardware: Dict[str, Any], datasets: Optional[Sequence[str]] = None,
) -> str:
    identity = {
        "params_2d": params_2d, "params_3d": params_3d, "mode": mode, "dimensions": list(dimensions),
        "trial_id": trial_id, "device": device, "hardware": hardware,
    }
    if datasets:
        # A run restricted to some datasets is a different (partial) result, not the manifest's.
        identity["datasets"] = sorted(datasets)
    return _content_checksum(identity)


def _prediction_identity(records: Sequence[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """The device and hardware the cached predictions were made on ('mixed' where they differ)."""
    if not records:
        return "cache", {}
    devices = sorted({str(record.get("device")) for record in records})
    accelerators = sorted({str((record.get("hardware") or {}).get("accelerator")) for record in records})
    hardware = dict(records[0].get("hardware") or {})
    if len(accelerators) > 1:
        hardware["accelerator"] = "mixed:" + "|".join(accelerators)
    return devices[0] if len(devices) == 1 else "mixed:" + "|".join(devices), hardware


def score_sample(
    sample: Dict[str, Any], context: Dict[str, Any], prediction: np.ndarray, labels: np.ndarray,
    valid: Optional[np.ndarray], record: Dict[str, Any], params: Dict[str, Any], model_type: str, n_threads: int,
    diagnostics: bool,
) -> Dict[str, Any]:
    """Post-process one cached prediction and score it; the row of `samples.csv`."""
    dense = context["postprocessing_mode"] == "dense"
    active = params["dense" if dense else "sparse"]
    started = time.perf_counter()
    segmentation = segment_prediction(prediction, active, dense, context["spacing"], model_type, n_threads)
    generation_seconds = time.perf_counter() - started
    if valid is not None:
        segmentation[~valid] = 0
    if context["ndim"] == 2:
        # Symmetric with the ground truth, which the loader filtered the same way.
        segmentation = drop_severed_objects(segmentation, context["border_min_size"])
    metrics = compute_metrics(segmentation, labels, context["metric_mode"], border_min_size=0)
    counts = object_counts(labels, segmentation)
    predict_seconds = float(record.get("predict_seconds", float("nan")))
    row = {
        "sample_id": sample["sample_id"],
        "dataset": sample["dataset"],
        "ndim": context["ndim"],
        "family": sample.get("family", sample["dataset"]),
        "seen_in_training": str(sample.get("seen_in_training", "")),
        "metric_mode": context["metric_mode"],
        "postprocessing_mode": context["postprocessing_mode"],
        "initialization_seconds": predict_seconds,
        "generation_seconds": generation_seconds,
        "total_seconds": (predict_seconds if np.isfinite(predict_seconds) else 0.0) + generation_seconds,
        "peak_cuda_memory_bytes": record.get("peak_cuda_memory_bytes"),
        **metrics,
        **counts,
    }
    if diagnostics and not dense:
        intermediates = sparse_pipeline(prediction, active, context["spacing"], n_threads)
        mirrored = intermediates["segmentation"]
        if valid is not None:
            mirrored[~valid] = 0
        if context["ndim"] == 2:
            mirrored = drop_severed_objects(mirrored, context["border_min_size"])
        row["pipeline_mismatch"] = int(not np.array_equal(mirrored, segmentation))
        row.update(seed_diagnostics(intermediates, labels, segmentation))
    return row


def summarize(samples: pd.DataFrame) -> pd.DataFrame:
    """Per-dataset means (metrics) and sums (counts, seconds), a balanced row and, for the 3d crop
    manifests, family and unseen macros in the style of `benchmark_apg_3d.summarize`."""
    metric_columns = [column for column in METRIC_COLUMNS if column in samples.columns]
    count_columns = [column for column in COUNT_COLUMNS if column in samples.columns]
    second_columns = ["initialization_seconds", "generation_seconds", "total_seconds"]
    rows = []
    for dataset, group in samples.groupby("dataset", sort=True):
        row: Dict[str, Any] = {
            "dataset": dataset,
            "family": group["family"].iloc[0] if "family" in group else dataset,
            "seen_in_training": str(group["seen_in_training"].iloc[0]) if "seen_in_training" in group else "",
            "n_samples": int(len(group)),
        }
        for column in second_columns:
            row[column] = float(group[column].sum())
        if "peak_cuda_memory_bytes" in group:
            values = group["peak_cuda_memory_bytes"].dropna()
            row["peak_cuda_memory_bytes"] = int(values.max()) if len(values) else np.nan
        for metric in metric_columns:
            values = group[metric].dropna().to_numpy(dtype="float64")
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=0)) if len(values) else np.nan
        if "msa" in group:
            row["msa_ci_low"], row["msa_ci_high"] = _bootstrap_ci(group["msa"].dropna().to_numpy())
        for column in count_columns:
            values = group[column].dropna()
            row[column] = int(values.sum()) if len(values) else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)

    def macro(name: str, selected: pd.DataFrame, by: str) -> Dict[str, Any]:
        row = {"dataset": name, "n_samples": int(selected["n_samples"].sum()) if len(selected) else 0}
        if selected.empty:
            return row
        groups = selected.groupby(by)
        for metric in metric_columns:
            means = groups[f"{metric}_mean"].mean().dropna()
            row[f"{metric}_mean"] = float(means.mean()) if len(means) else np.nan
            row[f"{metric}_std"] = float(means.std(ddof=0)) if len(means) else np.nan
        row["n_groups"] = int(groups.ngroups)
        for column in second_columns + [c for c in count_columns if c in selected]:
            row[column] = selected[column].sum()
        if "peak_cuda_memory_bytes" in selected:
            values = selected["peak_cuda_memory_bytes"].dropna()
            row["peak_cuda_memory_bytes"] = int(values.max()) if len(values) else np.nan
        return row

    macros = [macro(BALANCED_ROW, summary, "dataset")]
    if (summary["family"] != summary["dataset"]).any():
        macros.append(macro("__family_macro__", summary, "family"))
        macros.append(macro("__unseen_macro__", summary[summary["seen_in_training"] == "False"], "family"))
    return pd.concat([summary, pd.DataFrame(macros)], ignore_index=True)


def run_config(
    manifest: Dict[str, Any], output_root: Path, data_root: Path, model_type: str, joint_checkpoint: str,
    checkpoint_id: str, config_name: str, mode: str, params_2d: Dict[str, Any], params_3d: Dict[str, Any],
    dimensions: Sequence[int], trial_id: str, workers: int, n_threads: int, diagnostics: bool,
    predictor: Optional[Predictor] = None, datasets: Optional[Sequence[str]] = None, force: bool = False,
) -> Tuple[Path, pd.DataFrame, Dict[str, Any]]:
    """Run one configuration on the cached predictions of a manifest and write its run directory."""
    cache = PredictionCache(output_root, checkpoint_id, manifest["manifest_checksum"])
    loader = SampleLoader(manifest["kind"], data_root)
    samples = [sample for sample in manifest["samples"] if int(sample["ndim"]) in dimensions]
    if datasets:
        samples = [sample for sample in samples if sample["dataset"] in datasets]
    if not samples:
        raise ValueError("No samples selected.")
    for sample in samples:
        if not cache.has(sample) and predictor is None:
            raise FileNotFoundError(
                f"No cached prediction for '{sample['sample_id']}' under '{cache.root}'. Run 'predict' first, or pass "
                "--predict-missing."
            )
    device, hardware = _prediction_identity(cache.records(samples))
    config_checksum = run_identity(params_2d, params_3d, mode, dimensions, trial_id, device, hardware, datasets)
    epoch = implementation_checksum()
    run_dir = output_root / CAMPAIGN / model_type / checkpoint_id / (
        f"{manifest['manifest_checksum']}-{config_checksum}-{epoch}"
    )
    samples_path, summary_path = run_dir / "samples.csv", run_dir / "summary.csv"
    metadata_path = run_dir / "metadata.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    if metadata_path.exists() and not force:
        with open(metadata_path) as f:
            metadata = json.load(f)
        if metadata.get("status") == "complete" and samples_path.exists() and summary_path.exists():
            print(f"Completed result already exists at '{run_dir}'.")
            return run_dir, pd.read_csv(summary_path), metadata

    completed = pd.read_csv(samples_path) if samples_path.exists() and not force else pd.DataFrame()
    done = set(completed["sample_id"]) if not completed.empty else set()
    pending = [sample for sample in samples if sample["sample_id"] not in done]
    metadata = {
        "campaign": CAMPAIGN,
        "status": "running",
        "config_name": config_name,
        "config_checksum": config_checksum,
        "mode": mode,
        "manifest_kind": manifest["kind"],
        "subset": manifest.get("subset"),
        "manifest_checksum": manifest["manifest_checksum"],
        "implementation_checksum": epoch,
        "checkpoint_checksum": checkpoint_id,
        "checkpoint_name": joint_checkpoint,
        "model_type": model_type,
        "params_2d": params_2d,
        "params_3d": params_3d,
        "dimensions": list(dimensions),
        "datasets": sorted({sample["dataset"] for sample in samples}),
        "trial_id": trial_id,
        # The device and hardware of the cached predictions: the identity the comparator pairs runs by.
        "device": device,
        "hardware": hardware,
        "postprocessing_hardware": _hardware_identity("cpu"),
        "workers": workers,
        "n_threads": n_threads,
        "diagnostics": diagnostics,
        "prediction_cache": str(cache.root),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "git_revision": _git_revision(),
    }
    _atomic_write_json(metadata_path, metadata)
    params_by_dimension = {2: params_2d, 3: params_3d}
    started = time.perf_counter()

    def process(sample: Dict[str, Any]) -> Dict[str, Any]:
        context = sample_context(sample, manifest["kind"], mode)
        prediction, labels, valid, record = ensure_prediction(cache, sample, loader, predictor)
        row = score_sample(
            sample, context, prediction, labels, valid, record, params_by_dimension[context["ndim"]], model_type,
            n_threads, diagnostics,
        )
        row["trial_id"] = trial_id
        return row

    rows: List[Dict[str, Any]] = []

    def flush() -> None:
        nonlocal completed, rows
        if rows:
            completed = pd.concat([completed, pd.DataFrame(rows)], ignore_index=True)
            rows = []
            _atomic_write_csv(samples_path, completed)

    try:
        flush_every = 1 if any(int(s["ndim"]) == 3 for s in pending) else 20
        if workers <= 1 or predictor is not None:
            # Prediction needs the GPU and the source cache in one thread; a plain loop keeps it simple.
            results = map(process, pending)
            pool = None
        else:
            pool = futures.ThreadPoolExecutor(workers)
            results = pool.map(process, pending)
        try:
            for index, row in enumerate(results, start=1):
                rows.append(row)
                if len(rows) >= flush_every:
                    flush()
                    print(f"{config_name}: {index}/{len(pending)} samples, {time.perf_counter() - started:.0f} s")
        finally:
            if pool is not None:
                pool.shutdown()
        flush()
        expected = {sample["sample_id"] for sample in samples}
        if set(completed["sample_id"]) != expected:
            raise RuntimeError(f"Run finished with {len(completed)} of {len(expected)} samples.")
        summary = summarize(completed)
        _atomic_write_csv(summary_path, summary)
        metadata.update({
            "status": "complete", "wall_seconds": time.perf_counter() - started, "n_samples": int(len(completed)),
        })
        _atomic_write_json(metadata_path, metadata)
    except Exception as error:
        metadata.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        _atomic_write_json(metadata_path, metadata)
        raise
    return run_dir, summary, metadata


# ----------------------------------------------------------------------------------------------
# reports and the generalization gate


def dataset_scores(samples: pd.DataFrame) -> pd.Series:
    """Per-dataset quality: mean mSA, or the mean CREMI score (negated, so higher is better) on dense data."""
    scores = {}
    for dataset, group in samples.groupby("dataset"):
        dense = "metric_mode" in group and group["metric_mode"].iloc[0] == "dense"
        if dense and "cremi" in group and group["cremi"].notna().any():
            scores[dataset] = -float(group["cremi"].mean())
        else:
            scores[dataset] = float(group["msa"].mean())
    return pd.Series(scores).sort_index()


def gate_table(baseline: pd.Series, candidate: pd.Series, gate: Dict[str, float] = GATE) -> Dict[str, Any]:
    """The generalization gate: up on all but 'max_down' datasets, no dataset below both loss limits,
    balanced gain at least 'min_balanced_gain'. 'baseline' and 'candidate' are per-dataset scores."""
    datasets = sorted(set(baseline.index) & set(candidate.index))
    base = baseline[datasets].to_numpy(dtype="float64")
    cand = candidate[datasets].to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(base != 0, cand / np.where(base != 0, base, 1.0) - 1.0, np.nan)
    absolute = cand - base
    up = int((absolute > 0).sum())
    violates = (relative < gate["max_relative_loss"]) & (absolute < gate["max_absolute_loss"])
    balanced_gain = float(cand.mean() / base.mean() - 1.0) if base.mean() else float("nan")
    checks = {
        "up_on_all_but_two": bool(up >= len(datasets) - gate["max_down"]),
        "no_dataset_below_loss_limits": bool(not violates.any()),
        "balanced_gain_at_least_2_percent": bool(balanced_gain >= gate["min_balanced_gain"]),
    }
    return {
        "datasets": datasets, "n_up": up, "n_datasets": len(datasets),
        "relative": dict(zip(datasets, relative.tolist())),
        "balanced_baseline": float(base.mean()), "balanced_candidate": float(cand.mean()),
        "balanced_gain": balanced_gain,
        "worst_relative": float(np.nanmin(relative)) if len(relative) and np.isfinite(relative).any() else float("nan"),
        "checks": checks, "passed": bool(all(checks.values())),
    }


def load_run(run_dir: Path) -> Tuple[Dict[str, Any], pd.DataFrame]:
    with open(run_dir / "metadata.json") as f:
        metadata = json.load(f)
    if metadata.get("status") != "complete":
        raise RuntimeError(f"Run is not complete: '{run_dir}'.")
    return metadata, pd.read_csv(run_dir / "samples.csv")


def report(
    run_dirs_by_config: Dict[str, List[Path]], baseline_name: str, ndim: Optional[int] = None,
    datasets: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Join the sample tables of every configuration over its subsets and compare with the baseline.

    'ndim' and 'datasets' restrict the samples (the 2d screens read the eleven image datasets; a
    manifest's single volumes are too few to compare). Returns the per-configuration table (balanced
    score, gain, gate verdict, count sums) and the per-(configuration, dataset) table of relative changes.
    """
    joined: Dict[str, pd.DataFrame] = {}
    for name, run_dirs in run_dirs_by_config.items():
        samples = pd.concat([load_run(run_dir)[1] for run_dir in run_dirs], ignore_index=True)
        if ndim is not None:
            samples = samples[samples["ndim"] == ndim]
        if datasets:
            samples = samples[samples["dataset"].isin(datasets)]
        joined[name] = samples.reset_index(drop=True)
    if baseline_name not in joined:
        raise ValueError(f"Baseline '{baseline_name}' is not among the configurations {sorted(joined)}.")
    baseline_scores = dataset_scores(joined[baseline_name])
    baseline_counts = joined[baseline_name][[c for c in COUNT_COLUMNS if c in joined[baseline_name]]].sum()
    rows, details = [], []
    for name, samples in joined.items():
        scores = dataset_scores(samples)
        verdict = gate_table(baseline_scores, scores)
        counts = samples[[c for c in COUNT_COLUMNS if c in samples]].sum()
        row = {
            "config": name, "n_samples": int(len(samples)), "balanced": verdict["balanced_candidate"],
            "balanced_gain": verdict["balanced_gain"], "n_up": verdict["n_up"], "n_datasets": verdict["n_datasets"],
            "worst_relative": verdict["worst_relative"], "passed": verdict["passed"],
            "generation_seconds": float(samples["generation_seconds"].sum()),
        }
        for column in ("matched", "unmatched", "predicted_objects", "gt_with_0_seeds", "gt_with_2plus_seeds",
                       "background_seeds", "seeded_unmatched", "seeded_split", "seeded_merged", "seeded_undersized",
                       "seeded_oversized", "unseeded_absorbed", "unseeded_missing", "pipeline_mismatch"):
            if column in counts:
                row[column] = int(counts[column])
                row[f"{column}_delta"] = int(counts[column] - baseline_counts.get(column, 0))
        rows.append(row)
        for dataset in verdict["datasets"]:
            details.append({
                "config": name, "dataset": dataset, "baseline": float(baseline_scores[dataset]),
                "candidate": float(scores[dataset]), "relative": verdict["relative"][dataset],
            })
    table = pd.DataFrame(rows).sort_values("balanced", ascending=False).reset_index(drop=True)
    return table, pd.DataFrame(details)


def _format_relative(value: float) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{100 * value:+.1f}%"


def print_report(table: pd.DataFrame, details: pd.DataFrame) -> None:
    pivot = details.pivot(index="config", columns="dataset", values="relative").loc[table["config"]]
    columns = ["config", "balanced", "balanced_gain", "n_up", "n_datasets", "worst_relative", "passed"]
    columns += [c for c in ("matched_delta", "gt_with_0_seeds_delta", "gt_with_2plus_seeds_delta",
                            "background_seeds_delta", "seeded_split_delta", "seeded_merged_delta",
                            "seeded_undersized_delta", "seeded_oversized_delta", "pipeline_mismatch") if c in table]
    shown = table[columns].copy()
    for column in ("balanced_gain", "worst_relative"):
        shown[column] = shown[column].map(_format_relative)
    shown["balanced"] = shown["balanced"].map(lambda v: f"{v:.4f}")
    print(shown.to_string(index=False))
    print()
    print("Relative change per dataset:")
    print(pivot.map(_format_relative).to_string())


# ----------------------------------------------------------------------------------------------
# parameter sweeps on the cache


def grid_combinations(grid: Dict[str, List[Any]], mode: str) -> List[Dict[str, Any]]:
    keys = list(grid)
    allowed = SPARSE_KEYS if mode == "sparse" else DENSE_KEYS
    unknown = set(keys) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {mode} grid parameters: {sorted(unknown)}.")
    combinations = [dict(zip(keys, combo)) for combo in itertools.product(*[grid[key] for key in keys])]
    if mode == "sparse":
        combinations = deduplicate_flow_travel(combinations)
    return combinations


def sweep_dir(
    output_root: Path, checkpoint_id: str, manifest_checksum: str, grid_name: str, grid: Dict[str, Any],
) -> Path:
    identity = f"{grid_name}-{_content_checksum(grid)[:12]}-{implementation_checksum()[:12]}"
    return output_root / CAMPAIGN / "sweeps" / checkpoint_id / manifest_checksum / identity


def sweep_dataset(
    manifest: Dict[str, Any], cache: PredictionCache, dataset: str, mode: str, grid: Dict[str, List[Any]],
    model_type: str, n_threads: int, shard_index: int, num_shards: int, out_dir: Path,
) -> Path:
    """Score every grid combination of one dataset on the cache; writes the `parameter_search` CSV layout."""
    samples = [sample for sample in manifest["samples"] if sample["dataset"] == dataset]
    if not samples:
        raise ValueError(f"No samples of '{dataset}' in the manifest.")
    contexts = [sample_context(sample, manifest["kind"], mode) for sample in samples]
    postproc_mode = contexts[0]["postprocessing_mode"]
    # The grid keys the sweep did not name stay at the library defaults, and the row records them.
    defaults = default_postprocessing(model_type, postproc_mode, ndim=contexts[0]["ndim"])
    combinations = [{**defaults, **combo} for combo in grid_combinations(grid, postproc_mode)]
    if num_shards > 1:
        combinations = combinations[shard_index::num_shards]
    suffix = "" if num_shards <= 1 else f".shard{shard_index}of{num_shards}"
    out_path = out_dir / f"{dataset}{suffix}.csv"
    if out_path.exists():
        print(f"Sweep result exists: {out_path}")
        return out_path
    metric_lists: List[List[Dict[str, float]]] = [[] for _ in combinations]
    started = time.perf_counter()
    for index, (sample, context) in enumerate(zip(samples, contexts), start=1):
        prediction, labels, _, _ = cache.load(sample)
        # The scorers see no valid mask: invalid voxels are background in the labels, so a prediction there
        # costs precision the same way in every combination.
        if postproc_mode == "sparse":
            scores = score_image_sparse_cached(
                prediction, labels, combinations, n_threads=n_threads, spacing=context["spacing"],
                border_min_size=context["border_min_size"],
            )
        else:
            scores = score_image_dense_cached(prediction, labels, combinations, n_threads=n_threads, border_min_size=0)
        for metrics, collected in zip(scores, metric_lists):
            if metrics is not None:
                collected.append(metrics)
        print(f"{dataset}: {index}/{len(samples)} samples, {len(combinations)} combinations, "
              f"{time.perf_counter() - started:.0f} s")
    rows = []
    for combo, per_sample in zip(combinations, metric_lists):
        if not per_sample:
            continue
        row = {**combo, "n_images": len(per_sample)}
        for key in per_sample[0]:
            values = np.asarray([m[key] for m in per_sample], dtype="float64")
            row[f"{key}_mean"], row[f"{key}_std"] = float(values.mean()), float(values.std())
        rows.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(out_path, pd.DataFrame(rows))
    print(f"Saved {out_path} ({time.perf_counter() - started:.0f} s).")
    return out_path


def merge_sweep(out_dir: Path, dataset: str, num_shards: int) -> Path:
    out_path = out_dir / f"{dataset}.csv"
    if num_shards <= 1:
        if not out_path.exists():
            raise FileNotFoundError(f"Missing sweep result: {out_path}")
        return out_path
    paths = [out_dir / f"{dataset}.shard{i}of{num_shards}.csv" for i in range(num_shards)]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing shards: {missing}")
    _atomic_write_csv(out_path, pd.concat([pd.read_csv(p) for p in paths], ignore_index=True))
    return out_path


def shared_configuration(out_dir: Path, datasets: Sequence[str], criterion: str = "msa") -> pd.DataFrame:
    """Rank the combinations every dataset scored by how close they come to each dataset's own optimum.

    Columns: the parameters, per-dataset scores and relative-to-optimum ratios, 'mean_relative' (the
    selection criterion), 'min_relative' (the worst dataset) and 'balanced' (the equal-weight mean).
    """
    tables = []
    keys: Optional[List[str]] = None
    for dataset in datasets:
        table = pd.read_csv(out_dir / f"{dataset}.csv")
        params = [c for c in table.columns if not c.endswith(("_mean", "_std")) and c != "n_images"]
        keys = params if keys is None else keys
        column = f"{criterion}_mean"
        if criterion == "cremi":
            table[column] = -table[column]
        tables.append(table[params + [column]].rename(columns={column: dataset}))
    merged = tables[0]
    for table in tables[1:]:
        merged = merged.merge(table, on=keys, how="inner")
    for dataset in datasets:
        best = merged[dataset].max()
        merged[f"{dataset}_relative"] = merged[dataset] / best if best else np.nan
    relative = merged[[f"{d}_relative" for d in datasets]]
    merged["mean_relative"] = relative.mean(axis=1)
    merged["min_relative"] = relative.min(axis=1)
    merged["balanced"] = merged[list(datasets)].mean(axis=1)
    merged = merged.sort_values(["mean_relative", "min_relative"], ascending=False).reset_index(drop=True)
    _atomic_write_csv(out_dir / "shared_config.csv", merged)
    return merged


# ----------------------------------------------------------------------------------------------
# oracles: what the seeds, the height map and the foreground each cost


ORACLES = ("baseline", "gt_seeds", "gt_seeds_gt_fg", "gt_heightmap", "gt_fg", "gt_seeds_gt_heightmap")


def gt_seed_markers(labels: np.ndarray) -> np.ndarray:
    """One marker per ground-truth object around its deepest interior point, carrying the object's id.

    The marker is the point's 3-neighbourhood clipped to the object. A single pixel would not do: the
    geodesic field's magnitude is zero at the object's centre (the gradient vanishes at its source), so
    the inverted-magnitude height map has a one-pixel spike there, and the monotone flooding of
    `bioimage_cpp.segmentation.watershed` lets a seed sitting on a spike flood last.
    """
    from scipy.ndimage import grey_dilation
    from micro_sam.v2.automatic_prompt_generation import interior_points

    points = np.zeros(labels.shape, dtype="uint64")
    ids = np.unique(labels)
    ids = ids[ids != 0]
    for index, point in zip(ids, interior_points(labels)):
        points[tuple(int(c) for c in point)] = index
    dilated = grey_dilation(points, size=(3,) * labels.ndim)
    return np.where(labels.astype("uint64") == dilated, dilated, 0).astype("uint64")


def gt_ridge_heightmap(labels: np.ndarray) -> np.ndarray:
    """A height map whose only ridges are the ground-truth object boundaries."""
    from skimage.segmentation import find_boundaries

    return np.ascontiguousarray(find_boundaries(labels, mode="inner"), dtype="float32")


def _finish_watershed(before: np.ndarray, hmap: np.ndarray, fg_mask: np.ndarray, min_size: int) -> np.ndarray:
    """The size filter and refill of `flow_instance_segmentation`, applied to an oracle's watershed."""
    seg = before
    if min_size > 0:
        ids, sizes = np.unique(before, return_counts=True)
        discard = ids[(sizes < min_size) & (ids > 0)]
        seg = before.copy()
        seg[np.isin(seg, discard)] = 0
        seg = watershed(hmap, markers=seg, mask=fg_mask)
    return seg.astype("uint32")


def oracle_sample(
    sample: Dict[str, Any], context: Dict[str, Any], prediction: np.ndarray, labels: np.ndarray,
    valid: Optional[np.ndarray], params: Dict[str, Any], n_threads: int,
) -> Dict[str, Any]:
    """Score the sparse pipeline with parts of it replaced by the ground truth.

    'gt_seeds': ground-truth seeds, predicted height map and foreground (ceiling of any seed logic);
    'gt_heightmap': predicted seeds and foreground, ridges at the ground-truth boundaries (ceiling of
    any height-map / assignment logic); 'gt_fg': predicted seeds and height map inside the ground-truth
    foreground (ceiling of the foreground); and the two-part combinations.
    """
    active = params["sparse"]
    intermediates = sparse_pipeline(prediction, active, context["spacing"], n_threads)
    fg_pred, hmap_pred, seeds_pred = intermediates["fg_mask"], intermediates["heightmap"], intermediates["seeds"]
    gt_fg, gt_markers, gt_hmap = labels != 0, gt_seed_markers(labels), gt_ridge_heightmap(labels)
    min_size = int(active["min_size"])

    def finish(hmap: np.ndarray, markers: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return _finish_watershed(watershed(hmap, markers=markers, mask=mask), hmap, mask, min_size)

    variants = {
        "baseline": intermediates["segmentation"],
        "gt_seeds": finish(hmap_pred, gt_markers, fg_pred),
        "gt_seeds_gt_fg": finish(hmap_pred, gt_markers, gt_fg),
        "gt_heightmap": finish(gt_hmap, seeds_pred, fg_pred),
        "gt_fg": finish(hmap_pred, seeds_pred, gt_fg),
        "gt_seeds_gt_heightmap": finish(gt_hmap, gt_markers, fg_pred),
    }
    row = {
        "sample_id": sample["sample_id"], "dataset": sample["dataset"], "ndim": context["ndim"],
        "family": sample.get("family", sample["dataset"]), "metric_mode": context["metric_mode"],
        "gt_objects": int(len(np.unique(labels)) - 1),
    }
    for name, segmentation in variants.items():
        segmentation = segmentation.astype("uint32")
        if valid is not None:
            segmentation[~valid] = 0
        if context["ndim"] == 2:
            segmentation = drop_severed_objects(segmentation, context["border_min_size"])
        counts = object_counts(labels, segmentation)
        row[f"msa_{name}"] = compute_metrics(segmentation, labels, "sparse", border_min_size=0)["msa"]
        row[f"matched_{name}"] = counts["matched"]
        row[f"predicted_{name}"] = counts["predicted_objects"]
    return row


def summarize_oracles(samples: pd.DataFrame) -> pd.DataFrame:
    """Per-dataset means of every oracle, their gain over the baseline, and the balanced row."""
    rows = []
    for dataset, group in samples.groupby("dataset", sort=True):
        row: Dict[str, Any] = {
            "dataset": dataset, "n_samples": int(len(group)), "gt_objects": int(group["gt_objects"].sum()),
        }
        for name in ORACLES:
            row[f"msa_{name}"] = float(group[f"msa_{name}"].mean())
            row[f"matched_{name}"] = int(group[f"matched_{name}"].sum())
        rows.append(row)
    summary = pd.DataFrame(rows)
    balanced = {
        "dataset": BALANCED_ROW, "n_samples": int(summary["n_samples"].sum()),
        "gt_objects": int(summary["gt_objects"].sum()),
    }
    for name in ORACLES:
        balanced[f"msa_{name}"] = float(summary[f"msa_{name}"].mean())
        balanced[f"matched_{name}"] = int(summary[f"matched_{name}"].sum())
    summary = pd.concat([summary, pd.DataFrame([balanced])], ignore_index=True)
    for name in ORACLES[1:]:
        summary[f"gain_{name}"] = summary[f"msa_{name}"] / summary["msa_baseline"] - 1.0
    return summary


def cmd_oracle(args: argparse.Namespace) -> None:
    checkpoint_id = _checkpoint_identity(args.model_type, args.joint_checkpoint)
    name, _, params_2d, params_3d = load_config(args.config, args.model_type)
    params_by_dimension = {2: params_2d, 3: params_3d}
    for manifest in _manifests(args):
        cache = PredictionCache(args.output_root, checkpoint_id, manifest["manifest_checksum"])
        samples = [sample for sample in manifest["samples"] if int(sample["ndim"]) in _dimensions(args)]
        if args.datasets:
            samples = [sample for sample in samples if sample["dataset"] in args.datasets]
        identity = _content_checksum(
            {"params_2d": params_2d, "params_3d": params_3d, "datasets": sorted(args.datasets or [])}
        )
        out_dir = args.output_root / CAMPAIGN / "oracles" / checkpoint_id / manifest["manifest_checksum"] / (
            f"{name}-{identity[:12]}-{implementation_checksum()[:12]}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        samples_path = out_dir / "samples.csv"
        completed = pd.read_csv(samples_path) if samples_path.exists() else pd.DataFrame()
        done = set(completed["sample_id"]) if not completed.empty else set()
        pending = [sample for sample in samples if sample["sample_id"] not in done]

        def process(sample: Dict[str, Any]) -> Dict[str, Any]:
            # The oracles are about the sparse pipeline; every sample runs through it.
            context = sample_context(sample, manifest["kind"], "sparse")
            prediction, labels, valid, _ = cache.load(sample)
            return oracle_sample(
                sample, context, prediction, labels, valid, params_by_dimension[context["ndim"]], args.threads,
            )

        started = time.perf_counter()
        with futures.ThreadPoolExecutor(max(1, args.workers)) as pool:
            rows = []
            for index, row in enumerate(pool.map(process, pending), start=1):
                rows.append(row)
                if len(rows) >= 20:
                    completed = pd.concat([completed, pd.DataFrame(rows)], ignore_index=True)
                    rows = []
                    _atomic_write_csv(samples_path, completed)
                    elapsed = time.perf_counter() - started
                    print(f"oracle {manifest.get('subset')}: {index}/{len(pending)}, {elapsed:.0f} s")
            if rows:
                completed = pd.concat([completed, pd.DataFrame(rows)], ignore_index=True)
                _atomic_write_csv(samples_path, completed)
        summary = summarize_oracles(completed)
        _atomic_write_csv(out_dir / "summary.csv", summary)
        _atomic_write_json(out_dir / "metadata.json", {
            "campaign": CAMPAIGN, "kind": "oracle", "config_name": name, "params_2d": params_2d,
            "params_3d": params_3d, "manifest_checksum": manifest["manifest_checksum"],
            "subset": manifest.get("subset"),
            "checkpoint_checksum": checkpoint_id, "implementation_checksum": implementation_checksum(),
            "n_samples": int(len(completed)), "git_revision": _git_revision(),
        })
        shown = ["dataset", "n_samples"] + [f"msa_{n}" for n in ORACLES]
        print(f"\nOracles on {manifest['kind']}/{manifest.get('subset')}: {out_dir}")
        print(summary[shown].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(summary[["dataset"] + [f"gain_{n}" for n in ORACLES[1:]]].to_string(
            index=False, float_format=lambda v: f"{100 * v:+.1f}%"))


# ----------------------------------------------------------------------------------------------
# commands


def _checkpoint_identity(model_type: str, joint_checkpoint: str) -> str:
    return checkpoint_checksum(get_joint_checkpoint(model_type, joint_checkpoint))


def _manifests(args: argparse.Namespace) -> List[Dict[str, Any]]:
    return [
        load_campaign_manifest(args.kind, subset, args.output_root, args.data_root, args.campaign_root)
        for subset in args.subset
    ]


def _dimensions(args: argparse.Namespace) -> Tuple[int, ...]:
    return (2, 3) if args.ndim == "both" else (int(args.ndim),)


def cmd_predict(args: argparse.Namespace) -> None:
    checkpoint_id = _checkpoint_identity(args.model_type, args.joint_checkpoint)
    predictor = Predictor(args.model_type, args.joint_checkpoint, checkpoint_id, args.device, args.output_root)
    for manifest in _manifests(args):
        cache = PredictionCache(args.output_root, checkpoint_id, manifest["manifest_checksum"])
        loader = SampleLoader(manifest["kind"], args.data_root)
        samples = [sample for sample in manifest["samples"] if int(sample["ndim"]) in _dimensions(args)]
        if args.datasets:
            samples = [sample for sample in samples if sample["dataset"] in args.datasets]
        if args.sample_index is not None:
            samples = [samples[args.sample_index]]
        pending = [sample for sample in samples if args.force or not cache.has(sample)]
        print(f"{manifest['kind']}/{manifest.get('subset')}: {len(pending)} of {len(samples)} samples to predict "
              f"-> {cache.root}")
        started = time.perf_counter()
        for index, sample in enumerate(pending, start=1):
            raw, labels, valid = loader.load(sample)
            prediction, record = predictor.predict(raw, int(sample["ndim"]))
            record["sample_id"] = sample["sample_id"]
            cache.store(sample, prediction, labels, valid, record)
            print(f"  {sample['sample_id']:40s} {str(prediction.shape):24s} {record['predict_seconds']:6.2f} s "
                  f"({index}/{len(pending)}, {time.perf_counter() - started:.0f} s)")


def _run_configs(args: argparse.Namespace, config_paths: Sequence[Optional[Path]]) -> Dict[str, Dict[str, str]]:
    checkpoint_id = _checkpoint_identity(args.model_type, args.joint_checkpoint)
    predictor = None
    if args.predict_missing:
        predictor = Predictor(args.model_type, args.joint_checkpoint, checkpoint_id, args.device, args.output_root)
    index: Dict[str, Dict[str, str]] = {}
    for manifest in _manifests(args):
        for config_path in config_paths:
            name, mode, params_2d, params_3d = load_config(config_path, args.model_type)
            run_dir, summary, _ = run_config(
                manifest, args.output_root, args.data_root, args.model_type, args.joint_checkpoint, checkpoint_id,
                name, mode, params_2d, params_3d, _dimensions(args), args.trial_id, args.workers, args.threads,
                not args.no_diagnostics, predictor=predictor, datasets=args.datasets, force=args.force,
            )
            index.setdefault(name, {})[str(manifest.get("subset"))] = str(run_dir)
            shown = [c for c in ("dataset", "n_samples", "msa_mean", "cremi_mean", "matched", "unmatched",
                                 "gt_with_0_seeds", "gt_with_2plus_seeds", "background_seeds", "pipeline_mismatch",
                                 "generation_seconds") if c in summary]
            print(f"\n{name} on {manifest['kind']}/{manifest.get('subset')}: {run_dir}")
            print(summary[shown].to_string(index=False))
    return index


def cmd_run(args: argparse.Namespace) -> None:
    _run_configs(args, [args.config])


def cmd_screen(args: argparse.Namespace) -> None:
    config_paths: List[Path] = []
    for pattern in args.configs:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No configuration matches '{pattern}'.")
        config_paths.extend(Path(match) for match in matches)
    index = _run_configs(args, config_paths)
    screens = args.output_root / CAMPAIGN / "screens"
    screens.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    index_path = screens / f"{stamp}_{args.name}.json"
    _atomic_write_json(index_path, {
        "name": args.name, "kind": args.kind, "subsets": list(args.subset), "runs": index,
        "implementation_checksum": implementation_checksum(), "created": stamp,
    })
    print(f"\nScreen index: {index_path}")
    if args.baseline in index:
        table, details = report(
            {name: [Path(p) for p in runs.values()] for name, runs in index.items()}, args.baseline,
            ndim=None if args.ndim == "both" else int(args.ndim),
        )
        print_report(table, details)


def cmd_report(args: argparse.Namespace) -> None:
    runs_by_config: Dict[str, List[Path]] = {}
    for index_path in args.index or []:
        with open(index_path) as f:
            index = json.load(f)
        for name, runs in index["runs"].items():
            runs_by_config.setdefault(name, []).extend(Path(p) for p in runs.values())
    for run_dir in args.runs or []:
        metadata, _ = load_run(Path(run_dir))
        runs_by_config.setdefault(metadata["config_name"], []).append(Path(run_dir))
    if not runs_by_config:
        raise SystemExit("Pass --index and/or --runs.")
    table, details = report(
        runs_by_config, args.baseline, ndim=None if args.ndim == "both" else int(args.ndim), datasets=args.datasets,
    )
    print_report(table, details)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_csv(args.output, table)
        _atomic_write_csv(args.output.with_name(args.output.stem + "_datasets.csv"), details)
        print(f"\nReport: {args.output}")


def cmd_sweep(args: argparse.Namespace) -> None:
    checkpoint_id = _checkpoint_identity(args.model_type, args.joint_checkpoint)
    with open(args.grid) as f:
        grid = json.load(f)
    grid_name = args.grid.stem
    for manifest in _manifests(args):
        cache = PredictionCache(args.output_root, checkpoint_id, manifest["manifest_checksum"])
        out_dir = sweep_dir(args.output_root, checkpoint_id, manifest["manifest_checksum"], grid_name, grid)
        datasets = args.datasets or sorted({sample["dataset"] for sample in manifest["samples"]})
        if args.merge:
            for dataset in datasets:
                print(f"Merged: {merge_sweep(out_dir, dataset, args.num_shards)}")
            shared = shared_configuration(out_dir, datasets)
            print(shared.head(args.top).to_string(index=False))
            continue
        for dataset in datasets:
            sweep_dataset(
                manifest, cache, dataset, args.mode, grid, args.model_type, args.threads, args.shard_index,
                args.num_shards, out_dir,
            )
        print(f"Sweep directory: {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common_arguments(p: argparse.ArgumentParser) -> None:
        p.add_argument("--kind", choices=KINDS, default="v5", help="Manifest family: 2d subsets or deep 3d crops.")
        p.add_argument("--subset", nargs="+", default=["primary"])
        p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
        p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        p.add_argument("--campaign-root", type=Path, default=apg3d_manifest.CAMPAIGN_ROOT,
                       help="Where the deep 3d manifests live (--kind apg3d).")
        p.add_argument("--model-type", default="hvit_t", choices=common.MODEL_TYPES)
        p.add_argument("--joint-checkpoint", default="best")
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--ndim", choices=("2", "3", "both"), default="both")
        p.add_argument("--datasets", nargs="*", default=None, help="Restrict to these datasets.")

    predict = sub.add_parser("predict", help="Cache the decoder predictions of a manifest.")
    common_arguments(predict)
    predict.add_argument("--sample-index", type=int, default=None)
    predict.add_argument("--force", action="store_true", help="Re-predict cached samples.")

    def run_arguments(p: argparse.ArgumentParser) -> None:
        common_arguments(p)
        p.add_argument("--trial-id", default="trial-1")
        p.add_argument("--workers", type=int, default=1, help="Samples post-processed concurrently.")
        p.add_argument("--threads", type=int, default=4, help="Threads per post-processing call.")
        p.add_argument("--no-diagnostics", action="store_true", help="Skip the mirrored pipeline and seed columns.")
        p.add_argument("--predict-missing", action="store_true", help="Predict samples missing from the cache.")
        p.add_argument("--force", action="store_true", help="Recompute a finished run.")

    run = sub.add_parser("run", help="Run one configuration on the cache.")
    run_arguments(run)
    run.add_argument("--config", type=Path, default=None)

    screen = sub.add_parser("screen", help="Run several configurations on the cache and report them.")
    run_arguments(screen)
    screen.add_argument("--configs", nargs="+", required=True, help="Configuration files or globs.")
    screen.add_argument("--name", required=True, help="Names the screen index file.")
    screen.add_argument("--baseline", default="current-defaults", help="Configuration name the report compares to.")

    rep = sub.add_parser("report", help="Compare finished runs with a baseline under the generalization gate.")
    rep.add_argument("--index", type=Path, nargs="*", default=None, help="Screen index files.")
    rep.add_argument("--runs", type=Path, nargs="*", default=None, help="Run directories.")
    rep.add_argument("--baseline", default="current-defaults")
    rep.add_argument("--ndim", choices=("2", "3", "both"), default="both", help="Restrict to images or volumes.")
    rep.add_argument("--datasets", nargs="*", default=None, help="Restrict to these datasets.")
    rep.add_argument("--output", type=Path, default=None, help="CSV path for the tables.")

    oracle = sub.add_parser("oracle", help="Score the pipeline with ground-truth seeds, height map or foreground.")
    common_arguments(oracle)
    oracle.add_argument("--config", type=Path, default=None)
    oracle.add_argument("--workers", type=int, default=1)
    oracle.add_argument("--threads", type=int, default=4)

    sweep = sub.add_parser("sweep", help="Score a parameter grid on the cache, one dataset at a time.")
    common_arguments(sweep)
    sweep.add_argument("--grid", type=Path, required=True, help="JSON dict of parameter lists.")
    sweep.add_argument("--mode", choices=MODES, default="auto")
    sweep.add_argument("--threads", type=int, default=4)
    sweep.add_argument("--shard-index", type=int, default=0)
    sweep.add_argument("--num-shards", type=int, default=1)
    sweep.add_argument("--merge", action="store_true", help="Merge the shards and rank the shared configuration.")
    sweep.add_argument("--top", type=int, default=20)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "data_root"):
        args.data_root = args.data_root.expanduser().resolve(strict=True)
        args.output_root = args.output_root.expanduser().resolve()
        if args.output_root == args.data_root or args.data_root in args.output_root.parents:
            parser.error("The output root must not be inside the read-only data root.")
    commands = {
        "predict": cmd_predict, "run": cmd_run, "screen": cmd_screen, "report": cmd_report, "sweep": cmd_sweep,
        "oracle": cmd_oracle,
    }
    commands[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
