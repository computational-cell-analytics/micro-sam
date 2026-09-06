"""Run one APG configuration on the crops of an `apg3d_manifest` subset, one crop per invocation.

Each crop is scored, attributed and timed on its own so that a Slurm array can spread a subset over
many MIG slices, and `aggregate` folds the per-crop results into a summary with per-crop bootstrap
confidence intervals, a family macro and seen/unseen macros. `--serial` runs every crop of a subset
in one process, which is what a timing trial needs.

Recall attribution per crop, from the generation trace (`generate(keep_trace=True)`):
  seeded_<ladder>: ground-truth objects containing a candidate anchor of that density ladder,
  anchor_kept:     objects containing the anchor of a candidate that survived the anchor scoring,
  tracked:         objects some propagated record overlaps at IoU >= 0.5 before the merge,
  merged:          objects matched in the output; genuine_misses excludes the crop-severed ones.

Usage examples:
    python benchmark_apg_3d.py run --subset primary --config configs/apg3d_defaults.json --sample-index 3
    python benchmark_apg_3d.py run --subset primary --config configs/apg3d_defaults.json --serial
    python benchmark_apg_3d.py aggregate --subset primary --config configs/apg3d_defaults.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

EVALUATION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVALUATION_ROOT))

import common  # noqa
from common import VOLUME_SPEED_OPTIONS, build_apg_segmenter, checkpoint_checksum, get_joint_checkpoint  # noqa
from common import genuine_misses, severed_objects, unmatched_objects  # noqa
from parameter_search import compute_metrics  # noqa
from optimization.benchmark_apg_optimization import (  # noqa
    DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, _atomic_write_csv, _atomic_write_json, _content_checksum,
    _git_revision, _hardware_identity, _implementation_checksum,
)
from optimization.apg3d_manifest import CAMPAIGN_ROOT, load_manifest, load_normalized_source, load_sample  # noqa

DEFAULT_LADDERS = ((1.5, 10.0), (1.0, 3.0, 10.0), (0.5, 2.0, 10.0))
LEGACY_FAMILIES = ("celegans", "embedseg", "gonuclear", "cremi", "snemi")
STATS_KEYS = (
    "proposed_candidates", "scored_candidates", "unique_anchor_slices", "propagation_passes",
    "propagated_candidates", "pruned_candidates", "propagated_frame_steps", "early_stopped_frame_steps",
    "filtered_candidates", "budgeted_candidates", "candidate_scorer_seconds",
    "refined_candidates", "replaced_candidates", "gated_consistency", "gated_foreign", "refinement_negatives",
)
BOOTSTRAP_SAMPLES = 2000


VOLUME_PARAM_KEYS = (
    "candidate_threshold", "foreground_threshold", "n_iter", "dt", "sigma", "min_candidate_size",
    "score_threshold", "max_overlap", "min_size", "max_size_factor", "refinement", "refinement_kwargs",
    "multimasking", "n_objects_per_pass", "early_stop_patience", "propagation_waves", "batch_size", "n_threads",
    "candidate_scorer_threshold", "candidate_order", "candidate_budget",
)


def resolve_volume_params(overrides: Optional[Dict[str, Any]], model_type: str = "hvit_t") -> Dict[str, Any]:
    """The volume parameters `generate` would use, with 'overrides' on top.

    Unlike `common.resolve_params(ndim=3)`, which fills every key from the 2d per-model table and
    only swaps the candidate threshold, this starts from the library's *volume* defaults (sigma,
    minimum candidate size, overlap limit and size floor differ between an image and a volume), so a
    run with no overrides is exactly what `generate()` does on its own.
    """
    from micro_sam.v2.automatic_prompt_generation import DEFAULT_PROMPT_GENERATION, default_prompt_generation
    defaults = {**DEFAULT_PROMPT_GENERATION, **default_prompt_generation(model_type, is_volume=True)}
    defaults["foreground_threshold"], defaults["n_iter"] = 0.7, 50  # flat constants of `derive_volume_prompts`
    params = {key: defaults[key] for key in VOLUME_PARAM_KEYS if key in defaults}
    overrides = dict(overrides or {})
    if "candidate_threshold_3d" in overrides:
        overrides["candidate_threshold"] = overrides.pop("candidate_threshold_3d")
    unknown = set(overrides) - set(VOLUME_PARAM_KEYS)
    if unknown:
        raise ValueError(f"Unknown volume parameters: {sorted(unknown)}.")
    params.update(overrides)
    return params


def load_volume_config(path: Optional[Path], model_type: str = "hvit_t") -> Tuple[str, Dict[str, Any]]:
    """Read a benchmark-style configuration and resolve its 'params_3d' against the volume defaults."""
    if path is None:
        return "apg3d-defaults", resolve_volume_params({}, model_type)
    with open(path) as f:
        config = json.load(f)
    unknown = set(config) - {"name", "params_2d", "params_3d"}
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}.")
    return str(config.get("name", path.stem)), resolve_volume_params(config.get("params_3d", {}), model_type)


def _ladder_key(ladder: Sequence[float]) -> str:
    return "seeded_" + "_".join(f"{value:g}" for value in ladder).replace(".", "p")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_identity(config_name: str, params_3d: Dict[str, Any], artifacts: Dict[str, Path]) -> str:
    identity = {
        "params_3d": params_3d,
        "artifacts": {name: _sha256(path) for name, path in sorted(artifacts.items())},
    }
    return f"{config_name}-{_content_checksum(identity)[:12]}-{_implementation_checksum()[:12]}"


def run_dir(campaign_root: Path, subset: str, config_name: str, params_3d: Dict[str, Any],
            artifacts: Dict[str, Path]) -> Path:
    return campaign_root / "runs" / subset / run_identity(config_name, params_3d, artifacts)


def sibling_run_dirs(run_path: Path) -> List[Path]:
    """Run directories of the same configuration under other implementation checksums.

    A library edit that leaves the volume path's output unchanged (an epoch boundary) re-keys the
    directory while an array is still running, so one configuration's crops can end up in two of them.
    The aggregate reads them all and records every implementation checksum it saw.
    """
    prefix = run_path.name.rsplit("-", 1)[0] + "-"
    return sorted(path for path in run_path.parent.glob(f"{prefix}*") if path.is_dir())


# ----------------------------------------------------------------------------------------------
# attribution


def _objects_containing(labels: np.ndarray, anchors_zyx: np.ndarray) -> set:
    if len(anchors_zyx) == 0:
        return set()
    valid = np.all((anchors_zyx >= 0) & (anchors_zyx < np.asarray(labels.shape)), axis=1)
    hits = labels[tuple(anchors_zyx[valid].T)]
    return set(int(value) for value in np.unique(hits) if value != 0)


def _tracked_objects(labels: np.ndarray, records: List[dict], iou_threshold: float = 0.5) -> set:
    """Ground-truth ids some pre-merge record overlaps at IoU >= threshold."""
    sizes = np.bincount(labels.ravel())
    tracked = set()
    for record in records:
        mask = record["segmentation"]
        area = int(mask.sum())
        if area == 0:
            continue
        overlap = np.bincount(labels[record["bounding_box"]][mask], minlength=len(sizes))
        overlap[0] = 0
        best = int(overlap.argmax())
        if best == 0:
            continue
        intersection = int(overlap[best])
        iou = intersection / (area + int(sizes[best]) - intersection)
        if iou >= iou_threshold:
            tracked.add(best)
    return tracked


def _anchors_of(prompts: Optional[dict]) -> np.ndarray:
    if prompts is None:
        return np.zeros((0, 3), dtype="int64")
    points = np.asarray(prompts["points"])[:, 0]
    frames = np.asarray(prompts["frames"])
    return np.stack([frames, points[:, 1].astype("int64"), points[:, 0].astype("int64")], axis=1)


def attribute_recall(
    segmenter, labels: np.ndarray, segmentation: np.ndarray, trace: Optional[dict], ladders: Sequence[Sequence[float]],
    spacing: Optional[tuple],
) -> Dict[str, Any]:
    from micro_sam.v2.automatic_prompt_generation import derive_volume_prompts

    gt_ids = set(int(value) for value in np.unique(labels) if value != 0)
    _, severed_ids = severed_objects(labels)
    severed = set(int(value) for value in severed_ids)
    genuine = gt_ids - severed
    result = {"gt_objects": len(gt_ids), "severed_objects": len(severed)}
    prediction = segmenter._prediction
    for ladder in ladders:
        prompts = derive_volume_prompts(
            prediction[0], prediction[1:], model_type=segmenter._model_type, candidate_threshold=tuple(ladder),
            spacing=spacing,
        )
        result[_ladder_key(ladder)] = len(_objects_containing(labels, _anchors_of(prompts)) & genuine)
        result[_ladder_key(ladder).replace("seeded", "candidates")] = 0 if prompts is None else len(prompts["points"])
    if trace is not None:
        candidates = trace["candidates"]
        anchors = np.array(
            [(c["frame"], int(c["point"][1]), int(c["point"][0])) for c in candidates], dtype="int64",
        ).reshape(-1, 3)
        result["anchor_kept"] = len(_objects_containing(labels, anchors) & genuine)
        result["tracked"] = len(_tracked_objects(labels, trace["records"]) & genuine)
    unmatched = set(int(value) for value in np.unique(unmatched_objects(labels, segmentation)) if value != 0)
    result["merged"] = len(genuine - unmatched)
    result["unmatched"], result["genuine_misses"] = genuine_misses(labels, segmentation)
    return result


# ----------------------------------------------------------------------------------------------
# running


def _build(model_type: str, joint_checkpoint: str, device: str, export_root: Path, artifacts: Dict[str, Path]):
    checkpoint_id = checkpoint_checksum(get_joint_checkpoint(model_type, joint_checkpoint))
    segmenter = build_apg_segmenter(
        model_type, 3, device, joint_checkpoint=joint_checkpoint, joint_checksum=checkpoint_id,
        export_root=str(export_root),
    )
    if "volume_candidate_scorer" in artifacts:
        from optimization.train_apg_3d_filter import load_volume_candidate_scorer
        segmenter.set_multimask_models(
            volume_candidate_scorer=load_volume_candidate_scorer(artifacts["volume_candidate_scorer"], device=device),
        )
    return segmenter, checkpoint_id


def _save_outputs(path: Path, segmentation: np.ndarray, trace: Optional[dict]) -> None:
    """Keep what a visual inspection needs: the segmentation, every proposed anchor and which ones survived.

    Anchors are (z, y, x) voxel coordinates of the density-ladder candidates; 'scored_prompt_index' lists
    the anchors whose candidates passed the anchor scoring and were propagated, 'merged_prompt_index' those
    whose track made it into the output (in output instance id order, 'merged_instance_id').
    """
    dtype = "uint16" if segmentation.max() < np.iinfo("uint16").max else "uint32"
    arrays = {"segmentation": segmentation.astype(dtype)}
    if trace is not None:
        arrays["anchors"] = _anchors_of(trace.get("prompts"))
        candidates = trace.get("candidates") or []
        arrays["scored_prompt_index"] = np.array(
            [int(candidate.get("prompt_index", -1)) for candidate in candidates], dtype="int64",
        )
        records, matches = trace.get("records") or [], trace.get("matches") or {}
        arrays["merged_instance_id"] = np.array(sorted(matches), dtype="int64")
        arrays["merged_prompt_index"] = np.array(
            [int(records[matches[instance_id]].get("prompt_index", -1)) for instance_id in sorted(matches)],
            dtype="int64",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def run_crop(
    segmenter, sample: Dict[str, Any], raw: np.ndarray, labels: np.ndarray, valid: Optional[np.ndarray],
    params_3d: Dict[str, Any], device: str, ladders: Sequence[Sequence[float]], save_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    segmenter.clear_state()
    cuda_device = torch.device(device) if device.startswith("cuda") else None
    if cuda_device is not None:
        torch.cuda.reset_peak_memory_stats(cuda_device)
    spacing = tuple(sample["spacing"]) if sample.get("spacing") and tuple(sample["spacing"]) != (1, 1, 1) else None
    started = time.perf_counter()
    segmenter.initialize(raw, ndim=3, **VOLUME_SPEED_OPTIONS)
    initialized = time.perf_counter()
    segmentation = segmenter.generate(**params_3d, spacing=spacing, keep_trace=True).astype("uint32")
    generated = time.perf_counter()
    if valid is not None:
        segmentation[~valid] = 0
    metrics = compute_metrics(segmentation, labels, sample["metric_mode"], border_min_size=0)
    row = {
        "sample_id": sample["sample_id"],
        "dataset": sample["dataset"],
        "family": sample["family"],
        "seen_in_training": str(sample["seen_in_training"]),
        "depth_flag": sample["depth_flag"],
        "realized_depth": int(labels.shape[0]),
        "legacy_sample_id": sample.get("legacy_sample_id"),
        "predicted_objects": int(len(np.unique(segmentation)) - 1),
        "initialization_seconds": initialized - started,
        "generation_seconds": generated - initialized,
        "total_seconds": generated - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(cuda_device)) if cuda_device else None,
        **metrics,
    }
    stats = getattr(segmenter, "_last_generation_stats", {}) or {}
    row.update({key: stats.get(key, 0) for key in STATS_KEYS})
    row.update(attribute_recall(segmenter, labels, segmentation, segmenter._last_generation_trace, ladders, spacing))
    if save_dir is not None:
        _save_outputs(
            save_dir / f"{sample['sample_id'].replace(':', '_')}.npz", segmentation, segmenter._last_generation_trace,
        )
    segmenter._last_generation_trace = None
    return row


def _write_crop(run_path: Path, row: Dict[str, Any]) -> None:
    (run_path / "crops").mkdir(parents=True, exist_ok=True)
    _atomic_write_json(run_path / "crops" / f"{row['sample_id'].replace(':', '_')}.json", row)


def _write_metadata(run_path: Path, manifest: Dict[str, Any], config_name: str, params_3d: Dict[str, Any],
                    artifacts: Dict[str, Path], model_type: str, joint_checkpoint: str, checkpoint_id: str,
                    device: str, ladders: Sequence[Sequence[float]], status: str, extra: Optional[dict] = None) -> None:
    metadata = {
        "campaign": "apg3d",
        "status": status,
        "config_name": config_name,
        "params_3d": params_3d,
        "artifacts": {name: str(Path(path).resolve()) for name, path in artifacts.items()},
        "artifact_checksums": {name: _sha256(path) for name, path in artifacts.items()},
        "manifest_checksum": manifest["manifest_checksum"],
        "subset": manifest["subset"],
        "datasets": sorted({sample["dataset"] for sample in manifest["samples"]}),
        "implementation_checksum": _implementation_checksum(),
        "checkpoint_checksum": checkpoint_id,
        "checkpoint_name": joint_checkpoint,
        "model_type": model_type,
        "device": device,
        "hardware": _hardware_identity(device),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "git_revision": _git_revision(),
        "ladders": [list(ladder) for ladder in ladders],
        **(extra or {}),
    }
    _atomic_write_json(run_path / "metadata.json", metadata)


def run(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.subset, args.campaign_root, args.data_root)
    config_name, params_3d = load_volume_config(args.config, args.model_type)
    artifacts = {}
    if args.volume_candidate_scorer_artifact is not None:
        artifacts["volume_candidate_scorer"] = Path(args.volume_candidate_scorer_artifact)
    run_path = run_dir(args.campaign_root, args.subset, config_name, params_3d, artifacts)
    samples = manifest["samples"]
    if args.sample_index is not None:
        samples = [samples[args.sample_index]]
    elif args.sample_id is not None:
        samples = [sample for sample in samples if sample["sample_id"] == args.sample_id]
        if not samples:
            raise SystemExit(f"Unknown sample id {args.sample_id!r}.")
    elif not args.serial:
        raise SystemExit("Pass --sample-index, --sample-id or --serial.")
    pending = [
        sample for sample in samples
        if args.force or not (run_path / "crops" / f"{sample['sample_id'].replace(':', '_')}.json").exists()
    ]
    if not pending:
        print(f"All {len(samples)} crop(s) already done in {run_path}.")
        return
    ladders = tuple(tuple(ladder) for ladder in args.ladders) if args.ladders else DEFAULT_LADDERS
    segmenter, checkpoint_id = _build(
        args.model_type, args.joint_checkpoint, args.device, DEFAULT_OUTPUT_ROOT / "model_exports", artifacts,
    )
    if not (run_path / "metadata.json").exists():
        _write_metadata(
            run_path, manifest, config_name, params_3d, artifacts, args.model_type, args.joint_checkpoint,
            checkpoint_id, args.device, ladders, status="running",
        )
    source_cache: Dict[tuple, np.ndarray] = {}
    started = time.perf_counter()
    for sample in pending:
        key = (sample["raw_path"], tuple(sample["normalization_z_range"]))
        if key not in source_cache:
            source_cache.clear()
            source_cache[key] = load_normalized_source(sample, args.data_root)
        raw, labels, valid = load_sample(sample, args.data_root, source_cache[key])
        row = run_crop(
            segmenter, sample, raw, labels, valid, params_3d, args.device, ladders,
            save_dir=(run_path / "outputs") if args.save_outputs else None,
        )
        row["trial_id"] = args.trial_id
        row["hardware"] = _hardware_identity(args.device).get("accelerator")
        _write_crop(run_path, row)
        print(f"{sample['sample_id']:36s} msa={row.get('msa', float('nan')):.4f} "
              f"objects {row['gt_objects']}/{row['predicted_objects']} passes {row['propagation_passes']} "
              f"{row['total_seconds']:.1f} s")
        if args.time_budget_minutes and (time.perf_counter() - started) / 60 > args.time_budget_minutes:
            print("Time budget exhausted; stopping after this crop.")
            break
    segmenter.clear_state()
    print(f"Run directory: {run_path}")


# ----------------------------------------------------------------------------------------------
# aggregation


def _bootstrap_ci(values: np.ndarray, seed: int = 0, n_samples: int = BOOTSTRAP_SAMPLES) -> tuple:
    values = np.asarray(values, dtype="float64")
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_samples, len(values)))
    means = values[draws].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def summarize(samples: pd.DataFrame) -> pd.DataFrame:
    """Per-dataset means with bootstrap CIs, plus family, seen/unseen, legacy and balanced macros."""
    metric = "msa"
    rows = []
    numeric = [column for column in samples.columns if pd.api.types.is_numeric_dtype(samples[column])]
    sums = [column for column in numeric if column.startswith(("seeded_", "candidates_")) or column in (
        "gt_objects", "severed_objects", "anchor_kept", "tracked", "merged", "unmatched", "genuine_misses",
        "predicted_objects", "candidates", "tracks", "slice_instances", "chains", "hybrid_prompts", *STATS_KEYS,
    )]
    per_dataset = {}
    for dataset, group in samples.groupby("dataset", sort=True):
        values = group[metric].dropna().to_numpy() if metric in group else np.array([])
        low, high = _bootstrap_ci(values)
        row = {
            "dataset": dataset, "family": group["family"].iloc[0],
            "seen_in_training": group["seen_in_training"].iloc[0],
            "n_crops": len(group), f"{metric}_mean": float(values.mean()) if len(values) else np.nan,
            f"{metric}_std": float(values.std(ddof=0)) if len(values) else np.nan,
            f"{metric}_ci_low": low, f"{metric}_ci_high": high,
            "total_seconds": float(group["total_seconds"].sum()),
            "generation_seconds": float(group["generation_seconds"].sum()),
        }
        if "cremi" in group:
            row["cremi_mean"] = float(group["cremi"].dropna().mean()) if group["cremi"].notna().any() else np.nan
        for column in sums:
            row[column] = int(group[column].fillna(0).sum())
        rows.append(row)
        per_dataset[dataset] = row
    summary = pd.DataFrame(rows)

    def macro(name: str, selected: pd.DataFrame) -> Dict[str, Any]:
        if selected.empty:
            return {"dataset": name, "n_crops": 0, f"{metric}_mean": np.nan}
        families = selected.groupby("family")[f"{metric}_mean"].mean()
        return {
            "dataset": name, "n_crops": int(selected["n_crops"].sum()),
            f"{metric}_mean": float(families.mean()), "n_families": int(len(families)),
            "total_seconds": float(selected["total_seconds"].sum()),
            "generation_seconds": float(selected["generation_seconds"].sum()),
            **{column: int(selected[column].sum()) for column in sums if column in selected},
        }

    macros = [
        macro("__family_macro__", summary),
        macro("__dataset_balanced__", summary.assign(family=summary["dataset"])),
        macro("__unseen_macro__", summary[summary["seen_in_training"] == "False"]),
        macro("__legacy_macro__", summary[summary["family"].isin(LEGACY_FAMILIES)]),
    ]
    return pd.concat([summary, pd.DataFrame(macros)], ignore_index=True)


def aggregate(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.subset, args.campaign_root, args.data_root)
    config_name, params_3d = load_volume_config(args.config, args.model_type)
    artifacts = {}
    if args.volume_candidate_scorer_artifact is not None:
        artifacts["volume_candidate_scorer"] = Path(args.volume_candidate_scorer_artifact)
    run_path = run_dir(args.campaign_root, args.subset, config_name, params_3d, artifacts)
    by_sample: Dict[str, Dict[str, Any]] = {}
    implementations = []
    for sibling in sibling_run_dirs(run_path):
        implementation = sibling.name.rsplit("-", 1)[1]
        for path in sorted((sibling / "crops").glob("*.json")) if (sibling / "crops").exists() else []:
            row = json.load(open(path))
            row["implementation_checksum"] = implementation
            # The current implementation wins when a crop was run under both.
            if row["sample_id"] not in by_sample or sibling == run_path:
                by_sample[row["sample_id"]] = row
        if (sibling / "crops").exists():
            implementations.append(implementation)
    rows = list(by_sample.values())
    expected = {sample["sample_id"] for sample in manifest["samples"]}
    done = {row["sample_id"] for row in rows}
    if not rows:
        raise SystemExit(f"No crops found in {run_path} or its siblings.")
    samples = pd.DataFrame(rows)
    _atomic_write_csv(run_path / "samples.csv", samples)
    summary = summarize(samples)
    _atomic_write_csv(run_path / "summary.csv", summary)
    metadata_path = run_path / "metadata.json"
    metadata = json.load(open(metadata_path)) if metadata_path.exists() else {}
    metadata.update({
        "status": "complete" if done == expected else "partial",
        "n_crops": len(rows), "n_expected": len(expected), "missing": sorted(expected - done),
        "implementation_checksums": sorted(set(implementations)),
        "mixed_implementations": len(set(implementations)) > 1,
        "timing_comparable": (
            len(samples["hardware"].dropna().unique()) == 1 if "hardware" in samples else False
        ),
    })
    _atomic_write_json(metadata_path, metadata)
    columns = ["dataset", "n_crops", "msa_mean", "msa_ci_low", "msa_ci_high", "gt_objects", "merged", "genuine_misses",
               "propagation_passes", "total_seconds"]
    print(summary[[column for column in columns if column in summary]].to_string(index=False))
    print(f"{metadata['status']}: {len(rows)}/{len(expected)} crops in {run_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "aggregate"))
    parser.add_argument("--subset", required=True)
    parser.add_argument("--config", type=Path, default=None, help="Benchmark-style config JSON; defaults if omitted.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--campaign-root", type=Path, default=CAMPAIGN_ROOT)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--serial", action="store_true", help="Run every crop of the subset in this process.")
    parser.add_argument("--trial-id", default="trial-1")
    parser.add_argument("--force", action="store_true", help="Re-run crops that already have a result.")
    parser.add_argument("--model-type", default="hvit_t")
    parser.add_argument("--joint-checkpoint", default="best")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--time-budget-minutes", type=float, default=None)
    parser.add_argument("--ladders", type=json.loads, default=None, help='JSON, e.g. "[[1.5,10],[1,3,10]]".')
    parser.add_argument("--volume-candidate-scorer-artifact", type=Path, default=None)
    parser.add_argument(
        "--save-outputs", action="store_true",
        help="Also store each crop's segmentation and anchors under <run dir>/outputs/, for visual inspection.",
    )
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args)
    else:
        aggregate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
