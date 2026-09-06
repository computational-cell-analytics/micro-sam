"""Screen the structural, label-free 2d APG changes of the generalization campaign from cached proposals.

The campaign plan (`notes/APG_2D_GENERALIZATION_CAMPAIGN_PLAN.md`) asks for changes that improve on the
per-model registry defaults consistently across datasets, with nothing learned and nothing tuned. Every
candidate here is a `select`-level option of `AutomaticPromptGenerator` (AIS/APG fusion, decoder-arbitrated
merge, residual recovery) or a `propose`-level prompt type (box prompts), so one GPU pass per manifest
caches the decoder prediction and the proposals of every prompt type, and every selection variant is a CPU
replay of that cache. The registry-defaults replay has to reproduce the canonical benchmark bit for bit,
which the report checks.

Stages:
    cache    encode every image once (GPU), store the (4, Y, X) prediction and the proposals per prompt type
    oracle   P0 headroom: AIS vs APG per image and per object, recall ceiling from the seeded objects
    replay   the selection variants on the cache (CPU, one process per image)
    report   per-dataset deltas against the registry replay over one or several manifests, with the gate

Usage examples:
    python screen_apg_structural.py cache --subset primary
    python screen_apg_structural.py oracle --subset primary
    python screen_apg_structural.py replay --subset primary --workers 8
    python screen_apg_structural.py report --subsets primary training_extra
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from concurrent import futures
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EVALUATION_ROOT = Path(__file__).resolve().parent.parent
OPTIMIZATION_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALUATION_ROOT))
sys.path.insert(0, str(OPTIMIZATION_ROOT))

import common  # noqa
from common import GT_MIN_SIZE_2D, resolve_params, unmatched_objects  # noqa
from parameter_search import compute_metrics  # noqa
from benchmark_apg_optimization import (  # noqa
    DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, _atomic_write_csv, _atomic_write_json, _content_checksum,
    _default_manifest_path, _git_revision, _hardware_identity, _implementation_checksum, _load_2d_sample,
    _validate_roots, prepare_manifest,
)

MODEL_TYPE = "hvit_t"
CHECKPOINT = "best"
PROMPT_TYPES = ("point", "box", "point_box", "box_thin")
# The proposal half of the registry defaults, pinned explicitly (see CAMPAIGN_OPERATIONS.md).
PROPOSAL_PARAMS = {
    "candidate_threshold": 3.0, "dt": 0.5, "sigma": 0.5, "min_candidate_size": 4, "n_iter": 50,
    "foreground_threshold": 0.7, "multimasking": True, "multimask_scorer": "predicted_iou",
    "multimask_selection": "eager", "batch_size": 64,
}
# The selection half of the registry defaults.
SELECT_PARAMS = {"score_threshold": 0.6, "score_filter": "predicted_iou", "max_overlap": 0.3, "min_size": 50}
# The protocol: a candidate is up on at least this share of the datasets, no dataset below the minor
# regression line, and the balanced gain reaches the bar.
GATE_MIN_UP_FRACTION = 9 / 11
GATE_LOSS_LIMIT = -0.02
GATE_ABSOLUTE_ALLOWANCE = 0.005
GATE_BALANCED_GAIN = 0.02
ADAPTIVE_THRESHOLDS = (0.4, 0.5, 0.6, 0.7)
FUSION_SENSITIVITY = ((0.4, 0.9), (0.6, 0.9), (0.5, 0.85), (0.5, 0.95))


def structural_root(output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    return output_root / "structural_2d"


def cache_dir(output_root: Path, subset: str, checkpoint_id: str) -> Path:
    identity = _content_checksum({
        "proposal_params": PROPOSAL_PARAMS, "prompt_types": list(PROMPT_TYPES), "checkpoint": checkpoint_id,
        "implementation": _implementation_checksum(),
    })
    return structural_root(output_root) / "cache" / subset / identity


def sample_stem(sample_id: str) -> str:
    return sample_id.replace(":", "__").replace("/", "_")


def variant_grid() -> Dict[str, Dict[str, Any]]:
    """The fixed screening grid: every entry names the prompt type, the select overrides and any harness rule.

    Nothing here is tuned on a result: the list is the plan's variant list, with the sensitivity checks
    the plan asks to report rather than optimize.
    """
    grid: Dict[str, Dict[str, Any]] = {"registry": {"prompt_type": "point", "select": {}}}
    # P1: fusion with the decoder's instances.
    for mode in ("fallback", "conflict", "both"):
        grid[f"fusion-{mode}"] = {"prompt_type": "point", "select": {"fusion": mode}}
    for agreement, stability in FUSION_SENSITIVITY:
        grid[f"fusion-both-a{agreement:g}-s{stability:g}"] = {
            "prompt_type": "point", "select": {"fusion": "both"},
            "fusion_constants": {"agreement": agreement, "stability": stability},
        }
    # P2: the arbitrated merge.
    for arbitration in ("decoder", "euclidean"):
        for max_overlap in (0.3, 0.5, 1.0):
            grid[f"arb-{arbitration}-mo{max_overlap:g}"] = {
                "prompt_type": "point", "select": {"arbitration": arbitration, "max_overlap": max_overlap},
            }
    # P3a: box prompts, alone and with the two select-level changes.
    for prompt_type in ("box", "point_box", "box_thin"):
        grid[f"prompt-{prompt_type}"] = {"prompt_type": prompt_type, "select": {}}
        grid[f"prompt-{prompt_type}+fusion-both"] = {"prompt_type": prompt_type, "select": {"fusion": "both"}}
        grid[f"prompt-{prompt_type}+fusion-fallback"] = {
            "prompt_type": prompt_type, "select": {"fusion": "fallback"},
        }
        grid[f"prompt-{prompt_type}+arb-decoder-mo0.3"] = {
            "prompt_type": prompt_type, "select": {"arbitration": "decoder"},
        }
    # Combination of the two select-level changes.
    grid["fusion-both+arb-decoder-mo0.3"] = {
        "prompt_type": "point", "select": {"fusion": "both", "arbitration": "decoder"},
    }
    grid["fusion-fallback+arb-decoder-mo0.3"] = {
        "prompt_type": "point", "select": {"fusion": "fallback", "arbitration": "decoder"},
    }
    # P4: label-free per-image threshold from the agreement with the predicted foreground, and the fixed
    # thresholds of its grid as controls: the adaptation only counts if it beats the best fixed value.
    grid["adaptive-fg-agreement"] = {"prompt_type": "point", "select": {}, "adaptive": list(ADAPTIVE_THRESHOLDS)}
    for threshold in ADAPTIVE_THRESHOLDS:
        if threshold != SELECT_PARAMS["score_threshold"]:
            grid[f"fixed-t{threshold:g}"] = {"prompt_type": "point", "select": {"score_threshold": threshold}}
    grid["adaptive-fg-agreement-no0.4"] = {
        "prompt_type": "point", "select": {}, "adaptive": [t for t in ADAPTIVE_THRESHOLDS if t != 0.4],
    }
    return grid


def _headless_generator(prediction: np.ndarray):
    """An `AutomaticPromptGenerator` with only what `select` reads: the prediction and the model type."""
    from micro_sam.v2.automatic_prompt_generation import AutomaticPromptGenerator

    generator = object.__new__(AutomaticPromptGenerator)
    generator._prediction = prediction
    generator._model_type = MODEL_TYPE
    generator._last_generation_stats = {}
    generator._predictor = None
    generator._refinement_gate_model = None
    generator._microscopy_multimask_scorer = None
    generator._is_initialized = True
    return generator


def _select(generator, proposals: list, overrides: Dict[str, Any], constants: Optional[Dict[str, float]] = None):
    from micro_sam.v2.automatic_prompt_generation import fuse_with_instances

    params = {**SELECT_PARAMS, **overrides}
    fusion = params.pop("fusion", None)
    generator._last_generation_stats = {}
    if constants is None or fusion is None:
        return generator.select(proposals, fusion=fusion, **params)
    # The sensitivity variants call the fusion with explicit constants instead of the module's.
    segmentation = generator.select(proposals, **params)
    from micro_sam.v2.postprocessing import flow_instance_segmentation

    instances = flow_instance_segmentation(
        generator._prediction[0], generator._prediction[1:], model_type=MODEL_TYPE,
    )
    stability = _accepted_stability(generator, proposals, params)
    segmentation, stats = fuse_with_instances(
        segmentation, instances, stability, fusion, min_size=params["min_size"],
        agreement=constants["agreement"], stability_threshold=constants["stability"],
    )
    generator._last_generation_stats.update(stats)
    return segmentation


def _accepted_stability(generator, proposals: list, params: Dict[str, Any]) -> Dict[int, float]:
    """The stability per accepted instance, from a merge with context (same result as the plain one)."""
    shape = generator._prediction[0].shape
    _, context = generator._merge(
        proposals, shape, score_threshold=params["score_threshold"], max_overlap=params["max_overlap"],
        min_size=params["min_size"], return_context=True, score_filter=params["score_filter"],
        arbitration=params.get("arbitration", "drop"),
    )
    if context is None:
        return {}
    return {
        instance_id: float(context["records"][index]["stability_score"])
        for instance_id, index in context["matches"].items()
    }


def foreground_agreement(segmentation: np.ndarray, foreground: np.ndarray, threshold: float = 0.5) -> float:
    """Dice between the union of the accepted masks and the predicted foreground above the threshold."""
    masks = segmentation != 0
    fg = foreground > threshold
    denominator = int(masks.sum()) + int(fg.sum())
    return 2.0 * int((masks & fg).sum()) / denominator if denominator else 1.0


def select_adaptive(generator, proposals: list, overrides: Dict[str, Any], thresholds: Sequence[float]):
    """Pick the filter threshold per image by the foreground agreement, then return that selection."""
    best = None
    for threshold in thresholds:
        segmentation = _select(generator, proposals, {**overrides, "score_threshold": float(threshold)})
        agreement = foreground_agreement(segmentation, generator._prediction[0])
        if best is None or agreement > best[0]:
            best = (agreement, threshold, segmentation)
    generator._last_generation_stats["adaptive_threshold"] = best[1]
    generator._last_generation_stats["adaptive_agreement"] = best[0]
    return best[2]


def object_recall_counts(records: Sequence[dict], labels: np.ndarray, iou: float = 0.5) -> Tuple[int, int]:
    """How many ground-truth objects some prompt lands in ('seeded') and some proposal matches ('proposed')."""
    seeded, proposed = set(), set()
    for record in records:
        x, y = record["point"]
        y, x = int(np.clip(round(y), 0, labels.shape[0] - 1)), int(np.clip(round(x), 0, labels.shape[1] - 1))
        target = int(labels[y, x])
        if target == 0:
            continue
        seeded.add(target)
        if target in proposed:
            continue
        box = record["bounding_box"]
        mask = record["segmentation"]
        gt_crop = labels[box] == target
        intersection = int((mask & gt_crop).sum())
        union = int(mask.sum()) + int((labels == target).sum()) - intersection
        if union and intersection / union >= iou:
            proposed.add(target)
    return len(seeded), len(proposed)


def matched_objects(labels: np.ndarray, segmentation: np.ndarray) -> np.ndarray:
    """The ground-truth object ids a segmentation matches at IoU 0.5."""
    ids = np.unique(labels)
    ids = ids[ids != 0]
    missed = np.unique(unmatched_objects(labels, segmentation))
    return np.setdiff1d(ids, missed)


# --- cache ---------------------------------------------------------------------------------------------


def stage_cache(manifest: Dict[str, Any], data_root: Path, output_root: Path, device: str) -> Path:
    import torch

    checkpoint = common.get_joint_checkpoint(MODEL_TYPE, CHECKPOINT)
    checkpoint_id = common.checkpoint_checksum(checkpoint)
    root = cache_dir(output_root, manifest["subset"], checkpoint_id)
    root.mkdir(parents=True, exist_ok=True)
    samples = [sample for sample in manifest["samples"] if sample["ndim"] == 2]
    pending = [sample for sample in samples if not (root / f"{sample_stem(sample['sample_id'])}.pkl").exists()]
    _atomic_write_json(root / "metadata.json", {
        "subset": manifest["subset"], "manifest_checksum": manifest["manifest_checksum"],
        "checkpoint_checksum": checkpoint_id, "implementation_checksum": _implementation_checksum(),
        "proposal_params": PROPOSAL_PARAMS, "prompt_types": list(PROMPT_TYPES), "git_revision": _git_revision(),
        "hardware": _hardware_identity(device), "n_samples": len(samples), "status": "running",
    })
    if not pending:
        print(f"Cache complete at {root}")
    else:
        segmenter = common.build_apg_segmenter(
            MODEL_TYPE, 2, device, joint_checkpoint=CHECKPOINT, joint_checksum=checkpoint_id,
            export_root=str(output_root / "model_exports"),
        )
        started = time.perf_counter()
        try:
            for number, sample in enumerate(pending, 1):
                raw, _ = _load_2d_sample(sample, data_root)
                segmenter.clear_state()
                segmenter.initialize(raw, ndim=2)
                proposals = {}
                seconds = {}
                for prompt_type in PROMPT_TYPES:
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    proposals[prompt_type] = segmenter.propose(prompt_type=prompt_type, **PROPOSAL_PARAMS)
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    seconds[prompt_type] = time.perf_counter() - t0
                stem = root / sample_stem(sample["sample_id"])
                np.save(str(stem) + ".prediction.npy", np.asarray(segmenter._prediction, dtype="float32"))
                payload = {
                    "sample_id": sample["sample_id"], "dataset": sample["dataset"], "proposals": proposals,
                    "propose_seconds": seconds,
                }
                tmp = stem.with_suffix(".pkl.tmp")
                with open(tmp, "wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                tmp.replace(stem.with_suffix(".pkl"))
                elapsed = time.perf_counter() - started
                print(f"[{number}/{len(pending)}] {sample['sample_id']} ({elapsed / number:.1f} s/image)", flush=True)
        finally:
            segmenter.clear_state()
    metadata = json.load(open(root / "metadata.json"))
    metadata["status"] = "complete"
    _atomic_write_json(root / "metadata.json", metadata)
    return root


def load_cached(root: Path, sample_id: str) -> Tuple[np.ndarray, Dict[str, list], Dict[str, float]]:
    stem = root / sample_stem(sample_id)
    prediction = np.load(str(stem) + ".prediction.npy")
    with open(stem.with_suffix(".pkl"), "rb") as f:
        payload = pickle.load(f)
    return prediction, payload["proposals"], payload["propose_seconds"]


# --- oracle (P0) -----------------------------------------------------------------------------------------


def oracle_row(sample: Dict[str, Any], labels: np.ndarray, prediction: np.ndarray, proposals: Dict[str, list]):
    from micro_sam.v2.postprocessing import flow_instance_segmentation

    border = GT_MIN_SIZE_2D.get(sample["dataset"], 0)
    generator = _headless_generator(prediction)
    apg = _select(generator, proposals["point"], {})
    ais = flow_instance_segmentation(prediction[0], prediction[1:], model_type=MODEL_TYPE).astype("uint32")
    apg_msa = compute_metrics(apg, labels, "sparse", border_min_size=border)["msa"]
    ais_msa = compute_metrics(ais, labels, "sparse", border_min_size=border)["msa"]
    apg_matched = set(matched_objects(labels, apg).tolist())
    ais_matched = set(matched_objects(labels, ais).tolist())
    seeded, proposed = object_recall_counts(proposals["point"], labels)
    seeded_box, proposed_box = object_recall_counts(proposals["box"], labels)
    n_objects = int(len(np.unique(labels)) - 1)
    return {
        "sample_id": sample["sample_id"], "dataset": sample["dataset"], "gt_objects": n_objects,
        "apg_msa": apg_msa, "ais_msa": ais_msa, "max_msa": max(apg_msa, ais_msa),
        "apg_objects": int(len(np.unique(apg)) - 1), "ais_objects": int(len(np.unique(ais)) - 1),
        "apg_matched": len(apg_matched), "ais_matched": len(ais_matched),
        "either_matched": len(apg_matched | ais_matched), "ais_only_matched": len(ais_matched - apg_matched),
        "seeded": seeded, "proposed": proposed, "seeded_box": seeded_box, "proposed_box": proposed_box,
        "n_prompts": len(proposals["point"]),
    }


def _oracle_worker(args) -> Dict[str, Any]:
    sample, root, data_root = args
    _, labels = _load_2d_sample(sample, data_root)
    prediction, proposals, _ = load_cached(root, sample["sample_id"])
    return oracle_row(sample, labels, prediction, proposals)


def summarize_oracle(rows: pd.DataFrame) -> pd.DataFrame:
    sums = [
        "gt_objects", "apg_objects", "ais_objects", "apg_matched", "ais_matched", "either_matched",
        "ais_only_matched", "seeded", "proposed", "seeded_box", "proposed_box", "n_prompts",
    ]
    table = rows.groupby("dataset", sort=True).agg(
        n_samples=("sample_id", "count"), apg_msa=("apg_msa", "mean"), ais_msa=("ais_msa", "mean"),
        per_image_max_msa=("max_msa", "mean"), **{column: (column, "sum") for column in sums},
    ).reset_index()
    balanced = {
        "dataset": "__dataset_balanced__", "n_samples": int(len(rows)), "apg_msa": float(table["apg_msa"].mean()),
        "ais_msa": float(table["ais_msa"].mean()), "per_image_max_msa": float(table["per_image_max_msa"].mean()),
        **{column: int(table[column].sum()) for column in sums},
    }
    table = pd.concat([table, pd.DataFrame([balanced])], ignore_index=True)
    table["dataset_max_msa"] = table[["apg_msa", "ais_msa"]].max(axis=1)
    table["dataset_ceiling_rel"] = (table["dataset_max_msa"] - table["apg_msa"]) / table["apg_msa"]
    table["per_image_ceiling_rel"] = (table["per_image_max_msa"] - table["apg_msa"]) / table["apg_msa"]
    table["apg_recall"] = table["apg_matched"] / table["gt_objects"]
    table["ais_recall"] = table["ais_matched"] / table["gt_objects"]
    table["union_recall"] = table["either_matched"] / table["gt_objects"]
    table["seeded_fraction"] = table["seeded"] / table["gt_objects"]
    table["proposed_fraction"] = table["proposed"] / table["gt_objects"]
    table["proposed_fraction_box"] = table["proposed_box"] / table["gt_objects"]
    return table


def stage_oracle(manifest: Dict[str, Any], data_root: Path, output_root: Path, workers: int) -> Path:
    checkpoint_id = common.checkpoint_checksum(common.get_joint_checkpoint(MODEL_TYPE, CHECKPOINT))
    root = cache_dir(output_root, manifest["subset"], checkpoint_id)
    samples = [sample for sample in manifest["samples"] if sample["ndim"] == 2]
    out_dir = structural_root(output_root) / "oracle" / manifest["subset"] / root.name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _map_samples(_oracle_worker, [(sample, root, data_root) for sample in samples], workers)
    table = pd.DataFrame(rows)
    _atomic_write_csv(out_dir / "oracle_samples.csv", table)
    summary = summarize_oracle(table)
    _atomic_write_csv(out_dir / "oracle_summary.csv", summary)
    columns = [
        "dataset", "apg_msa", "ais_msa", "dataset_ceiling_rel", "per_image_ceiling_rel", "apg_recall", "ais_recall",
        "union_recall", "seeded_fraction", "proposed_fraction", "proposed_fraction_box",
    ]
    print(summary[columns].round(4).to_string(index=False))
    print(f"Oracle: {out_dir}")
    return out_dir


# --- replay ----------------------------------------------------------------------------------------------


def replay_rows(sample: Dict[str, Any], labels: np.ndarray, prediction: np.ndarray, proposals: Dict[str, list],
                grid: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    border = GT_MIN_SIZE_2D.get(sample["dataset"], 0)
    n_objects = int(len(np.unique(labels)) - 1)
    generator = _headless_generator(prediction)
    recall = {prompt_type: object_recall_counts(records, labels) for prompt_type, records in proposals.items()}
    rows = []
    for name, variant in grid.items():
        records = proposals[variant["prompt_type"]]
        started = time.perf_counter()
        if variant.get("adaptive"):
            segmentation = select_adaptive(generator, records, variant["select"], variant["adaptive"])
        else:
            segmentation = _select(generator, records, variant["select"], variant.get("fusion_constants"))
        seconds = time.perf_counter() - started
        segmentation = segmentation.astype("uint32")
        metrics = compute_metrics(segmentation, labels, "sparse", border_min_size=border)
        stats = generator._last_generation_stats
        merged = len(matched_objects(labels, segmentation))
        seeded, proposed = recall[variant["prompt_type"]]
        rows.append({
            "sample_id": sample["sample_id"], "dataset": sample["dataset"], "variant": name,
            "prompt_type": variant["prompt_type"], "gt_objects": n_objects,
            "predicted_objects": int(len(np.unique(segmentation)) - 1), "n_prompts": len(records),
            "seeded": seeded, "proposed": proposed, "merged": merged, "select_seconds": seconds,
            "fusion_fallback_added": int(stats.get("fusion_fallback_added", 0)),
            "fusion_conflicts": int(stats.get("fusion_conflicts", 0)),
            "fusion_conflicts_split": int(stats.get("fusion_conflicts_split", 0)),
            "arbitration_dropped": int(stats.get("arbitration_dropped", 0)),
            "adaptive_threshold": float(stats.get("adaptive_threshold", np.nan)),
            **metrics,
        })
    return rows


def _replay_worker(args) -> List[Dict[str, Any]]:
    sample, root, data_root, grid = args
    _, labels = _load_2d_sample(sample, data_root)
    prediction, proposals, _ = load_cached(root, sample["sample_id"])
    return replay_rows(sample, labels, prediction, proposals, grid)


def _map_samples(worker, tasks: Sequence[Any], workers: int) -> list:
    results = []
    if workers <= 1:
        for number, task in enumerate(tasks, 1):
            results.append(worker(task))
            print(f"[{number}/{len(tasks)}]", flush=True)
        return results
    with futures.ProcessPoolExecutor(workers) as pool:
        for number, result in enumerate(pool.map(worker, tasks, chunksize=1), 1):
            results.append(result)
            if number % 10 == 0 or number == len(tasks):
                print(f"[{number}/{len(tasks)}]", flush=True)
    return results


def summarize_replay(rows: pd.DataFrame) -> pd.DataFrame:
    sums = (
        "gt_objects", "predicted_objects", "n_prompts", "seeded", "proposed", "merged", "fusion_fallback_added",
        "fusion_conflicts", "fusion_conflicts_split", "arbitration_dropped",
    )
    parts = []
    for name, frame in rows.groupby("variant", sort=False):
        table = frame.groupby("dataset", sort=True).agg(
            n_samples=("sample_id", "count"), msa_mean=("msa", "mean"), select_seconds=("select_seconds", "sum"),
            **{column: (column, "sum") for column in sums},
        ).reset_index()
        table.insert(0, "variant", name)
        parts.append(table)
        parts.append(pd.DataFrame([{
            "variant": name, "dataset": "__dataset_balanced__", "n_samples": int(len(frame)),
            "msa_mean": float(table["msa_mean"].mean()), "select_seconds": float(table["select_seconds"].sum()),
            **{column: int(table[column].sum()) for column in sums},
        }]))
    return pd.concat(parts, ignore_index=True)


def stage_replay(manifest: Dict[str, Any], data_root: Path, output_root: Path, workers: int,
                 variants: Optional[Sequence[str]] = None) -> Path:
    checkpoint_id = common.checkpoint_checksum(common.get_joint_checkpoint(MODEL_TYPE, CHECKPOINT))
    root = cache_dir(output_root, manifest["subset"], checkpoint_id)
    if not (root / "metadata.json").exists():
        raise SystemExit(f"No cache at {root}; run the cache stage first.")
    grid = variant_grid()
    if variants:
        unknown = set(variants) - set(grid)
        if unknown:
            raise SystemExit(f"Unknown variants: {sorted(unknown)}.")
        grid = {name: grid[name] for name in grid if name in set(variants) | {"registry"}}
    samples = [sample for sample in manifest["samples"] if sample["ndim"] == 2]
    identity = _content_checksum({"grid": grid, "cache": root.name, "manifest": manifest["manifest_checksum"]})
    out_dir = structural_root(output_root) / "replay" / manifest["subset"] / identity
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(out_dir / "metadata.json", {
        "subset": manifest["subset"], "manifest_checksum": manifest["manifest_checksum"], "cache": str(root),
        "implementation_checksum": _implementation_checksum(), "grid": grid, "git_revision": _git_revision(),
        "status": "running",
    })
    started = time.perf_counter()
    rows = _map_samples(_replay_worker, [(sample, root, data_root, grid) for sample in samples], workers)
    table = pd.DataFrame([row for rows_of_sample in rows for row in rows_of_sample])
    _atomic_write_csv(out_dir / "samples.csv", table)
    summary = summarize_replay(table)
    _atomic_write_csv(out_dir / "summary.csv", summary)
    metadata = json.load(open(out_dir / "metadata.json"))
    metadata.update({"status": "complete", "wall_seconds": time.perf_counter() - started})
    _atomic_write_json(out_dir / "metadata.json", metadata)
    balanced = summary[summary["dataset"] == "__dataset_balanced__"].sort_values("msa_mean", ascending=False)
    print(balanced[["variant", "msa_mean", "predicted_objects", "gt_objects", "merged"]].to_string(index=False))
    print(f"Replay: {out_dir}")
    return out_dir


# --- report ----------------------------------------------------------------------------------------------


def latest_replay(output_root: Path, subset: str) -> Optional[Path]:
    """The newest complete replay of a subset whose cache came from the checkpoint the environment selects."""
    checkpoint = common.checkpoint_checksum(common.get_joint_checkpoint(MODEL_TYPE, CHECKPOINT))
    candidates = []
    for metadata_path in (structural_root(output_root) / "replay" / subset).glob("*/metadata.json"):
        metadata = json.load(open(metadata_path))
        if metadata.get("status") != "complete":
            continue
        cache_metadata = Path(metadata["cache"]) / "metadata.json"
        if cache_metadata.exists() and json.load(open(cache_metadata))["checkpoint_checksum"] != checkpoint:
            continue
        candidates.append((metadata_path.stat().st_mtime, metadata_path.parent))
    return max(candidates)[1] if candidates else None


def find_reference_run(
    output_root: Path, manifest_checksum: str, implementation: Optional[str] = None,
    checkpoint_checksum: Optional[str] = None,
) -> Optional[Path]:
    """The canonical registry-defaults benchmark run of a manifest, preferring the current implementation."""
    registry = resolve_params({}, ndim=2, model_type=MODEL_TYPE)
    matches = []
    pattern = f"{checkpoint_checksum or '*'}/{manifest_checksum}-*/metadata.json"
    for metadata_path in (output_root / MODEL_TYPE).glob(pattern):
        metadata = json.load(open(metadata_path))
        if metadata.get("status") != "complete" or metadata.get("params_2d") != registry:
            continue
        current = metadata.get("implementation_checksum") == (implementation or _implementation_checksum())
        matches.append((current, metadata_path.stat().st_mtime, metadata_path.parent))
    return max(matches)[2] if matches else None


def identity_check(replay: pd.DataFrame, reference: pd.DataFrame) -> Dict[str, Any]:
    """Whether the registry replay reproduces the canonical run per image (bit-identical selection)."""
    registry = replay[replay["variant"] == "registry"].set_index("sample_id")
    reference = reference.set_index("sample_id")
    shared = registry.index.intersection(reference.index)
    differences = (registry.loc[shared, "msa"] - reference.loc[shared, "msa"]).abs()
    objects = (registry.loc[shared, "predicted_objects"] - reference.loc[shared, "predicted_objects"]).abs()
    return {
        "n_compared": int(len(shared)), "max_abs_msa_difference": float(differences.max()) if len(shared) else None,
        "n_object_count_differences": int((objects > 0).sum()),
        "identical": bool(len(shared) and differences.max() < 1e-9),
    }


def gate_table(summary: pd.DataFrame, control: str = "registry") -> pd.DataFrame:
    """Per variant: balanced gain, datasets up, worst regression and the protocol gate over all datasets given."""
    datasets = sorted(set(summary["dataset"]) - {"__dataset_balanced__"})
    per_dataset = summary[summary["dataset"] != "__dataset_balanced__"].pivot(
        index="dataset", columns="variant", values="msa_mean",
    )
    counts = summary[summary["dataset"] != "__dataset_balanced__"].pivot(
        index="dataset", columns="variant", values="predicted_objects",
    )
    gt = summary[(summary["dataset"] != "__dataset_balanced__") & (summary["variant"] == control)].set_index("dataset")
    rows = []
    for variant in per_dataset.columns:
        base, candidate = per_dataset[control], per_dataset[variant]
        delta = candidate - base
        relative = delta / base.replace(0, np.nan)
        up = int((delta > 0).sum())
        regressions = [
            dataset for dataset in datasets
            if relative[dataset] < GATE_LOSS_LIMIT and delta[dataset] < -GATE_ABSOLUTE_ALLOWANCE
        ]
        balanced_gain = (candidate.mean() - base.mean()) / base.mean()
        rows.append({
            "variant": variant, "n_datasets": len(datasets), "balanced_msa": float(candidate.mean()),
            "balanced_gain": float(balanced_gain), "datasets_up": up, "datasets_down": int((delta < 0).sum()),
            "worst_relative": float(relative.min()), "worst_dataset": str(relative.idxmin()),
            "best_relative": float(relative.max()), "best_dataset": str(relative.idxmax()),
            "regressions": ",".join(regressions),
            "objects_ratio": (
                float(counts[variant].sum() / gt["gt_objects"].sum())
                if variant in counts and "gt_objects" in gt else np.nan
            ),
            "gate": bool(
                up >= np.ceil(GATE_MIN_UP_FRACTION * len(datasets) - 1e-9) and not regressions
                and balanced_gain >= GATE_BALANCED_GAIN
            ),
        })
    return pd.DataFrame(rows).sort_values("balanced_gain", ascending=False).reset_index(drop=True)


def stage_report(output_root: Path, subsets: Sequence[str], replay_dirs: Optional[Sequence[Path]] = None) -> Path:
    tables, identities, checkpoints = [], {}, set()
    for index, subset in enumerate(subsets):
        replay_dir = Path(replay_dirs[index]) if replay_dirs else latest_replay(output_root, subset)
        if replay_dir is None:
            raise SystemExit(f"No complete replay for subset '{subset}'.")
        metadata = json.load(open(replay_dir / "metadata.json"))
        # The cache records which checkpoint proposed; the canonical run to compare with has to match it.
        checkpoint = json.load(open(Path(metadata["cache"]) / "metadata.json"))["checkpoint_checksum"]
        checkpoints.add(checkpoint)
        samples = pd.read_csv(replay_dir / "samples.csv")
        samples["subset"] = subset
        tables.append(samples)
        reference = find_reference_run(output_root, metadata["manifest_checksum"], checkpoint_checksum=checkpoint)
        if reference is not None:
            identities[subset] = {
                "reference_run": str(reference), **identity_check(samples, pd.read_csv(reference / "samples.csv")),
            }
    if len(checkpoints) != 1:
        raise SystemExit(f"The replays come from different checkpoints: {sorted(checkpoints)}.")
    rows = pd.concat(tables, ignore_index=True)
    summary = summarize_replay(rows)
    gates = gate_table(summary)
    per_dataset = summary[summary["dataset"] != "__dataset_balanced__"].pivot(
        index="dataset", columns="variant", values="msa_mean",
    )
    relative = (per_dataset.sub(per_dataset["registry"], axis=0)).div(per_dataset["registry"], axis=0)
    out_dir = structural_root(output_root) / "reports" / next(iter(checkpoints)) / "+".join(subsets)
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(out_dir / "summary.csv", summary)
    _atomic_write_csv(out_dir / "gates.csv", gates)
    _atomic_write_csv(out_dir / "per_dataset_msa.csv", per_dataset.reset_index())
    _atomic_write_csv(out_dir / "per_dataset_relative.csv", relative.reset_index())
    _atomic_write_json(out_dir / "identity.json", identities)
    pd.set_option("display.width", 250)
    print("Identity of the registry replay against the canonical runs:")
    print(json.dumps(identities, indent=2))
    print(gates.round(4).to_string(index=False))
    print(f"Report: {out_dir}")
    return out_dir


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("cache", "oracle", "replay"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--subset", default="primary", choices=("primary", "training_extra", "holdout"))
        sub.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
        sub.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        if name == "cache":
            sub.add_argument("--device", default="cuda")
        else:
            sub.add_argument("--workers", type=int, default=8)
        if name == "replay":
            sub.add_argument("--variants", nargs="*", default=None)
    report = subparsers.add_parser("report")
    report.add_argument("--subsets", nargs="+", default=("primary", "training_extra"))
    report.add_argument("--replay-dirs", nargs="*", type=Path, default=None)
    report.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "report":
        stage_report(args.output_root, list(args.subsets), args.replay_dirs)
        return 0
    manifest_path = _default_manifest_path(args.output_root, "standard", args.subset)
    data_root, output_root, manifest_path = _validate_roots(args.data_root, args.output_root, manifest_path)
    manifest = prepare_manifest(data_root, manifest_path, "standard", subset=args.subset)
    manifest["subset"] = args.subset
    if args.command == "cache":
        stage_cache(manifest, data_root, output_root, args.device)
    elif args.command == "oracle":
        stage_oracle(manifest, data_root, output_root, args.workers)
    else:
        stage_replay(manifest, data_root, output_root, args.workers, args.variants)
    return 0


if __name__ == "__main__":
    sys.exit(main())
