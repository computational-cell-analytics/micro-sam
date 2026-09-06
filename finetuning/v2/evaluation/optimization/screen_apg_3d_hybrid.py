"""Screen the slice-wise hybrid: 2d APG with the learned selector on every slice, linked across z.

The 2d APG is now far ahead of its predicted-IoU baseline because a learned score both selects the
mask alternative and filters the candidates, and in 2d that selected mask *is* the output. In the
volumetric pipeline the selected anchor mask only gates the propagation, which restarts from the
point, so the same learning never reached the output. This screen makes the 2d decision the output
again: every slice is segmented by the 2d APG on the volume's own per-slice embeddings (no
re-encoding), and the slices are linked into objects by overlap - with a multicut or greedy matching.
There is no propagation at all, which makes it a candidate efficiency mode as well.

A second variant feeds the linked chains back into the propagation as candidates: each chain's best
slice (by learned score) becomes a prompt, by point or by mask conditioning, so recall from
slice-wise density maxima reaches the propagation under a pass budget.

Usage examples:
    python screen_apg_3d_hybrid.py run --subset primary --variant hybrid-2d --linker multicut --beta 0.5 \\
        --sample-index 3 --selector-artifact <token_lowres_v1 selector .pt>
    python screen_apg_3d_hybrid.py aggregate --subset primary --variant hybrid-2d --linker multicut --beta 0.5 \\
        --selector-artifact <selector .pt>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

EVALUATION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVALUATION_ROOT))

import common  # noqa
from common import VOLUME_SPEED_OPTIONS, build_apg_segmenter, checkpoint_checksum, get_joint_checkpoint  # noqa
from common import genuine_misses  # noqa
from parameter_search import compute_metrics  # noqa
from optimization.benchmark_apg_optimization import (  # noqa
    DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, _atomic_write_csv, _atomic_write_json, _content_checksum,
    _hardware_identity, _implementation_checksum,
)
from optimization.apg3d_manifest import CAMPAIGN_ROOT, load_manifest, load_normalized_source, load_sample  # noqa
from optimization.benchmark_apg_3d import (  # noqa
    STATS_KEYS, attribute_recall, summarize, DEFAULT_LADDERS, load_volume_config,
)

VARIANTS = ("hybrid-2d", "hybrid-3dpred", "candidates-point", "candidates-mask", "union-point")
LINKERS = ("multicut", "greedy")
ENCODINGS = ("embeddings", "standalone")
SCORINGS = ("selector", "plain")
# The pinned campaign defaults with SAM2's own predicted-IoU scoring, the learned selector's control.
PLAIN_2D = {
    "candidate_threshold": 1.5, "dt": 0.25, "sigma": 0.5, "min_candidate_size": 4, "foreground_threshold": 0.7,
    "max_overlap": 0.15, "min_size": 50, "multimasking": True, "multimask_scorer": "predicted_iou",
    "multimask_selection": "eager", "score_filter": "predicted_iou", "score_threshold": 0.6,
}
# The accepted 2d configuration, pinned to the parameters the accepted runs used.
ACCEPTED_2D = {
    "candidate_threshold": 1.5, "dt": 0.25, "sigma": 0.5, "min_candidate_size": 4, "foreground_threshold": 0.7,
    "max_overlap": 0.15, "min_size": 50, "multimasking": True, "multimask_scorer": "microscopy",
    "multimask_selection": "eager", "score_filter": "selection_score", "score_threshold": 0.375,
}


# ----------------------------------------------------------------------------------------------
# linking


def relabel_stack(stack: np.ndarray) -> Tuple[np.ndarray, List[Dict[int, int]]]:
    """Make the slice labels unique across z; return the stack and, per slice, {new id: old id}."""
    out = np.zeros_like(stack, dtype="uint32")
    offset = 0
    maps = []
    for z in range(stack.shape[0]):
        ids = np.unique(stack[z])
        ids = ids[ids != 0]
        lookup = np.zeros(int(stack[z].max()) + 1, dtype="uint32")
        lookup[ids] = np.arange(offset + 1, offset + 1 + len(ids), dtype="uint32")
        out[z] = lookup[stack[z]]
        maps.append({int(offset + 1 + index): int(old) for index, old in enumerate(ids)})
        offset += len(ids)
    return out, maps


def _overlap_matrix(first: np.ndarray, second: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IoU between the labels of two consecutive slices; returns (ids_a, ids_b, iou[a, b])."""
    ids_a = np.unique(first)
    ids_a = ids_a[ids_a != 0]
    ids_b = np.unique(second)
    ids_b = ids_b[ids_b != 0]
    if len(ids_a) == 0 or len(ids_b) == 0:
        return ids_a, ids_b, np.zeros((len(ids_a), len(ids_b)), dtype="float64")
    index_a = np.zeros(int(first.max()) + 1, dtype="int64")
    index_a[ids_a] = np.arange(len(ids_a))
    index_b = np.zeros(int(second.max()) + 1, dtype="int64")
    index_b[ids_b] = np.arange(len(ids_b))
    both = (first != 0) & (second != 0)
    pair = index_a[first[both]] * len(ids_b) + index_b[second[both]]
    intersection = np.bincount(pair, minlength=len(ids_a) * len(ids_b)).reshape(len(ids_a), len(ids_b))
    size_a = np.bincount(first.ravel(), minlength=int(first.max()) + 1)[ids_a]
    size_b = np.bincount(second.ravel(), minlength=int(second.max()) + 1)[ids_b]
    union = size_a[:, None] + size_b[None, :] - intersection
    return ids_a, ids_b, intersection / np.maximum(union, 1)


def link_greedy(stack: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """Chain slice instances by one-to-one IoU matching between consecutive slices."""
    parent = {}
    for z in range(stack.shape[0] - 1):
        ids_a, ids_b, iou = _overlap_matrix(stack[z], stack[z + 1])
        if iou.size == 0:
            continue
        rows, cols = linear_sum_assignment(-iou)
        for row, col in zip(rows, cols):
            if iou[row, col] >= iou_threshold:
                parent[int(ids_b[col])] = int(ids_a[row])
    roots = {}

    def root(node):
        while node in parent:
            node = parent[node]
        return node

    lookup = np.zeros(int(stack.max()) + 1, dtype="uint32")
    next_id = 1
    for node in np.unique(stack):
        if node == 0:
            continue
        key = root(int(node))
        if key not in roots:
            roots[key] = next_id
            next_id += 1
        lookup[node] = roots[key]
    return lookup[stack]


def link_multicut(stack: np.ndarray, beta: float = 0.5) -> np.ndarray:
    """The v1 `merge_instance_segmentation_3d` recipe: overlap edges, cost transform, multicut."""
    from bioimage_cpp.graph import UndirectedGraph
    from bioimage_py.segmentation import multicut as mc
    from elf.tracking.tracking_utils import compute_edges_from_overlap

    edges = compute_edges_from_overlap(stack, verbose=False)
    if not edges:
        return stack
    uv_ids = np.array([[edge["source"], edge["target"]] for edge in edges], dtype="uint64")
    overlaps = np.array([edge["score"] for edge in edges], dtype="float64")
    n_nodes = int(stack.max() + 1)
    graph = UndirectedGraph(n_nodes)
    graph.insert_edges(uv_ids)
    # The overlap is a merge affinity; the cost transform expects a boundary (cut) probability, so it
    # gets its complement. Positive costs attract, and 'beta' shifts the prior towards merging (< 0.5)
    # or splitting (> 0.5). Edges to the background are maximally repulsive.
    costs = mc.compute_edge_costs(np.clip(1.0 - overlaps, 1e-6, 1 - 1e-6), beta=beta)
    costs[(uv_ids == 0).any(axis=1)] = -8.0
    node_labels = mc.multicut_decomposition(graph, costs)
    node_labels = np.asarray(node_labels)
    node_labels[0] = 0
    return node_labels[stack].astype("uint32")


def filter_z_extent(segmentation: np.ndarray, min_z_extent: int) -> np.ndarray:
    if min_z_extent <= 1:
        return segmentation
    present = [np.unique(segmentation[z]) for z in range(segmentation.shape[0])]
    counts = {}
    for ids in present:
        for value in ids:
            if value:
                counts[int(value)] = counts.get(int(value), 0) + 1
    drop = [value for value, count in counts.items() if count < min_z_extent]
    if drop:
        segmentation[np.isin(segmentation, drop)] = 0
    return segmentation


def link_slices(stack: np.ndarray, linker: str, beta: float, iou_threshold: float, min_z_extent: int) -> np.ndarray:
    unique, _ = relabel_stack(stack)
    linked = link_multicut(unique, beta) if linker == "multicut" else link_greedy(unique, iou_threshold)
    linked = filter_z_extent(linked, min_z_extent)
    # Consecutive ids.
    ids = np.unique(linked)
    lookup = np.zeros(int(linked.max()) + 1, dtype="uint32")
    lookup[ids] = np.arange(len(ids), dtype="uint32")
    return lookup[linked]


# ----------------------------------------------------------------------------------------------
# per-slice 2d APG on the volume's embeddings


def build_hybrid_2d(segmenter3d, selector_path: Path, device: str):
    from micro_sam.v2.automatic_prompt_generation import AutomaticPromptGenerator
    from micro_sam.v2.multimask_selection import load_feature_scorer
    from micro_sam.v2.util import get_sam2_image_predictor

    predictor = get_sam2_image_predictor(segmenter3d._video_predictor)
    hybrid = AutomaticPromptGenerator(segmenter3d._model, predictor, device=device)
    hybrid.set_multimask_models(scorer=load_feature_scorer(selector_path, device=device))
    return hybrid


def segment_slices(segmenter3d, hybrid, raw: np.ndarray, params_2d: Dict[str, Any], use_3d_prediction: bool,
                   encoding: str = "embeddings"):
    """Run the 2d APG on every slice; return the label stack and the per-slice instance records.

    'encoding' decides where the slice features come from: the volume's own per-slice embeddings (no
    re-encoding, the video model's preprocessing) or a fresh 2d encode of the slice (the image path
    the 2d selector was fitted on).
    """
    depth = raw.shape[0]
    stack = np.zeros(raw.shape, dtype="uint32")
    instances: List[Dict[str, Any]] = []
    propose_keys = ("candidate_threshold", "foreground_threshold", "n_iter", "dt", "sigma", "min_candidate_size",
                    "multimasking", "multimask_scorer", "multimask_selection", "batch_size", "n_threads")
    propose_kwargs = {key: params_2d[key] for key in propose_keys if key in params_2d}
    for z in range(depth):
        hybrid.clear_state()
        if encoding == "standalone":
            hybrid.initialize(raw[z], ndim=2)
        else:
            hybrid.initialize(raw[z], ndim=2, image_embeddings=segmenter3d._image_embeddings, i=z)
        if use_3d_prediction:
            hybrid._prediction = np.ascontiguousarray(segmenter3d._prediction[:, z])
        proposals = hybrid.propose(**propose_kwargs)
        segmentation, context = hybrid._merge(
            proposals, raw.shape[1:], score_threshold=params_2d["score_threshold"],
            max_overlap=params_2d["max_overlap"], min_size=params_2d["min_size"], return_context=True,
            score_filter=params_2d["score_filter"],
        )
        stack[z] = segmentation
        if context is not None:
            for instance_id, record_index in context["matches"].items():
                record = context["records"][record_index]
                instances.append({
                    "z": z, "instance_id": int(instance_id), "selection_score": float(record["selection_score"]),
                    "predicted_iou": float(record["predicted_iou"]), "point": tuple(float(v) for v in record["point"]),
                })
    return stack, instances


def slice_cache_dir(campaign_root: Path, subset: str, args: argparse.Namespace) -> Path:
    """Where a crop's per-slice 2d result is kept, so every linker replays it without the GPU."""
    identity = _content_checksum({
        "encoding": args.encoding, "scoring": args.scoring, "params_2d": params_2d_for(args),
        "variant_pred": args.variant == "hybrid-3dpred", "selector": Path(args.selector_artifact).name,
        "implementation": _implementation_checksum(),
    })
    return campaign_root / "hybrid" / "slices" / subset / f"{args.encoding}-{args.scoring}-{identity[:12]}"


def load_slice_cache(path: Path):
    data = np.load(path.with_suffix(".npz"), allow_pickle=False)
    instances = json.loads(str(data["instances"]))
    for entry in instances:
        entry["point"] = tuple(entry["point"])
    return data["stack"], instances, dict(zip(data["timing_keys"].tolist(), data["timing_values"].tolist()))


def save_slice_cache(path: Path, stack: np.ndarray, instances: List[Dict[str, Any]], timings: Dict[str, float]):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path.with_suffix(".npz"), stack=stack.astype("uint32"), instances=np.asarray(json.dumps(instances)),
        timing_keys=np.asarray(list(timings)), timing_values=np.asarray(list(timings.values()), dtype="float64"),
    )


def chains_to_prompts(linked: np.ndarray, stack: np.ndarray, instances: List[Dict[str, Any]], with_masks: bool):
    """One prompt per linked chain, anchored on its slice of highest learned score."""
    by_slice_instance = {(entry["z"], entry["instance_id"]): entry for entry in instances}
    best: Dict[int, Tuple[float, Dict[str, Any]]] = {}
    for z in range(linked.shape[0]):
        ids_linked = linked[z]
        ids_slice = stack[z]
        both = (ids_linked != 0) & (ids_slice != 0)
        pairs = np.unique(np.stack([ids_linked[both], ids_slice[both]], axis=1), axis=0) if both.any() else []
        for chain_id, slice_id in pairs:
            entry = by_slice_instance.get((z, int(slice_id)))
            if entry is None:
                continue
            score = entry["selection_score"]
            if int(chain_id) not in best or score > best[int(chain_id)][0]:
                best[int(chain_id)] = (score, {**entry, "slice_id": int(slice_id)})
    points, frames, conditioning = [], [], []
    for chain_id, (_, entry) in sorted(best.items()):
        points.append(entry["point"])
        frames.append(entry["z"])
        if with_masks:
            conditioning.append({"mask": stack[entry["z"]] == entry["slice_id"]})
    prompts = {
        "points": np.array(points, dtype="float32").reshape(-1, 1, 2),
        "point_labels": np.ones((len(points), 1), dtype="int32"),
        "frames": np.array(frames, dtype="int64"),
    }
    if with_masks:
        prompts["conditioning"] = conditioning
    return prompts


def union_prompts(density_prompts: Optional[dict], hybrid_prompts: dict, stack: np.ndarray) -> dict:
    """Density candidates plus the hybrid ones whose anchor no density candidate already covers."""
    if density_prompts is None:
        return hybrid_prompts
    covered = set()
    for point, frame in zip(density_prompts["points"][:, 0], density_prompts["frames"]):
        x, y = int(point[0]), int(point[1])
        covered.add((int(frame), int(stack[int(frame), y, x])))
    keep = []
    for index, (point, frame) in enumerate(zip(hybrid_prompts["points"][:, 0], hybrid_prompts["frames"])):
        x, y = int(point[0]), int(point[1])
        slice_id = int(stack[int(frame), y, x])
        if slice_id == 0 or (int(frame), slice_id) not in covered:
            keep.append(index)
    return {
        "points": np.concatenate([density_prompts["points"], hybrid_prompts["points"][keep]]),
        "point_labels": np.concatenate([density_prompts["point_labels"], hybrid_prompts["point_labels"][keep]]),
        "frames": np.concatenate([density_prompts["frames"], hybrid_prompts["frames"][keep]]),
    }


# ----------------------------------------------------------------------------------------------
# running


def params_2d_for(args: argparse.Namespace) -> Dict[str, Any]:
    return dict(ACCEPTED_2D if args.scoring == "selector" else PLAIN_2D)


def config_identity(args: argparse.Namespace, params_3d: Dict[str, Any]) -> str:
    identity = {
        "variant": args.variant, "linker": args.linker, "beta": args.beta, "iou_threshold": args.iou_threshold,
        "min_z_extent": args.min_z_extent, "budget_factor": args.budget_factor, "params_3d": params_3d,
        "selector": Path(args.selector_artifact).name, "params_2d": params_2d_for(args), "encoding": args.encoding,
        "scoring": args.scoring,
    }
    tag = f"{args.variant}-{args.encoding}-{args.scoring}-{args.linker}"
    return f"{tag}-{_content_checksum(identity)[:12]}-{_implementation_checksum()[:12]}"


def run_crop(segmenter3d, hybrid, sample, raw, labels, valid, args, params_3d, device,
             slice_cache: Optional[Path] = None) -> Dict[str, Any]:
    from micro_sam.v2.automatic_prompt_generation import derive_volume_prompts

    if segmenter3d is not None:
        segmenter3d.clear_state()
    cuda_device = torch.device(device) if device.startswith("cuda") and torch.cuda.is_available() else None
    if cuda_device is not None:
        torch.cuda.reset_peak_memory_stats(cuda_device)
    spacing = tuple(sample["spacing"]) if sample.get("spacing") and tuple(sample["spacing"]) != (1, 1, 1) else None
    started = time.perf_counter()
    cached = slice_cache is not None and slice_cache.with_suffix(".npz").exists()
    hybrid_only = args.variant in ("hybrid-2d", "hybrid-3dpred")
    if cached and hybrid_only:
        # The linking is the only thing that varies; the 2d pass is replayed from the cache, GPU-free.
        stack, instances, timings = load_slice_cache(slice_cache)
        initialized = started + timings["initialize"]
        sliced = initialized + timings["slices"]
    else:
        segmenter3d.initialize(raw, ndim=3, **VOLUME_SPEED_OPTIONS)
        initialized = time.perf_counter()
        stack, instances = segment_slices(
            segmenter3d, hybrid, raw, params_2d_for(args), args.variant == "hybrid-3dpred", encoding=args.encoding,
        )
        sliced = time.perf_counter()
        if slice_cache is not None:
            save_slice_cache(slice_cache, stack, instances,
                             {"initialize": initialized - started, "slices": sliced - initialized})
    linked = link_slices(stack, args.linker, args.beta, args.iou_threshold, args.min_z_extent)
    linked_at = time.perf_counter()
    row = {
        "sample_id": sample["sample_id"], "dataset": sample["dataset"], "family": sample["family"],
        "seen_in_training": str(sample["seen_in_training"]), "depth_flag": sample["depth_flag"],
        "realized_depth": int(labels.shape[0]), "legacy_sample_id": sample.get("legacy_sample_id"),
        "slice_instances": len(instances), "chains": int(len(np.unique(linked)) - 1),
        "initialization_seconds": initialized - started, "slice_seconds": sliced - initialized,
        "link_seconds": linked_at - sliced, "slices_from_cache": bool(cached and hybrid_only),
    }
    trace = None
    if args.variant in ("hybrid-2d", "hybrid-3dpred"):
        segmentation = linked
        generation_seconds = linked_at - initialized
        row.update({key: 0 for key in STATS_KEYS})
    else:
        prompts = chains_to_prompts(linked, stack, instances, with_masks=args.variant == "candidates-mask")
        if args.variant == "union-point":
            density = derive_volume_prompts(
                segmenter3d._prediction[0], segmenter3d._prediction[1:], model_type=segmenter3d._model_type,
                spacing=spacing,
            )
            prompts = union_prompts(density, prompts, stack)
        budget = None
        if args.budget_factor is not None:
            reference = derive_volume_prompts(
                segmenter3d._prediction[0], segmenter3d._prediction[1:], model_type=segmenter3d._model_type,
                spacing=spacing,
            )
            budget = int(np.ceil(args.budget_factor * (0 if reference is None else len(reference["points"]))))
        excluded = ("candidate_budget", "candidate_order", "candidate_scorer_threshold")
        generate_params = {k: v for k, v in params_3d.items() if k not in excluded}
        segmentation = segmenter3d.generate(
            **generate_params, spacing=spacing, prompts=prompts, candidate_budget=budget, keep_trace=True,
        ).astype("uint32")
        generation_seconds = time.perf_counter() - initialized
        trace = segmenter3d._last_generation_trace
        row["hybrid_prompts"] = int(len(prompts["points"]))
        stats = getattr(segmenter3d, "_last_generation_stats", {}) or {}
        row.update({key: stats.get(key, 0) for key in STATS_KEYS})
    if valid is not None:
        segmentation[~valid] = 0
    if cached and hybrid_only:
        generation_seconds = (sliced - initialized) + (linked_at - sliced)
    row.update({
        "generation_seconds": generation_seconds,
        "total_seconds": (initialized - started) + generation_seconds if cached and hybrid_only
        else time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(cuda_device)) if cuda_device else None,
        "predicted_objects": int(len(np.unique(segmentation)) - 1),
        **compute_metrics(segmentation, labels, sample["metric_mode"], border_min_size=0),
    })
    if segmenter3d is not None and segmenter3d._prediction is not None:
        row.update(attribute_recall(segmenter3d, labels, segmentation, trace, DEFAULT_LADDERS, spacing))
        segmenter3d._last_generation_trace = None
    else:
        gt_ids = set(int(v) for v in np.unique(labels) if v != 0)
        row["gt_objects"] = len(gt_ids)
        row["unmatched"], row["genuine_misses"] = genuine_misses(labels, segmentation)
        row["merged"] = len(gt_ids) - int(row["unmatched"])
    return row


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "aggregate"))
    parser.add_argument("--subset", required=True)
    parser.add_argument("--variant", choices=VARIANTS, default="hybrid-2d")
    parser.add_argument("--linker", choices=LINKERS, default="multicut")
    parser.add_argument("--encoding", choices=ENCODINGS, default="standalone")
    parser.add_argument("--scoring", choices=SCORINGS, default="selector")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--min-z-extent", type=int, default=1)
    parser.add_argument("--budget-factor", type=float, default=None,
                        help="Candidate budget as a multiple of the density ladder's candidate count.")
    parser.add_argument("--config", type=Path, default=None, help="3d parameters for the propagation variants.")
    parser.add_argument("--selector-artifact", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--campaign-root", type=Path, default=CAMPAIGN_ROOT)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--model-type", default="hvit_t")
    parser.add_argument("--joint-checkpoint", default="best")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.subset, args.campaign_root, args.data_root)
    _, params_3d = load_volume_config(args.config, args.model_type)
    run_path = args.campaign_root / "hybrid" / args.subset / config_identity(args, params_3d)
    if args.command == "aggregate":
        from optimization.benchmark_apg_3d import sibling_run_dirs
        by_sample = {}
        for sibling in sibling_run_dirs(run_path):
            for path in sorted((sibling / "crops").glob("*.json")) if (sibling / "crops").exists() else []:
                row = json.load(open(path))
                row["implementation_checksum"] = sibling.name.rsplit("-", 1)[1]
                if row["sample_id"] not in by_sample or sibling == run_path:
                    by_sample[row["sample_id"]] = row
        rows = list(by_sample.values())
        if not rows:
            raise SystemExit(f"No crops in {run_path} or its siblings.")
        samples = pd.DataFrame(rows)
        _atomic_write_csv(run_path / "samples.csv", samples)
        summary = summarize(samples)
        _atomic_write_csv(run_path / "summary.csv", summary)
        expected = {sample["sample_id"] for sample in manifest["samples"]}
        metadata = json.load(open(run_path / "metadata.json")) if (run_path / "metadata.json").exists() else {}
        metadata.update({"status": "complete" if {r["sample_id"] for r in rows} == expected else "partial",
                         "n_crops": len(rows), "n_expected": len(expected)})
        _atomic_write_json(run_path / "metadata.json", metadata)
        columns = ["dataset", "n_crops", "msa_mean", "msa_ci_low", "msa_ci_high", "gt_objects", "merged",
                   "genuine_misses", "total_seconds"]
        print(summary[[c for c in columns if c in summary]].to_string(index=False))
        print(f"{metadata['status']}: {run_path}")
        return 0

    samples = manifest["samples"]
    if args.sample_index is not None:
        samples = [samples[args.sample_index]]
    elif args.sample_id is not None:
        samples = [sample for sample in samples if sample["sample_id"] == args.sample_id]
    elif not args.serial:
        raise SystemExit("Pass --sample-index, --sample-id or --serial.")
    pending = [
        s for s in samples
        if args.force or not (run_path / "crops" / f"{s['sample_id'].replace(':', '_')}.json").exists()
    ]
    if not pending:
        print(f"All {len(samples)} crop(s) already done in {run_path}.")
        return 0
    checkpoint_id = checkpoint_checksum(get_joint_checkpoint(args.model_type, args.joint_checkpoint))
    cache_dir = slice_cache_dir(args.campaign_root, args.subset, args)
    hybrid_only = args.variant in ("hybrid-2d", "hybrid-3dpred")
    needs_gpu = not hybrid_only or any(
        not (cache_dir / s["sample_id"].replace(":", "_")).with_suffix(".npz").exists() for s in pending
    )
    segmenter3d = hybrid = None
    if needs_gpu:
        segmenter3d = build_apg_segmenter(
            args.model_type, 3, args.device, joint_checkpoint=args.joint_checkpoint, joint_checksum=checkpoint_id,
            export_root=str(DEFAULT_OUTPUT_ROOT / "model_exports"),
        )
        hybrid = build_hybrid_2d(segmenter3d, args.selector_artifact, args.device)
    (run_path / "crops").mkdir(parents=True, exist_ok=True)
    if not (run_path / "metadata.json").exists():
        _atomic_write_json(run_path / "metadata.json", {
            "campaign": "apg3d-hybrid", "status": "running", "variant": args.variant, "linker": args.linker,
            "beta": args.beta, "iou_threshold": args.iou_threshold, "min_z_extent": args.min_z_extent,
            "budget_factor": args.budget_factor, "params_2d": params_2d_for(args), "params_3d": params_3d,
            "encoding": args.encoding, "scoring": args.scoring,
            "selector_artifact": str(Path(args.selector_artifact).resolve()),
            "manifest_checksum": manifest["manifest_checksum"], "subset": args.subset,
            "datasets": sorted({s["dataset"] for s in manifest["samples"]}),
            "implementation_checksum": _implementation_checksum(), "checkpoint_checksum": checkpoint_id,
            "model_type": args.model_type, "device": args.device, "hardware": _hardware_identity(args.device),
        })
    cache: Dict[tuple, np.ndarray] = {}
    for sample in pending:
        key = (sample["raw_path"], tuple(sample["normalization_z_range"]))
        if key not in cache:
            cache.clear()
            cache[key] = load_normalized_source(sample, args.data_root)
        raw, labels, valid = load_sample(sample, args.data_root, cache[key])
        row = run_crop(segmenter3d, hybrid, sample, raw, labels, valid, args, params_3d, args.device,
                       slice_cache=cache_dir / sample["sample_id"].replace(":", "_"))
        row["hardware"] = _hardware_identity(args.device).get("accelerator")
        _atomic_write_json(run_path / "crops" / f"{sample['sample_id'].replace(':', '_')}.json", row)
        print(f"{sample['sample_id']:36s} msa={row.get('msa', float('nan')):.4f} objects {row['gt_objects']}/"
              f"{row['predicted_objects']} chains {row['chains']} {row['total_seconds']:.1f} s")
    print(f"Run directory: {run_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
