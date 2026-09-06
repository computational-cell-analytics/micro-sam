"""Cache every candidate's anchor evidence and its propagated track for one 3d crop.

A volumetric sweep pays a full propagation per configuration, which is what made the stopped 3d
campaign cache its tracks. This extractor rebuilds that cache on the new manifests, richer and
policy-free: for the union of several density ladders it records each candidate's ladder metadata,
its three anchor alternatives' selector features, its anchor mask and score, and the point-conditioned
track the propagation produces for it. The historical decision (predicted IoU >= score_threshold,
then the in-plane merge) is *not* applied here; the replay reconstructs it per ladder from the cached
anchor masks, so one cache serves the control, every learned filter and every recall expansion.

Usage examples:
    python extract_apg_3d_tracks.py --subset primary --sample-index 3
    python extract_apg_3d_tracks.py --subset primary --sample-index 3 --ladders "[[1.5,10],[1,3,10]]"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

EVALUATION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVALUATION_ROOT))

import common  # noqa
from common import VOLUME_SPEED_OPTIONS, build_apg_segmenter, checkpoint_checksum, get_joint_checkpoint  # noqa
from optimization.benchmark_apg_optimization import (  # noqa
    DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, _atomic_write_json, _content_checksum, _git_revision,
    _hardware_identity, _implementation_checksum,
)
from optimization.apg3d_manifest import CAMPAIGN_ROOT, load_manifest, load_normalized_source, load_sample  # noqa

DEFAULT_LADDERS = ((1.5, 10.0), (1.0, 3.0, 10.0), (0.5, 2.0, 10.0))
SCHEMA = "token_lowres_v1"
CACHE_VERSION = "apg3d-tracks-v1"
N_OBJECTS_PER_PASS = 16
EARLY_STOP_PATIENCE = 2
MAX_OVERLAP = 0.15
# Every replayed policy applies the anchor-slice predicted-IoU filter first, so a candidate below it
# is never propagated by any of them; propagating it here would only cost time. The in-plane merge
# is not applied, because its outcome depends on which ladder's candidates are present.
PROPAGATED_MIN_ANCHOR_IOU = 0.6


def pack_masks(masks: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bit-pack a list of boolean arrays into one payload with offsets and shapes."""
    payload, offsets, shapes = [], [0], []
    for mask in masks:
        packed = np.packbits(np.asarray(mask, dtype=bool).ravel())
        payload.append(packed)
        offsets.append(offsets[-1] + len(packed))
        shapes.append(mask.shape)
    return (
        np.concatenate(payload) if payload else np.zeros(0, dtype="uint8"),
        np.asarray(offsets, dtype="int64"),
        np.asarray(shapes, dtype="int64").reshape(len(masks), -1),
    )


def unpack_mask(payload: np.ndarray, offsets: np.ndarray, shapes: np.ndarray, index: int) -> np.ndarray:
    shape = tuple(int(side) for side in shapes[index])
    packed = payload[offsets[index]:offsets[index + 1]]
    return np.unpackbits(packed)[:int(np.prod(shape))].reshape(shape).astype(bool)


def _anchor_key(frame: int, point: Sequence[float]) -> Tuple[int, int, int]:
    return int(frame), int(round(float(point[0]))), int(round(float(point[1])))


def union_prompts(per_ladder: List[Tuple[dict, dict]]) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Merge the ladders' prompts; each anchor voxel once, with the metadata of its first ladder.

    Returns the prompts, the (N, n_ladders) membership matrix, the (N, F) metadata features and the
    ladder index that supplied each candidate's metadata.
    """
    keys: Dict[Tuple[int, int, int], int] = {}
    points, frames, features, origin = [], [], [], []
    membership: List[List[bool]] = []
    for ladder_index, (prompts, metadata) in enumerate(per_ladder):
        if prompts is None:
            continue
        for index, (point, frame) in enumerate(zip(prompts["points"][:, 0], prompts["frames"])):
            key = _anchor_key(frame, point)
            if key not in keys:
                keys[key] = len(points)
                points.append(point)
                frames.append(int(frame))
                features.append(metadata["features"][index])
                origin.append(ladder_index)
                membership.append([False] * len(per_ladder))
            membership[keys[key]][ladder_index] = True
    prompts = {
        "points": np.asarray(points, dtype="float32").reshape(-1, 1, 2),
        "point_labels": np.ones((len(points), 1), dtype="int32"),
        "frames": np.asarray(frames, dtype="int64"),
    }
    return (
        prompts, np.asarray(membership, dtype=bool).reshape(len(points), len(per_ladder)),
        np.asarray(features, dtype="float32").reshape(len(points), -1), np.asarray(origin, dtype="int64"),
    )


def score_all_candidates(segmenter, prompts: dict, batch_size: int = 64) -> List[dict]:
    """Prompt every candidate on its anchor slice and keep all of them, with alternative features.

    Mirrors `_score_candidates` without its decision: no predicted-IoU threshold and no in-plane merge,
    so the replay can apply either per ladder from the cached anchor masks.
    """
    from micro_sam.v2.instance_segmentation import _set_image_predictor_from_3d_embeddings

    points, labels, frames = prompts["points"], prompts["point_labels"], prompts["frames"]
    candidates = []
    predictor = segmenter._predictor
    for frame in np.unique(frames):
        indices = np.where(frames == frame)[0]
        _set_image_predictor_from_3d_embeddings(predictor, segmenter._image_embeddings, int(frame))
        frame_prompts = {"points": points[indices], "point_labels": labels[indices]}
        records = segmenter._apply_prompts(predictor, frame_prompts, multimasking=True, batch_size=batch_size)
        features = segmenter._anchor_alternative_features(predictor, frame_prompts, int(frame), SCHEMA, batch_size)
        for record in records:
            candidate = segmenter._anchor_candidate(int(frame), record)
            local = int(record["prompt_index"])
            candidate["prompt_index"] = int(indices[local])
            candidate.update(features[local])
            candidates.append(candidate)
    candidates.sort(key=lambda candidate: candidate["prompt_index"])
    return candidates


def track_targets(records: List[dict], labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Best-matching ground-truth object and IoU of every propagated track."""
    sizes = np.bincount(labels.ravel())
    ious = np.zeros(len(records), dtype="float32")
    gt_ids = np.zeros(len(records), dtype="int64")
    for index, record in enumerate(records):
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
        ious[index] = intersection / (area + int(sizes[best]) - intersection)
        gt_ids[index] = best
    return ious, gt_ids


@torch.no_grad()
def extract_crop(
    segmenter, sample: Dict[str, Any], raw: np.ndarray, labels: np.ndarray, ladders: Sequence[Sequence[float]],
    out_dir: Path, device: str,
) -> Dict[str, Any]:
    from micro_sam.v2.automatic_prompt_generation import VOLUME_CANDIDATE_FEATURE_NAMES, derive_volume_prompts

    spacing = tuple(sample["spacing"]) if sample.get("spacing") and tuple(sample["spacing"]) != (1, 1, 1) else None
    timings = {}
    started = time.perf_counter()
    segmenter.clear_state()
    segmenter.initialize(raw, ndim=3, **VOLUME_SPEED_OPTIONS)
    timings["initialize"] = time.perf_counter() - started
    prediction = segmenter._prediction

    step = time.perf_counter()
    per_ladder = []
    for ladder in ladders:
        result = derive_volume_prompts(
            prediction[0], prediction[1:], model_type=segmenter._model_type, candidate_threshold=tuple(ladder),
            spacing=spacing, return_metadata=True,
        )
        per_ladder.append(result if result != (None, None) else (None, None))
    prompts, membership, component_features, origin = union_prompts(per_ladder)
    timings["derive"] = time.perf_counter() - step
    n_candidates = len(prompts["points"])

    step = time.perf_counter()
    segmenter._last_generation_stats = {}
    candidates = score_all_candidates(segmenter, prompts) if n_candidates else []
    timings["score"] = time.perf_counter() - step

    step = time.perf_counter()
    records = []
    propagated = [candidate for candidate in candidates if candidate["score"] >= PROPAGATED_MIN_ANCHOR_IOU]
    if propagated:
        records = segmenter._propagate_candidates(
            propagated, n_objects_per_pass=N_OBJECTS_PER_PASS, early_stop_patience=EARLY_STOP_PATIENCE,
            verbose=False, max_overlap=MAX_OVERLAP, propagation_waves=1,
        )
    timings["propagate"] = time.perf_counter() - step
    stats = dict(segmenter._last_generation_stats)

    # Candidates: one row per scored prompt (prompts with an empty anchor mask have no row).
    n_features = 0
    for candidate in candidates:
        n_features = candidate["alternative_features"].shape[1]
        break
    anchor_payload, anchor_offsets, anchor_shapes = pack_masks([candidate["mask"] for candidate in candidates])
    np.savez_compressed(
        out_dir / "candidates.npz",
        prompt_index=np.asarray([c["prompt_index"] for c in candidates], dtype="int64"),
        frame=np.asarray([c["frame"] for c in candidates], dtype="int64"),
        point_xy=np.asarray([c["point"] for c in candidates], dtype="float32").reshape(-1, 2),
        anchor_predicted_iou=np.asarray([c["score"] for c in candidates], dtype="float32"),
        anchor_stability=np.asarray([c["stability"] for c in candidates], dtype="float32"),
        alternative_features=np.asarray(
            [c["alternative_features"] for c in candidates], dtype="float32",
        ).reshape(len(candidates), 3, n_features),
        alternative_scores=np.asarray([c["alternative_scores"] for c in candidates], dtype="float32").reshape(-1, 3),
        alternative_stability=np.asarray(
            [c["alternative_stability"] for c in candidates], dtype="float32",
        ).reshape(-1, 3),
        anchor_mask_payload=anchor_payload, anchor_mask_offsets=anchor_offsets, anchor_mask_shapes=anchor_shapes,
        anchor_box_start=np.asarray([[c["mask_box"][0].start, c["mask_box"][1].start] for c in candidates],
                                    dtype="int64").reshape(-1, 2),
        # Per prompt (indexed by prompt_index): ladder membership and component features.
        prompt_frame=prompts["frames"], prompt_point_xy=prompts["points"][:, 0],
        ladder_membership=membership, component_features=component_features, component_origin_ladder=origin,
        component_feature_names=np.asarray(VOLUME_CANDIDATE_FEATURE_NAMES),
        ladders=np.asarray([json.dumps(list(ladder)) for ladder in ladders]),
        feature_schema=np.asarray(SCHEMA),
    )

    # Tracks: one row per propagated record, linked to its candidate by prompt index.
    ious, gt_ids = track_targets(records, labels)
    payload, offsets, shapes = pack_masks([record["segmentation"] for record in records])
    np.savez_compressed(
        out_dir / "tracks.npz",
        prompt_index=np.asarray([record["prompt_index"] for record in records], dtype="int64"),
        box_start=np.asarray([[axis.start for axis in record["bounding_box"]] for record in records],
                             dtype="int64").reshape(-1, 3),
        box_stop=np.asarray([[axis.stop for axis in record["bounding_box"]] for record in records],
                            dtype="int64").reshape(-1, 3),
        mask_payload=payload, mask_offsets=offsets, mask_shapes=shapes,
        track_iou=ious, track_gt_id=gt_ids,
        volume_shape=np.asarray(labels.shape, dtype="int64"),
    )
    gt_sizes = np.bincount(labels.ravel())
    np.savez_compressed(
        out_dir / "labels.npz", gt_ids=np.arange(1, len(gt_sizes), dtype="int64")[gt_sizes[1:] > 0],
        gt_sizes=gt_sizes[1:][gt_sizes[1:] > 0],
    )
    return {
        "sample_id": sample["sample_id"], "dataset": sample["dataset"], "family": sample["family"],
        "n_prompts": int(n_candidates), "n_candidates": len(candidates), "n_tracks": len(records),
        "n_propagated": len(propagated), "propagated_min_anchor_iou": PROPAGATED_MIN_ANCHOR_IOU,
        "per_ladder_prompts": [0 if p is None else int(len(p["points"])) for p, _ in per_ladder],
        "stats": stats, "timings": timings, "total_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else None,
        "volume_shape": list(labels.shape),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--ladders", type=json.loads, default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--campaign-root", type=Path, default=CAMPAIGN_ROOT)
    parser.add_argument("--model-type", default="hvit_t")
    parser.add_argument("--joint-checkpoint", default="best")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.subset, args.campaign_root, args.data_root)
    ladders = tuple(tuple(float(v) for v in ladder) for ladder in (args.ladders or DEFAULT_LADDERS))
    samples = manifest["samples"]
    if args.sample_index is not None:
        samples = [samples[args.sample_index]]
    elif args.sample_id is not None:
        samples = [sample for sample in samples if sample["sample_id"] == args.sample_id]
    else:
        raise SystemExit("Pass --sample-index or --sample-id.")
    identity = {
        "cache_version": CACHE_VERSION, "ladders": [list(ladder) for ladder in ladders], "schema": SCHEMA,
        "manifest_checksum": manifest["manifest_checksum"], "implementation_checksum": _implementation_checksum(),
        "n_objects_per_pass": N_OBJECTS_PER_PASS, "early_stop_patience": EARLY_STOP_PATIENCE,
        "max_overlap": MAX_OVERLAP,
    }
    cache_root = args.campaign_root / "cache" / args.subset / _content_checksum(identity)[:12]
    checkpoint_id = checkpoint_checksum(get_joint_checkpoint(args.model_type, args.joint_checkpoint))
    identity["checkpoint_checksum"] = checkpoint_id
    cache_root.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(cache_root / "identity.json", identity)
    segmenter = None
    cache: Dict[tuple, np.ndarray] = {}
    for sample in samples:
        out_dir = cache_root / sample["sample_id"].replace(":", "_")
        if (out_dir / "complete.json").exists() and not args.force:
            print(f"{sample['sample_id']} is cached.")
            continue
        if segmenter is None:
            segmenter = build_apg_segmenter(
                args.model_type, 3, args.device, joint_checkpoint=args.joint_checkpoint, joint_checksum=checkpoint_id,
                export_root=str(DEFAULT_OUTPUT_ROOT / "model_exports"),
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        key = (sample["raw_path"], tuple(sample["normalization_z_range"]))
        if key not in cache:
            cache.clear()
            cache[key] = load_normalized_source(sample, args.data_root)
        raw, labels, valid = load_sample(sample, args.data_root, cache[key])
        if valid is not None:
            labels = labels.copy()
            labels[~valid] = 0
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        summary = extract_crop(segmenter, sample, raw, labels, ladders, out_dir, args.device)
        summary.update({"identity": identity, "git_revision": _git_revision(),
                        "hardware": _hardware_identity(args.device)})
        _atomic_write_json(out_dir / "complete.json", summary)
        print(f"{sample['sample_id']:36s} prompts {summary['n_prompts']} candidates {summary['n_candidates']} "
              f"tracks {summary['n_tracks']} passes {summary['stats'].get('propagation_passes')} "
              f"{summary['total_seconds']:.1f} s")
    if segmenter is not None:
        segmenter.clear_state()
    print(f"Cache: {cache_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
