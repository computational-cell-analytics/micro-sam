"""Benchmark multi-anchor prompt replay against isolated SAM2 predictor states.

This benchmark targets the prompt-state contract of ``PromptableSegmentation3D``. It uses one
deterministic crop from the standard APG manifest and the four largest objects that do not touch a
z border and have distinct anchor slices. Embeddings are cached outside the repository.

Two protocols exercise the failure modes that prompted this benchmark:

* ``joint`` sends a box, one positive point and up to four negative points in one decoder call.
* ``replacement`` sends an initial box/point set, replaces it with the joint set, then appends the
  cleared initial point again.

The production path puts all objects in one ``PromptableSegmentation3D`` state and therefore uses
its multi-anchor replay. The oracle gives each anchor its own state and sends the logical predictor
calls directly, without using prompt bookkeeping or replay. Exact agreement is the correctness
criterion; runtime and ground-truth metrics describe the practical effect.

Example:
    python finetuning/v2/evaluation/optimization/benchmark_prompt_state_replay.py --label baseline
    python finetuning/v2/evaluation/optimization/benchmark_prompt_state_replay.py --label fixed --expect-exact
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import xxhash
from elf.evaluation import mean_segmentation_accuracy
from scipy.ndimage import distance_transform_edt

from micro_sam.v1.prompt_based_segmentation import _process_box
from micro_sam.v2.prompt_based_segmentation import PromptableSegmentation3D, _crop_to_original_shape
from micro_sam.v2.util import get_sam2_model, precompute_image_embeddings

EVALUATION_ROOT = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = EVALUATION_ROOT.parents[2]
sys.path.insert(0, str(EVALUATION_ROOT))

import common  # noqa
from optimization import benchmark_apg_optimization as apg_benchmark  # noqa


DEFAULT_SAMPLE_ID = "celegans_atlas:8db1fb8b4013"
DEFAULT_OUTPUT_ROOT = apg_benchmark.DEFAULT_OUTPUT_ROOT / "prompt_state_replay"
EXPECTED_OBJECTS = 4


def _synchronize(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def _implementation_checksum() -> str:
    checksum = xxhash.xxh128()
    paths = (Path(__file__), REPOSITORY_ROOT / "micro_sam/v2/prompt_based_segmentation.py")
    for path in paths:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                checksum.update(block)
        checksum.update(b"\0")
    return checksum.hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _deepest_point(mask: np.ndarray, excluded: np.ndarray | None = None) -> np.ndarray:
    distances = distance_transform_edt(mask)
    if excluded is not None:
        distances = distances.copy()
        distances[excluded] = -1
    if distances.max() <= 0:
        coordinates = np.argwhere(mask)
        if len(coordinates) == 0:
            raise ValueError("Cannot derive a point from an empty mask.")
        return coordinates[0].astype("float32")
    return np.asarray(np.unravel_index(int(np.argmax(distances)), mask.shape), dtype="float32")


def _select_objects(labels: np.ndarray, n_objects: int) -> List[Dict[str, int]]:
    ids, counts = np.unique(labels, return_counts=True)
    order = [int(ids[i]) for i in np.argsort(counts)[::-1] if ids[i] != 0]
    selected, anchors = [], set()
    for object_id in order:
        zs = np.flatnonzero(np.any(labels == object_id, axis=(1, 2)))
        if len(zs) == 0 or zs[0] == 0 or zs[-1] == labels.shape[0] - 1:
            continue
        anchor = int((int(zs[0]) + int(zs[-1])) // 2)
        if anchor in anchors:
            continue
        selected.append({"label_id": object_id, "object_id": len(selected) + 1, "anchor": anchor})
        anchors.add(anchor)
        if len(selected) == n_objects:
            return selected
    raise RuntimeError(
        f"Could only find {len(selected)} non-border objects with distinct anchors; requested {n_objects}."
    )


def _prompt_spec(labels: np.ndarray, selected: Sequence[Dict[str, int]]) -> Dict[int, Dict[str, Any]]:
    specs: Dict[int, Dict[str, Any]] = {}
    height, width = labels.shape[-2:]
    for item in selected:
        label_id, object_id, anchor = item["label_id"], item["object_id"], item["anchor"]
        frame_labels = labels[anchor]
        mask = frame_labels == label_id
        ys, xs = np.nonzero(mask)
        box = np.array([ys.min(), xs.min(), ys.max(), xs.max()], dtype="float32")
        positive = _deepest_point(mask)

        foreign = []
        for foreign_id in np.unique(frame_labels):
            if foreign_id in (0, label_id):
                continue
            point = _deepest_point(frame_labels == foreign_id)
            distance = float(np.sum((point - positive) ** 2))
            foreign.append((distance, int(foreign_id), point))
        negatives = [point for _, _, point in sorted(foreign, key=lambda value: (value[0], value[1]))[:4]]
        points = np.vstack([positive, *negatives]).astype("float32")
        labels_ = np.array([1] + [0] * len(negatives), dtype="int32")

        yy, xx = np.ogrid[:height, :width]
        excluded = (yy - positive[0]) ** 2 + (xx - positive[1]) ** 2 <= 16
        alternate = _deepest_point(mask, excluded=excluded)
        grown_box = box + np.array([-8, -8, 8, 8], dtype="float32")
        grown_box[[0, 2]] = np.clip(grown_box[[0, 2]], 0, height - 1)
        grown_box[[1, 3]] = np.clip(grown_box[[1, 3]], 0, width - 1)

        specs[object_id] = {
            "label_id": label_id,
            "anchor": anchor,
            "points": points,
            "labels": labels_,
            "box": box,
            "alternate": alternate[None],
            "grown_box": grown_box,
        }
    return specs


def _operations(spec: Dict[str, Any], protocol: str) -> List[Dict[str, Any]]:
    joint = {
        "points": spec["points"], "labels": spec["labels"], "box": spec["box"],
        "clear_old_points": True,
    }
    if protocol == "joint":
        return [joint]
    if protocol == "replacement":
        return [
            {
                "points": spec["alternate"], "labels": np.array([1], dtype="int32"),
                "box": spec["grown_box"], "clear_old_points": True,
            },
            joint,
            {
                "points": spec["alternate"], "labels": np.array([1], dtype="int32"),
                "box": None, "clear_old_points": False,
            },
        ]
    raise ValueError(f"Unknown protocol: {protocol}")


def _apply_public(segmenter: PromptableSegmentation3D, specs: Dict[int, Dict[str, Any]], protocol: str) -> None:
    for object_id, spec in specs.items():
        operations = _operations(spec, protocol)
        for index, operation in enumerate(operations):
            if protocol == "replacement" and index == 2:
                segmenter.add_point_prompts(
                    frame_ids=spec["anchor"], points=operation["points"],
                    point_labels=operation["labels"], object_id=object_id,
                )
            else:
                segmenter.add_prompt_set(
                    frame_id=spec["anchor"], points=operation["points"],
                    point_labels=operation["labels"], box=operation["box"],
                    object_id=object_id, clear_old_points=operation["clear_old_points"],
                )


def _apply_direct(predictor, inference_state: Dict[str, Any], object_id: int, frame_id: int,
                  operation: Dict[str, Any], shape: Tuple[int, int]) -> None:
    kwargs: Dict[str, Any] = {}
    points = operation["points"]
    if points is not None and len(points):
        kwargs["points"] = np.array(np.asarray(points, dtype="float32")[:, ::-1], dtype="float32")
        kwargs["labels"] = np.array(operation["labels"], dtype="int32")
    if operation["box"] is not None:
        kwargs["box"] = np.array([_process_box(operation["box"], shape)])
    skip_output = getattr(predictor, "skip_prompt_output", None)
    with (skip_output() if skip_output is not None else contextlib.nullcontext()):
        predictor.add_new_points_or_box(
            inference_state=inference_state, frame_idx=frame_id, obj_id=object_id,
            clear_old_points=operation["clear_old_points"], **kwargs,
        )


class _CallCounter:
    def __init__(self, predictor):
        self.predictor = predictor
        self.original = predictor.add_new_points_or_box
        self.count = 0

    def __enter__(self):
        def counted(*args, **kwargs):
            self.count += 1
            return self.original(*args, **kwargs)

        self.predictor.add_new_points_or_box = counted
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.predictor.add_new_points_or_box = self.original


def _new_segmenter(predictor, raw: np.ndarray, embeddings, device: str) -> PromptableSegmentation3D:
    return PromptableSegmentation3D(
        predictor, raw, embeddings, device=device, offload_state_to_cpu=False,
        max_cached_frames=raw.shape[0],
    )


def _extract_masks(video_segments: Dict[int, Dict[int, np.ndarray]], object_ids: Iterable[int],
                   shape: Tuple[int, int, int]) -> Dict[int, np.ndarray]:
    masks = {object_id: np.zeros(shape, dtype=bool) for object_id in object_ids}
    for frame, per_object in video_segments.items():
        for object_id, mask in per_object.items():
            if object_id in masks:
                masks[object_id][frame] = _crop_to_original_shape(np.asarray(mask).squeeze(), shape[-2:])
    return masks


def _time_replay(predictor, raw: np.ndarray, embeddings, device: str,
                 specs: Dict[int, Dict[str, Any]], protocol: str, repetitions: int) -> Dict[str, Any]:
    timings, calls = [], []
    for repetition in range(repetitions + 1):
        segmenter = _new_segmenter(predictor, raw, embeddings, device)
        # Exercise grouping and replay while excluding video propagation from this decoder benchmark.
        segmenter._propagate_both_directions = lambda *args, **kwargs: {}
        _synchronize(device)
        start = time.perf_counter()
        with _CallCounter(predictor) as counter:
            _apply_public(segmenter, specs, protocol)
            segmenter.propagate_prompts(early_stop_patience=None)
        _synchronize(device)
        elapsed = time.perf_counter() - start
        if repetition:
            timings.append(elapsed)
            calls.append(counter.count)
        segmenter.reset_tracking()
    if len(set(calls)) != 1:
        raise RuntimeError(f"Prompt call count changed between repetitions: {calls}")
    return {
        "warmup_repetitions": 1,
        "measured_repetitions": repetitions,
        "seconds": timings,
        "median_seconds": float(np.median(timings)),
        "predictor_calls": calls[0],
    }


def _run_production(predictor, raw: np.ndarray, embeddings, device: str,
                    specs: Dict[int, Dict[str, Any]], protocol: str) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
    segmenter = _new_segmenter(predictor, raw, embeddings, device)
    with _CallCounter(predictor) as counter:
        _synchronize(device)
        start = time.perf_counter()
        _apply_public(segmenter, specs, protocol)
        _synchronize(device)
        prompt_seconds = time.perf_counter() - start
        start = time.perf_counter()
        video_segments = segmenter.propagate_prompts(early_stop_patience=None)
        _synchronize(device)
        propagation_seconds = time.perf_counter() - start
    masks = _extract_masks(video_segments, specs, raw.shape)
    segmenter.reset_tracking()
    return masks, {
        "prompt_seconds": prompt_seconds,
        "propagation_seconds": propagation_seconds,
        "total_seconds": prompt_seconds + propagation_seconds,
        "predictor_calls": counter.count,
    }


def _run_oracle(predictor, raw: np.ndarray, embeddings, device: str,
                specs: Dict[int, Dict[str, Any]], protocol: str) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
    masks: Dict[int, np.ndarray] = {}
    calls, prompt_seconds, propagation_seconds = 0, 0.0, 0.0
    with _CallCounter(predictor) as counter:
        for object_id, spec in specs.items():
            segmenter = _new_segmenter(predictor, raw, embeddings, device)
            _synchronize(device)
            start = time.perf_counter()
            for operation in _operations(spec, protocol):
                _apply_direct(
                    predictor, segmenter.inference_state, object_id, spec["anchor"], operation, raw.shape[-2:],
                )
            _synchronize(device)
            prompt_seconds += time.perf_counter() - start
            start = time.perf_counter()
            video_segments = segmenter._propagate_both_directions(early_stop_patience=None)
            _synchronize(device)
            propagation_seconds += time.perf_counter() - start
            masks.update(_extract_masks(video_segments, [object_id], raw.shape))
            segmenter.reset_tracking()
        calls = counter.count
    return masks, {
        "prompt_seconds": prompt_seconds,
        "propagation_seconds": propagation_seconds,
        "total_seconds": prompt_seconds + propagation_seconds,
        "predictor_calls": calls,
    }


def _quality(masks: Dict[int, np.ndarray], specs: Dict[int, Dict[str, Any]], labels: np.ndarray) -> Dict[str, Any]:
    per_object_iou: Dict[str, float] = {}
    segmentation = np.zeros(labels.shape, dtype="uint32")
    selected_labels = np.zeros(labels.shape, dtype="uint32")
    for object_id, spec in specs.items():
        prediction = masks[object_id]
        ground_truth = labels == spec["label_id"]
        union = np.logical_or(prediction, ground_truth).sum()
        per_object_iou[str(object_id)] = float(np.logical_and(prediction, ground_truth).sum() / max(1, union))
        segmentation[np.logical_and(prediction, segmentation == 0)] = object_id
        selected_labels[ground_truth] = object_id
    return {
        "per_object_iou": per_object_iou,
        "mean_object_iou": float(np.mean(list(per_object_iou.values()))),
        "msa": float(mean_segmentation_accuracy(segmentation, selected_labels)),
    }


def _comparison(production: Dict[int, np.ndarray], oracle: Dict[int, np.ndarray]) -> Dict[str, Any]:
    per_object: Dict[str, Any] = {}
    exact = True
    for object_id in oracle:
        same = production[object_id] == oracle[object_id]
        agreement = float(same.mean())
        object_exact = bool(same.all())
        per_object[str(object_id)] = {"voxel_agreement": agreement, "exact": object_exact}
        exact = exact and object_exact
    return {"exact": exact, "per_object": per_object}


def _load_sample(data_root: Path, manifest_path: Path, sample_id: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    with open(manifest_path) as f:
        manifest = json.load(f)
    matches = [sample for sample in manifest["samples"] if sample["sample_id"] == sample_id]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one manifest sample '{sample_id}', found {len(matches)}.")
    sample = matches[0]
    normalized = apg_benchmark._load_normalized_3d_source(sample, data_root)
    raw, labels = apg_benchmark._load_3d_sample(sample, data_root, normalized)
    return raw, labels, manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Run label, e.g. baseline or fixed.")
    parser.add_argument("--data-root", type=Path, default=apg_benchmark.DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--sample-id", default=DEFAULT_SAMPLE_ID)
    parser.add_argument("--model-type", default="hvit_t")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-objects", type=int, default=EXPECTED_OBJECTS)
    parser.add_argument("--timing-repetitions", type=int, default=5)
    parser.add_argument("--expect-exact", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_root = args.data_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    manifest_path = args.manifest or apg_benchmark._default_manifest_path(
        apg_benchmark.DEFAULT_OUTPUT_ROOT, "standard",
    )
    manifest_path = manifest_path.expanduser().resolve(strict=True)
    output_root.mkdir(parents=True, exist_ok=True)

    raw, labels, manifest = _load_sample(data_root, manifest_path, args.sample_id)
    selected = _select_objects(labels, args.n_objects)
    specs = _prompt_spec(labels, selected)

    source_checkpoint = common.get_joint_checkpoint(args.model_type, args.checkpoint)
    source_checksum = common.checkpoint_checksum(source_checkpoint)
    interactive_checkpoint, _ = common.export_joint_checkpoint(
        args.model_type, args.checkpoint, source_checksum=source_checksum,
    )
    predictor = get_sam2_model(
        model_type=args.model_type, device=args.device, checkpoint_path=interactive_checkpoint, input_type="videos",
    )
    embedding_path = output_root / "embeddings" / f"{args.sample_id.replace(':', '_')}_{source_checksum}.zarr"
    embeddings = precompute_image_embeddings(
        predictor, raw, save_path=embedding_path, lazy_loading=True, ndim=3, verbose=True,
    )

    result: Dict[str, Any] = {
        "schema_version": 1,
        "label": args.label,
        "sample_id": args.sample_id,
        "manifest": str(manifest_path),
        "manifest_checksum": manifest["manifest_checksum"],
        "shape": list(raw.shape),
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "checkpoint_checksum": source_checksum,
        "implementation_checksum": _implementation_checksum(),
        "git_revision": _git_revision(),
        "device": args.device,
        "hardware": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(torch.device(args.device))
            if torch.device(args.device).type == "cuda" else None,
        },
        "objects": selected,
        "protocols": {},
    }

    failures = []
    for protocol in ("joint", "replacement"):
        timing = _time_replay(
            predictor, raw, embeddings, args.device, specs, protocol, args.timing_repetitions,
        )
        production, production_runtime = _run_production(
            predictor, raw, embeddings, args.device, specs, protocol,
        )
        oracle, oracle_runtime = _run_oracle(predictor, raw, embeddings, args.device, specs, protocol)
        comparison = _comparison(production, oracle)
        logical_calls = sum(len(_operations(spec, protocol)) for spec in specs.values())
        expected_production_calls = 3 * logical_calls  # initial push, grouped replay, final restoration
        call_count_exact = production_runtime["predictor_calls"] == expected_production_calls
        if not comparison["exact"]:
            failures.append(f"{protocol}: masks differ from isolated-state oracle")
        if not call_count_exact:
            failures.append(
                f"{protocol}: {production_runtime['predictor_calls']} predictor calls, "
                f"expected {expected_production_calls}"
            )
        result["protocols"][protocol] = {
            "timing": timing,
            "production_runtime": production_runtime,
            "oracle_runtime": oracle_runtime,
            "logical_predictor_calls": logical_calls,
            "expected_production_predictor_calls": expected_production_calls,
            "production_call_count_exact": call_count_exact,
            "comparison": comparison,
            "production_quality": _quality(production, specs, labels),
            "oracle_quality": _quality(oracle, specs, labels),
        }

    digest = hashlib.sha1(args.sample_id.encode("utf-8")).hexdigest()[:8]
    output_path = output_root / f"prompt_state_replay_{args.label}_{digest}.json"
    apg_benchmark._atomic_write_json(output_path, result)
    getattr(embeddings, "close", lambda: None)()
    print(json.dumps({"output": str(output_path), "failures": failures}, indent=2))
    if args.expect_exact and failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    main()
