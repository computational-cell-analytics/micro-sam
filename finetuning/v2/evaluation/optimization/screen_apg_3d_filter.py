"""Replay candidate policies on the cached 3d tracks, without touching a GPU.

Every policy decides which cached candidates are propagated and in which order their tracks enter
the 3d merge; the tracks themselves are cached, so a policy costs one `merge_by_score` per crop. The
control reproduces the pipeline: the base ladder's candidates, predicted IoU >= score_threshold, the
in-plane merge on every anchor slice, and the anchor score as the merge order. A learned policy adds a
filter (out-of-fold scores at a retention fraction, thresholds from the other folds), a learned merge
order, a candidate budget, or a wider ladder.

Usage examples:
    python screen_apg_3d_filter.py --subset primary --cache <cache dir> --output <screen dir>
    python screen_apg_3d_filter.py --subset primary --cache <cache dir> --output <screen dir> \\
        --oof <models dir>/volume-candidate-token_lowres_v1-comp-h64-d0p1_oof.npz --retention 0.9 0.8 0.7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EVALUATION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVALUATION_ROOT))

import common  # noqa
from common import genuine_misses  # noqa
from parameter_search import compute_metrics  # noqa
from optimization.benchmark_apg_optimization import _atomic_write_csv, _atomic_write_json, _content_checksum  # noqa
from optimization.apg3d_manifest import CAMPAIGN_ROOT, DEFAULT_DATA_ROOT, load_manifest, load_labels  # noqa
from optimization.benchmark_apg_3d import summarize, BOOTSTRAP_SAMPLES  # noqa
from optimization.extract_apg_3d_tracks import unpack_mask  # noqa
from micro_sam.v2.automatic_prompt_generation import merge_by_score  # noqa

SCORE_THRESHOLD = 0.6
MAX_OVERLAP = 0.15
MIN_SIZE_2D = 50
MIN_SIZE_3D = 100
N_OBJECTS_PER_PASS = 16


class CropCache:
    """One crop's cached candidates and tracks, unpacked lazily."""

    def __init__(self, crop_dir: Path):
        self.dir = crop_dir
        self.candidates = np.load(crop_dir / "candidates.npz", allow_pickle=False)
        self.tracks = np.load(crop_dir / "tracks.npz", allow_pickle=False)
        self.summary = json.load(open(crop_dir / "complete.json"))
        self.shape = tuple(int(side) for side in self.tracks["volume_shape"])
        self.track_of_prompt = {int(p): i for i, p in enumerate(self.tracks["prompt_index"].tolist())}

    def anchor_record(self, index: int) -> dict:
        c = self.candidates
        mask = unpack_mask(c["anchor_mask_payload"], c["anchor_mask_offsets"], c["anchor_mask_shapes"], index)
        y0, x0 = (int(v) for v in c["anchor_box_start"][index])
        return {
            "segmentation": mask, "bounding_box": (slice(y0, y0 + mask.shape[0]), slice(x0, x0 + mask.shape[1])),
            "predicted_iou": float(c["anchor_predicted_iou"][index]),
            "stability_score": float(c["anchor_stability"][index]), "index": index,
        }

    def track_record(self, index: int, merge_score: Optional[float] = None) -> Optional[dict]:
        prompt = int(self.candidates["prompt_index"][index])
        track = self.track_of_prompt.get(prompt)
        if track is None:
            return None
        t = self.tracks
        mask = unpack_mask(t["mask_payload"], t["mask_offsets"], t["mask_shapes"], track)
        start, stop = t["box_start"][track], t["box_stop"][track]
        record = {
            "segmentation": mask,
            "bounding_box": tuple(slice(int(a), int(b)) for a, b in zip(start, stop)),
            "predicted_iou": float(self.candidates["anchor_predicted_iou"][index]),
            "stability_score": float(self.candidates["anchor_stability"][index]),
        }
        if merge_score is not None:
            record["merge_score"] = float(merge_score)
        return record


def anchor_survivors(cache: CropCache, ladder_index: int, score_threshold: float = SCORE_THRESHOLD) -> np.ndarray:
    """The candidate rows of one ladder that pass the historical anchor decision."""
    c = cache.candidates
    prompt_index = c["prompt_index"]
    member = c["ladder_membership"][prompt_index][:, ladder_index]
    strong = c["anchor_predicted_iou"] >= score_threshold
    eligible = np.flatnonzero(member & strong)
    survivors = []
    frames = c["frame"]
    for frame in np.unique(frames[eligible]):
        rows = eligible[frames[eligible] == frame]
        records = [cache.anchor_record(int(row)) for row in rows]
        shape = tuple(
            max(record["bounding_box"][axis].stop for record in records) for axis in range(2)
        )
        _, kept = merge_by_score(records, shape, max_overlap=MAX_OVERLAP, min_size=MIN_SIZE_2D, return_matches=True)
        survivors.extend(int(records[record_index]["index"]) for record_index in kept.values())
    return np.asarray(sorted(survivors), dtype="int64")


def passes_for(cache: CropCache, rows: np.ndarray) -> int:
    frames = cache.candidates["frame"][rows]
    return int(sum(int(np.ceil(count / N_OBJECTS_PER_PASS)) for count in np.bincount(frames) if count))


def replay(cache: CropCache, rows: np.ndarray, labels: np.ndarray, merge_scores: Optional[np.ndarray],
           metric_mode: str) -> Dict[str, Any]:
    records = []
    for position, row in enumerate(rows):
        record = cache.track_record(int(row), None if merge_scores is None else merge_scores[position])
        if record is not None:
            records.append(record)
    if records:
        segmentation = merge_by_score(records, cache.shape, max_overlap=MAX_OVERLAP, min_size=MIN_SIZE_3D)
    else:
        segmentation = np.zeros(cache.shape, dtype="uint32")
    segmentation = segmentation.astype("uint32")
    result = compute_metrics(segmentation, labels, metric_mode, border_min_size=0)
    result["unmatched"], result["genuine_misses"] = genuine_misses(labels, segmentation)
    result["predicted_objects"] = int(len(np.unique(segmentation)) - 1)
    result["candidates"] = int(len(rows))
    result["tracks"] = len(records)
    result["propagation_passes"] = passes_for(cache, rows)
    return result


def fold_thresholds(scores: np.ndarray, folds: np.ndarray, eligible: np.ndarray, retention: float) -> Dict[int, float]:
    """Per fold, the score below which the other folds' eligible candidates would be cut at 'retention'."""
    thresholds = {}
    for fold in np.unique(folds):
        pool = scores[eligible & (folds != fold) & np.isfinite(scores)]
        thresholds[int(fold)] = float(np.quantile(pool, 1.0 - retention)) if len(pool) else -np.inf
    return thresholds


def load_oof(path: Path) -> Dict[Tuple[str, int], float]:
    data = np.load(path, allow_pickle=False)
    key = "oof"
    return {(str(s), int(p)): float(v) for s, p, v in zip(data["sample_id"], data["prompt_index"], data[key])}


def _bootstrap_delta(control: pd.DataFrame, candidate: pd.DataFrame, seed: int = 0) -> Dict[str, float]:
    """Paired bootstrap over crops of the family-macro mSA difference."""
    merged = control[["sample_id", "family", "msa"]].merge(
        candidate[["sample_id", "msa"]], on="sample_id", suffixes=("_control", "_candidate"),
    )
    if merged.empty:
        return {}
    rng = np.random.default_rng(seed)
    families = merged["family"].to_numpy()
    deltas = (merged["msa_candidate"] - merged["msa_control"]).to_numpy()

    def macro(index):
        table = pd.DataFrame({"family": families[index], "delta": deltas[index]})
        return float(table.groupby("family")["delta"].mean().mean())

    n = len(merged)
    draws = np.array([macro(rng.integers(0, n, n)) for _ in range(BOOTSTRAP_SAMPLES)])
    return {"delta": macro(np.arange(n)), "ci_low": float(np.percentile(draws, 2.5)),
            "ci_high": float(np.percentile(draws, 97.5))}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", default="primary")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, default=CAMPAIGN_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--oof", type=Path, nargs="*", default=[], help="OOF prediction files of trained filters.")
    parser.add_argument("--retention", type=float, nargs="*", default=[1.0, 0.95, 0.9, 0.85, 0.8, 0.7])
    parser.add_argument("--ladders", type=int, nargs="*", default=None,
                        help="Ladder indices to replay; all by default.")
    parser.add_argument("--budget-factor", type=float, nargs="*", default=[])
    args = parser.parse_args(argv)

    manifest = load_manifest(args.subset, args.campaign_root, args.data_root)
    oof_sets = {path.stem.replace("_oof", ""): load_oof(path) for path in args.oof}
    samples = [
        s for s in manifest["samples"] if (args.cache / s["sample_id"].replace(":", "_") / "complete.json").exists()
    ]
    if not samples:
        raise SystemExit("No cached crops.")
    caches = {s["sample_id"]: CropCache(args.cache / s["sample_id"].replace(":", "_")) for s in samples}
    first_cache = next(iter(caches.values()))
    ladders = [json.loads(str(ladder)) for ladder in first_cache.candidates["ladders"]]
    ladder_indices = args.ladders if args.ladders else list(range(len(ladders)))

    policies: List[Dict[str, Any]] = []
    for ladder_index in ladder_indices:
        base = {"ladder": ladder_index, "ladder_values": ladders[ladder_index]}
        policies.append({**base, "name": f"L{ladder_index}-control", "filter": None, "order": "anchor", "budget": None})
        for oof_name in oof_sets:
            for retention in args.retention:
                for order in ("anchor", "learned"):
                    if retention == 1.0 and order == "anchor":
                        continue
                    policies.append({**base, "name": f"L{ladder_index}-{oof_name}-r{retention:g}-{order}",
                                     "filter": (oof_name, retention), "order": order, "budget": None})
        for factor in args.budget_factor:
            policies.append({**base, "name": f"L{ladder_index}-budget{factor:g}", "filter": None, "order": "anchor",
                             "budget": factor})

    labels_cache: Dict[str, np.ndarray] = {}
    control_passes: Dict[str, int] = {}
    results = []
    for policy in policies:
        eligible_by_crop = {}
        scores_by_crop = {}
        for s in samples:
            cache = caches[s["sample_id"]]
            survivors = anchor_survivors(cache, policy["ladder"])
            eligible_by_crop[s["sample_id"]] = survivors
            if policy["filter"] is not None:
                oof = oof_sets[policy["filter"][0]]
                scores_by_crop[s["sample_id"]] = np.asarray([
                    oof.get((s["sample_id"], int(cache.candidates["prompt_index"][row])), np.nan) for row in survivors
                ], dtype="float32")
        thresholds = None
        if policy["filter"] is not None:
            flat_scores = np.concatenate([scores_by_crop[s["sample_id"]] for s in samples])
            flat_folds = np.concatenate([
                np.full(len(eligible_by_crop[s["sample_id"]]), int(s["fold"])) for s in samples
            ])
            thresholds = fold_thresholds(flat_scores, flat_folds, np.ones(len(flat_scores), dtype=bool),
                                         policy["filter"][1])
        for s in samples:
            cache = caches[s["sample_id"]]
            rows = eligible_by_crop[s["sample_id"]]
            merge_scores = None
            if policy["filter"] is not None:
                scores = scores_by_crop[s["sample_id"]]
                keep = np.isfinite(scores) & (scores >= thresholds[int(s["fold"])])
                rows, scores = rows[keep], scores[keep]
                if policy["order"] == "learned":
                    merge_scores = scores
            if policy["budget"] is not None:
                budget = int(np.ceil(policy["budget"] * len(eligible_by_crop[s["sample_id"]])))
                candidates = cache.candidates
                anchor_scores = candidates["anchor_predicted_iou"][rows] * candidates["anchor_stability"][rows]
                order = np.argsort(-(merge_scores if merge_scores is not None else anchor_scores))
                rows = rows[order[:budget]]
                merge_scores = None if merge_scores is None else merge_scores[order[:budget]]
            if s["sample_id"] not in labels_cache:
                labels_cache[s["sample_id"]] = load_labels(s, args.data_root)
            result = replay(cache, rows, labels_cache[s["sample_id"]], merge_scores, s["metric_mode"])
            if policy["name"].endswith("-control") and policy["ladder"] == ladder_indices[0]:
                control_passes[s["sample_id"]] = result["propagation_passes"]
            results.append({
                "policy": policy["name"], "ladder": policy["ladder"], "sample_id": s["sample_id"],
                "dataset": s["dataset"], "family": s["family"], "seen_in_training": str(s["seen_in_training"]),
                "gt_objects": int(len(np.unique(labels_cache[s["sample_id"]])) - 1),
                "total_seconds": 0.0, "generation_seconds": 0.0, **result,
            })
        print(f"{policy['name']}: done", flush=True)

    table = pd.DataFrame(results)
    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(args.output / "samples.csv", table)
    summaries = []
    control_name = f"L{ladder_indices[0]}-control"
    control = table[table["policy"] == control_name]
    for name, group in table.groupby("policy", sort=False):
        summary = summarize(group.drop(columns=["policy"]))
        summary.insert(0, "policy", name)
        bootstrap = _bootstrap_delta(control, group) if name != control_name else {}
        for key, value in bootstrap.items():
            summary.loc[summary["dataset"] == "__family_macro__", f"macro_delta_{key}"] = value
        summaries.append(summary)
    summary = pd.concat(summaries, ignore_index=True)
    _atomic_write_csv(args.output / "summary.csv", summary)
    wanted = ["policy", "msa_mean", "propagation_passes", "candidates", "tracks", "genuine_misses"]
    macro = summary[summary["dataset"] == "__family_macro__"][
        [c for c in wanted if c in summary.columns] + [c for c in summary.columns if c.startswith("macro_delta")]
    ]
    print(macro.to_string(index=False))
    _atomic_write_json(args.output / "metadata.json", {
        "subset": args.subset, "cache": str(args.cache), "oof": [str(p) for p in args.oof], "retention": args.retention,
        "ladders": ladders, "n_crops": len(samples), "policies": [p["name"] for p in policies],
        "identity": _content_checksum({"cache": str(args.cache), "oof": [str(p) for p in args.oof]}),
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
