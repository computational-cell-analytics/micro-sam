"""Screen APG refinement configurations on the 2d benchmark subset, reusing one round of proposals.

A canonical benchmark run re-prompts SAM2 from scratch for every configuration, which costs 15-26
minutes per configuration. Every refinement configuration shares the first round, so this screening
runs `propose` once per image and only the merge and the second-round re-prompt per configuration:
the marginal cost of a configuration is its own refinement forwards.

Screening ranks quality only. The per-configuration select seconds are recorded as a rough cost
signal, but they are not comparable to the canonical benchmark's serialized timings: final numbers,
and any gate decision, come from `benchmark_apg_optimization.py` runs of the shortlisted
configurations.

Run with the built-in grid (the refinement sweep of the current experiment) or a JSON list of
configurations, each `{"name": ..., "params_2d": {...}}` as in the benchmark:

```bash
python finetuning/v2/evaluation/screen_apg_refinement.py --device cuda
python finetuning/v2/evaluation/screen_apg_refinement.py --configs my_configs.json
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

import common
from common import (
    GENERATE_PARAM_KEYS,
    GT_MIN_SIZE_2D,
    build_apg_segmenter,
    checkpoint_checksum,
    get_joint_checkpoint,
    resolve_params,
)
from benchmark_apg_optimization import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    MANIFEST_SUBSETS,
    _atomic_write_csv,
    _atomic_write_json,
    _content_checksum,
    _default_manifest_path,
    _git_revision,
    _implementation_checksum,
    _load_2d_sample,
    _validate_roots,
    prepare_manifest,
)
from evaluate_automatic_segmentation import compute_metrics
from micro_sam.v2.multimask_selection import load_feature_scorer
from screen_apg_multimask import (
    _configured_records, _load_oof_lookup, _oof_predictions_for_sample,
)

# The half of the parameters that decides the proposals, which is the half that prompts SAM2 from
# scratch. Every screened configuration must share these, so one `propose` serves all of them.
PROPOSE_KEYS = (
    "candidate_threshold", "foreground_threshold", "n_iter", "dt", "sigma", "min_candidate_size",
    "multimasking", "multimask_scorer", "multimask_selection", "batch_size", "n_threads",
)
SELECT_KEYS = (
    "score_threshold", "score_filter", "max_overlap", "min_size",
    "refinement", "refinement_kwargs", "batch_size",
)

# Flattened from `_last_generation_stats`, so a gain can be attributed to a measured failure mode.
STAT_COLUMNS = (
    "proposed_candidates", "scored_candidates", "refinement_eligible_instances",
    "uncertainty_selected_instances", "refined_instances", "replaced_instances",
    "merged_kept", "merged_duplicate", "merged_too_small", "merged_truncated",
)


def default_screening_configs() -> List[Dict[str, Any]]:
    """The refinement screening grid: the point-prompt sweep plus the box/mask baselines.

    The control ('refinement-none') verifies that the shared proposals reproduce the plain APG on
    every sample. The combined modes run at the point defaults; the shortlisted settings replace
    them in the canonical follow-up runs.
    """
    configs = [{"name": "refinement-none", "params_2d": {}}]
    for n_positives in (2, 3, 5):
        for n_negatives in (0, 2, 4):
            for policy in ("replace", "keep-if-better"):
                configs.append({
                    "name": f"points-p{n_positives}-n{n_negatives}-{policy}",
                    "params_2d": {
                        "refinement": "points",
                        "refinement_kwargs": {
                            "n_positives": n_positives, "n_negatives": n_negatives, "policy": policy,
                        },
                    },
                })
    for mode in ("boxes", "points+boxes", "points+masks", "boxes+masks"):
        for policy in ("replace", "keep-if-better"):
            configs.append({
                "name": f"{mode.replace('+', '-')}-{policy}",
                "params_2d": {"refinement": mode, "refinement_kwargs": {"policy": policy}},
            })
    return configs


def _load_configs(path: Path | None) -> List[Dict[str, Any]]:
    if path is None:
        configs = default_screening_configs()
    else:
        with open(path) as f:
            configs = json.load(f)
    if not isinstance(configs, list) or not configs:
        raise ValueError("Expected a non-empty JSON list of configurations.")

    resolved, names = [], set()
    for config in configs:
        unknown_top_level = set(config) - {"name", "params_2d"}
        if unknown_top_level:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown_top_level)}.")
        name = config.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"Every configuration needs a unique non-empty name, got {name!r}.")
        names.add(name)
        overrides = config.get("params_2d", {})
        unknown = set(overrides) - set(GENERATE_PARAM_KEYS)
        if unknown:
            raise ValueError(f"Unknown APG parameters in '{name}': {sorted(unknown)}.")
        resolved.append({"name": name, "params_2d": resolve_params(overrides, ndim=2)})

    shared = {key: resolved[0]["params_2d"][key] for key in PROPOSE_KEYS}
    for config in resolved[1:]:
        if any(config["params_2d"][key] != shared[key] for key in PROPOSE_KEYS):
            raise ValueError(
                f"Configuration '{config['name']}' changes a proposal parameter. Screening reuses "
                f"one round of proposals, so all configurations must share {PROPOSE_KEYS}."
            )
    return resolved


def _flatten_stats(stats: Dict[str, Any]) -> Dict[str, int]:
    reasons = stats.get("merge_reasons", {})
    return {
        "proposed_candidates": int(stats.get("proposed_candidates", 0)),
        "scored_candidates": int(stats.get("scored_candidates", 0)),
        "refinement_eligible_instances": int(stats.get("refinement_eligible_instances", 0)),
        "uncertainty_selected_instances": int(stats.get("uncertainty_selected_instances", 0)),
        "refined_instances": int(stats.get("refined_instances", 0)),
        "replaced_instances": int(stats.get("replaced_instances", 0)),
        "merged_kept": int(reasons.get("kept", 0)),
        "merged_duplicate": int(reasons.get("duplicate", 0)),
        "merged_too_small": int(reasons.get("too small", 0)),
        "merged_truncated": int(reasons.get("truncated below min size", 0)),
    }


def _summarize(samples: pd.DataFrame) -> pd.DataFrame:
    """Per configuration and dataset, plus the dataset-balanced row that ranks the configurations."""
    rows = []
    for config_name, config_frame in samples.groupby("config_name", sort=False):
        by_dataset = config_frame.groupby("dataset", sort=True)
        per_dataset = by_dataset.agg(
            n_samples=("sample_id", "count"), msa_mean=("msa", "mean"), msa_std=("msa", "std"),
            select_seconds=("select_seconds", "sum"),
        ).reset_index()
        per_dataset.insert(0, "config_name", config_name)
        rows.append(per_dataset)
        rows.append(pd.DataFrame([{
            "config_name": config_name,
            "dataset": "__dataset_balanced__",
            "n_samples": len(config_frame),
            "msa_mean": float(per_dataset["msa_mean"].mean()),
            "msa_std": float("nan"),
            "select_seconds": float(per_dataset["select_seconds"].sum()),
        }]))
    summary = pd.concat(rows, ignore_index=True)
    # Best dataset-balanced configuration first, its per-dataset rows directly below it.
    balanced = summary[summary["dataset"] == "__dataset_balanced__"].sort_values("msa_mean", ascending=False)
    order = {name: rank for rank, name in enumerate(balanced["config_name"])}
    summary["__rank__"] = summary["config_name"].map(order)
    summary = summary.sort_values(["__rank__", "dataset"], kind="stable").drop(columns="__rank__")
    return summary.reset_index(drop=True)


def _load_gate_oof_lookup(dataset_path, predictions_path, manifest_checksum):
    data = np.load(dataset_path, allow_pickle=False)
    if str(data["manifest_checksum"]) != manifest_checksum:
        raise ValueError(
            "The refinement-gate dataset was extracted from a different manifest: "
            f"{data['manifest_checksum']} != {manifest_checksum}."
        )
    required = {"sample_ids", "prompt_indices", "multimask_indices"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"Refinement-gate dataset is missing lookup fields: {sorted(missing)}.")
    predictions = np.load(predictions_path, allow_pickle=False).astype("float32", copy=False)
    if predictions.shape != (len(data["sample_ids"]),):
        raise ValueError(
            f"Gate OOF predictions have shape {predictions.shape}, expected {(len(data['sample_ids']),)}."
        )
    lookup = {}
    for index, values in enumerate(zip(
        data["sample_ids"], data["prompt_indices"], data["multimask_indices"],
    )):
        key = (str(values[0]), int(values[1]), int(values[2]))
        if key in lookup:
            raise ValueError(f"Duplicate gate-dataset key: {key}.")
        lookup[key] = float(predictions[index])
    return lookup


def _inject_gate_oof_scores(segmenter, proposals, sample_id, shape, configs, lookup):
    if not proposals:
        return
    for record in proposals:
        key = (sample_id, int(record["prompt_index"]), int(record["multimask_index"]))
        record["uncertainty_score"] = lookup.get(key, float("nan"))

    # A gate dataset only contains first-round instances that survived the merge. Verify that every
    # source record accepted by each screened gate configuration has an OOF score; a mismatch means
    # the gate data and first-round strategy are not replay-compatible.
    merge_settings = set()
    for config in configs:
        params = config["params_2d"]
        if (params.get("refinement_kwargs") or {}).get("gate") != "uncertainty":
            continue
        merge_settings.add((
            params["score_threshold"], params["score_filter"],
            params["max_overlap"], params["min_size"],
        ))
    for score_threshold, score_filter, max_overlap, min_size in merge_settings:
        _, context = segmenter._merge(
            proposals, shape, score_threshold=score_threshold, score_filter=score_filter,
            max_overlap=max_overlap, min_size=min_size, return_context=True,
        )
        if context is None:
            continue
        for instance_id, record_index in context["matches"].items():
            record = context["records"][record_index]
            if not np.isfinite(record["uncertainty_score"]):
                key = (sample_id, int(record["prompt_index"]), int(record["multimask_index"]))
                raise ValueError(
                    f"Accepted instance {instance_id} with source {key} has no gate OOF prediction."
                )


def run_screening(
    manifest: Dict[str, Any], data_root: Path, output_root: Path, model_type: str,
    joint_checkpoint: str, configs: List[Dict[str, Any]], device: str, subset: str = "primary",
    multimask_scorer_artifact: Path | None = None, refinement_gate_artifact: Path | None = None,
    selector_oof_dataset: Path | None = None, selector_oof_predictions: Path | None = None,
    gate_oof_dataset: Path | None = None, gate_oof_predictions: Path | None = None,
) -> Tuple[Path, pd.DataFrame]:
    checkpoint_path = get_joint_checkpoint(model_type, joint_checkpoint)
    checkpoint_id = checkpoint_checksum(checkpoint_path)
    implementation_checksum = _implementation_checksum()
    artifact_paths = {
        "multimask_scorer": multimask_scorer_artifact,
        "refinement_gate": refinement_gate_artifact,
        "selector_oof_dataset": selector_oof_dataset,
        "selector_oof_predictions": selector_oof_predictions,
        "gate_oof_dataset": gate_oof_dataset,
        "gate_oof_predictions": gate_oof_predictions,
    }
    artifact_checksums = {
        name: hashlib.sha256(Path(path).resolve(strict=True).read_bytes()).hexdigest()
        for name, path in artifact_paths.items() if path is not None
    }
    screen_implementation_checksum = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    configs_checksum = _content_checksum({
        "configs": configs, "model_artifacts": artifact_checksums,
        "screen_implementation_checksum": screen_implementation_checksum,
    })
    manifest_checksum = manifest["manifest_checksum"]

    run_dir = output_root / "refinement_screening" / model_type / checkpoint_id / (
        f"{manifest_checksum}-{configs_checksum}-{implementation_checksum}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    samples_path = run_dir / "samples.csv"
    summary_path = run_dir / "summary.csv"

    _atomic_write_json(run_dir / "metadata.json", {
        # Not a benchmark result: quality is canonical, the timings are not serialized trials.
        "screening": True,
        "configs": configs,
        "configs_checksum": configs_checksum,
        "manifest_checksum": manifest_checksum,
        "implementation_checksum": implementation_checksum,
        "checkpoint_checksum": checkpoint_id,
        "checkpoint_name": joint_checkpoint,
        "model_type": model_type,
        "device": device,
        "subset": subset,
        "git_revision": _git_revision(),
        "model_artifacts": artifact_checksums,
        "screen_implementation_checksum": screen_implementation_checksum,
    })

    completed = pd.read_csv(samples_path) if samples_path.exists() else pd.DataFrame()
    completed_ids = set(completed["sample_id"]) if not completed.empty else set()
    samples = [sample for sample in manifest["samples"] if sample["ndim"] == 2]
    pending = [sample for sample in samples if sample["sample_id"] not in completed_ids]

    propose_params = {key: configs[0]["params_2d"][key] for key in PROPOSE_KEYS}
    desired_scorer = propose_params["multimask_scorer"]
    desired_selection = propose_params["multimask_selection"]
    use_selector_oof = selector_oof_predictions is not None
    use_gate_oof = gate_oof_predictions is not None
    if (selector_oof_dataset is None) != (selector_oof_predictions is None):
        raise ValueError("Selector OOF replay requires both its feature dataset and predictions.")
    if (gate_oof_dataset is None) != (gate_oof_predictions is None):
        raise ValueError("Gate OOF replay requires both its feature dataset and predictions.")
    needs_uncertainty = any(
        (config["params_2d"].get("refinement_kwargs") or {}).get("gate") == "uncertainty"
        for config in configs
    )
    if subset == "primary" and desired_scorer == "microscopy" and not use_selector_oof:
        raise ValueError("Primary microscopy-selector screening requires OOF selector predictions.")
    if subset == "primary" and needs_uncertainty and not use_gate_oof:
        raise ValueError("Primary uncertainty-gate screening requires OOF gate predictions.")
    if subset != "primary" and (use_selector_oof or use_gate_oof):
        raise ValueError("OOF prediction replay is only valid for the primary subset.")

    if use_selector_oof:
        selector_rows, selector_values, selector_lookup = _load_oof_lookup(
            selector_oof_dataset, {"selector": selector_oof_predictions}, manifest_checksum,
        )
    else:
        selector_rows = selector_values = selector_lookup = None
    gate_lookup = (
        _load_gate_oof_lookup(gate_oof_dataset, gate_oof_predictions, manifest_checksum)
        if use_gate_oof else None
    )
    if use_selector_oof or use_gate_oof:
        propose_params = dict(propose_params)
        propose_params.update({
            "multimask_scorer": "predicted_iou", "multimask_selection": "deferred",
        })
    segmenter = build_apg_segmenter(
        model_type, 2, device, joint_checkpoint=joint_checkpoint, joint_checksum=checkpoint_id,
        export_root=str(output_root / "model_exports"),
    )
    if ((multimask_scorer_artifact is not None and not use_selector_oof)
            or (refinement_gate_artifact is not None and not use_gate_oof)):
        segmenter.set_multimask_models(
            scorer=(
                load_feature_scorer(multimask_scorer_artifact, device=device)
                if multimask_scorer_artifact is not None and not use_selector_oof else None
            ),
            refinement_gate=(
                load_feature_scorer(refinement_gate_artifact, device=device)
                if refinement_gate_artifact is not None and not use_gate_oof else None
            ),
        )

    for index, sample in enumerate(pending, start=1):
        raw, labels = _load_2d_sample(sample, data_root)
        segmenter.clear_state()
        segmenter.initialize(raw, ndim=2)
        proposals = segmenter.propose(
            **propose_params, compute_multimask_uncertainty=needs_uncertainty and not use_gate_oof,
        )
        if use_selector_oof or use_gate_oof:
            if proposals:
                if use_selector_oof:
                    selection_scores = _oof_predictions_for_sample(
                        sample["sample_id"], proposals, selector_rows, selector_values, selector_lookup,
                    )["selector"]
                else:
                    selection_scores = None
                proposals = _configured_records(
                    proposals,
                    {
                        "selection": desired_selection,
                        "merge": "learned" if desired_scorer == "microscopy" else "raw",
                    },
                    selection_scores,
                )
        if use_gate_oof:
            _inject_gate_oof_scores(
                segmenter, proposals, sample["sample_id"], labels.shape, configs, gate_lookup,
            )

        rows = []
        # The metric setup of the benchmark's `_sample_row`, so a screening mSA matches a canonical one.
        border_min_size = GT_MIN_SIZE_2D.get(sample["dataset"], 0)
        for config in configs:
            select_params = {key: config["params_2d"][key] for key in SELECT_KEYS}
            segmenter._last_generation_stats = {}
            start = time.perf_counter()
            segmentation = segmenter.select(proposals, **select_params).astype("uint32")
            select_seconds = time.perf_counter() - start
            metrics = compute_metrics(segmentation, labels, "sparse", border_min_size=border_min_size)
            rows.append({
                "sample_id": sample["sample_id"],
                "dataset": sample["dataset"],
                "config_name": config["name"],
                "msa": metrics["msa"],
                "select_seconds": select_seconds,
                "predicted_objects": int(len(np.unique(segmentation)) - 1),
                **_flatten_stats(segmenter._last_generation_stats),
            })

        completed = pd.concat([completed, pd.DataFrame(rows)], ignore_index=True)
        _atomic_write_csv(samples_path, completed)
        print(f"[{index}/{len(pending)}] {sample['sample_id']}", flush=True)

    segmenter.clear_state()
    summary = _summarize(completed)
    _atomic_write_csv(summary_path, summary)
    return run_dir, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Read-only dataset root.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None, help="Subset manifest; defaults below output-root.")
    parser.add_argument(
        "--configs", type=Path, default=None,
        help="JSON list of configurations; without one the built-in refinement grid is screened.",
    )
    parser.add_argument("--model-type", default="hvit_t", choices=common.MODEL_TYPES)
    parser.add_argument("--joint-checkpoint", default="best", help="Joint checkpoint name without '.pt'.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--multimask-scorer-artifact", type=Path, default=None)
    parser.add_argument("--refinement-gate-artifact", type=Path, default=None)
    parser.add_argument("--selector-oof-dataset", type=Path, default=None)
    parser.add_argument("--selector-oof-predictions", type=Path, default=None)
    parser.add_argument("--gate-oof-dataset", type=Path, default=None)
    parser.add_argument("--gate-oof-predictions", type=Path, default=None)
    parser.add_argument(
        "--subset", choices=MANIFEST_SUBSETS, default="primary",
        help="The validation subset. Tuning stays on 'primary'; 'holdout' confirms shortlisted "
             "configurations on images the tuning never saw.",
    )
    args = parser.parse_args()

    manifest_path = args.manifest or _default_manifest_path(args.output_root, "standard", args.subset)
    data_root, output_root, manifest_path = _validate_roots(args.data_root, args.output_root, manifest_path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("A CUDA device was requested, but CUDA is not available.")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = prepare_manifest(data_root, manifest_path, "standard", subset=args.subset)
    configs = _load_configs(args.configs)
    print(
        f"Manifest: {manifest_path} ({manifest['manifest_checksum']}, subset {args.subset})\n"
        f"Screening {len(configs)} configurations on "
        f"{sum(sample['ndim'] == 2 for sample in manifest['samples'])} 2d samples.",
        file=sys.stderr,
    )

    run_dir, summary = run_screening(
        manifest, data_root, output_root, args.model_type, args.joint_checkpoint, configs, args.device,
        subset=args.subset, multimask_scorer_artifact=args.multimask_scorer_artifact,
        refinement_gate_artifact=args.refinement_gate_artifact,
        selector_oof_dataset=args.selector_oof_dataset,
        selector_oof_predictions=args.selector_oof_predictions,
        gate_oof_dataset=args.gate_oof_dataset, gate_oof_predictions=args.gate_oof_predictions,
    )
    balanced = summary[summary["dataset"] == "__dataset_balanced__"]
    print(balanced.to_string(index=False))
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
