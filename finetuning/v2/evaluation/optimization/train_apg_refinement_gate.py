"""Extract refinement utility features and train the selected direct H128x64 MLP gate.

The extractor runs the established blanket refinement on the primary manifest and records, for each
first-round instance, only evidence available before the second decoder call. The target is the
positive IoU improvement delivered by the accepted refined mask. Five image-level folds produce
leakage-safe OOF predictions before one full-primary refit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from micro_sam.v2.automatic_prompt_generation import (
    _parse_refinement, derive_refinement_prompts, postmerge_refinement_gate_features,
)
from micro_sam.v2.multimask_selection import (
    MULTIMASK_FEATURE_NAMES, MULTIMASK_FEATURE_VERSION, POSTMERGE_REFINEMENT_GATE_FEATURE_NAMES,
    REFINEMENT_GATE_FEATURE_NAMES, REFINEMENT_GATE_STAGES, load_feature_scorer, refinement_gate_features_torch,
    selector_input_schema,
)

EVALUATION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVALUATION_ROOT))

import common  # noqa
from optimization.benchmark_apg_optimization import (  # noqa
    DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_ROOT, _default_manifest_path, _load_2d_sample,
    _validate_roots, prepare_manifest,
)
from optimization.screen_apg_multimask import (  # noqa
    _configured_records, _load_oof_lookup, _oof_predictions_for_sample, _predict_records,
)
from optimization.train_apg_multimask_selector import _stable_folds  # noqa


ARCHITECTURE = {"hidden_sizes": (128, 64), "dropout": 0.1}


def _iou(mask: np.ndarray, target: np.ndarray) -> float:
    intersection = int(np.count_nonzero(mask & target))
    union = int(mask.sum()) + int(target.sum()) - intersection
    return intersection / union if union else 0.0


def _target_for_instance(segmentation, labels, instance_id, point):
    x, y = np.round(point).astype("int64")
    x, y = int(np.clip(x, 0, labels.shape[1] - 1)), int(np.clip(y, 0, labels.shape[0] - 1))
    object_id = int(labels[y, x])
    if object_id == 0:
        overlaps = labels[segmentation == instance_id]
        overlaps = overlaps[overlaps != 0]
        if len(overlaps):
            object_id = int(np.bincount(overlaps).argmax())
    return labels == object_id if object_id else np.zeros_like(labels, dtype=bool)


def _gate_row(raw_proposals, selection_scores, source_record):
    prompt_index = source_record["prompt_index"]
    group_indices = [
        index for index, record in enumerate(raw_proposals) if record["prompt_index"] == prompt_index
    ]
    group_indices.sort(key=lambda index: raw_proposals[index]["multimask_index"])
    features = torch.as_tensor(
        np.stack([raw_proposals[index]["multimask_features"] for index in group_indices]),
        dtype=torch.float32,
    )
    # Compact selector datasets may carry a token suffix, but the established pre-merge gate uses
    # the same 19 low-resolution mask statistics as the dense implementation.
    features = features[:, :len(MULTIMASK_FEATURE_NAMES)]
    scores = torch.as_tensor(selection_scores[group_indices], dtype=torch.float32)
    alternatives = [raw_proposals[index]["multimask_index"] for index in group_indices]
    selected = alternatives.index(source_record["multimask_index"])
    return refinement_gate_features_torch(
        features[None], scores[None], torch.as_tensor([selected]),
    )[0].numpy()


def extract_gate_dataset(
    manifest, data_root, output, device, selector_artifact=None, selection="eager", merge="raw",
    score_filter="predicted_iou", score_threshold=0.6,
    selector_oof_dataset=None, selector_oof_predictions=None,
    gate_stage="premerge", target_mode="positive",
):
    if gate_stage not in REFINEMENT_GATE_STAGES:
        raise ValueError(f"Invalid gate stage {gate_stage!r}.")
    if target_mode not in ("positive", "signed"):
        raise ValueError(f"Invalid target mode {target_mode!r}.")
    samples = [sample for sample in manifest["samples"] if sample["ndim"] == 2]
    folds = _stable_folds(samples)
    scorer = load_feature_scorer(selector_artifact, device=device) if selector_artifact else None
    if selector_artifact is not None and selector_oof_predictions is not None:
        raise ValueError("Use either a refit selector artifact or OOF selector predictions, not both.")
    if selector_oof_predictions is not None:
        if selector_oof_dataset is None:
            raise ValueError("OOF selector predictions require their extracted feature dataset.")
        selector_rows, selector_predictions, selector_lookup = _load_oof_lookup(
            selector_oof_dataset, {"selector": selector_oof_predictions}, manifest["manifest_checksum"],
        )
        selector_data = np.load(selector_oof_dataset, allow_pickle=False)
        proposal_schema = str(selector_data["input_schema"]) if "input_schema" in selector_data.files else "dense_v1"
    else:
        selector_rows = selector_predictions = selector_lookup = None
        proposal_schema = selector_input_schema(scorer) if scorer is not None else "dense_v1"
    checkpoint = common.get_joint_checkpoint("hvit_t", "best")
    segmenter = common.build_apg_segmenter(
        "hvit_t", 2, device, joint_checkpoint="best",
        joint_checksum=common.checkpoint_checksum(checkpoint),
        export_root=str(output.parent / "model_exports"),
    )
    components, refinement_kwargs = _parse_refinement("points+boxes", None)
    rows = []
    try:
        for number, sample in enumerate(samples, 1):
            raw, labels = _load_2d_sample(sample, data_root)
            segmenter.clear_state()
            segmenter.initialize(raw, ndim=2)
            raw_proposals = segmenter.propose(
                multimasking=True, multimask_scorer="predicted_iou", multimask_selection="deferred",
                return_multimask_features=True, multimask_feature_schema=proposal_schema,
            )
            if not raw_proposals:
                continue
            if selector_predictions is not None:
                selection_scores = _oof_predictions_for_sample(
                    sample["sample_id"], raw_proposals, selector_rows,
                    selector_predictions, selector_lookup,
                )["selector"]
            else:
                selection_scores = np.asarray(
                    _predict_records(scorer, raw_proposals) if scorer is not None
                    else [record["predicted_iou"] for record in raw_proposals],
                    dtype="float32",
                )
            config = {
                "selection": selection, "merge": merge,
            }
            configured = _configured_records(
                raw_proposals, config,
                selection_scores if scorer is not None or selector_predictions is not None else None,
            )
            first, context = segmenter._merge(
                configured, labels.shape, score_threshold=score_threshold,
                score_filter=score_filter, max_overlap=0.15, min_size=50, return_context=True,
            )
            if context is None or first.max() == 0:
                continue
            if gate_stage == "postmerge":
                all_points_list, seen_groups = [], set()
                for record_index, record in enumerate(context["proposals"]):
                    group = record.get("multimask_group", ("record", record_index))
                    if group in seen_groups:
                        continue
                    seen_groups.add(group)
                    all_points_list.append(record["point"])
                point_prompts = derive_refinement_prompts(
                    first, np.asarray(all_points_list, dtype="float32"),
                    {
                        instance_id: context["records"][record_index]["point"]
                        for instance_id, record_index in context["matches"].items()
                    },
                    n_positives=refinement_kwargs["n_positives"],
                    n_negatives=refinement_kwargs["n_negatives"],
                    max_negative_distance=refinement_kwargs["max_negative_distance"],
                    negative_source=refinement_kwargs["negative_source"],
                    min_negative_distance=refinement_kwargs["min_negative_distance"],
                )
                gate_features, gate_instance_ids = postmerge_refinement_gate_features(
                    first, context, point_prompts, segmenter._prediction[0], float(
                        context["records"][next(iter(context["matches"].values()))].get(
                            "foreground_threshold", 0.5,
                        )
                    ),
                )
                postmerge_rows = {
                    int(instance_id): features
                    for instance_id, features in zip(gate_instance_ids, gate_features)
                }
            instance_rows = []
            for instance_id, record_index in context["matches"].items():
                source = context["records"][record_index]
                target = _target_for_instance(first, labels, instance_id, source["point"])
                instance_rows.append({
                    "instance_id": instance_id,
                    "features": (
                        postmerge_rows[instance_id] if gate_stage == "postmerge"
                        else _gate_row(raw_proposals, selection_scores, source)
                    ),
                    "first_iou": _iou(first == instance_id, target),
                    "target": target,
                    "prompt_index": source["prompt_index"],
                    "multimask_index": source["multimask_index"],
                })
            refined = segmenter._refine(
                first, context, components, refinement_kwargs, batch_size=64,
            )
            for item in instance_rows:
                delta = _iou(refined == item["instance_id"], item["target"]) - item["first_iou"]
                rows.append({
                    "features": item["features"],
                    "target": delta if target_mode == "signed" else max(delta, 0.0),
                    "raw_delta": delta,
                    "sample_id": sample["sample_id"], "dataset": sample["dataset"],
                    "fold": folds[sample["sample_id"]],
                    "group": f"{sample['sample_id']}:{item['instance_id']}",
                    "prompt_index": item["prompt_index"], "multimask_index": item["multimask_index"],
                })
            print(f"[{number}/{len(samples)}] {sample['sample_id']} instances={len(instance_rows)}", flush=True)
    finally:
        segmenter.clear_state()

    features = np.stack([row["features"] for row in rows]).astype("float32")
    targets = np.asarray([row["target"] for row in rows], dtype="float32")
    raw_delta = np.asarray([row["raw_delta"] for row in rows], dtype="float32")
    sample_ids = np.asarray([row["sample_id"] for row in rows])
    datasets = np.asarray([row["dataset"] for row in rows])
    groups = np.asarray([row["group"] for row in rows])
    fold_array = np.asarray([row["fold"] for row in rows], dtype="int8")
    prompt_indices = np.asarray([row["prompt_index"] for row in rows], dtype="int32")
    multimask_indices = np.asarray([row["multimask_index"] for row in rows], dtype="int8")
    weights = np.zeros(len(rows), dtype="float64")
    for dataset in np.unique(datasets):
        dataset_rows = np.flatnonzero(datasets == dataset)
        dataset_samples = np.unique(sample_ids[dataset_rows])
        for sample_id in dataset_samples:
            image_rows = dataset_rows[sample_ids[dataset_rows] == sample_id]
            weights[image_rows] = 1.0 / (len(np.unique(datasets)) * len(dataset_samples) * len(image_rows))
    weights /= weights.mean()
    output.parent.mkdir(parents=True, exist_ok=True)
    feature_names = (
        POSTMERGE_REFINEMENT_GATE_FEATURE_NAMES if gate_stage == "postmerge"
        else REFINEMENT_GATE_FEATURE_NAMES
    )
    np.savez_compressed(
        output, features=features, targets=targets, raw_delta=raw_delta, sample_ids=sample_ids,
        datasets=datasets, groups=groups, folds=fold_array, weights=weights.astype("float32"),
        prompt_indices=prompt_indices, multimask_indices=multimask_indices,
        feature_version=np.asarray(MULTIMASK_FEATURE_VERSION),
        feature_names=np.asarray(feature_names), gate_stage=np.asarray(gate_stage),
        target_mode=np.asarray(target_mode),
        manifest_checksum=np.asarray(manifest["manifest_checksum"]),
        selector_prediction_source=np.asarray(
            "out-of-fold" if selector_predictions is not None else (
                "refit-model" if scorer is not None else "predicted-iou"
            )
        ),
        first_pass_policy=np.asarray(json.dumps({
            "selection": selection, "merge": merge, "score_filter": score_filter,
            "score_threshold": score_threshold, "max_overlap": 0.15, "min_size": 50,
        }, sort_keys=True)),
    )
    return output


def _load_gate_dataset(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    if int(data["feature_version"]) != MULTIMASK_FEATURE_VERSION:
        raise ValueError("The gate feature dataset has a different runtime schema version.")
    gate_stage = str(data["gate_stage"]) if "gate_stage" in data.files else "premerge"
    target_mode = str(data["target_mode"]) if "target_mode" in data.files else "positive"
    if gate_stage not in REFINEMENT_GATE_STAGES:
        raise ValueError(f"Unsupported gate stage {gate_stage!r} in the feature dataset.")
    expected_names = (
        POSTMERGE_REFINEMENT_GATE_FEATURE_NAMES if gate_stage == "postmerge"
        else REFINEMENT_GATE_FEATURE_NAMES
    )
    if tuple(data["feature_names"].tolist()) != expected_names:
        raise ValueError("The gate feature dataset does not match the runtime schema.")
    if target_mode not in ("positive", "signed"):
        raise ValueError(f"Unsupported gate target mode {target_mode!r}.")
    return {
        "features": data["features"].astype("float32", copy=False),
        "targets": data["targets"].astype("float32", copy=False),
        "raw_delta": data["raw_delta"].astype("float32", copy=False),
        "weights": data["weights"].astype("float32", copy=False),
        "folds": data["folds"].astype("int8", copy=False),
        "manifest_checksum": str(data["manifest_checksum"]),
        "feature_names": expected_names, "gate_stage": gate_stage, "target_mode": target_mode,
        "first_pass_policy": (
            json.loads(str(data["first_pass_policy"])) if "first_pass_policy" in data.files else None
        ),
    }


def _make_mlp(input_size: int) -> torch.nn.Module:
    layers, width = [], input_size
    for hidden in ARCHITECTURE["hidden_sizes"]:
        layers.extend((torch.nn.Linear(width, hidden), torch.nn.ReLU()))
        layers.append(torch.nn.Dropout(ARCHITECTURE["dropout"]))
        width = hidden
    layers.append(torch.nn.Linear(width, 1))
    return torch.nn.Sequential(*layers)


def _normalization(features, weights):
    mean = np.average(features, axis=0, weights=weights).astype("float32")
    variance = np.average((features - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance).astype("float32")
    scale[scale == 0] = 1.0
    return mean, scale


def _loss(prediction, target, weights):
    per_row = F.smooth_l1_loss(prediction, target, reduction="none")
    return (per_row * weights).sum() / weights.sum()


def _fit_gate(data, train, validation, device, max_epochs=200):
    mean, scale = _normalization(data["features"][train], data["weights"][train])
    x = torch.as_tensor((data["features"] - mean) / scale, dtype=torch.float32, device=device)
    y = torch.as_tensor(data["targets"], dtype=torch.float32, device=device)
    weights = torch.as_tensor(data["weights"], dtype=torch.float32, device=device)
    torch.manual_seed(17)
    model = _make_mlp(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(17)
    train_indices = torch.as_tensor(np.flatnonzero(train), dtype=torch.int64)
    validation_indices = torch.as_tensor(np.flatnonzero(validation), dtype=torch.int64, device=device)
    best_state, best_loss, best_epoch, stale = None, float("inf"), 0, 0
    for epoch in range(max_epochs):
        model.train()
        order = train_indices[torch.randperm(len(train_indices), generator=generator)]
        for start in range(0, len(order), 1024):
            index = order[start:start + 1024].to(device)
            loss = _loss(model(x[index]).reshape(-1), y[index], weights[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(_loss(
                model(x[validation_indices]).reshape(-1), y[validation_indices],
                weights[validation_indices],
            ).cpu())
        if validation_loss < best_loss - 1e-7:
            best_loss, best_epoch, stale = validation_loss, epoch + 1, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 15:
                break
    model.load_state_dict(best_state)
    return model.eval(), mean, scale, best_epoch


def _fit_gate_full(data, device, epochs):
    mean, scale = _normalization(data["features"], data["weights"])
    x = torch.as_tensor((data["features"] - mean) / scale, dtype=torch.float32, device=device)
    y = torch.as_tensor(data["targets"], dtype=torch.float32, device=device)
    weights = torch.as_tensor(data["weights"], dtype=torch.float32, device=device)
    torch.manual_seed(17)
    model = _make_mlp(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(17)
    indices = torch.arange(len(x), dtype=torch.int64)
    for _ in range(epochs):
        order = indices[torch.randperm(len(indices), generator=generator)]
        for start in range(0, len(order), 1024):
            index = order[start:start + 1024].to(device)
            loss = _loss(model(x[index]).reshape(-1), y[index], weights[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval(), mean, scale


def train_gate(dataset: Path, output_dir: Path, device: str, target_mode: str | None = None) -> Path:
    data = _load_gate_dataset(dataset)
    if target_mode is not None:
        if target_mode not in ("positive", "signed"):
            raise ValueError(f"Invalid target mode {target_mode!r}.")
        data["target_mode"] = target_mode
        data["targets"] = (
            data["raw_delta"].copy() if target_mode == "signed"
            else np.maximum(data["raw_delta"], 0.0)
        ).astype("float32", copy=False)
    predictions = np.zeros_like(data["targets"])
    fold_epochs = []
    for outer in range(5):
        validation_fold = (outer + 1) % 5
        train = (data["folds"] != outer) & (data["folds"] != validation_fold)
        validation, test = data["folds"] == validation_fold, data["folds"] == outer
        model, mean, scale, best_epoch = _fit_gate(data, train, validation, device)
        values = torch.as_tensor(
            (data["features"][test] - mean) / scale, dtype=torch.float32, device=device,
        )
        with torch.no_grad():
            fold_predictions = model(values).reshape(-1)
            if data["target_mode"] == "positive":
                fold_predictions = fold_predictions.clamp(0, 1)
            predictions[test] = fold_predictions.cpu().numpy()
        fold_epochs.append(best_epoch)
        print(f"gate fold {outer + 1}/5 epoch={best_epoch}", flush=True)

    error = predictions - data["targets"]
    metrics = {
        "weighted_mse": float(np.average(error * error, weights=data["weights"])),
        "weighted_mae": float(np.average(np.abs(error), weights=data["weights"])),
        "correlation": float(np.corrcoef(predictions, data["targets"])[0, 1]),
        "fold_epochs": fold_epochs,
    }
    thresholds = {
        str(fraction): float(np.quantile(predictions, 1.0 - fraction))
        for fraction in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
    }
    metrics["fraction_thresholds"] = thresholds
    refit_epochs = max(1, int(round(float(np.mean(fold_epochs)))))
    model, mean, scale = _fit_gate_full(data, device, refit_epochs)
    refit_values = torch.as_tensor(
        (data["features"] - mean) / scale, dtype=torch.float32, device=device,
    )
    with torch.no_grad():
        refit_predictions = model(refit_values).reshape(-1)
        if data["target_mode"] == "positive":
            refit_predictions = refit_predictions.clamp(0, 1)
        refit_predictions = refit_predictions.cpu().numpy()
    metrics["refit_fraction_thresholds"] = {
        str(fraction): float(np.quantile(refit_predictions, 1.0 - fraction))
        for fraction in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
    }

    prefix = "postmerge-" if data["gate_stage"] == "postmerge" else ""
    suffix = "-signed" if data["target_mode"] == "signed" else ""
    name = f"{prefix}gate-mlp-h128x64-d0p1-regression{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / f"{name}.pt"
    torch.save({
        "kind": "mlp", "feature_version": MULTIMASK_FEATURE_VERSION,
        "feature_names": list(data["feature_names"]),
        "hidden_sizes": list(ARCHITECTURE["hidden_sizes"]), "dropout": ARCHITECTURE["dropout"],
        "mean": mean, "scale": scale,
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "metadata": {
            "architecture": ARCHITECTURE, "loss": "direct-regression", "epochs": refit_epochs,
            "manifest_checksum": data["manifest_checksum"],
            "first_pass_policy": data["first_pass_policy"], "oof_metrics": metrics,
            "gate_stage": data["gate_stage"], "target_mode": data["target_mode"],
            "output_activation": "identity" if data["target_mode"] == "signed" else "clamp",
        },
    }, artifact)
    np.save(output_dir / f"{name}_oof.npy", predictions.astype("float32"))
    with open(output_dir / "gate_training_results.json", "w") as f:
        json.dump({
            "artifact": str(artifact), "metrics": metrics, "refit_epochs": refit_epochs,
        }, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("extract", "train", "all"), default="all")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--selector-artifact", type=Path, default=None)
    parser.add_argument("--selector-oof-dataset", type=Path, default=None)
    parser.add_argument("--selector-oof-predictions", type=Path, default=None)
    parser.add_argument("--selection", choices=("eager", "deferred"), default="eager")
    parser.add_argument("--merge", choices=("raw", "learned"), default="raw")
    parser.add_argument(
        "--score-filter", choices=("predicted_iou", "selection_score", "none"),
        default="predicted_iou",
    )
    parser.add_argument("--score-threshold", type=float, default=0.6)
    parser.add_argument(
        "--gate-stage", choices=("premerge", "postmerge"), default="premerge",
        help="Feature stage. Post-merge sees the accepted mask and its assembled refinement prompts.",
    )
    parser.add_argument(
        "--target", choices=("positive", "signed"), default="positive",
        help="Fit clipped positive gain or the signed IoU change from refinement.",
    )
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    manifest_path = args.manifest or _default_manifest_path(args.output_root, "standard", "primary")
    data_root, output_root, manifest_path = _validate_roots(args.data_root, args.output_root, manifest_path)
    manifest = prepare_manifest(data_root, manifest_path, "standard", subset="primary")
    root = output_root / "multimask_selection" / "groupwise_v1" / "refinement_gate"
    if args.gate_stage != "premerge" or args.target != "positive":
        root = root / f"{args.gate_stage}_{args.target}"
    dataset = args.dataset or root / "primary_features.npz"
    artifact_dir = args.artifact_dir or root / "models"
    if args.stage in ("extract", "all"):
        extract_gate_dataset(
            manifest, data_root, dataset, args.device, args.selector_artifact,
            selection=args.selection, merge=args.merge, score_filter=args.score_filter,
            score_threshold=args.score_threshold,
            selector_oof_dataset=args.selector_oof_dataset,
            selector_oof_predictions=args.selector_oof_predictions,
            gate_stage=args.gate_stage, target_mode=args.target,
        )
    if args.stage in ("train", "all"):
        artifact = train_gate(dataset.resolve(strict=True), artifact_dir, args.device, target_mode=args.target)
        print(f"Artifact: {artifact}")


if __name__ == "__main__":
    main()
