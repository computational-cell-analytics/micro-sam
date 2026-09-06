"""Fit the pre-propagation candidate filter of the 3d APG on the cached tracks, leakage-safe.

One row per cached candidate: its three anchor alternatives' selector features (and optionally the
ladder's component features), its target the IoU of the point-conditioned track that the propagation
produced for it. A groupwise MLP scores the three alternatives jointly and reduces them to one score
per candidate. Folds are the manifest's source-grouped folds; the out-of-fold predictions are what the
replay screens, and a leave-one-dataset-out pass reports how much of the signal is dataset identity.

Usage examples:
    python train_apg_3d_filter.py aggregate --cache <cache dir>  --output <training dir>
    python train_apg_3d_filter.py train --dataset <training dir>/candidates.npz --schema token_lowres_v1 \\
        --component-features all --hidden-size 64 --output <training dir>/models
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

EVALUATION_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVALUATION_ROOT))

from optimization.train_apg_multimask_selector import _fit, _fit_full  # noqa
from optimization.apg3d_manifest import load_manifest, CAMPAIGN_ROOT  # noqa
from micro_sam.v2.multimask_selection import GroupwiseMLP, SELECTOR_FEATURE_SCHEMAS  # noqa

SCHEMAS = ("token_lowres_v1", "token_v1", "lowres_v1")
KIND = "volume_candidate_mlp"


# ----------------------------------------------------------------------------------------------
# aggregation of the per-crop caches into one training table


def _schema_columns(feature_names: Sequence[str], schema: str) -> np.ndarray:
    """Columns of the cached `token_lowres_v1` rows that make up a (sub)schema."""
    names = [str(name) for name in feature_names]
    return np.asarray([names.index(name) for name in SELECTOR_FEATURE_SCHEMAS[schema]], dtype="int64")


def aggregate(cache_root: Path, manifest: Dict[str, Any], output: Path, base_ladder: int = 0) -> Path:
    """Stack the crops' candidates into one table; the target is the cached track IoU."""
    rows: Dict[str, List[Any]] = {
        "features": [], "component_features": [], "target": [], "anchor_predicted_iou": [], "dataset": [],
        "family": [], "source": [], "fold": [], "seen": [], "sample_id": [], "prompt_index": [],
        "ladder_membership": [], "crop_weight": [],
    }
    feature_names = component_names = None
    by_dataset: Dict[str, int] = {}
    crops = []
    for sample in manifest["samples"]:
        crop_dir = cache_root / sample["sample_id"].replace(":", "_")
        if not (crop_dir / "complete.json").exists():
            continue
        crops.append((sample, crop_dir))
        by_dataset[sample["dataset"]] = by_dataset.get(sample["dataset"], 0) + 1
    if not crops:
        raise SystemExit(f"No complete crops in {cache_root}.")
    for sample, crop_dir in crops:
        candidates = np.load(crop_dir / "candidates.npz", allow_pickle=False)
        tracks = np.load(crop_dir / "tracks.npz", allow_pickle=False)
        if feature_names is None:
            component_names = tuple(str(name) for name in candidates["component_feature_names"])
            feature_names = tuple(SELECTOR_FEATURE_SCHEMAS[str(candidates["feature_schema"])])
        track_iou = dict(zip(tracks["prompt_index"].tolist(), tracks["track_iou"].tolist()))
        # Read every array once: indexing the NpzFile decompresses the whole array on each access and
        # a per-candidate loop over it kept one full copy alive per candidate (128 GB was not enough).
        prompt_index = np.asarray(candidates["prompt_index"], dtype="int64")
        n = len(prompt_index)
        if n == 0:
            continue
        alternative_features = np.asarray(candidates["alternative_features"], dtype="float32")
        component_features = np.asarray(candidates["component_features"], dtype="float32")
        anchor_predicted_iou = np.asarray(candidates["anchor_predicted_iou"], dtype="float32")
        ladder_membership = np.asarray(candidates["ladder_membership"], dtype=bool)
        weight = 1.0 / (len(by_dataset) * by_dataset[sample["dataset"]] * n)
        rows["features"].append(alternative_features)
        rows["component_features"].append(component_features[prompt_index])
        rows["target"].append(np.asarray([track_iou.get(int(p), 0.0) for p in prompt_index], dtype="float32"))
        rows["anchor_predicted_iou"].append(anchor_predicted_iou)
        rows["dataset"].append(np.full(n, sample["dataset"]))
        rows["family"].append(np.full(n, sample["family"]))
        rows["source"].append(np.full(n, sample["source_id"]))
        rows["fold"].append(np.full(n, int(sample["fold"]), dtype="int64"))
        rows["seen"].append(np.full(n, str(sample["seen_in_training"])))
        rows["sample_id"].append(np.full(n, sample["sample_id"]))
        rows["prompt_index"].append(prompt_index)
        rows["ladder_membership"].append(ladder_membership[prompt_index])
        rows["crop_weight"].append(np.full(n, weight, dtype="float32"))
    rows = {key: np.concatenate(value) if value else np.asarray(value) for key, value in rows.items()}
    output.mkdir(parents=True, exist_ok=True)
    path = output / "candidates.npz"
    features = np.asarray(rows["features"], dtype="float32")
    # An alternative whose mask came back empty has no features; give it the group mean so the
    # normalization and the MLP see finite numbers, and mark it in a separate column.
    missing = ~np.isfinite(features).all(axis=2)
    if missing.any():
        group_mean = np.nanmean(features, axis=1, keepdims=True)
        group_mean = np.where(np.isfinite(group_mean), group_mean, 0.0)
        features = np.where(missing[..., None], np.broadcast_to(group_mean, features.shape), features)
    np.savez_compressed(
        path, features=features, missing_alternative=missing,
        component_features=np.asarray(rows["component_features"], dtype="float32"),
        target=np.asarray(rows["target"], dtype="float32"),
        anchor_predicted_iou=np.asarray(rows["anchor_predicted_iou"], dtype="float32"),
        dataset=np.asarray(rows["dataset"]), family=np.asarray(rows["family"]), source=np.asarray(rows["source"]),
        fold=np.asarray(rows["fold"], dtype="int64"), seen=np.asarray(rows["seen"]),
        sample_id=np.asarray(rows["sample_id"]), prompt_index=np.asarray(rows["prompt_index"], dtype="int64"),
        ladder_membership=np.asarray(rows["ladder_membership"], dtype=bool),
        weight=np.asarray(rows["crop_weight"], dtype="float32"),
        feature_names=np.asarray(feature_names), component_feature_names=np.asarray(component_names),
        manifest_checksum=np.asarray(manifest["manifest_checksum"]), cache_root=np.asarray(str(cache_root)),
        n_crops=np.asarray(len(crops)),
    )
    print(f"{len(features)} candidates from {len(crops)} crops -> {path}")
    return path


# ----------------------------------------------------------------------------------------------
# training


def _inputs(data, schema: str, component_names: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    columns = _schema_columns(data["feature_names"], schema)
    features = data["features"][:, :, columns]
    names = [str(data["feature_names"][index]) for index in columns]
    if component_names:
        all_names = [str(name) for name in data["component_feature_names"]]
        selected = [all_names.index(name) for name in component_names]
        components = data["component_features"][:, selected]
        # Broadcast the candidate-level ladder features onto every alternative row.
        features = np.concatenate([features, np.repeat(components[:, None, :], 3, axis=1)], axis=2)
        names = names + [f"component_{name}" for name in component_names]
    features = np.nan_to_num(features.astype("float32"), nan=0.0, posinf=0.0, neginf=0.0)
    return features, names


def _weights(data, balance: str) -> np.ndarray:
    if balance == "crop":
        return data["weight"].astype("float64")
    return np.ones(len(data["target"]), dtype="float64")


def _grouped_targets(targets: np.ndarray) -> np.ndarray:
    # The groupwise MLP predicts one value per alternative; the track target is shared by the group.
    return np.repeat(targets[:, None], 3, axis=1).astype("float32")


def _reduce(predictions: np.ndarray) -> np.ndarray:
    """One score per candidate from the three alternative scores: the mean, which is what the
    installed scorer computes too (see `VolumeCandidateScorer`)."""
    return predictions.mean(axis=1)


def train(
    dataset: Path, output: Path, schema: str, component_names: Sequence[str], hidden_size: int, dropout: float,
    device: str, balance: str = "crop", lodo: bool = True, unseen_only: bool = False,
) -> Path:
    data = np.load(dataset, allow_pickle=False)
    features, names = _inputs(data, schema, component_names)
    targets = data["target"].astype("float32")
    weights = _weights(data, balance)
    folds = data["fold"].astype("int64")
    datasets = data["dataset"]
    keep = np.ones(len(targets), dtype=bool)
    if unseen_only:
        keep = data["seen"] == "False"
    architecture = {"hidden_size": int(hidden_size), "dropout": float(dropout)}
    grouped_targets = _grouped_targets(targets)

    oof = np.full(len(targets), np.nan, dtype="float32")
    fold_epochs = []
    for outer in range(5):
        validation_fold = (outer + 1) % 5
        train_mask = keep & (folds != outer) & (folds != validation_fold)
        validation = keep & (folds == validation_fold)
        test = folds == outer
        if train_mask.sum() == 0 or validation.sum() == 0 or test.sum() == 0:
            continue
        model, mean, scale, best_epoch = _fit(
            features, grouped_targets, weights, train_mask, validation, device, architecture,
        )
        values = torch.as_tensor((features[test] - mean) / scale, dtype=torch.float32, device=device)
        with torch.no_grad():
            oof[test] = _reduce(model(values).cpu().numpy())
        fold_epochs.append(best_epoch)
        print(f"fold {outer + 1}/5 epoch={best_epoch} rows={int(test.sum())}", flush=True)

    lodo_predictions = np.full(len(targets), np.nan, dtype="float32")
    lodo_metrics = {}
    if lodo:
        for held_out in np.unique(datasets):
            test = datasets == held_out
            others = keep & ~test
            validation = others & (folds == 0)
            train_mask = others & (folds != 0)
            if train_mask.sum() == 0 or validation.sum() == 0:
                continue
            model, mean, scale, _ = _fit(
                features, grouped_targets, weights, train_mask, validation, device, architecture,
            )
            values = torch.as_tensor((features[test] - mean) / scale, dtype=torch.float32, device=device)
            with torch.no_grad():
                lodo_predictions[test] = _reduce(model(values).cpu().numpy())
            lodo_metrics[str(held_out)] = _metrics(targets[test], lodo_predictions[test], weights[test])
            print(f"lodo {held_out}: {lodo_metrics[str(held_out)]}", flush=True)

    valid = np.isfinite(oof)
    metrics = {
        "oof": _metrics(targets[valid], oof[valid], weights[valid]),
        "oof_by_dataset": {
            str(name): _metrics(targets[valid & (datasets == name)], oof[valid & (datasets == name)],
                                weights[valid & (datasets == name)])
            for name in np.unique(datasets)
        },
        "anchor_predicted_iou": _metrics(targets, data["anchor_predicted_iou"], weights),
        "lodo": lodo_metrics, "fold_epochs": fold_epochs,
    }
    refit_epochs = max(1, int(round(float(np.mean(fold_epochs))))) if fold_epochs else 20
    model, mean, scale = _fit_full(
        features[keep], grouped_targets[keep], weights[keep], device, refit_epochs, architecture,
    )

    component_tag = "comp" if component_names else "nocomp"
    name = f"volume-candidate-{schema}-{component_tag}-h{hidden_size}-d{str(dropout).replace('.', 'p')}"
    if unseen_only:
        name += "-unseen"
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / f"{name}.pt"
    torch.save({
        "kind": KIND, "input_schema": schema, "feature_names": names, "component_feature_names": list(component_names),
        "n_alternatives": 3, "hidden_size": architecture["hidden_size"], "dropout": architecture["dropout"],
        "mean": mean, "scale": scale, "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "metadata": {"architecture": architecture, "loss": "direct-track-iou", "epochs": refit_epochs,
                     "balance": balance, "unseen_only": unseen_only, "dataset": str(dataset),
                     "manifest_checksum": str(data["manifest_checksum"]), "metrics": metrics},
    }, artifact)
    np.savez_compressed(output / f"{name}_oof.npz", oof=oof, lodo=lodo_predictions, target=targets,
                        sample_id=data["sample_id"], prompt_index=data["prompt_index"])
    with open(output / f"{name}_training_results.json", "w") as f:
        json.dump({"artifact": str(artifact), "metrics": metrics, "refit_epochs": refit_epochs}, f, indent=2,
                  sort_keys=True, default=float)
        f.write("\n")
    print(json.dumps(metrics["oof"], indent=2, sort_keys=True))
    return artifact


def _metrics(targets: np.ndarray, predictions: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
    if len(targets) == 0:
        return {}
    finite = np.isfinite(predictions)
    targets, predictions, weights = targets[finite], predictions[finite], weights[finite]
    if len(targets) < 2:
        return {"n": int(len(targets))}
    mse = float(np.average((predictions - targets) ** 2, weights=weights))
    correlation = float(np.corrcoef(predictions, targets)[0, 1]) if np.std(predictions) > 0 else 0.0
    return {"n": int(len(targets)), "weighted_mse": mse, "correlation": correlation}


# ----------------------------------------------------------------------------------------------
# the installed scorer


class VolumeCandidateScorer:
    """The `set_multimask_models(volume_candidate_scorer=...)` protocol around a fitted artifact."""

    def __init__(self, state: Dict[str, Any], device: str = "cpu"):
        if state.get("kind") != KIND:
            raise ValueError(f"Expected a {KIND!r} artifact, got {state.get('kind')!r}.")
        self.input_schema = str(state["input_schema"])
        if self.input_schema not in SELECTOR_FEATURE_SCHEMAS:
            raise ValueError(f"Unknown input schema {self.input_schema!r}.")
        self.component_feature_names = tuple(state["component_feature_names"])
        self.feature_names = list(state["feature_names"])
        self.device = torch.device(device)
        self.mean = torch.as_tensor(np.asarray(state["mean"]), dtype=torch.float32, device=self.device)
        self.scale = torch.as_tensor(np.asarray(state["scale"]), dtype=torch.float32, device=self.device)
        n_features = int(self.mean.shape[-1])
        self.model = GroupwiseMLP(n_features, hidden_size=int(state["hidden_size"]), dropout=float(state["dropout"]))
        self.model.load_state_dict(state["state_dict"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict_candidates(self, features: torch.Tensor, component_features: Optional[torch.Tensor]) -> torch.Tensor:
        features = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        if self.component_feature_names:
            if component_features is None:
                raise ValueError("This scorer needs the ladder's component features.")
            components = torch.as_tensor(component_features, dtype=torch.float32, device=self.device)
            components = torch.nan_to_num(components, nan=0.0, posinf=0.0, neginf=0.0)
            features = torch.cat([features, components[:, None, :].expand(-1, features.shape[1], -1)], dim=2)
        normalized = (features - self.mean) / self.scale
        return self.model(normalized).mean(dim=1)


def load_volume_candidate_scorer(path: Path, device: str = "cpu") -> VolumeCandidateScorer:
    state = torch.load(path, map_location="cpu", weights_only=False)
    return VolumeCandidateScorer(state, device=device)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("aggregate", "train"))
    parser.add_argument("--subset", default="primary")
    parser.add_argument("--cache", type=Path, default=None, help="The extractor's cache directory.")
    parser.add_argument("--dataset", type=Path, default=None, help="The aggregated candidates.npz.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, default=CAMPAIGN_ROOT)
    parser.add_argument("--schema", choices=SCHEMAS, default="token_lowres_v1")
    parser.add_argument("--component-features", default="all", help="'all', 'none' or a comma-separated list.")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--balance", choices=("crop", "none"), default="crop")
    parser.add_argument("--no-lodo", action="store_true")
    parser.add_argument("--unseen-only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if args.command == "aggregate":
        manifest = load_manifest(args.subset, args.campaign_root)
        aggregate(args.cache, manifest, args.output)
        return 0
    data = np.load(args.dataset, allow_pickle=False)
    all_components = [str(name) for name in data["component_feature_names"]]
    if args.component_features == "all":
        components = all_components
    elif args.component_features == "none":
        components = []
    else:
        components = [name.strip() for name in args.component_features.split(",") if name.strip()]
    train(args.dataset, args.output, args.schema, components, args.hidden_size, args.dropout, args.device,
          balance=args.balance, lodo=not args.no_lodo, unseen_only=args.unseen_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
