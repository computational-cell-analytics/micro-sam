"""Paired, hierarchical comparison of Dice-foreground baseline and boundary AIS checkpoints.

Each checkpoint must be evaluated with its own development-selected post-processing configuration on
the exact same manifest. The report balances acquisition strata within a dataset, balances datasets
in the macro score, and resamples datasets and paired images for its confidence interval. It also
audits the sealed OOD paths against the two decoder-training manifests.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

OPTIMIZATION_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(OPTIMIZATION_ROOT))

import benchmark_ais_optimization as ais  # noqa
import benchmark_apg_optimization as apg  # noqa


DEFAULT_TRAINING_ROOT = (
    Path("/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/apg_optimization")
    / "ais_decoder_training/checkpoints"
)
DEFAULT_TRAINING_MANIFESTS = (
    DEFAULT_TRAINING_ROOT / "ais_decoder_baseline/data_manifest.json",
    DEFAULT_TRAINING_ROOT / "ais_decoder_boundary/data_manifest.json",
)
FATE_COLUMNS = (
    "matched", "unmatched", "genuine_misses", "gt_with_0_seeds", "gt_with_1_seed",
    "gt_with_2plus_seeds", "seeded_unmatched", "seeded_split", "seeded_merged",
    "seeded_undersized", "seeded_oversized", "unseeded_absorbed", "unseeded_missing",
)
OTHER_COUNT_COLUMNS = ("predicted_objects", "n_seeds", "background_seeds", "pipeline_mismatch")
EXTENT_COLUMNS = ("fg_iou", "fg_area_ratio", "matched_iou")
TIME_COLUMNS = ("initialization_seconds", "generation_seconds", "total_seconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")


def _atomic_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(temporary, "w") as f:
        json.dump(value, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")
    os.replace(temporary, path)


def _atomic_csv(path: Path, value: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    value.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _single(values: Iterable[Any], description: str) -> Any:
    unique = {json.dumps(value, sort_keys=True, default=_json_default): value for value in values}
    if len(unique) != 1:
        raise ValueError(f"Expected one {description}, found {len(unique)} distinct values.")
    return next(iter(unique.values()))


def load_side(run_dirs: Sequence[Path], side: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Load compatible complete runs belonging to one checkpoint/configuration."""
    if not run_dirs:
        raise ValueError(f"No {side} run directories were supplied.")
    loaded = [(path.resolve(strict=True), *ais.load_run(path.resolve(strict=True))) for path in run_dirs]
    metadata = [entry[1] for entry in loaded]
    for field in (
        "config_name", "config_checksum", "checkpoint_checksum", "implementation_checksum", "model_type",
        "params_2d", "dimensions",
    ):
        _single((item.get(field) for item in metadata), f"{side} {field}")
    if any(item.get("mode") not in ("sparse", "auto") for item in metadata):
        raise ValueError(f"Every {side} run must use sparse AIS post-processing.")
    if any(item.get("dimensions") != [2] for item in metadata):
        raise ValueError(f"Every {side} run must be 2d-only (dimensions=[2]).")
    if len({item.get("manifest_checksum") for item in metadata}) != len(metadata):
        raise ValueError(f"The {side} inputs contain duplicate runs for a manifest.")

    samples = pd.concat([entry[2] for entry in loaded], ignore_index=True)
    if samples["sample_id"].duplicated().any():
        duplicated = samples.loc[samples["sample_id"].duplicated(), "sample_id"].tolist()
        raise ValueError(f"The {side} runs contain duplicate sample ids: {duplicated[:5]}.")
    if not (samples["ndim"] == 2).all():
        raise ValueError(f"The {side} sample table contains non-2d rows.")
    if "stratum" not in samples:
        samples["stratum"] = ""
    samples["stratum"] = samples["stratum"].fillna("").astype(str)
    summary = {
        "side": side,
        "run_dirs": [str(entry[0]) for entry in loaded],
        "config_name": _single((item["config_name"] for item in metadata), f"{side} config name"),
        "config_checksum": _single((item["config_checksum"] for item in metadata), f"{side} config checksum"),
        "checkpoint_checksum": _single(
            (item["checkpoint_checksum"] for item in metadata), f"{side} checkpoint checksum",
        ),
        "checkpoint_name": _single((item.get("checkpoint_name") for item in metadata), f"{side} checkpoint name"),
        "implementation_checksum": _single(
            (item["implementation_checksum"] for item in metadata), f"{side} implementation checksum",
        ),
        "model_type": _single((item["model_type"] for item in metadata), f"{side} model type"),
        "params_2d": _single((item["params_2d"] for item in metadata), f"{side} 2d parameters"),
        "manifest_checksums": sorted(item["manifest_checksum"] for item in metadata),
        "subsets": sorted(str(item.get("subset")) for item in metadata),
        "hardware": _single((item.get("hardware", {}) for item in metadata), f"{side} prediction hardware"),
        "postprocessing_hardware": _single(
            (item.get("postprocessing_hardware", {}) for item in metadata),
            f"{side} post-processing hardware",
        ),
    }
    return summary, samples


def pair_samples(baseline: pd.DataFrame, boundary: pd.DataFrame) -> pd.DataFrame:
    """Pair the exact same source samples, retaining metrics and diagnostics from both sides."""
    identity = ["sample_id", "dataset", "stratum"]
    base_ids = set(map(tuple, baseline[identity].itertuples(index=False, name=None)))
    boundary_ids = set(map(tuple, boundary[identity].itertuples(index=False, name=None)))
    if base_ids != boundary_ids:
        raise ValueError(
            "Baseline and boundary runs do not contain the same samples: "
            f"{len(base_ids - boundary_ids)} baseline-only, {len(boundary_ids - base_ids)} boundary-only."
        )
    required = ["msa", *EXTENT_COLUMNS, *TIME_COLUMNS, "gt_objects", *FATE_COLUMNS, *OTHER_COUNT_COLUMNS]
    missing = [column for column in required if column not in baseline or column not in boundary]
    if missing:
        raise ValueError(
            f"Both runs must include full AIS diagnostics; missing shared columns: {missing}. "
            "Do not use --no-diagnostics for the confirmation runs."
        )
    paired = baseline[identity + required].merge(
        boundary[identity + required], on=identity, how="inner", validate="one_to_one",
        suffixes=("_baseline", "_boundary"),
    )
    if paired[["msa_baseline", "msa_boundary"]].isna().any().any():
        raise ValueError("Paired mSA values must all be finite.")
    if not np.array_equal(paired["gt_objects_baseline"], paired["gt_objects_boundary"]):
        raise ValueError("The two runs disagree on ground-truth object counts.")
    return paired.sort_values(identity).reset_index(drop=True)


def validate_manifest_coverage(paired: pd.DataFrame, manifest: Dict[str, Any]) -> None:
    """Require the paired table to cover every sealed manifest sample, with matching domain metadata."""
    identity = ("sample_id", "dataset", "stratum")
    expected = {
        (sample["sample_id"], sample["dataset"], str(sample.get("stratum", "")))
        for sample in manifest["samples"]
    }
    actual = set(map(tuple, paired[list(identity)].itertuples(index=False, name=None)))
    if actual != expected:
        raise ValueError(
            "The paired runs do not exactly cover the sealed manifest: "
            f"{len(expected - actual)} missing and {len(actual - expected)} unexpected sample identities."
        )


def _strata(group: pd.DataFrame) -> List[pd.DataFrame]:
    if (group["stratum"] != "").any():
        if (group["stratum"] == "").any():
            raise ValueError(f"Dataset '{group['dataset'].iloc[0]}' mixes declared and missing strata.")
        return [part for _, part in group.groupby("stratum", sort=True)]
    return [group]


def balanced_scores(group: pd.DataFrame) -> Tuple[float, float]:
    """Return baseline/boundary mSA, giving declared strata equal weight."""
    scores = np.asarray([
        [part["msa_baseline"].mean(), part["msa_boundary"].mean()] for part in _strata(group)
    ])
    return float(scores[:, 0].mean()), float(scores[:, 1].mean())


def _balanced_column(group: pd.DataFrame, column: str) -> float:
    values = np.asarray([part[column].dropna().mean() for part in _strata(group)], dtype="float64")
    return float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else np.nan


def _stratified_bootstrap(group: pd.DataFrame, n_bootstrap: int, rng: np.random.Generator) -> np.ndarray:
    scores = np.zeros((n_bootstrap, 2), dtype="float64")
    strata = _strata(group)
    for part in strata:
        values = part[["msa_baseline", "msa_boundary"]].to_numpy(dtype="float64")
        indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
        scores += values[indices].mean(axis=1)
    return scores / len(strata)


def bootstrap(
    paired: pd.DataFrame, n_bootstrap: int = 20_000, seed: int = 0,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Paired hierarchical bootstrap over domains and images within each domain/stratum."""
    if n_bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replicates.")
    rng = np.random.default_rng(seed)
    datasets = sorted(paired["dataset"].unique())
    domain_samples = np.stack([
        _stratified_bootstrap(paired[paired["dataset"] == dataset], n_bootstrap, rng)
        for dataset in datasets
    ], axis=1)
    domain_indices = rng.integers(0, len(datasets), size=(n_bootstrap, len(datasets)))
    rows = np.arange(n_bootstrap)[:, None]
    macro = domain_samples[rows, domain_indices].mean(axis=1)

    def interval(values: np.ndarray) -> Tuple[float, float]:
        values = values[np.isfinite(values)]
        if not len(values):
            return np.nan, np.nan
        low, high = np.quantile(values, (0.025, 0.975))
        return float(low), float(high)

    delta = macro[:, 1] - macro[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = macro[:, 1] / macro[:, 0] - 1.0
    delta_low, delta_high = interval(delta)
    relative_low, relative_high = interval(relative)
    overall = {
        "absolute_ci_low": delta_low,
        "absolute_ci_high": delta_high,
        "relative_ci_low": relative_low,
        "relative_ci_high": relative_high,
        "probability_boundary_better": float((delta > 0).mean()),
    }
    by_dataset = {}
    for index, dataset in enumerate(datasets):
        sample = domain_samples[:, index]
        domain_delta = sample[:, 1] - sample[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            domain_relative = sample[:, 1] / sample[:, 0] - 1.0
        low, high = interval(domain_delta)
        rel_low, rel_high = interval(domain_relative)
        by_dataset[dataset] = {
            "absolute_ci_low": low, "absolute_ci_high": high,
            "relative_ci_low": rel_low, "relative_ci_high": rel_high,
            "probability_boundary_better": float((domain_delta > 0).mean()),
        }
    return overall, by_dataset


def dataset_table(paired: pd.DataFrame, intervals: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    count_columns = [column for column in (*FATE_COLUMNS, *OTHER_COUNT_COLUMNS) if f"{column}_baseline" in paired]
    for dataset, group in paired.groupby("dataset", sort=True):
        baseline, boundary = balanced_scores(group)
        relative = boundary / baseline - 1.0 if baseline else np.nan
        row: Dict[str, Any] = {
            "dataset": dataset,
            "n_samples": int(len(group)),
            "n_strata": len(_strata(group)),
            "baseline_msa": baseline,
            "boundary_msa": boundary,
            "absolute_delta": boundary - baseline,
            "relative_gain": relative,
            "improved": bool(boundary > baseline),
            "material_loss": bool(
                boundary - baseline < ais.GATE["max_absolute_loss"]
                and relative < ais.GATE["max_relative_loss"]
            ),
            **intervals[dataset],
        }
        for column in EXTENT_COLUMNS:
            base = _balanced_column(group, f"{column}_baseline")
            candidate = _balanced_column(group, f"{column}_boundary")
            row[f"baseline_{column}"] = base
            row[f"boundary_{column}"] = candidate
            row[f"delta_{column}"] = candidate - base
        for column in TIME_COLUMNS:
            row[f"baseline_{column}"] = float(group[f"{column}_baseline"].sum())
            row[f"boundary_{column}"] = float(group[f"{column}_boundary"].sum())
        gt_total = float(group.get("gt_objects_baseline", pd.Series(dtype=float)).sum())
        for column in count_columns:
            base = float(group[f"{column}_baseline"].sum())
            candidate = float(group[f"{column}_boundary"].sum())
            row[f"baseline_{column}"] = base
            row[f"boundary_{column}"] = candidate
            row[f"delta_{column}"] = candidate - base
            if column in FATE_COLUMNS and gt_total:
                row[f"delta_{column}_per_gt"] = (candidate - base) / gt_total
        rows.append(row)
    return pd.DataFrame(rows)


def audit_training_disjointness(
    manifest: Dict[str, Any], data_root: Path, training_manifest_paths: Sequence[Path],
) -> Dict[str, Any]:
    """Audit both dataset identities and resolved source paths against decoder training manifests."""
    ood_datasets = {sample["dataset"] for sample in manifest["samples"]}
    ood_paths = {
        (data_root / sample[key]).resolve()
        for sample in manifest["samples"]
        for key in ("raw_path", "label_path")
    }
    training_datasets, training_paths = set(), set()
    manifests = []
    for path in training_manifest_paths:
        path = path.resolve(strict=True)
        with open(path) as f:
            record = json.load(f)
        datasets = record.get("datasets")
        if not isinstance(datasets, dict):
            raise ValueError(f"Training manifest '{path}' has no dataset mapping.")
        training_datasets.update(datasets)
        for splits in datasets.values():
            if not isinstance(splits, dict):
                continue
            for paths in splits.values():
                if isinstance(paths, list):
                    training_paths.update(
                        Path(value).expanduser().resolve() for value in paths if isinstance(value, str)
                    )
        manifests.append({"path": str(path), "variant": record.get("variant"), "n_datasets": len(datasets)})
    variants = {record["variant"] for record in manifests}
    if variants != {"baseline", "boundary"}:
        raise ValueError(
            "The disjointness audit needs the Dice-foreground baseline and boundary training manifests; "
            f"found variants {sorted(variants)}."
        )
    dataset_overlap = sorted(
        dataset for dataset in ood_datasets
        if any(name == dataset or name.startswith(f"{dataset}_") for name in training_datasets)
    )
    path_overlap = sorted(map(str, ood_paths & training_paths))
    if dataset_overlap or path_overlap:
        raise RuntimeError(
            f"The OOD set overlaps decoder training: datasets={dataset_overlap}, paths={path_overlap[:5]}."
        )
    return {
        "passed": True,
        "ood_datasets": sorted(ood_datasets),
        "training_manifests": manifests,
        "dataset_overlap": dataset_overlap,
        "path_overlap": path_overlap,
    }


def compare(
    baseline_runs: Sequence[Path], boundary_runs: Sequence[Path], manifest_path: Path,
    training_manifest_paths: Sequence[Path], n_bootstrap: int = 20_000, seed: int = 0,
    expected_subset: str = "ood_extended", expected_baseline_config: str = "baseline-dice-optimum",
    expected_boundary_config: str = "boundary-dice-optimum",
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    baseline_meta, baseline = load_side(baseline_runs, "baseline")
    boundary_meta, boundary = load_side(boundary_runs, "boundary")
    if baseline_meta["checkpoint_checksum"] == boundary_meta["checkpoint_checksum"]:
        raise ValueError("Baseline and boundary runs unexpectedly use the same checkpoint.")
    for field in ("implementation_checksum", "model_type", "manifest_checksums", "subsets"):
        if baseline_meta[field] != boundary_meta[field]:
            raise ValueError(f"Baseline and boundary {field} differ.")
    if baseline_meta["subsets"] != [expected_subset]:
        raise ValueError(
            f"Expected only subset '{expected_subset}', found {baseline_meta['subsets']}."
        )
    expected_configs = {"baseline": expected_baseline_config, "boundary": expected_boundary_config}
    actual_configs = {
        "baseline": baseline_meta["config_name"], "boundary": boundary_meta["config_name"],
    }
    if actual_configs != expected_configs:
        raise ValueError(
            "The confirmation must compare each checkpoint at its own frozen optimum: "
            f"expected {expected_configs}, found {actual_configs}."
        )

    with open(manifest_path.resolve(strict=True)) as f:
        manifest = json.load(f)
    data_root = Path(manifest["data_root"]).resolve(strict=True)
    apg._validate_manifest(manifest, data_root, "standard", expected_subset)
    if baseline_meta["manifest_checksums"] != [manifest["manifest_checksum"]]:
        raise ValueError("Run metadata does not match the supplied manifest checksum.")

    disjointness = audit_training_disjointness(manifest, data_root, training_manifest_paths)
    paired = pair_samples(baseline, boundary)
    validate_manifest_coverage(paired, manifest)
    overall_interval, dataset_intervals = bootstrap(paired, n_bootstrap, seed)
    domains = dataset_table(paired, dataset_intervals)
    baseline_macro = float(domains["baseline_msa"].mean())
    boundary_macro = float(domains["boundary_msa"].mean())
    relative_gain = boundary_macro / baseline_macro - 1.0 if baseline_macro else np.nan
    checks = {
        "ci_excludes_zero": bool(overall_interval["absolute_ci_low"] > 0),
        "at_least_four_of_five_domains_improve": bool(
            len(domains) == 5 and int(domains["improved"].sum()) >= 4
        ),
        "no_material_domain_loss": bool(not domains["material_loss"].any()),
        "relative_macro_gain_at_least_two_percent": bool(relative_gain >= ais.GATE["min_balanced_gain"]),
    }
    timing_comparable = (
        baseline_meta["hardware"] == boundary_meta["hardware"]
        and baseline_meta["postprocessing_hardware"] == boundary_meta["postprocessing_hardware"]
    )
    report = {
        "comparison": (
            "Dice-foreground boundary checkpoint at its own optimum vs Dice-foreground baseline at its own optimum"
        ),
        "baseline": baseline_meta,
        "boundary": boundary_meta,
        "manifest": str(manifest_path.resolve()),
        "manifest_checksum": manifest["manifest_checksum"],
        "n_samples": int(len(paired)),
        "n_domains": int(len(domains)),
        "domain_weighting": "equal; acquisition strata are equal-weight within each declared domain",
        "baseline_macro_msa": baseline_macro,
        "boundary_macro_msa": boundary_macro,
        "absolute_delta": boundary_macro - baseline_macro,
        "relative_gain": relative_gain,
        "bootstrap": {"replicates": n_bootstrap, "seed": seed, **overall_interval},
        "claim_checks": checks,
        "strong_improvement_claim_supported": bool(all(checks.values())),
        "timing_comparable": timing_comparable,
        "timing_seconds": {
            side: {column: float(paired[f"{column}_{side}"].sum()) for column in TIME_COLUMNS}
            for side in ("baseline", "boundary")
        },
        "disjointness_audit": disjointness,
        "caveats": ["microbeSEG has only two official manual test images and is a stress-test domain."],
    }
    return report, domains, paired


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--boundary-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--training-manifests", type=Path, nargs="+", default=list(DEFAULT_TRAINING_MANIFESTS))
    parser.add_argument("--expected-subset", default="ood_extended")
    parser.add_argument("--baseline-config-name", default="baseline-dice-optimum")
    parser.add_argument("--boundary-config-name", default="boundary-dice-optimum")
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True, help="JSON path; CSV tables are written beside it.")
    args = parser.parse_args(argv)
    report, domains, paired = compare(
        args.baseline_runs, args.boundary_runs, args.manifest, args.training_manifests,
        args.bootstrap, args.seed, args.expected_subset, args.baseline_config_name, args.boundary_config_name,
    )
    _atomic_json(args.output, report)
    _atomic_csv(args.output.with_name(f"{args.output.stem}_domains.csv"), domains)
    _atomic_csv(args.output.with_name(f"{args.output.stem}_paired_samples.csv"), paired)
    print(domains[[
        "dataset", "n_samples", "n_strata", "baseline_msa", "boundary_msa", "absolute_delta",
        "relative_gain", "absolute_ci_low", "absolute_ci_high", "material_loss",
    ]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(
        f"\nMacro mSA: {report['baseline_macro_msa']:.4f} -> {report['boundary_macro_msa']:.4f} "
        f"({100 * report['relative_gain']:+.2f}%); 95% paired hierarchical CI "
        f"[{report['bootstrap']['absolute_ci_low']:+.4f}, {report['bootstrap']['absolute_ci_high']:+.4f}]."
    )
    print(f"Strong improvement claim supported: {report['strong_improvement_claim_supported']}")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
