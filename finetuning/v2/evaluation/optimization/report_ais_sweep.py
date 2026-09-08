"""Rank the combinations of an AIS parameter sweep as shared defaults under the generalization gate.

Reads the per-dataset CSVs that `benchmark_ais_optimization.py sweep` wrote for one grid on one or more
manifests (e.g. primary and training_extra), joins them over the datasets, and reports for every
combination the balanced mSA, the per-dataset change against a reference combination (the current
library defaults by default), the generalization gate verdict and the mean ratio to each dataset's own
optimum. Not part of the implementation checksum: it only reads results.

Usage:
    python report_ais_sweep.py --grid configs/ais_grid_lm_v4.json --subset primary training_extra \\
        --output <root>/ais/reports/a1_sweep_dev.csv --top 25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

OPTIMIZATION_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(OPTIMIZATION_ROOT.parent))

from optimization import benchmark_ais_optimization as ais  # noqa


def load_sweep_tables(
    grid_path: Path, subsets: Sequence[str], output_root: Path, data_root: Path, campaign_root: Path,
    model_type: str, joint_checkpoint: str, datasets: Optional[Sequence[str]] = None, kind: str = "v5",
) -> Dict[str, pd.DataFrame]:
    """The per-dataset sweep tables of one grid over the given manifests, keyed by dataset."""
    with open(grid_path) as f:
        grid = json.load(f)
    checkpoint_id = ais._checkpoint_identity(model_type, joint_checkpoint)
    tables: Dict[str, pd.DataFrame] = {}
    for subset in subsets:
        manifest = ais.load_campaign_manifest(kind, subset, output_root, data_root, campaign_root)
        sweep_dir = ais.sweep_dir(output_root, checkpoint_id, manifest["manifest_checksum"], grid_path.stem, grid)
        for path in sorted(sweep_dir.glob("*.csv")):
            if ".shard" in path.name or path.stem in ("shared_config",):
                continue
            if datasets and path.stem not in datasets:
                continue
            if path.stem in tables:
                raise ValueError(
                    f"Dataset '{path.stem}' appears in more than one requested subset for grid '{grid_path.stem}'."
                )
            tables[path.stem] = pd.read_csv(path)
    if not tables:
        sweeps = output_root / ais.CAMPAIGN / "sweeps"
        raise FileNotFoundError(f"No sweep tables for grid '{grid_path.stem}' under {sweeps}.")
    if datasets:
        missing = sorted(set(datasets) - set(tables))
        if missing:
            raise FileNotFoundError(f"Grid '{grid_path.stem}' is missing requested datasets: {missing}.")
    return tables


def _parameter_columns(table: pd.DataFrame) -> List[str]:
    return [c for c in table.columns if not c.endswith(("_mean", "_std")) and c != "n_images"]


def load_sweep_tables_many(
    grid_paths: Sequence[Path], subsets: Sequence[str], output_root: Path, data_root: Path, campaign_root: Path,
    model_type: str, joint_checkpoint: str, datasets: Optional[Sequence[str]] = None, kind: str = "v5",
) -> Dict[str, pd.DataFrame]:
    """Load and union compatible sweep families, preserving the grid name as a categorical parameter."""
    collected: Dict[str, List[pd.DataFrame]] = {}
    parameter_columns = set()
    for grid_path in grid_paths:
        tables = load_sweep_tables(
            grid_path, subsets, output_root, data_root, campaign_root, model_type, joint_checkpoint, datasets, kind,
        )
        for dataset, table in tables.items():
            table = table.copy()
            if "mechanism_family" not in table:
                table["mechanism_family"] = grid_path.stem
            parameter_columns.update(_parameter_columns(table))
            collected.setdefault(dataset, []).append(table)
    if not collected:
        raise FileNotFoundError("No sweep tables were found for the requested grids.")

    combined = {}
    for dataset, parts in collected.items():
        normalized = []
        for table in parts:
            table = table.copy()
            for column in parameter_columns:
                if column not in table:
                    table[column] = "none"
            normalized.append(table)
        combined[dataset] = pd.concat(
            normalized, ignore_index=True, sort=False,
        ).drop_duplicates().reset_index(drop=True)
    return combined


def rank_shared(tables: Dict[str, pd.DataFrame], reference: Optional[Dict[str, object]] = None) -> pd.DataFrame:
    """Join the datasets on the parameter columns and score every combination as a shared default."""
    datasets = sorted(tables)
    keys = _parameter_columns(tables[datasets[0]])
    merged = None
    for dataset in datasets:
        table = tables[dataset][keys + ["msa_mean"]].rename(columns={"msa_mean": dataset}).copy()
        # NaN-safe join key for the optional parameters.
        for key in keys:
            table[key] = table[key].astype(object).where(table[key].notna(), "none")
        if table.duplicated(keys).any():
            raise ValueError(f"Dataset '{dataset}' contains duplicate resolved parameter combinations.")
        merged = table if merged is None else merged.merge(table, on=keys, how="inner")
    if merged is None or merged.empty:
        raise ValueError("The datasets share no combination.")
    scores = merged[datasets].to_numpy(dtype="float64")
    merged["balanced"] = scores.mean(axis=1)
    optimum = scores.max(axis=0)
    merged["mean_relative_optimum"] = (scores / optimum).mean(axis=1)
    merged["min_relative_optimum"] = (scores / optimum).min(axis=1)
    if reference is not None:
        mask = np.ones(len(merged), dtype=bool)
        for key, value in reference.items():
            if key not in keys:
                continue
            column = merged[key]
            wanted = "none" if value is None else value
            mask &= np.array([_same(v, wanted) for v in column])
        if mask.sum() != 1:
            raise ValueError(f"The reference combination matches {int(mask.sum())} rows, expected one: {reference}.")
        base = scores[mask][0]
        relative = scores / np.where(base > 0, base, np.nan) - 1.0
        absolute = scores - base
        merged["balanced_gain"] = merged["balanced"] / base.mean() - 1.0
        merged["n_up"] = (absolute > 0).sum(axis=1)
        merged["worst_relative"] = np.nanmin(relative, axis=1)
        violates = (relative < ais.GATE["max_relative_loss"]) & (absolute < ais.GATE["max_absolute_loss"])
        merged["passed"] = (
            (merged["n_up"] >= len(datasets) - ais.GATE["max_down"]) & ~violates.any(axis=1)
            & (merged["balanced_gain"] >= ais.GATE["min_balanced_gain"])
        )
        for index, dataset in enumerate(datasets):
            merged[f"rel_{dataset}"] = relative[:, index]
    return merged


def select_plateau(ranked: pd.DataFrame, tolerance: float = 0.001) -> pd.Series:
    """Select the robust, cheaper member of the near-optimal balanced-mSA plateau."""
    if ranked.empty:
        raise ValueError("Cannot select from an empty sweep ranking.")
    best = float(ranked["balanced"].max())
    plateau = ranked[ranked["balanced"] >= best - tolerance].copy()

    def numeric_column(name: str) -> pd.Series:
        values = plateau[name] if name in plateau else pd.Series(0, index=plateau.index)
        return pd.to_numeric(values, errors="coerce").fillna(0)

    plateau["_n_iter"] = numeric_column("n_iter")
    contact_weight = numeric_column("contact_weight")
    contact_mask = plateau.get("contact_mask_threshold", pd.Series("none", index=plateau.index))
    plateau["_active_controls"] = (
        (contact_weight != 0).astype(int) + (~contact_mask.isin(("none", None))).astype(int)
    )
    plateau["_contact_weight"] = contact_weight
    return plateau.sort_values(
        ["min_relative_optimum", "_n_iter", "_active_controls", "_contact_weight", "balanced"],
        ascending=[False, True, True, True, False],
    ).iloc[0]


def selected_config(row: pd.Series, name: str) -> Dict[str, object]:
    """Turn one ranked sweep row into a run-compatible sparse configuration."""
    params = {}
    for key in ais.SPARSE_KEYS:
        if key not in row or row[key] == "none" or pd.isna(row[key]):
            continue
        value = row[key]
        if isinstance(value, np.generic):
            value = value.item()
        if key in ("n_iter", "min_size"):
            value = int(value)
        elif key == "boundary_magnitude_max" and np.isinf(value):
            value = ais.EXPLICIT_OFF
        params[key] = value
    return {"name": name, "mode": "sparse", "params_2d": {"sparse": params}}


def _same(a: object, b: object) -> bool:
    try:
        return bool(np.isclose(float(a), float(b)))
    except (TypeError, ValueError):
        return str(a) == str(b)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", type=Path, nargs="+", required=True)
    parser.add_argument("--kind", choices=ais.KINDS, default="v5", help="Manifest family the sweep ran on.")
    parser.add_argument("--subset", nargs="+", default=["primary", "training_extra"])
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--data-root", type=Path, default=ais.DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=ais.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--campaign-root", type=Path, default=ais.apg3d_manifest.CAMPAIGN_ROOT)
    parser.add_argument("--model-type", default="hvit_t")
    parser.add_argument("--joint-checkpoint", default="best")
    parser.add_argument("--no-reference", action="store_true", help="Do not compare against the library defaults.")
    parser.add_argument("--sort", choices=("balanced", "mean_relative_optimum", "balanced_gain"), default="balanced")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--select-config", type=Path, default=None,
                        help="Write the plateau-selected row as a run-compatible configuration.")
    parser.add_argument("--config-name", default="dice-reoptimized")
    parser.add_argument("--plateau-tolerance", type=float, default=0.001)
    args = parser.parse_args(argv)

    tables = load_sweep_tables_many(
        args.grid, args.subset, args.output_root.resolve(), args.data_root.resolve(), args.campaign_root,
        args.model_type, args.joint_checkpoint, args.datasets, kind=args.kind,
    )
    reference = None if args.no_reference else ais.default_postprocessing(args.model_type, "sparse")
    ranked = rank_shared(tables, reference)
    order = [args.sort] + (["passed"] if "passed" in ranked else [])
    ranked = ranked.sort_values(order, ascending=False).reset_index(drop=True)
    keys = _parameter_columns(tables[sorted(tables)[0]])
    shown = keys + ["balanced", "mean_relative_optimum", "min_relative_optimum"]
    if "passed" in ranked:
        shown += ["balanced_gain", "n_up", "worst_relative", "passed"]
        print(f"{int(ranked['passed'].sum())} of {len(ranked)} combinations pass the gate against the defaults.")
    pd.set_option("display.width", 250)
    print(ranked[shown].head(args.top).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if "passed" in ranked and ranked["passed"].any():
        best = ranked[ranked["passed"]].sort_values("balanced", ascending=False).iloc[0]
        print("\nBest passing combination:", {k: best[k] for k in keys})
        print("Per-dataset change:", {d: f"{100 * best[f'rel_{d}']:+.1f}%" for d in sorted(tables)})
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(args.output, index=False)
        print(f"\nRanking: {args.output}")
    if args.select_config is not None:
        selected = select_plateau(ranked, args.plateau_tolerance)
        config = selected_config(selected, args.config_name)
        args.select_config.parent.mkdir(parents=True, exist_ok=True)
        with open(args.select_config, "w") as f:
            json.dump(config, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"Selected configuration: {args.select_config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
