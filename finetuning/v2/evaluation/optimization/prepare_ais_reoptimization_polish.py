"""Generate the bounded second-stage AIS grid from a finished coarse Dice-foreground ranking.

The coarse search captures interactions between seed, watershed and boundary-use parameters. This
script takes the three best rows of every mechanism family and varies one coordinate at a time. Its
output uses the explicit-candidate grid format understood by ``benchmark_ais_optimization.py sweep``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

import benchmark_ais_optimization as ais


def _parameter_row(row: pd.Series) -> Dict[str, object]:
    params = {}
    for key in ais.SPARSE_KEYS:
        if key not in row:
            continue
        value = row[key]
        if pd.isna(value) or (isinstance(value, str) and value.lower() == "none"):
            continue
        # Mixed optional columns (for example ridge and mask controls in the same ranking) are
        # read by pandas as strings because they also contain the sentinel "none". Convert their
        # numeric entries back here so the generated JSON cannot pass string thresholds to numpy.
        if isinstance(value, str):
            if key == "boundary_magnitude_max" and value.lower() == ais.EXPLICIT_OFF:
                value = float("inf")
            else:
                try:
                    value = float(value)
                except ValueError as error:
                    raise ValueError(f"Invalid value {value!r} for AIS parameter '{key}'.") from error
        if isinstance(value, np.generic):
            value = value.item()
        if key in ("n_iter", "min_size"):
            value = int(value)
        elif key == "boundary_magnitude_max" and np.isinf(value):
            value = ais.EXPLICIT_OFF
        params[key] = value
    return params


def _with_values(base: Dict[str, object], key: str, values: Iterable[object]) -> Iterable[Dict[str, object]]:
    for value in values:
        candidate = dict(base)
        candidate[key] = value
        yield candidate


def _edge_iterations(family: pd.DataFrame, margin: float) -> List[int]:
    by_iteration = family.groupby("n_iter")["balanced"].max()
    extra = []
    if 1600 in by_iteration and 1200 in by_iteration and by_iteration[1600] - by_iteration[1200] >= margin:
        extra.append(2400)
    if 400 in by_iteration and 800 in by_iteration and by_iteration[400] - by_iteration[800] >= margin:
        extra.append(200)
    return extra


def polish_combinations(ranking: pd.DataFrame, top_per_family: int = 3, edge_margin: float = 0.001) -> List[Dict]:
    """Return unique one-coordinate refinements of the top coarse rows."""
    if "balanced" not in ranking:
        raise ValueError("The ranking needs a 'balanced' column.")
    if "mechanism_family" not in ranking:
        ranking = ranking.copy()
        ranking["mechanism_family"] = "single-grid"

    candidates = {}
    for _, family in ranking.groupby("mechanism_family", sort=True, dropna=False):
        family = family.sort_values("balanced", ascending=False)
        extra_iterations = _edge_iterations(family, edge_margin)
        for _, row in family.head(top_per_family).iterrows():
            base = _parameter_row(row)
            variants = [base]
            threshold = float(base["foreground_threshold"])
            variants.extend(_with_values(
                base, "foreground_threshold",
                sorted({round(max(0.25, min(0.65, threshold + offset)), 3) for offset in (-0.025, 0, 0.025)}),
            ))
            variants.extend(_with_values(base, "foreground_weight", (0.5, 0.625, 0.75, 0.875, 1.0)))
            variants.extend(_with_values(base, "min_size", (0, 10, 25, 50, 75, 100)))
            variants.extend(_with_values(
                base, "boundary_magnitude_max", (ais.EXPLICIT_OFF, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6),
            ))
            if extra_iterations:
                variants.extend(_with_values(base, "n_iter", extra_iterations))
            if "contact_weight" in base:
                variants.extend(_with_values(base, "contact_weight", (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)))
            if "contact_mask_threshold" in base:
                variants.extend(_with_values(
                    base, "contact_mask_threshold", (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
                ))
            for candidate in variants:
                identity = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
                candidates[identity] = candidate
    return list(candidates.values())


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-per-family", type=int, default=3)
    parser.add_argument("--edge-margin", type=float, default=0.001)
    args = parser.parse_args(argv)
    if args.top_per_family < 1:
        parser.error("--top-per-family must be positive.")
    ranking = pd.read_csv(args.ranking)
    combinations = polish_combinations(ranking, args.top_per_family, args.edge_margin)
    resolved = [ais.resolve_postprocessing({"sparse": combo}, "hvit_t")["sparse"] for combo in combinations]
    cache_keys = ais.SWEEP_CACHE_KEYS["sparse"]
    n_cache_groups = len({tuple(combo[key] for key in cache_keys) for combo in resolved})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"combinations": combinations}, f, indent=2, sort_keys=True)
        f.write("\n")
    print(
        f"Wrote {len(combinations)} polish candidates in {n_cache_groups} flow-cache groups to {args.output}. "
        f"Use no more than {n_cache_groups} sweep shards."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
