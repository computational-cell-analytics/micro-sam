"""Read the refinement screens of the 2026-09 campaign: gate table, identity checks and cost columns.

`screen_apg_refinement.py` writes one run directory per (manifest, configuration list, checkpoint). This reader
joins the run directories of one checkpoint, applies the campaign rule of `screen_apg_structural.gate_table`
(most datasets up, no dataset below the minor-regression line, balanced gain over the bar) against the `none`
control, checks two identities image by image - the `none` entry against the canonical registry benchmark of the
same checkpoint and the `pb` entry against the canonical `apg_s_refine_pb` run - and reports, per dataset and
configuration, how many instances took a full-prompt second pass, a box-only one, or none.

Usage:
    python report_refinement_screen.py <run dir> [<run dir> ...] [--pb-config-name s-refine-pb]
    python report_refinement_screen.py --latest --subsets primary training_extra   # newest run per subset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

OPTIMIZATION_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(OPTIMIZATION_ROOT))
sys.path.insert(0, str(OPTIMIZATION_ROOT.parent))

import common  # noqa
from benchmark_apg_optimization import DEFAULT_OUTPUT_ROOT, _atomic_write_csv, _atomic_write_json  # noqa
from screen_apg_structural import (  # noqa
    MODEL_TYPE, CHECKPOINT, find_reference_run, gate_table, identity_check, structural_root,
)

CONTROL = "none"
COST_COLUMNS = (
    "refinement_eligible_instances", "refined_instances", "refinement_fallback_instances",
    "refinement_isolated_instances", "replaced_instances", "refinement_protected_pixels", "refinement_negatives",
    "gated_consistency", "gated_foreign",
)


def latest_screen_runs(output_root: Path, subsets: Sequence[str], checkpoint: str) -> List[Path]:
    """The newest complete refinement screen per subset for one checkpoint."""
    root = output_root / "refinement_screening" / MODEL_TYPE / checkpoint
    chosen = []
    for subset in subsets:
        candidates = []
        for metadata_path in root.glob("*/metadata.json"):
            metadata = json.load(open(metadata_path))
            # The screen writes its summary last, so its presence is the completion marker.
            if metadata.get("subset") == subset and (metadata_path.parent / "summary.csv").exists():
                candidates.append((metadata_path.stat().st_mtime, metadata_path.parent))
        if not candidates:
            raise SystemExit(f"No complete refinement screen for subset '{subset}' under {root}.")
        chosen.append(max(candidates)[1])
    return chosen


def find_candidate_run(output_root: Path, manifest_checksum: str, checkpoint: str, config_name: str) -> Optional[Path]:
    """The newest complete canonical benchmark run of a named configuration on one manifest and checkpoint."""
    matches = []
    for metadata_path in (output_root / MODEL_TYPE / checkpoint).glob(f"{manifest_checksum}-*/metadata.json"):
        metadata = json.load(open(metadata_path))
        if metadata.get("status") == "complete" and metadata.get("config_name") == config_name:
            matches.append((metadata_path.stat().st_mtime, metadata_path.parent))
    return max(matches)[1] if matches else None


def load_screens(run_dirs: Sequence[Path]) -> tuple:
    tables, checkpoints, manifests = [], set(), {}
    for run_dir in run_dirs:
        metadata = json.load(open(Path(run_dir) / "metadata.json"))
        samples = pd.read_csv(Path(run_dir) / "samples.csv").rename(columns={"config_name": "variant"})
        samples["subset"] = metadata.get("subset", "?")
        tables.append(samples)
        checkpoints.add(metadata["checkpoint_checksum"])
        manifests[metadata.get("subset", "?")] = metadata["manifest_checksum"]
    if len(checkpoints) != 1:
        raise SystemExit(f"The screens come from different checkpoints: {sorted(checkpoints)}.")
    return pd.concat(tables, ignore_index=True), next(iter(checkpoints)), manifests


def summarize(samples: pd.DataFrame) -> pd.DataFrame:
    """Per variant and dataset: mean mSA, its std, and the summed instance counts; plus a balanced row."""
    sums = [column for column in COST_COLUMNS if column in samples.columns] + ["predicted_objects"]
    parts = []
    for variant, frame in samples.groupby("variant", sort=False):
        table = frame.groupby("dataset", sort=True).agg(
            n_samples=("sample_id", "count"), msa_mean=("msa", "mean"), msa_std=("msa", "std"),
            select_seconds=("select_seconds", "sum"), **{column: (column, "sum") for column in sums},
        ).reset_index()
        table.insert(0, "variant", variant)
        parts.append(table)
        parts.append(pd.DataFrame([{
            "variant": variant, "dataset": "__dataset_balanced__", "n_samples": int(len(frame)),
            "msa_mean": float(table["msa_mean"].mean()), "msa_std": float("nan"),
            "select_seconds": float(table["select_seconds"].sum()),
            **{column: int(table[column].sum()) for column in sums},
        }]))
    return pd.concat(parts, ignore_index=True)


def cost_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Second-pass forwards per dataset and variant, as fractions of the eligible instances."""
    rows = summary[summary["dataset"] != "__dataset_balanced__"].copy()
    eligible = rows["refinement_eligible_instances"].replace(0, float("nan"))
    rows["full_prompt_fraction"] = (rows["refined_instances"] - rows["refinement_fallback_instances"]) / eligible
    rows["box_only_fraction"] = rows["refinement_fallback_instances"] / eligible
    rows["isolated_fraction"] = rows["refinement_isolated_instances"] / eligible
    rows["replaced_fraction"] = rows["replaced_instances"] / eligible
    rows["negatives_per_instance"] = rows["refinement_negatives"] / eligible
    columns = [
        "variant", "dataset", "msa_mean", "full_prompt_fraction", "box_only_fraction", "isolated_fraction",
        "replaced_fraction", "negatives_per_instance", "refinement_protected_pixels", "gated_consistency",
        "gated_foreign", "select_seconds",
    ]
    return rows[columns]


def report(run_dirs: Sequence[Path], output_root: Path, pb_config_name: str) -> Path:
    samples, checkpoint, manifests = load_screens(run_dirs)
    summary = summarize(samples)
    gates = gate_table(summary, control=CONTROL)
    per_dataset = summary[summary["dataset"] != "__dataset_balanced__"].pivot(
        index="dataset", columns="variant", values="msa_mean",
    )
    relative = per_dataset.sub(per_dataset[CONTROL], axis=0).div(per_dataset[CONTROL], axis=0)
    identities: Dict[str, dict] = {}
    for subset, manifest_checksum in manifests.items():
        subset_samples = samples[samples["subset"] == subset]
        registry = find_reference_run(output_root, manifest_checksum, checkpoint_checksum=checkpoint)
        if registry is not None:
            identities[f"{subset}:none-vs-registry"] = {
                "reference_run": str(registry),
                **identity_check(
                    subset_samples.assign(variant=subset_samples["variant"].where(
                        subset_samples["variant"] != CONTROL, "registry",
                    )),
                    pd.read_csv(registry / "samples.csv"),
                ),
            }
        pb_run = find_candidate_run(output_root, manifest_checksum, checkpoint, pb_config_name)
        if pb_run is not None and (subset_samples["variant"] == "pb").any():
            identities[f"{subset}:pb-vs-canonical"] = {
                "reference_run": str(pb_run),
                **identity_check(
                    subset_samples.assign(variant=subset_samples["variant"].where(
                        subset_samples["variant"] != "pb", "registry",
                    )),
                    pd.read_csv(pb_run / "samples.csv"),
                ),
            }
    out_dir = structural_root(output_root) / "refinement_reports" / checkpoint / "+".join(sorted(manifests))
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(out_dir / "summary.csv", summary)
    _atomic_write_csv(out_dir / "gates.csv", gates)
    _atomic_write_csv(out_dir / "per_dataset_msa.csv", per_dataset.reset_index())
    _atomic_write_csv(out_dir / "per_dataset_relative.csv", relative.reset_index())
    _atomic_write_csv(out_dir / "costs.csv", cost_table(summary))
    _atomic_write_json(out_dir / "identity.json", identities)
    pd.set_option("display.width", 250)
    print("Identity checks (per image):")
    print(json.dumps(identities, indent=2))
    print("Relative mSA change vs the control (%):")
    print((relative.drop(columns=[CONTROL]) * 100).round(2).to_string())
    print(gates.round(4).to_string(index=False))
    print(f"Report: {out_dir}")
    return out_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="*", type=Path)
    parser.add_argument("--latest", action="store_true", help="Newest complete screen per subset, current checkpoint.")
    parser.add_argument("--subsets", nargs="+", default=("primary", "training_extra"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pb-config-name", default="s-refine-pb")
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_dirs = list(args.run_dirs)
    if args.latest:
        checkpoint = common.checkpoint_checksum(common.get_joint_checkpoint(MODEL_TYPE, CHECKPOINT))
        run_dirs.extend(latest_screen_runs(args.output_root, args.subsets, checkpoint))
    if not run_dirs:
        parser.error("Give run directories or --latest.")
    report(run_dirs, args.output_root, args.pb_config_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
