"""Select and package 3d crops for visual inspection in napari.

Reads the per-crop results of the 3d benchmark (`benchmark_apg_3d.py run --save-outputs`) for two checkpoints
(joint/v2 and joint/v4 geodesic) and two configurations (volume defaults, `points+boxes` refinement), ranks the
crops of every dataset by (a) the refinement's effect on v4 and (b) the checkpoint's effect with the defaults, and
writes one HDF5 file per selected crop with the raw volume, the ground truth and the four segmentations, plus a
`cases.csv` index. Outputs written on the `apg-optim-fable` branch also carry the anchors of each run (all
proposed, the scored ones, the merged ones); they are packaged when present, the current runner does not record
them. Open a file with `view_apg3d_cases.py <file.h5>`.

Usage:
    python package_apg3d_cases.py --subset primary --n 1
    python package_apg3d_cases.py --subset primary --n 2 --out /some/where
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np
import pandas as pd

EVALUATION_ROOT = Path(__file__).resolve().parent.parent
OPTIMIZATION_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALUATION_ROOT))
sys.path.insert(0, str(OPTIMIZATION_ROOT))

DEFAULT_OUTPUT_ROOT = Path("/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/apg_optimization")
CHECKPOINTS = {
    # name -> (campaign root, checkpoint root env value or None)
    "v2": (DEFAULT_OUTPUT_ROOT / "3d_v2", None),
    "v4": (DEFAULT_OUTPUT_ROOT / "3d_v4geo", DEFAULT_OUTPUT_ROOT / "v4_geodesic_checkpoints"),
}
CONFIGS = {
    "defaults": OPTIMIZATION_ROOT / "configs" / "apg3d_defaults.json",
    "refine": OPTIMIZATION_ROOT / "configs" / "apg3d_refine_points_boxes.json",
}
SCORE_KEYS = (
    "msa", "gt_objects", "predicted_objects", "merged", "genuine_misses", "propagation_passes", "total_seconds",
)


def load_run(checkpoint: str, config: str, subset: str) -> tuple:
    """The run directory and the per-crop table of one (checkpoint, configuration) on a subset.

    Like `benchmark_apg_3d.aggregate`, the crops are read from the run directory and its siblings under
    other implementation checksums, the current implementation winning when a crop was run under both.
    """
    from benchmark_apg_3d import load_volume_config, run_dir, sibling_run_dirs

    campaign_root, checkpoint_root = CHECKPOINTS[checkpoint]
    if checkpoint_root is not None:
        os.environ["MICRO_SAM2_JOINT_CHECKPOINT_ROOT"] = str(checkpoint_root)
    else:
        os.environ.pop("MICRO_SAM2_JOINT_CHECKPOINT_ROOT", None)
    config_name, params_3d = load_volume_config(CONFIGS[config])
    path = run_dir(campaign_root, subset, config_name, params_3d)
    rows: Dict[str, dict] = {}
    for sibling in sibling_run_dirs(path):
        for crop in sorted((sibling / "crops").glob("*.json")):
            row = json.load(open(crop))
            if row["sample_id"] not in rows or sibling == path:
                rows[row["sample_id"]] = row
    if not rows:
        raise SystemExit(f"No crop results under {path} or its siblings.")
    table = pd.DataFrame(list(rows.values())).set_index("sample_id")
    return path, table


def _output_path(run_path: Path, stem: str) -> Optional[Path]:
    """The saved outputs of one crop, from the run directory or the sibling that holds them."""
    from benchmark_apg_3d import sibling_run_dirs

    for candidate in (run_path, *sibling_run_dirs(run_path)):
        output = candidate / "outputs" / f"{stem}.npz"
        if output.exists():
            return output
    return None


def select_cases(tables: Dict[tuple, pd.DataFrame], n: int) -> pd.DataFrame:
    """Per dataset: the n largest and smallest refinement effects on v4, and v4-vs-v2 defaults effects."""
    v4_def, v4_ref, v2_def = tables[("v4", "defaults")], tables[("v4", "refine")], tables[("v2", "defaults")]
    common_ids = v4_def.index.intersection(v4_ref.index).intersection(v2_def.index)
    frame = pd.DataFrame({
        "dataset": v4_def.loc[common_ids, "dataset"],
        "msa_v2_defaults": v2_def.loc[common_ids, "msa"], "msa_v4_defaults": v4_def.loc[common_ids, "msa"],
        "msa_v4_refine": v4_ref.loc[common_ids, "msa"],
        "merged_v2_defaults": v2_def.loc[common_ids, "merged"], "merged_v4_defaults": v4_def.loc[common_ids, "merged"],
        "merged_v4_refine": v4_ref.loc[common_ids, "merged"], "gt_objects": v4_def.loc[common_ids, "gt_objects"],
    })
    frame["refine_delta_msa"] = frame["msa_v4_refine"] - frame["msa_v4_defaults"]
    frame["refine_delta_merged"] = frame["merged_v4_refine"] - frame["merged_v4_defaults"]
    frame["checkpoint_delta_msa"] = frame["msa_v4_defaults"] - frame["msa_v2_defaults"]
    frame["checkpoint_delta_merged"] = frame["merged_v4_defaults"] - frame["merged_v2_defaults"]
    reasons: Dict[str, List[str]] = {}
    for dataset, group in frame.groupby("dataset"):
        for column, label in (("refine_delta_msa", "refinement"), ("checkpoint_delta_msa", "checkpoint")):
            ordered = group.sort_values(column)
            for sample_id in ordered.index[:n]:
                reasons.setdefault(sample_id, []).append(f"{label}-worst")
            for sample_id in ordered.index[::-1][:n]:
                reasons.setdefault(sample_id, []).append(f"{label}-best")
    selected = frame.loc[sorted(reasons)].copy()
    selected["reasons"] = [",".join(reasons[sample_id]) for sample_id in selected.index]
    return selected.sort_values(["dataset", "refine_delta_msa"])


def package_case(
    sample: dict, data_root: Path, runs: Dict[tuple, Path], tables: Dict[tuple, pd.DataFrame], path: Path,
    reasons: str,
) -> None:
    from apg3d_manifest import load_normalized_source, load_sample

    source = load_normalized_source(sample, data_root)
    raw, labels, valid = load_sample(sample, data_root, source)
    stem = sample["sample_id"].replace(":", "_")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["sample_id"] = sample["sample_id"]
        f.attrs["dataset"] = sample["dataset"]
        f.attrs["spacing"] = json.dumps(sample.get("spacing"))
        f.attrs["roi"] = json.dumps(sample.get("roi"))
        f.attrs["reasons"] = reasons
        f.create_dataset("raw", data=np.asarray(raw), compression="gzip", compression_opts=4)
        label_dtype = "uint16" if labels.max() < np.iinfo("uint16").max else "uint32"
        f.create_dataset("labels", data=labels.astype(label_dtype), compression="gzip", compression_opts=4)
        if valid is not None:
            f.create_dataset("valid", data=valid.astype("uint8"), compression="gzip", compression_opts=4)
        for (checkpoint, config), run_path in runs.items():
            name = f"{checkpoint}_{config}"
            output = _output_path(run_path, stem)
            if output is None:
                print(f"  missing outputs for {name}: {run_path / 'outputs' / f'{stem}.npz'}")
                continue
            with np.load(output) as arrays:
                f.create_dataset(
                    f"segmentation/{name}", data=arrays["segmentation"], compression="gzip", compression_opts=4,
                )
                for key in ("anchors", "scored_prompt_index", "merged_prompt_index", "merged_instance_id"):
                    if key in arrays:
                        f.create_dataset(f"prompts/{name}/{key}", data=arrays[key])
            row = tables[(checkpoint, config)].loc[sample["sample_id"]]
            for key in SCORE_KEYS:
                if key in row and pd.notna(row[key]):
                    f.attrs[f"{name}/{key}"] = float(row[key])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", default="primary")
    parser.add_argument("--n", type=int, default=1, help="Best and worst crops per dataset and criterion.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_ROOT / "3d_cases")
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/vast-nhr/projects/cidas/cca/data"))
    parser.add_argument("--sample-ids", nargs="*", default=None, help="Package exactly these crops instead of ranking.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    from apg3d_manifest import load_manifest

    runs, tables = {}, {}
    for checkpoint in CHECKPOINTS:
        for config in CONFIGS:
            runs[(checkpoint, config)], tables[(checkpoint, config)] = load_run(checkpoint, config, args.subset)
    os.environ.pop("MICRO_SAM2_JOINT_CHECKPOINT_ROOT", None)
    manifest = load_manifest(args.subset, CHECKPOINTS["v2"][0], args.data_root)
    by_id = {sample["sample_id"]: sample for sample in manifest["samples"]}

    selected = select_cases(tables, args.n)
    if args.sample_ids:
        selected = selected.reindex([s for s in args.sample_ids if s in selected.index]).dropna(how="all")
        for sample_id in args.sample_ids:
            if sample_id not in selected.index:
                selected.loc[sample_id, "dataset"] = by_id[sample_id]["dataset"]
                selected.loc[sample_id, "reasons"] = "requested"
    out_dir = args.out / args.subset
    out_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_dir / "cases.csv")
    print(selected[["dataset", "msa_v2_defaults", "msa_v4_defaults", "msa_v4_refine", "refine_delta_msa",
                    "checkpoint_delta_msa", "merged_v4_defaults", "merged_v4_refine", "gt_objects", "reasons"]]
          .round(4).to_string())
    for sample_id, row in selected.iterrows():
        path = out_dir / f"{row['dataset']}__{sample_id.replace(':', '_')}.h5"
        print(f"packaging {sample_id} -> {path}")
        package_case(by_id[sample_id], args.data_root, runs, tables, path, str(row["reasons"]))
    print(f"Cases: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
