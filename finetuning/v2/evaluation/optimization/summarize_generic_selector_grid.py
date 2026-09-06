"""Rank generic-feature selector fits by their leave-one-dataset-out proxies.

Every ``*_training_results.json`` written by ``train_apg_multimask_selector.py --lodo`` carries, per held-out
dataset, the matched-AUC and the selected-alternative IoU of the model's out-of-fold (in-domain) and
leave-one-dataset-out (out-of-domain) predictions next to the same two numbers for SAM2's predicted IoU.
This script tabulates them per configuration (dataset-balanced means and worst dataset) so the GPU replay
screens can be limited to the configurations whose out-of-domain proxies beat predicted IoU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _parse_name(name: str) -> dict:
    fields = {"model": "mlp", "target": "regression", "feature_set": "all", "per_image": "none"}
    if "-linear-" in name:
        fields["model"] = "linear"
    if "-matched" in name:
        fields["target"] = "matched"
    for token in name.split("-"):
        if token.startswith("fs_"):
            fields["feature_set"] = token[3:]
        elif token.startswith("z_"):
            fields["per_image"] = token[2:]
        elif token.startswith("h") and token[1:].isdigit():
            fields["model"] = f"mlp-h{token[1:]}"
    return fields


def summarize(model_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(model_dir.glob("*_training_results.json")):
        results = json.loads(path.read_text())
        lodo = results["metrics"].get("lodo")
        if not lodo:
            continue
        name = path.name.removesuffix("_training_results.json")
        per_dataset = pd.DataFrame(lodo).T
        auc_delta = per_dataset["lodo_matched_auc"] - per_dataset["predicted_iou_matched_auc"]
        iou_delta = per_dataset["lodo_selected_iou"] - per_dataset["predicted_iou_selected_iou"]
        oof_auc_delta = per_dataset["oof_matched_auc"] - per_dataset["predicted_iou_matched_auc"]
        oof_iou_delta = per_dataset["oof_selected_iou"] - per_dataset["predicted_iou_selected_iou"]
        rows.append({
            "name": name, **_parse_name(name), "n_datasets": len(per_dataset),
            "lodo_auc_delta_mean": float(auc_delta.mean()), "lodo_auc_delta_min": float(auc_delta.min()),
            "lodo_auc_delta_min_dataset": str(auc_delta.idxmin()),
            "lodo_auc_wins": int((auc_delta > 0).sum()),
            "lodo_selected_iou_delta_mean": float(iou_delta.mean()),
            "lodo_selected_iou_delta_min": float(iou_delta.min()),
            "oof_auc_delta_mean": float(oof_auc_delta.mean()),
            "oof_selected_iou_delta_mean": float(oof_iou_delta.mean()),
            "lodo_auc_mean": float(per_dataset["lodo_matched_auc"].mean()),
            "predicted_iou_auc_mean": float(per_dataset["predicted_iou_matched_auc"].mean()),
        })
    if not rows:
        raise FileNotFoundError(f"No LODO training results below {model_dir}.")
    table = pd.DataFrame(rows).sort_values(["lodo_auc_delta_mean", "lodo_auc_delta_min"], ascending=False)
    return table.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    table = summarize(args.model_dir)
    output = args.output or args.model_dir / "g1_proxy_summary.csv"
    table.to_csv(output, index=False)
    columns = [
        "feature_set", "per_image", "model", "target", "lodo_auc_delta_mean", "lodo_auc_delta_min",
        "lodo_auc_delta_min_dataset", "lodo_auc_wins", "lodo_selected_iou_delta_mean", "oof_auc_delta_mean",
    ]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table[columns].head(args.top).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
