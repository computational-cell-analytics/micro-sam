"""Field diagnostics of cached decoder predictions: contact geometry and foreground extent.

For every cached sample of a manifest (`benchmark_ais_optimization.py predict` must have run) and per dataset:
the cosine between the predicted flow one pixel on either side of a ground-truth contact pixel (the field of a
well separated pair flips, so the cosine is negative), the same cosine one pixel apart inside objects (a
smooth field gives +1), the median distance magnitude at contacts and inside, the foreground area ratio
`area(fg > threshold) / area(gt)` and, for five channel predictions, the Dice of `contact > 0.5` with the
ground-truth contact target. The proposal's "what would show that it worked" figures. CPU only, reader only.

    export MICRO_SAM2_JOINT_CHECKPOINT_ROOT=<staged root>
    python diagnose_decoder_fields.py --joint-checkpoint contact --subset primary training_extra --output <csv>
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import benchmark_ais_optimization as ais  # noqa: E402
from micro_sam.v2.transforms.labels import touching_boundaries  # noqa: E402


def _shift(array: np.ndarray, axis: int, step: int) -> np.ndarray:
    """The array shifted by 'step' along 'axis' with edge replication (so a difference at the border is zero)."""
    shifted = np.roll(array, -step, axis=axis)
    index = [slice(None)] * array.ndim
    if step > 0:
        index[axis] = slice(-step, None)
        shifted[tuple(index)] = np.take(array, [-1], axis=axis)
    else:
        index[axis] = slice(None, -step)
        shifted[tuple(index)] = np.take(array, [0], axis=axis)
    return shifted


def flow_cosines(directed: np.ndarray, where: np.ndarray, offset: int) -> np.ndarray:
    """Cosine between the flow 'offset' pixels before and after every pixel of 'where', along the axis of the
    stronger local label change (both in-plane axes are tried and the smaller cosine kept: the flip axis)."""
    norms = np.linalg.norm(directed, axis=0) + 1e-6
    unit = directed / norms
    cosines = []
    for axis in range(1, unit.ndim):
        before = _shift(unit, axis, -offset)
        after = _shift(unit, axis, offset)
        cosines.append((before * after).sum(axis=0))
    cosine = np.minimum.reduce(cosines)
    return cosine[where]


def sample_row(prediction: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    ndim = labels.ndim
    foreground, directed = prediction[0], prediction[1:4][-ndim:]
    contact_gt = touching_boundaries(labels, radius=1, dilation=0)
    interior = (labels > 0) & ~touching_boundaries(labels, radius=2, dilation=0)
    from skimage.segmentation import find_boundaries
    interior &= ~find_boundaries(labels, mode="inner")
    magnitude = np.linalg.norm(directed, axis=0)
    fg_mask = foreground > threshold
    row = {
        "gt_objects": int(len(np.unique(labels)) - 1),
        "contact_pixels": int(contact_gt.sum()),
        "fg_area_ratio": float(fg_mask.sum() / max(1, (labels > 0).sum())),
        "fg_iou": float((fg_mask & (labels > 0)).sum() / max(1, (fg_mask | (labels > 0)).sum())),
        "magnitude_bg_median": float(np.median(magnitude[labels == 0])) if (labels == 0).any() else float("nan"),
        "magnitude_interior_median": float(np.median(magnitude[interior])) if interior.any() else float("nan"),
    }
    if contact_gt.any():
        row["magnitude_contact_median"] = float(np.median(magnitude[contact_gt]))
        for offset in (1, 3):
            row[f"cosine_contact_{offset}px"] = float(np.median(flow_cosines(directed, contact_gt, offset)))
    else:
        row["magnitude_contact_median"] = float("nan")
        row["cosine_contact_1px"] = row["cosine_contact_3px"] = float("nan")
    if interior.any():
        for offset in (1, 3):
            row[f"cosine_interior_{offset}px"] = float(np.median(flow_cosines(directed, interior, offset)))
    if prediction.shape[0] > 4:
        contact_pred = prediction[4] > 0.5
        target = touching_boundaries(labels, radius=1, dilation=1)
        denominator = contact_pred.sum() + target.sum()
        row["contact_dice"] = float(2 * (contact_pred & target).sum() / denominator) if denominator else float("nan")
        row["contact_pred_pixels"] = int(contact_pred.sum())
        # Share of the predicted contact mass that lies within two pixels of a true contact.
        near = touching_boundaries(labels, radius=1, dilation=2)
        row["contact_precision_2px"] = float((contact_pred & near).sum() / max(1, contact_pred.sum()))
        row["contact_recall"] = float((contact_pred & target).sum() / max(1, target.sum()))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ais.common_arguments(parser)
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    checkpoint_id = ais._checkpoint_identity(args.model_type, args.joint_checkpoint)
    dimensions = ais._dimensions(args)
    rows = []
    for manifest in ais._manifests(args):
        cache = ais.PredictionCache(args.output_root, checkpoint_id, manifest["manifest_checksum"])
        samples = [s for s in manifest["samples"] if int(s["ndim"]) in dimensions]
        if args.datasets:
            samples = [s for s in samples if s["dataset"] in args.datasets]
        for index, sample in enumerate(samples, start=1):
            if not cache.has(sample):
                raise FileNotFoundError(f"No cached prediction for '{sample['sample_id']}' under '{cache.root}'.")
            prediction, labels, valid, _ = cache.load(sample)
            if valid is not None:
                labels = np.where(valid, labels, 0)
            row = sample_row(prediction, labels.astype("int64"), args.foreground_threshold)
            row.update({
                "sample_id": sample["sample_id"], "dataset": sample["dataset"], "subset": manifest.get("subset"),
            })
            rows.append(row)
            if index % 50 == 0:
                print(f"{manifest.get('subset')}: {index}/{len(samples)}")
    table = pd.DataFrame(rows)
    numeric = [c for c in table.columns if c not in ("sample_id", "dataset", "subset")]
    summary = table.groupby("dataset")[numeric].median(numeric_only=True)
    summary["n_samples"] = table.groupby("dataset").size()
    pd.set_option("display.width", 250)
    print(f"\nCheckpoint {checkpoint_id[:8]}: per-dataset medians")
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
        summary.to_csv(str(Path(args.output).with_name(Path(args.output).stem + "_summary.csv")))
        print(f"written {args.output}")


if __name__ == "__main__":
    main()
