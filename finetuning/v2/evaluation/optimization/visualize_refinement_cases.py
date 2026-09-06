"""Show the images a refinement variant helps most and hurts most, with masks and prompts on the raw data.

The per-image mSA of every screened variant is in the refinement screen's `samples.csv`
(`screen_apg_refinement.py`); this script ranks one dataset by the change of one variant against the
`none` control, takes the N largest improvements and the N largest decreases, recomputes the first
round, the refinement prompts and the refined result for those images with the real model, and writes
one figure per image into `improvements/` and `decreases/`. Each figure has six panels: the image, the
first-round APG masks, the refined masks, the refinement prompts (positives, negatives, boxes), the
pixel-level change (gained / lost / re-assigned), and the per-object IoU change on the ground-truth
footprints. Ground-truth boundaries are drawn on every mask panel; the title carries the score change.

Usage:
    python visualize_refinement_cases.py --dataset puma --variant pb --n 5 --checkpoint v4
    python visualize_refinement_cases.py --dataset puma --variant pb-isolated-boxes --checkpoint v2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EVALUATION_ROOT = Path(__file__).resolve().parent.parent
OPTIMIZATION_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EVALUATION_ROOT))
sys.path.insert(0, str(OPTIMIZATION_ROOT))

DEFAULT_OUTPUT_ROOT = Path("/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/apg_optimization")
V4_CHECKPOINT_ROOT = DEFAULT_OUTPUT_ROOT / "v4_geodesic_checkpoints"
MODEL_TYPE = "hvit_t"
CONTROL = "none"
# The proposal keys the screen shares across its entries and the selection keys it varies.
PROPOSE_KEYS = (
    "candidate_threshold", "foreground_threshold", "n_iter", "dt", "sigma", "min_candidate_size",
    "multimasking", "multimask_scorer", "multimask_selection", "batch_size", "n_threads",
)
SELECT_KEYS = ("score_threshold", "score_filter", "max_overlap", "min_size", "refinement", "refinement_kwargs")
# Overlay colours: masks get a per-instance palette; the prompt and change colours are chosen to
# stay apart from each other and from the ground-truth outline (white).
COLOR_POSITIVE = "#1a9850"   # filled circle
COLOR_NEGATIVE = "#f46d43"   # cross
COLOR_BOX = "#ffd92f"        # rectangle
COLOR_GT = "white"
COLOR_GAINED = "#2c7bb6"     # pixels the refinement added
COLOR_LOST = "#d7191c"       # pixels the refinement removed
COLOR_MOVED = "#fdae61"      # pixels that changed owner


def select_checkpoint(checkpoint: str) -> str:
    """Point `common` at the requested joint checkpoint and return its checksum."""
    if checkpoint == "v4":
        os.environ["MICRO_SAM2_JOINT_CHECKPOINT_ROOT"] = str(V4_CHECKPOINT_ROOT)
    else:
        os.environ.pop("MICRO_SAM2_JOINT_CHECKPOINT_ROOT", None)
    import common

    return common.checkpoint_checksum(common.get_joint_checkpoint(MODEL_TYPE, "best"))


def find_screen(output_root: Path, checkpoint_id: str, dataset: str, variant: str) -> Tuple[Path, dict]:
    """The newest complete refinement screen of the manifest holding 'dataset' that screened 'variant'."""
    root = output_root / "refinement_screening" / MODEL_TYPE / checkpoint_id
    candidates = []
    for metadata_path in root.glob("*/metadata.json"):
        if not (metadata_path.parent / "summary.csv").exists():
            continue
        metadata = json.load(open(metadata_path))
        names = {entry["name"] for entry in metadata.get("configs", [])}
        if variant not in names or CONTROL not in names:
            continue
        samples = pd.read_csv(metadata_path.parent / "samples.csv", usecols=["dataset"])
        if dataset in set(samples["dataset"]):
            candidates.append((metadata_path.stat().st_mtime, metadata_path.parent, metadata))
    if not candidates:
        raise SystemExit(
            f"No refinement screen with variants '{variant}' and '{CONTROL}' covers '{dataset}' under {root}."
        )
    _, run_dir, metadata = max(candidates, key=lambda entry: entry[0])
    return run_dir, metadata


def rank_images(run_dir: Path, dataset: str, variant: str) -> pd.DataFrame:
    samples = pd.read_csv(run_dir / "samples.csv")
    samples = samples[samples["dataset"] == dataset]
    table = samples.pivot(index="sample_id", columns="config_name", values="msa")
    ranking = pd.DataFrame({
        "msa_first": table[CONTROL], "msa_refined": table[variant], "delta": table[variant] - table[CONTROL],
    })
    ranking["relative"] = ranking["delta"] / ranking["msa_first"].replace(0, np.nan)
    return ranking.sort_values("delta", ascending=False)


def config_params(metadata: dict, variant: str) -> dict:
    for entry in metadata["configs"]:
        if entry["name"] == variant:
            return entry["params_2d"]
    raise KeyError(variant)


def display_image(raw: np.ndarray) -> np.ndarray:
    """The raw data as a float RGB image in [0, 1], whatever its channel layout."""
    raw = np.asarray(raw)
    if raw.ndim == 3 and raw.shape[0] in (1, 2, 3, 4) and raw.shape[0] < raw.shape[-1]:
        raw = np.moveaxis(raw, 0, -1)
    if raw.ndim == 3:
        raw = raw[..., :3]
        if raw.shape[-1] == 1:
            raw = np.repeat(raw, 3, axis=-1)
        elif raw.shape[-1] == 2:
            raw = np.concatenate([raw, np.zeros_like(raw[..., :1])], axis=-1)
    else:
        raw = np.repeat(raw[..., None], 3, axis=-1)
    image = raw.astype("float32")
    low, high = np.percentile(image, 1), np.percentile(image, 99.5)
    image = np.clip((image - low) / max(high - low, 1e-6), 0, 1)
    return image


def instance_palette(n_instances: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hues = rng.permutation(np.linspace(0, 1, max(n_instances, 1), endpoint=False))
    from matplotlib.colors import hsv_to_rgb

    return hsv_to_rgb(np.stack([hues, np.full_like(hues, 0.85), np.full_like(hues, 0.95)], axis=1))


def overlay_masks(image: np.ndarray, segmentation: np.ndarray, palette: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    out = image.copy()
    ids = np.unique(segmentation)
    for index in ids[ids != 0]:
        mask = segmentation == index
        out[mask] = (1 - alpha) * out[mask] + alpha * palette[(int(index) - 1) % len(palette)]
    return out


def draw_boundaries(axis, labels: np.ndarray, color: str, linewidth: float = 0.8) -> None:
    from skimage.segmentation import find_boundaries

    boundary = find_boundaries(labels, mode="inner")
    rgba = np.zeros((*labels.shape, 4), dtype="float32")
    from matplotlib.colors import to_rgb

    rgba[boundary, :3] = to_rgb(color)
    rgba[boundary, 3] = 1.0
    axis.imshow(rgba, interpolation="nearest")
    del linewidth  # boundaries are one pixel wide by construction


def per_object_iou(labels: np.ndarray, segmentation: np.ndarray) -> Dict[int, float]:
    """IoU of every ground-truth object with its best-overlapping predicted instance (0 if none)."""
    areas = dict(zip(*np.unique(segmentation[segmentation != 0], return_counts=True)))
    ious = {}
    for index in np.unique(labels):
        if index == 0:
            continue
        mask = labels == index
        overlapping = segmentation[mask]
        overlapping = overlapping[overlapping != 0]
        if overlapping.size == 0:
            ious[int(index)] = 0.0
            continue
        candidates, counts = np.unique(overlapping, return_counts=True)
        best = int(np.argmax(counts))
        intersection = int(counts[best])
        ious[int(index)] = intersection / (int(mask.sum()) + int(areas[candidates[best]]) - intersection)
    return ious


def area_ratios(labels: np.ndarray, segmentation: np.ndarray) -> Tuple[float, int]:
    """Median predicted / ground-truth area over the matched objects (IoU >= 0.5), and their count."""
    areas = dict(zip(*np.unique(segmentation[segmentation != 0], return_counts=True)))
    ratios = []
    for index in np.unique(labels):
        if index == 0:
            continue
        mask = labels == index
        overlapping = segmentation[mask]
        overlapping = overlapping[overlapping != 0]
        if overlapping.size == 0:
            continue
        candidates, counts = np.unique(overlapping, return_counts=True)
        best = int(np.argmax(counts))
        intersection, gt_area, predicted_area = int(counts[best]), int(mask.sum()), int(areas[candidates[best]])
        if intersection / (gt_area + predicted_area - intersection) >= 0.5:
            ratios.append(predicted_area / gt_area)
    return (float(np.median(ratios)) if ratios else float("nan")), len(ratios)


def refinement_prompts_for(generator, proposals: list, context: dict, segmentation: np.ndarray, params: dict):
    """Reproduce the prompts `_reprompt_instances` derives, and which instances it re-prompts."""
    from micro_sam.v2.automatic_prompt_generation import (
        _parse_refinement, _touching_instances, derive_refinement_prompts,
    )

    components, kwargs = _parse_refinement(params["refinement"], params.get("refinement_kwargs"))
    instance_ids = sorted(context["matches"])
    touching = None
    if kwargs.get("gate") == "isolated" or kwargs.get("negative_scope") == "touching":
        touching = _touching_instances(segmentation, int(kwargs.get("touch_radius", 2)))
    full, box_only, untouched = list(instance_ids), [], []
    if kwargs.get("gate") == "isolated":
        full = [index for index in instance_ids if not touching[index]]
        rest = [index for index in instance_ids if touching[index]]
        if kwargs.get("isolated_fallback") == "boxes":
            box_only = rest
        else:
            untouched = rest
    points = None
    if "points" in components:
        all_points, seen = [], set()
        for record_index, record in enumerate(proposals):
            group = record.get("multimask_group", ("record", record_index))
            if group in seen:
                continue
            seen.add(group)
            all_points.append(record["point"])
        surviving = {
            index: context["records"][record_index]["point"] for index, record_index in context["matches"].items()
        }
        points = derive_refinement_prompts(
            segmentation, np.array(all_points, dtype="float32"), surviving,
            n_positives=kwargs["n_positives"], n_negatives=kwargs["n_negatives"],
            max_negative_distance=kwargs["max_negative_distance"], negative_source=kwargs["negative_source"],
            min_negative_distance=kwargs["min_negative_distance"],
            negative_scope=kwargs.get("negative_scope", "nearest"), touching=touching,
        )
    return {
        "components": components, "kwargs": kwargs, "points": points,
        "full": full, "box_only": box_only, "untouched": untouched, "boxes": "boxes" in components,
    }


def render(
    path: Path, dataset: str, sample_id: str, raw: np.ndarray, labels: np.ndarray, first: np.ndarray,
    refined: np.ndarray, prompts: dict, scores: dict, variant: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from scipy.ndimage import find_objects

    image = display_image(raw)
    palette = instance_palette(int(max(first.max(), refined.max(), 1)))
    figure, axes = plt.subplots(2, 3, figsize=(19, 12.5))
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])

    axes[0, 0].imshow(image, interpolation="nearest")
    axes[0, 0].set_title("image\n", fontsize=11)

    axes[0, 1].imshow(overlay_masks(image, first, palette), interpolation="nearest")
    draw_boundaries(axes[0, 1], labels, COLOR_GT)
    n_first = int(len(np.unique(first)) - 1)
    axes[0, 1].set_title(
        f"APG first round\n{n_first} instances, mSA {scores['first']:.3f}; white = ground truth", fontsize=11,
    )

    axes[0, 2].imshow(overlay_masks(image, refined, palette), interpolation="nearest")
    draw_boundaries(axes[0, 2], labels, COLOR_GT)
    n_refined = int(len(np.unique(refined)) - 1)
    axes[0, 2].set_title(
        f"after refinement\n{n_refined} instances, mSA {scores['refined']:.3f}; white = ground truth", fontsize=11,
    )

    # Prompts on the first-round outlines.
    axis = axes[1, 0]
    axis.imshow(image, interpolation="nearest")
    draw_boundaries(axis, first, "#9ecae1")
    n_positive = n_negative = 0
    if prompts["boxes"]:
        for index, box in enumerate(find_objects(first), start=1):
            if box is None or index not in prompts["full"] + prompts["box_only"]:
                continue
            rectangle = Rectangle(
                (box[1].start - 0.5, box[0].start - 0.5), box[1].stop - box[1].start, box[0].stop - box[0].start,
                fill=False, edgecolor=COLOR_BOX, linewidth=0.9, linestyle="-" if index in prompts["full"] else "--",
            )
            axis.add_patch(rectangle)
    if prompts["points"] is not None:
        for index in prompts["full"]:
            prompt = prompts["points"].get(index)
            if prompt is None:
                continue
            positive = prompt["points"][prompt["point_labels"] == 1]
            negative = prompt["points"][prompt["point_labels"] == 0]
            n_positive += len(positive)
            n_negative += len(negative)
            axis.scatter(
                positive[:, 0], positive[:, 1], s=28, c=COLOR_POSITIVE, edgecolors="black", linewidths=0.4, zorder=3,
            )
            axis.scatter(
                negative[:, 0], negative[:, 1], s=30, c=COLOR_NEGATIVE, marker="x", linewidths=1.2, zorder=3,
            )
    handles = [
        Line2D(
            [], [], marker="o", color=COLOR_POSITIVE, markeredgecolor="black", linestyle="", label="positive point",
        ),
        Line2D([], [], marker="x", color=COLOR_NEGATIVE, linestyle="", label="negative point"),
        Patch(facecolor="none", edgecolor=COLOR_BOX, label="box prompt"),
        Line2D([], [], color="#9ecae1", label="first-round outline"),
    ]
    axis.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.85)
    gate_note = ""
    if prompts["untouched"] or prompts["box_only"]:
        gate_note = (
            f"; {len(prompts['full'])} full, {len(prompts['box_only'])} box-only, "
            f"{len(prompts['untouched'])} kept"
        )
    axis.set_title(f"refinement prompts\n{n_positive} positives, {n_negative} negatives{gate_note}", fontsize=11)

    # Pixel-level change.
    axis = axes[1, 1]
    change = image.copy()
    gained = (first == 0) & (refined != 0)
    lost = (first != 0) & (refined == 0)
    moved = (first != 0) & (refined != 0) & (first != refined)
    from matplotlib.colors import to_rgb

    for mask, color in ((gained, COLOR_GAINED), (lost, COLOR_LOST), (moved, COLOR_MOVED)):
        change[mask] = 0.25 * change[mask] + 0.75 * np.array(to_rgb(color))
    axis.imshow(change, interpolation="nearest")
    draw_boundaries(axis, labels, COLOR_GT)
    handles = [
        Patch(facecolor=COLOR_GAINED, label=f"gained ({int(gained.sum())} px)"),
        Patch(facecolor=COLOR_LOST, label=f"lost ({int(lost.sum())} px)"),
        Patch(facecolor=COLOR_MOVED, label=f"re-assigned ({int(moved.sum())} px)"),
        Line2D([], [], color=COLOR_GT, label="ground truth"),
    ]
    axis.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.85)
    ratio_first, matched_first = area_ratios(labels, first)
    ratio_refined, matched_refined = area_ratios(labels, refined)
    axis.set_title(
        "what the refinement changed\nmask area / truth, median over matched objects: "
        f"{ratio_first:.2f} (n={matched_first}) → {ratio_refined:.2f} (n={matched_refined})", fontsize=11,
    )

    # Per-object IoU change on the ground-truth footprints.
    axis = axes[1, 2]
    before, after = per_object_iou(labels, first), per_object_iou(labels, refined)
    delta = np.zeros(labels.shape, dtype="float32")
    for index, iou in before.items():
        delta[labels == index] = after[index] - iou
    shown = np.ma.masked_where(labels == 0, delta)
    axis.imshow(image, interpolation="nearest")
    mappable = axis.imshow(shown, cmap="RdBu", vmin=-0.3, vmax=0.3, interpolation="nearest", alpha=0.85)
    colorbar = figure.colorbar(mappable, ax=axis, fraction=0.035, pad=0.02)
    colorbar.set_label("IoU after − before (per ground-truth object)")
    ups = sum(after[index] > iou + 1e-6 for index, iou in before.items())
    downs = sum(after[index] < iou - 1e-6 for index, iou in before.items())
    axis.set_title(
        f"per-object IoU change\n{ups} up, {downs} down, {len(before) - ups - downs} unchanged", fontsize=11,
    )

    delta_msa = scores["refined"] - scores["first"]
    relative = delta_msa / scores["first"] if scores["first"] else float("nan")
    figure.suptitle(
        f"{dataset}  {sample_id}  |  {variant}: mSA {scores['first']:.4f} → {scores['refined']:.4f}  "
        f"(Δ {delta_msa:+.4f}, {relative:+.1%})  |  {int(len(before))} ground-truth objects",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110)
    plt.close(figure)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--variant", default="pb", help="A configuration name of the refinement screen.")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--checkpoint", choices=("v2", "v4"), default="v4")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(list(argv) if argv is not None else None)

    checkpoint_id = select_checkpoint(args.checkpoint)
    import common
    from benchmark_apg_optimization import DEFAULT_DATA_ROOT, _load_2d_sample, prepare_manifest, _default_manifest_path
    from common import GT_MIN_SIZE_2D
    from parameter_search import compute_metrics

    run_dir, metadata = find_screen(args.output_root, checkpoint_id, args.dataset, args.variant)
    ranking = rank_images(run_dir, args.dataset, args.variant)
    improvements = ranking[ranking["delta"] > 0].head(args.n)
    decreases = ranking[ranking["delta"] < 0].sort_values("delta").head(args.n)
    out_dir = args.output_root / "structural_2d" / "visual" / args.checkpoint / args.dataset / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(out_dir / "ranking.csv")
    print(f"Screen: {run_dir}\n{len(ranking)} images; {int((ranking.delta > 0).sum())} up, "
          f"{int((ranking.delta < 0).sum())} down, mean Δ {ranking.delta.mean():+.4f}")

    params = config_params(metadata, args.variant)
    control_params = config_params(metadata, CONTROL)
    propose_params = {key: params[key] for key in PROPOSE_KEYS if key in params}
    select_params = {key: params[key] for key in SELECT_KEYS if key in params}
    manifest = prepare_manifest(
        DEFAULT_DATA_ROOT, _default_manifest_path(args.output_root, "standard", metadata["subset"]), "standard",
        subset=metadata["subset"],
    )
    by_id = {sample["sample_id"]: sample for sample in manifest["samples"]}
    segmenter = common.build_apg_segmenter(
        MODEL_TYPE, 2, args.device, joint_checkpoint="best", joint_checksum=checkpoint_id,
        export_root=str(args.output_root / "model_exports"),
    )
    border = GT_MIN_SIZE_2D.get(args.dataset, 0)
    try:
        for folder, table in (("improvements", improvements), ("decreases", decreases)):
            for rank, (sample_id, row) in enumerate(table.iterrows(), start=1):
                raw, labels = _load_2d_sample(by_id[sample_id], DEFAULT_DATA_ROOT)
                segmenter.clear_state()
                segmenter.initialize(raw, ndim=2)
                proposals = segmenter.propose(**propose_params)
                first, context = segmenter._merge(
                    proposals, labels.shape, score_threshold=control_params["score_threshold"],
                    max_overlap=control_params["max_overlap"], min_size=control_params["min_size"],
                    return_context=True, score_filter=control_params["score_filter"],
                )
                first = first.astype("uint32")
                refined = segmenter.select(proposals, **select_params).astype("uint32")
                prompts = refinement_prompts_for(segmenter, proposals, context, first, params)
                scores = {
                    "first": compute_metrics(first, labels, "sparse", border_min_size=border)["msa"],
                    "refined": compute_metrics(refined, labels, "sparse", border_min_size=border)["msa"],
                }
                mismatch = (
                    abs(scores["first"] - row["msa_first"]) > 1e-6
                    or abs(scores["refined"] - row["msa_refined"]) > 1e-6
                )
                if mismatch:
                    print(f"  warning: {sample_id} recomputed {scores} differs from the screen "
                          f"({row['msa_first']:.6f}, {row['msa_refined']:.6f})")
                name = f"{rank:02d}_{sample_id.replace(':', '_')}_d{row['delta']:+.4f}.png"
                render(out_dir / folder / name, args.dataset, sample_id, raw, labels, first, refined, prompts, scores,
                       args.variant)
                print(f"  {folder} {rank}: {sample_id} Δ {row['delta']:+.4f} ({row['relative']:+.1%})")
    finally:
        segmenter.clear_state()
    print(f"Figures: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
