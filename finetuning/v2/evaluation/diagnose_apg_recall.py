"""Locate where automatic prompt generation loses its recall, stage by stage.

Recall sits well below precision, so objects are being lost somewhere. This walks a ground-truth
object through the pipeline and reports at which stage it drops out:

    seeded: a candidate point lands inside the object at all
    proposed: some prompted mask matches it at IoU >= 0.5, before any filtering
    scored: that match survives the 'score_threshold' filter
    merged: it is still there after the overlap merge, i.e. it is in the output

The gap between 'seeded' and 1.0 is what proposing more candidates could recover. The gap between
'proposed' and 'merged' is what the selection is throwing away, and no extra candidate helps there.

Usage:
    python diagnose_apg_recall.py -d livecell -m hvit_t -n 8
"""

import argparse

import numpy as np
import pandas as pd

from bioimage_cpp.segmentation import label as connected_components

from common import DATA_ROOT, DATASET_SPACING, VOLUME_SPEED_OPTIONS
from baselines_common import _load_data
from evaluate_apg import build_segmenter


def match_iou(mask: np.ndarray, gt_crop: np.ndarray, gt_sizes: dict) -> tuple:
    """Return the ground-truth id a mask overlaps most and their IoU."""
    overlapping = gt_crop[mask]
    overlapping = overlapping[overlapping != 0]
    if overlapping.size == 0:
        return 0, 0.0
    ids, counts = np.unique(overlapping, return_counts=True)
    best = int(np.argmax(counts))
    gt_id, intersection = int(ids[best]), int(counts[best])
    union = int(mask.sum()) + gt_sizes[gt_id] - intersection
    return gt_id, intersection / union


def multimask_batches(segmenter, prompts, batch_size):
    """Yield the binarised masks of every multimask output per prompt, with their predicted IoUs.

    Mirrors `AutomaticPromptGenerator._apply_prompts`, but keeps all of SAM2's mask proposals instead
    of only the best-scoring one, which is what lets the oracle below measure what the argmax discards.
    """
    from sam2.utils.amg import calculate_stability_score
    from micro_sam.v2.automatic_prompt_generation import STABILITY_SCORE_OFFSET

    predictor = segmenter._predictor
    points, point_labels = prompts["points"], prompts["point_labels"]
    mask_threshold = getattr(predictor, "mask_threshold", 0.0)

    for start in range(0, len(points), batch_size):
        stop = start + batch_size
        batch_points, batch_labels = points[start:stop], point_labels[start:stop]
        n_prompts = len(batch_points)
        mask_input, coords, labels, _ = predictor._prep_prompts(batch_points, batch_labels, None, None, True)
        logits, scores, _ = predictor._predict(coords, labels, None, mask_input, True, return_logits=True)
        logits = logits.reshape(n_prompts, -1, *logits.shape[-2:])
        scores = scores.reshape(n_prompts, -1)
        stability = calculate_stability_score(logits, mask_threshold, STABILITY_SCORE_OFFSET)
        yield (
            start, (logits > mask_threshold).cpu().numpy(),
            scores.float().cpu().numpy(), stability.float().cpu().numpy(),
        )


# How a multimask output is chosen. 'score' is what the generator does today; 'oracle' is the ceiling
# that any choice made without the ground truth has to be judged against.
SELECTORS = ("score", "score_x_stability", "smallest", "fewest_foreign", "oracle")


def select_mask(selector, scores, stability, areas, foreign, ious):
    """Return the index of the multimask output a selector keeps."""
    if selector == "score":
        return int(np.argmax(scores))
    if selector == "score_x_stability":
        return int(np.argmax(scores * stability))
    if selector == "oracle":
        return int(np.argmax(ious))
    if selector == "smallest":
        # A cluster is larger than the object, so the smallest mask that the model still rates
        # plausibly is the one biased away from it.
        pool = np.where(scores >= 0.5)[0]
        pool = pool if pool.size else np.arange(len(scores))
        return int(pool[np.argmin(areas[pool])])
    if selector == "fewest_foreign":
        # The decoder already said how many objects are in the region: a mask that swallows other
        # candidates' points is covering more than the object its own point sits in.
        pool = np.where(foreign == foreign.min())[0]
        return int(pool[np.argmax(scores[pool])])
    raise ValueError(f"Unknown selector: '{selector}'. Choose from {SELECTORS}.")


def diagnose_prompts(segmenter, prompts, labels, gt_sizes, batch_size, iou_threshold=0.5):
    """Ask, per prompt that lands inside an object, whether SAM2 returned that object.

    Every prompt has an unambiguous target: the object its point falls in. The generator keeps the
    multimask output with the highest predicted IoU, so comparing that mask against the target says
    whether the interactive branch answered the prompt, and comparing the best of all outputs against
    it says whether the answer was there but discarded. The selectors in between say how much of that
    gap a rule without access to the ground truth can close.

    A failure is classified by area, since the two ways a single point goes wrong need opposite fixes:
    a mask far larger than the target is the confluent cluster, and a far smaller one is a fragment.
    """
    all_points = prompts["points"][:, 0, :]
    counts = {
        "n_prompts_on_object": 0, "fail_cluster": 0, "fail_fragment": 0, "fail_misaligned": 0,
        "fragment_components": 0, "fragment_largest_share": 0.0, "fragment_holds_point": 0,
    }
    counts.update({f"matched_{name}": 0 for name in SELECTORS})
    matched_ids = {name: set() for name in SELECTORS}

    for offset, masks, batch_scores, batch_stability in multimask_batches(segmenter, prompts, batch_size):
        for index, per_prompt in enumerate(masks):
            x, y = all_points[offset + index]
            target = int(labels[int(np.clip(y, 0, labels.shape[0] - 1)), int(np.clip(x, 0, labels.shape[1] - 1))])
            if target == 0:
                continue
            counts["n_prompts_on_object"] += 1

            target_size = gt_sizes[target]
            target_mask = labels == target
            # Every other candidate's point, which a mask covering only this object should not contain.
            others = np.delete(all_points, offset + index, axis=0)
            rows = np.clip(others[:, 1].astype(int), 0, labels.shape[0] - 1)
            columns = np.clip(others[:, 0].astype(int), 0, labels.shape[1] - 1)

            ious, areas, foreign = [], [], []
            for mask in per_prompt:
                intersection = int(np.count_nonzero(mask & target_mask))
                area = int(mask.sum())
                ious.append(intersection / (area + target_size - intersection) if area else 0.0)
                areas.append(area)
                foreign.append(int(np.count_nonzero(mask[rows, columns])))
            ious, areas, foreign = np.array(ious), np.array(areas), np.array(foreign)
            scores, stability = batch_scores[index], batch_stability[index]

            for name in SELECTORS:
                chosen = select_mask(name, scores, stability, areas, foreign, ious)
                if ious[chosen] >= iou_threshold:
                    counts[f"matched_{name}"] += 1
                    matched_ids[name].add(target)

            chosen = select_mask("score", scores, stability, areas, foreign, ious)
            if ious[chosen] < iou_threshold:
                ratio = areas[chosen] / target_size if target_size else 0.0
                if ratio > 1.5:
                    counts["fail_cluster"] += 1
                elif ratio < 0.67:
                    counts["fail_fragment"] += 1
                    # Whether an under-segmentation is one coherent piece of the object or scattered
                    # debris, which decides whether a box prompt can recover it.
                    mask = per_prompt[chosen]
                    components = connected_components(mask.astype("uint32"))
                    counts["fragment_components"] += int(components.max())
                    largest = np.bincount(components[mask])[1:]
                    counts["fragment_largest_share"] += float(largest.max() / mask.sum()) if largest.size else 0.0
                    counts["fragment_holds_point"] += int(
                        mask[int(np.clip(y, 0, mask.shape[0] - 1)), int(np.clip(x, 0, mask.shape[1] - 1))]
                    )
                else:
                    counts["fail_misaligned"] += 1

    counts.update({f"objects_{name}": len(matched_ids[name]) for name in SELECTORS})
    return counts


def diagnose_image(segmenter, raw, labels, params, iou_threshold=0.5):
    """Return the per-stage set of ground-truth ids that survive, for one image."""
    from micro_sam.v2.automatic_prompt_generation import derive_point_prompts

    gt_ids = np.unique(labels)
    gt_ids = gt_ids[gt_ids != 0]
    if gt_ids.size == 0:
        return None
    gt_sizes = {int(i): int(s) for i, s in zip(*np.unique(labels[labels != 0], return_counts=True))}

    segmenter.clear_state()
    segmenter.initialize(raw, ndim=2)
    prompts = derive_point_prompts(
        segmenter._prediction[0], segmenter._prediction[1:],
        candidate_threshold=params["candidate_threshold"], foreground_threshold=params["foreground_threshold"],
        n_iter=params["n_iter"], sigma=params["sigma"], min_candidate_size=params["min_candidate_size"],
    )
    if prompts is None:
        return {"n_gt": len(gt_ids), "n_candidates": 0, "seeded": 0, "proposed": 0, "scored": 0, "merged": 0}

    # A ground-truth object is seeded when a candidate point falls inside it.
    points = prompts["points"][:, 0, :]
    inside = labels[np.clip(points[:, 1].astype(int), 0, labels.shape[0] - 1),
                    np.clip(points[:, 0].astype(int), 0, labels.shape[1] - 1)]
    seeded = {int(i) for i in np.unique(inside) if i != 0}

    prompt_counts = diagnose_prompts(
        segmenter, prompts, labels, gt_sizes, params["batch_size"], iou_threshold=iou_threshold,
    )

    proposals = segmenter._apply(prompts, multimasking=True, batch_size=params["batch_size"])
    proposed, scored = set(), set()
    for record in proposals:
        box = record["bounding_box"]
        gt_id, iou = match_iou(record["segmentation"], labels[box], gt_sizes)
        if iou >= iou_threshold:
            proposed.add(gt_id)
            if record["predicted_iou"] >= params["score_threshold"]:
                scored.add(gt_id)

    segmentation = segmenter.select(
        proposals, score_threshold=params["score_threshold"], max_overlap=params["max_overlap"],
        min_size=params["min_size"], refine_with_box_prompts=False,
    )
    merged = set()
    for instance_id in np.unique(segmentation):
        if instance_id == 0:
            continue
        mask = segmentation == instance_id
        gt_id, iou = match_iou(mask, labels, gt_sizes)
        if iou >= iou_threshold:
            merged.add(gt_id)

    return {
        "n_gt": len(gt_ids), "n_candidates": len(points), "seeded": len(seeded & set(gt_sizes)),
        "proposed": len(proposed), "scored": len(scored), "merged": len(merged),
        **prompt_counts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--dataset_name", default="livecell", help="Dataset to diagnose.")
    parser.add_argument("-m", "--model_type", default="hvit_t", help="SAM2 backbone of the joint model.")
    parser.add_argument("-n", "--n_samples", type=int, default=8, help="Images to diagnose.")
    parser.add_argument("--device", default="cuda", help="Device to run on.")
    parser.add_argument("--candidate_threshold", type=float, default=None, help="Override the candidate threshold.")
    parser.add_argument("--foreground_threshold", type=float, default=None, help="Override the foreground threshold.")
    parser.add_argument("--min_candidate_size", type=int, default=None, help="Override the minimum candidate size.")
    parser.add_argument("--ndim", type=int, default=2, choices=(2, 3), help="Diagnose images or volumes.")
    args = parser.parse_args()

    from micro_sam.v2.automatic_prompt_generation import DEFAULT_PROMPT_GENERATION
    params = dict(DEFAULT_PROMPT_GENERATION)
    params["batch_size"] = 64
    for key in ("candidate_threshold", "foreground_threshold", "min_candidate_size"):
        if getattr(args, key) is not None:
            params[key] = getattr(args, key)

    segmenter = build_segmenter(args.model_type, args.ndim, args.device)
    spacing = DATASET_SPACING.get(args.dataset_name)
    stages = ("seeded", "proposed", "scored", "merged") if args.ndim == 2 \
        else ("seeded", "scored", "propagated", "merged")
    rows = []
    for index, (raw, labels, _) in enumerate(_load_data(args.dataset_name, DATA_ROOT, ndim=args.ndim)):
        if index >= args.n_samples:
            break
        if args.ndim == 3:
            result = diagnose_volume(segmenter, raw, labels, params, spacing=spacing)
        else:
            result = diagnose_image(segmenter, raw, labels, params)
        if result is not None:
            rows.append(result)
            print(f"sample {index}: {result}")

    table = pd.DataFrame(rows).sum()
    n_gt = table["n_gt"]
    print()
    print(f"{args.dataset_name}: {len(rows)} sample(s), {n_gt} ground-truth objects, "
          f"{table['n_candidates']} candidates "
          f"({table['n_candidates'] / n_gt:.2f} per object)")
    print("candidate_threshold={candidate_threshold}, foreground_threshold={foreground_threshold}, "
          "min_candidate_size={min_candidate_size}".format(**params))
    previous = n_gt
    for stage in stages:
        value = table[stage]
        print(f"{stage}: {value / n_gt:.3f} of ground truth (lost {(previous - value) / n_gt:+.3f} at this stage)")
        previous = value

    if args.ndim == 3:
        print()
        n_unseeded = table["n_unseeded"]
        print(f"Why objects are never seeded (foreground_threshold={params['foreground_threshold']}):")
        print(f"objects whose peak foreground is below the gate: {table['below_gate']} "
              f"({table['below_gate'] / n_gt:.3f} of ground truth)")
        if n_unseeded:
            print(f"of the {n_unseeded} unseeded objects, {table['unseeded_below_gate']} are below the gate "
                  f"({table['unseeded_below_gate'] / n_unseeded:.3f})")
            print(f"mean size unseeded {table['unseeded_size_sum'] / n_unseeded:.0f} px vs "
                  f"seeded {table['seeded_size_sum'] / max(table['n_seeded_objects'], 1):.0f} px")
        return

    n_on_object = table["n_prompts_on_object"]
    print()
    print(f"Per prompt that lands inside an object ({n_on_object} of {table['n_candidates']} candidates), "
          f"the fraction whose kept mask matches its target, by how that mask is chosen:")
    baseline = table["matched_score"]
    for name in SELECTORS:
        matched = table[f"matched_{name}"]
        print(f"{name}: {matched / n_on_object:.3f} of prompts ({matched / baseline - 1:+.1%} over 'score'), "
              f"{table[f'objects_{name}']} of {n_gt} objects")

    n_failed = table["fail_cluster"] + table["fail_fragment"] + table["fail_misaligned"]
    if n_failed:
        print()
        print(f"How the {n_failed} prompts that 'score' gets wrong fail:")
        for kind in ("cluster", "fragment", "misaligned"):
            value = table[f"fail_{kind}"]
            print(f"{kind}: {value} ({value / n_failed:.3f} of failures)")

    n_fragments = table["fail_fragment"]
    if n_fragments:
        print()
        print("Shape of the under-segmentations, which says whether a box prompt could recover them:")
        print(f"connected components per fragment: {table['fragment_components'] / n_fragments:.2f}")
        print(f"share of the fragment in its largest component: {table['fragment_largest_share'] / n_fragments:.3f}")
        print(f"fragments containing their own prompt point: {table['fragment_holds_point'] / n_fragments:.3f}")


def diagnose_volume(segmenter, raw, labels, params, spacing=None, iou_threshold=0.5):
    """Return the per-stage set of ground-truth ids that survive, for one volume.

    The volumetric pipeline loses objects at different places than the 2d one, so the stages differ:

        seeded: a candidate point lands inside the object
        scored: that candidate survives 'score_threshold' and the merge on its anchor slice, which is
            what gates the propagation
        propagated: the mask propagated from it matches the object at IoU >= 'iou_threshold'
        merged: it is still there after the volumetric merge, i.e. it is in the output

    The gap between 'scored' and 'propagated' is the one the 2d pipeline has no counterpart for: the
    prompt was good enough to be kept, but tracking it through z did not reproduce the object.
    """
    from micro_sam.v2.automatic_prompt_generation import derive_volume_prompts, merge_by_score

    gt_ids = np.unique(labels)
    gt_ids = gt_ids[gt_ids != 0]
    if gt_ids.size == 0:
        return None
    gt_sizes = {int(i): int(s) for i, s in zip(*np.unique(labels[labels != 0], return_counts=True))}

    segmenter.clear_state()
    segmenter.initialize(raw, ndim=3, **VOLUME_SPEED_OPTIONS)
    prompts = derive_volume_prompts(
        segmenter._prediction[0], segmenter._prediction[1:],
        candidate_threshold=params["candidate_threshold_3d"], foreground_threshold=params["foreground_threshold"],
        n_iter=params["n_iter"], dt=params["dt"], sigma=params["sigma"], spacing=spacing,
        min_candidate_size=params["min_candidate_size"],
    )
    if prompts is None:
        return {"n_gt": len(gt_ids), "n_candidates": 0, "seeded": 0, "scored": 0, "propagated": 0, "merged": 0}

    points, frames = prompts["points"][:, 0, :], prompts["frames"]
    seeded = set()
    for (x, y), z in zip(points, frames):
        gt_id = int(labels[int(z), int(np.clip(y, 0, labels.shape[1] - 1)), int(np.clip(x, 0, labels.shape[2] - 1))])
        if gt_id:
            seeded.add(gt_id)

    candidates = segmenter._score_candidates(
        prompts, multimasking=params["multimasking"], batch_size=params["batch_size"],
        score_threshold=params["score_threshold"], max_overlap=params["max_overlap"],
    )
    scored = set()
    for candidate in candidates:
        x, y = candidate["point"]
        gt_id = int(labels[candidate["frame"], int(np.clip(y, 0, labels.shape[1] - 1)),
                           int(np.clip(x, 0, labels.shape[2] - 1))])
        if gt_id:
            scored.add(gt_id)

    records = segmenter._propagate_candidates(
        candidates, n_objects_per_pass=params["n_objects_per_pass"],
        early_stop_patience=params["early_stop_patience"], verbose=False,
    )
    propagated = set()
    for record in records:
        box = record["bounding_box"]
        gt_id, iou = match_iou(record["segmentation"], labels[box], gt_sizes)
        if iou >= iou_threshold:
            propagated.add(gt_id)

    segmentation = merge_by_score(
        records, labels.shape, max_overlap=params["max_overlap"], min_size=params["min_size"],
    )
    merged = set()
    for instance_id in np.unique(segmentation):
        if instance_id == 0:
            continue
        gt_id, iou = match_iou(segmentation == instance_id, labels, gt_sizes)
        if iou >= iou_threshold:
            merged.add(gt_id)

    # Why an object is never seeded: 'foreground_threshold' bounds which voxels a candidate can come
    # from at all, so an object the decoder never scores above it cannot be proposed at any density.
    from scipy.ndimage import maximum as label_maximum
    ids = np.array(sorted(gt_sizes))
    peak_foreground = np.asarray(label_maximum(segmenter._prediction[0], labels, index=ids), dtype="float64")
    gate = params["foreground_threshold"]
    unseeded = np.array([i not in seeded for i in ids])
    sizes = np.array([gt_sizes[int(i)] for i in ids])

    return {
        "n_gt": len(gt_ids), "n_candidates": len(points), "seeded": len(seeded & set(gt_sizes)),
        "scored": len(scored), "propagated": len(propagated), "merged": len(merged),
        "below_gate": int((peak_foreground < gate).sum()),
        "unseeded_below_gate": int((unseeded & (peak_foreground < gate)).sum()),
        "unseeded_size_sum": int(sizes[unseeded].sum()), "n_unseeded": int(unseeded.sum()),
        "seeded_size_sum": int(sizes[~unseeded].sum()), "n_seeded_objects": int((~unseeded).sum()),
    }


if __name__ == "__main__":
    main()
