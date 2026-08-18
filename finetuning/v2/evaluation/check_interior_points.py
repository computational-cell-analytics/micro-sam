"""Check that the fast interior points are the points the whole-image distance transform gives.

`micro_sam.v2.automatic_prompt_generation.interior_points` replaces a full-image boundary and
distance transform with one per component. That is only a speed-up if it proposes the very same
prompts, so this compares the two on real candidate maps and on real segmentations, and times them.

Usage:
    python check_interior_points.py -n 20
"""

import time
import argparse

import numpy as np
from tqdm import tqdm

from micro_sam.v1.instance_segmentation import _get_centers
from micro_sam.v2.postprocessing import _compute_flow_density
from micro_sam.v2.automatic_prompt_generation import interior_points, DEFAULT_PROMPT_GENERATION

from bioimage_cpp.segmentation import label

from benchmark_apg_2d import build_segmenter, load_livecell_subset


def candidate_map(prediction, params):
    """The labelled candidate components `derive_point_prompts` proposes from."""
    foreground, directed_distances = prediction[0], prediction[1:][-2:]
    density = _compute_flow_density(
        directed_distances, foreground > params["foreground_threshold"],
        n_iter=int(params["n_iter"]), dt=params["dt"], sigma=params["sigma"], n_threads=8,
    )
    candidates = label(density > params["candidate_threshold"])
    ids, sizes = np.unique(candidates, return_counts=True)
    discard = ids[(sizes < params["min_candidate_size"]) & (ids > 0)]
    if discard.size:
        candidates[np.isin(candidates, discard)] = 0
    return candidates


def n_outside(labels, points):
    """How many points do not lie in the component they were derived for."""
    ids = np.unique(labels)
    ids = ids[ids != 0]
    return sum(1 for label_id, point in zip(ids, points) if labels[tuple(point)] != label_id)


def compare(labels, times, totals):
    """Compare both implementations on one labelled image, accumulating runtimes and mismatches."""
    start = time.perf_counter()
    reference = _get_centers(labels)
    times["_get_centers"] += time.perf_counter() - start

    start = time.perf_counter()
    fast = interior_points(labels)
    times["interior_points"] += time.perf_counter() - start

    totals["points"] += len(reference)
    totals["different"] += int((reference != fast).any(axis=1).sum())
    totals["outside_v1"] += n_outside(labels, reference)
    totals["outside_v2"] += n_outside(labels, fast)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--n_images", type=int, default=20, help="LIVECell test images to check.")
    parser.add_argument("-m", "--model_type", default="hvit_t", help="SAM2 backbone of the joint model.")
    parser.add_argument("--device", default="cuda", help="Device to run on.")
    args = parser.parse_args()

    params = dict(DEFAULT_PROMPT_GENERATION)
    segmenter = build_segmenter(args.model_type, args.device)
    samples = load_livecell_subset(1)[:args.n_images]

    keys = ("points", "different", "outside_v1", "outside_v2")
    totals = {kind: dict.fromkeys(keys, 0) for kind in ("candidates", "segmentation")}
    times = {"_get_centers": 0.0, "interior_points": 0.0}
    for _, image, _ in tqdm(samples, desc="Check interior points"):
        segmenter.initialize(image, ndim=2)
        candidates = candidate_map(segmenter._prediction, params)
        segmentation = segmenter.generate()
        segmenter.clear_state()

        compare(candidates, times, totals["candidates"])
        compare(segmentation, times, totals["segmentation"])

    print()
    for kind, counts in totals.items():
        print(f"{kind}: {counts['different']} of {counts['points']} points differ, "
              f"{counts['outside_v1']} of the v1 points and {counts['outside_v2']} of the new ones "
              f"lie outside their component")
    for name, seconds in times.items():
        print(f"{name}: {seconds / (2 * len(samples)):.4f} s per call")


if __name__ == "__main__":
    main()
