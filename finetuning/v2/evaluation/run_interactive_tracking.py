"""Interactive tracking with micro-sam2: a simulated user drives a headless copy of the tracking annotator.

The user annotates every ground-truth lineage of a movie as described in `interactive_tracking.SimulatedUser`: a box
on the first frame of each track, a stop where it ends, a division where it divides, and corrections on the worst
frame. The committed result for each correction budget is written in CTC format and scored with traccuracy (TRA, DET,
LNK, divisions with an IoU matching) and the SEG score over all frames.

Datasets (see `tracking_util.INTERACTIVE_DATASETS`): 'dynamicnuclearnet' (in-domain) and 'tracking_fm_real'
(out-of-domain).

Usage examples:
    python run_interactive_tracking.py -d dynamicnuclearnet -s HeLa-S3_nuc_pos4_q1_5c
    python run_interactive_tracking.py -d tracking_fm_real -s microb_s3z1 --n_corrections 2
    python run_interactive_tracking.py -d dynamicnuclearnet --list
"""

import os
import glob
import json
import time
import pickle
import shutil
import argparse
import tempfile
import warnings

import tifffile
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from micro_sam.v1.multi_dimensional_segmentation import export_tracking_result_to_ctc

from interactive_tracking import SimulatedUser, TrackingAnnotator, V1TrackingAnnotator
from tracking_util import (
    EXPERIMENT_ROOT, INTERACTIVE_DATASETS, dense_seg_score, interactive_samples, load_interactive_sample,
    set_norm_percentiles, split_track_gaps, tracking_metrics, use_tmpdir_for_embeddings, write_ctc_tracks,
)


def export_budgets(user, gt_masks, gt_tracks, budgets, pred_root, info):
    """Export the ground-truth and the result of each correction budget in CTC format.

    The prompt counts and timings go to 'info.json', and 'complete' marks that all budgets are written, so a later
    run can score again without annotating again.
    """
    write_ctc_tracks(gt_masks, gt_tracks, os.path.join(pred_root, "gt", "TRA"))
    info = dict(info, n_prompts={})
    for budget, committed, lineages, n_prompts in user.results_per_budget(budgets):
        tracks = committed.copy()
        pred_folder = os.path.join(pred_root, f"corrections_{budget}")
        export_tracking_result_to_ctc(tracks, split_track_gaps(tracks, lineages), pred_folder)
        info["n_prompts"][str(budget)] = n_prompts
    with open(os.path.join(pred_root, "info.json"), "w") as f:
        json.dump(info, f)
    open(os.path.join(pred_root, "complete"), "w").close()


def score_budgets(gt_masks, budgets, pred_root):
    """Score the exported result of each correction budget. A budget whose scoring fails gets NaN, with a warning."""
    with open(os.path.join(pred_root, "info.json")) as f:
        info = json.load(f)
    gt_folder = os.path.join(pred_root, "gt", "TRA")
    rows = []
    for budget in budgets:
        pred_folder = os.path.join(pred_root, f"corrections_{budget}", "TRA")
        try:
            scores = tracking_metrics(gt_folder, pred_folder, dense_gt=True)
        except Exception as e:
            warnings.warn(f"The tracking metrics of budget {budget} in {pred_root} failed: {e}")
            scores = {"TRA": np.nan, "DET": np.nan, "LNK": np.nan}
        paths = sorted(glob.glob(os.path.join(pred_folder, "man_track*.tif")))
        pred = np.stack([tifffile.imread(path) for path in paths])
        scores["SEG"] = dense_seg_score(gt_masks, pred)
        rows.append({"n_corrections": budget, "n_prompts": info["n_prompts"][str(budget)], **scores})
    return info, rows


# The interactive tools: the micro-sam2 annotator with the micro-sam2 model or stock SAM2, and the micro-sam v1
# annotator with its light microscopy model. Each has its default model.
METHODS = {"microsam2": "hvit_l_cells", "sam2": "hvit_l", "microsam_v1": "vit_l_lm"}


def merge_parts(user, parts_folder):
    """Put the lineage records of all annotated ranges into 'user', in the order of its roots.

    Returns:
        The summed embedding and annotation times, or None if a range is missing.
    """
    n_roots = len(user.roots())
    parts = sorted(glob.glob(os.path.join(parts_folder, "part_*.pkl")))
    covered, records, times = 0, [], {"embedding_time": 0.0, "annotation_time": 0.0}
    for path in parts:
        with open(path, "rb") as f:
            part = pickle.load(f)
        if part["start"] != covered:
            break
        covered = part["end"]
        records += part["records"]
        for key in times:
            times[key] += part[key]
    if covered < n_roots:
        print(f"The ranges in '{parts_folder}' cover {covered} of {n_roots} lineages.")
        return None
    user.lineage_records = records
    return times


def run_sample(
    dataset_name, sample, model_type, first_prompt, n_corrections, accept_iou, output_root, correction,
    norm_percentiles=None, method="microsam2", lineages=None, merge=False,
):
    """Annotate one movie with the simulated user and return its rows of scores.

    With 'lineages' (start, end), only that range of the lineages is annotated and its records are stored; 'merge'
    then joins the records of all ranges, exports and scores them. The result equals that of a run in one piece.
    """
    # The point corrections keep the plain name of the first runs.
    correction_tag = "" if correction == "points" else f"_{correction}corr"
    tag = f"{dataset_name}_{sample}_{model_type}_{first_prompt}{correction_tag}_c{n_corrections}_iou{accept_iou}"
    if norm_percentiles is not None:
        tag += f"_norm{norm_percentiles[0]}-{norm_percentiles[1]}"
    result_path = os.path.join(output_root, "results", f"{tag}.csv")
    if os.path.exists(result_path):
        print(f"The result is already stored at '{result_path}'.")
        return pd.read_csv(result_path)

    raw, gt_masks, gt_tracks = load_interactive_sample(dataset_name, sample)
    print(f"{dataset_name} {sample}: {raw.shape} {raw.dtype}, {len(gt_tracks)} tracks")
    budgets = range(n_corrections + 1)
    pred_root = os.path.join(output_root, "predictions", tag)
    parts_folder = os.path.join(pred_root, "lineage_parts")
    if lineages is not None:
        part_path = os.path.join(parts_folder, f"part_{lineages[0]:05d}-{lineages[1]:05d}.pkl")
        if os.path.exists(part_path):
            print(f"The lineages {lineages} are already stored at '{part_path}'.")
            return None
    if os.path.exists(os.path.join(pred_root, "complete")):
        print(f"The annotation is already exported at '{pred_root}'. It is only scored.")
    elif merge:
        user = SimulatedUser(
            None, gt_masks, gt_tracks, first_prompt=first_prompt, n_corrections=n_corrections, accept_iou=accept_iou,
            correction=correction,
        )
        info = merge_parts(user, parts_folder)
        if info is None:
            raise RuntimeError(f"Not all lineage ranges of {tag} are annotated yet.")
        export_budgets(user, gt_masks, gt_tracks, budgets, pred_root, info)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_dir = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
        try:
            start = time.time()
            annotator_class = V1TrackingAnnotator if method == "microsam_v1" else TrackingAnnotator
            annotator = annotator_class(raw, model_type, os.path.join(embedding_dir, "embeddings.zarr"), device)
            embedding_time = time.time() - start

            start = time.time()
            user = SimulatedUser(
                annotator, gt_masks, gt_tracks, first_prompt=first_prompt, n_corrections=n_corrections,
                accept_iou=accept_iou, correction=correction,
            )
            user.run(progress=lambda roots: tqdm(roots, desc="Annotate lineages"), lineages=lineages)
            annotation_time = time.time() - start
        finally:
            shutil.rmtree(embedding_dir, ignore_errors=True)
        info = {"embedding_time": embedding_time, "annotation_time": annotation_time}
        if lineages is not None:
            os.makedirs(parts_folder, exist_ok=True)
            part = {
                "start": lineages[0], "end": min(lineages[1], len(user.roots())), "records": user.lineage_records,
                **info
            }
            with open(part_path + ".partial", "wb") as f:
                pickle.dump(part, f)
            os.replace(part_path + ".partial", part_path)
            print(f"Stored the lineages {lineages} at '{part_path}'.")
            return None
        export_budgets(user, gt_masks, gt_tracks, budgets, pred_root, info)

    info, rows = score_budgets(gt_masks, budgets, pred_root)
    result = pd.DataFrame([
        {
            "dataset": dataset_name, "domain": INTERACTIVE_DATASETS[dataset_name], "sample": sample,
            "method": method, "model": model_type, "first_prompt": first_prompt, "correction": correction,
            "accept_iou": accept_iou,
            "norm_percentiles": "2,98" if norm_percentiles is None else ",".join(map(str, norm_percentiles)),
            "n_frames": raw.shape[0], "n_gt_tracks": len(gt_tracks), "embedding_time": info["embedding_time"],
            "annotation_time": info["annotation_time"], **row,
        }
        for row in rows
    ])
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    result.to_csv(result_path, index=False)
    print(result.T.to_string(header=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--dataset_name", type=str, required=True, choices=sorted(INTERACTIVE_DATASETS))
    parser.add_argument("-s", "--sample", type=str, default=None, help="The movie. All movies by default.")
    parser.add_argument(
        "--method", type=str, default="microsam2", choices=sorted(METHODS),
        help="The tool: the micro-sam2 annotator ('microsam2', 'sam2' with stock SAM2) or the micro-sam v1 annotator.",
    )
    parser.add_argument("-m", "--model_type", type=str, default=None, help="By default the model of the method.")
    parser.add_argument("--first_prompt", type=str, default="box", choices=("box", "point"))
    parser.add_argument(
        "--lineages", type=int, nargs=2, default=None,
        help="Only annotate the lineages START to END (exclusive) and store them, to split a long movie.",
    )
    parser.add_argument("--merge", action="store_true", help="Merge, export and score the stored lineage ranges.")
    parser.add_argument("--n_corrections", type=int, default=4, help="The largest number of corrections per track.")
    parser.add_argument(
        "--correction", type=str, default="box", choices=("points", "box"),
        help="How the user corrects the worst frame: a positive and a negative point, or a new box.",
    )
    parser.add_argument(
        "--norm_percentiles", type=float, nargs=2, default=None,
        help="Lower and upper percentile of the input normalization instead of 2 and 98, e.g. 1 99.8.",
    )
    parser.add_argument("--accept_iou", type=float, default=0.9, help="The user accepts a track above this IoU.")
    parser.add_argument("-o", "--output_root", type=str, default=os.path.join(EXPERIMENT_ROOT, "interactive"))
    parser.add_argument("--list", action="store_true", help="Print the movies of the dataset and exit.")
    args = parser.parse_args()

    samples = interactive_samples(args.dataset_name)
    if args.list:
        print("\n".join(samples))
        return
    if args.sample is not None and args.sample not in samples:
        parser.error(f"Choose one of the movies: {samples}")

    use_tmpdir_for_embeddings()
    if args.norm_percentiles is not None:
        set_norm_percentiles(*args.norm_percentiles)
    model_type = args.model_type or METHODS[args.method]
    for sample in samples if args.sample is None else [args.sample]:
        run_sample(
            args.dataset_name, sample, model_type, args.first_prompt, args.n_corrections, args.accept_iou,
            args.output_root, args.correction,
            norm_percentiles=None if args.norm_percentiles is None else tuple(args.norm_percentiles),
            method=args.method, lineages=None if args.lineages is None else tuple(args.lineages), merge=args.merge,
        )


if __name__ == "__main__":
    main()
