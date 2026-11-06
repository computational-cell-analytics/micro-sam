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

from interactive_tracking import SimulatedUser, TrackingAnnotator
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


def run_sample(
    dataset_name, sample, model_type, first_prompt, n_corrections, accept_iou, output_root, correction,
    norm_percentiles=None,
):
    """Annotate one movie with the simulated user and return its rows of scores."""
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
    if os.path.exists(os.path.join(pred_root, "complete")):
        print(f"The annotation is already exported at '{pred_root}'. It is only scored.")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_dir = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
        try:
            start = time.time()
            annotator = TrackingAnnotator(raw, model_type, os.path.join(embedding_dir, "embeddings.zarr"), device)
            embedding_time = time.time() - start

            start = time.time()
            user = SimulatedUser(
                annotator, gt_masks, gt_tracks, first_prompt=first_prompt, n_corrections=n_corrections,
                accept_iou=accept_iou, correction=correction,
            )
            user.run(progress=lambda roots: tqdm(roots, desc="Annotate lineages"))
            annotation_time = time.time() - start
        finally:
            shutil.rmtree(embedding_dir, ignore_errors=True)
        info = {"embedding_time": embedding_time, "annotation_time": annotation_time}
        export_budgets(user, gt_masks, gt_tracks, budgets, pred_root, info)

    info, rows = score_budgets(gt_masks, budgets, pred_root)
    result = pd.DataFrame([
        {
            "dataset": dataset_name, "domain": INTERACTIVE_DATASETS[dataset_name], "sample": sample,
            "model": model_type, "first_prompt": first_prompt, "correction": correction, "accept_iou": accept_iou,
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
    parser.add_argument("-m", "--model_type", type=str, default="hvit_l_cells")
    parser.add_argument("--first_prompt", type=str, default="box", choices=("box", "point"))
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
    for sample in samples if args.sample is None else [args.sample]:
        run_sample(
            args.dataset_name, sample, args.model_type, args.first_prompt, args.n_corrections, args.accept_iou,
            args.output_root, args.correction,
            norm_percentiles=None if args.norm_percentiles is None else tuple(args.norm_percentiles),
        )


if __name__ == "__main__":
    main()
