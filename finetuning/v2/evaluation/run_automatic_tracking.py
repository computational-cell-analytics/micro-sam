"""Automatic tracking with micro-sam2 and Trackastra on the Cell Tracking Challenge (CTC) data, 2d+t and 3d+t.

micro-sam2 segments each frame on its own (AIS or APG, the library defaults of the model), then Trackastra links the
objects over time. The result is written in CTC format and scored against the CTC ground truth: the tracking metrics
of traccuracy (TRA, DET, LNK, divisions) on the TRA markers, and the CTC SEG score on the annotated frames.

Besides the CTC sets it runs on the BacMother test videos and the TOIAM test movie, which are in the CTC format too.
The result table marks the datasets that micro-sam2 trained on ('in_training').

Usage examples:
    python run_automatic_tracking.py -d Fluo-N2DH-GOWT1 -s 01 --mode ais
    python run_automatic_tracking.py -d Fluo-C3DH-A549 -s 02 --mode apg
    python run_automatic_tracking.py --list
"""

import os
import time
import argparse
import warnings

import numpy as np
import pandas as pd
from tqdm import trange

import torch

from micro_sam.v1.multi_dimensional_segmentation import track_across_frames
from micro_sam.v2.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter

from tracking_util import (
    EXPERIMENT_ROOT, SPARSE_GT_DATASETS, LazyFrames, automatic_datasets, ctc_seg_score, gt_folder_of, in_training,
    is_3d, load_sequence, restrict_to_annotated, sequences_of, set_norm_percentiles, tracking_metrics,
    use_tmpdir_for_embeddings,
)


def segment_frames(timeseries, predictor, segmenter, ndim, device):
    """Segment every frame on its own and give the objects ids that are unique over all frames."""
    segmentation = np.zeros(timeseries.shape, dtype="uint32")
    offset = 0
    for t in trange(timeseries.shape[0], desc="Segment frames"):
        frame_seg = automatic_instance_segmentation(
            predictor=predictor, segmenter=segmenter, input_path=timeseries[t], ndim=ndim, device=device,
            verbose=False,
        ).astype("uint32")
        max_id = int(frame_seg.max())
        frame_seg[frame_seg != 0] += offset
        offset += max_id
        segmentation[t] = frame_seg
    return segmentation


def score_tracking(dataset_name, sequence, pred_folder, tracked):
    """Score a CTC prediction; 'tracked' gives its frames for the SEG score.

    For a dataset in `SPARSE_GT_DATASETS` the tracking metrics ignore the predictions of unannotated cells.
    """
    gt_folder = gt_folder_of(dataset_name, sequence)
    if not os.path.exists(os.path.join(pred_folder, "man_track.txt")):
        # Trackastra writes nothing for an empty segmentation, which detects and links nothing.
        warnings.warn(f"The tracking result in {pred_folder} is empty. It scores 0.")
        scores = {"TRA": 0.0, "DET": 0.0, "LNK": 0.0}
    else:
        eval_folder = pred_folder
        if dataset_name in SPARSE_GT_DATASETS:
            eval_folder = restrict_to_annotated(pred_folder, os.path.join(gt_folder, "TRA"), f"{pred_folder}_annotated")
        scores = tracking_metrics(os.path.join(gt_folder, "TRA"), eval_folder)
    scores["SEG"] = ctc_seg_score(gt_folder, tracked)
    scores["annotated_only"] = dataset_name in SPARSE_GT_DATASETS
    return scores


def rescore_tracking(dataset_name, sequence, pred_folder, result_path):
    """Score the stored prediction of a run again and update its result table."""
    old = pd.read_csv(result_path)
    scores = score_tracking(dataset_name, sequence, pred_folder, LazyFrames(pred_folder))
    keep = [c for c in ("dataset", "sequence", "ndim", "in_training", "model", "mode", "tracking_model", "linking_mode",
                        "n_frames", "n_segmented_objects", "segmentation_time", "tracking_time") if c in old.columns]
    result = pd.DataFrame([{**old.iloc[0][keep].to_dict(), **scores}])
    result.to_csv(result_path, index=False)
    print(result.T.to_string(header=False))
    return result


def run_tracking(
    dataset_name, sequence, model_type, mode, output_root, linking_mode, tracking_model, score_only=False,
    norm_percentiles=None,
):
    """Segment and track one CTC sequence, write the CTC result and return the row of scores.

    With 'score_only', the stored prediction of an earlier run is scored again instead.
    """
    volumetric = is_3d(dataset_name)
    tag = f"{dataset_name}_{sequence}_{model_type}_{mode}_{tracking_model}_{linking_mode}"
    if norm_percentiles is not None:
        tag += f"_norm{norm_percentiles[0]}-{norm_percentiles[1]}"
    pred_folder = os.path.join(output_root, "predictions", tag)
    result_path = os.path.join(output_root, "results", f"{tag}.csv")
    if score_only:
        return rescore_tracking(dataset_name, sequence, pred_folder, result_path)
    if os.path.exists(result_path):
        print(f"The result is already stored at '{result_path}'.")
        return pd.read_csv(result_path)

    timeseries = load_sequence(dataset_name, sequence)
    print(f"{dataset_name} {sequence}: {timeseries.shape} {timeseries.dtype}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ndim = 3 if volumetric else 2
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type, segmentation_mode=mode, ndim=ndim, device=device,
    )

    start = time.time()
    segmentation = segment_frames(timeseries, predictor, segmenter, ndim, device)
    segmentation_time = time.time() - start

    start = time.time()
    tracked, _ = track_across_frames(
        timeseries=timeseries, segmentation=segmentation, mode=linking_mode, tracking_model=tracking_model,
        output_folder=pred_folder,
    )
    tracking_time = time.time() - start

    scores = score_tracking(dataset_name, sequence, pred_folder, tracked)
    row = {
        "dataset": dataset_name, "sequence": sequence, "ndim": "3d+t" if volumetric else "2d+t",
        "in_training": in_training(dataset_name), "model": model_type, "mode": mode,
        "tracking_model": tracking_model, "linking_mode": linking_mode, "n_frames": timeseries.shape[0],
        "norm_percentiles": "2,98" if norm_percentiles is None else ",".join(map(str, norm_percentiles)),
        "n_segmented_objects": int(len(np.unique(segmentation)) - 1),
        "segmentation_time": segmentation_time, "tracking_time": tracking_time, **scores,
    }
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    result = pd.DataFrame([row])
    result.to_csv(result_path, index=False)
    print(result.T.to_string(header=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-d", "--dataset_name", type=str, default=None, help="A CTC set like 'Fluo-N2DH-GOWT1', 'BacMother' or 'TOIAM'."
    )
    parser.add_argument("-s", "--sequence", type=str, default="01", help="The sequence, see --list.")
    parser.add_argument("-m", "--model_type", type=str, default="hvit_l_cells")
    parser.add_argument("--mode", type=str, default="ais", choices=("ais", "apg"))
    parser.add_argument("-o", "--output_root", type=str, default=os.path.join(EXPERIMENT_ROOT, "automatic"))
    parser.add_argument("--linking_mode", type=str, default="greedy", choices=("greedy", "greedy_nodiv", "ilp"))
    parser.add_argument(
        "--tracking_model", type=str, default=None,
        help="The Trackastra model. By default 'general_2d' for 2d+t and 'ctc' for 3d+t (the only 3d model).",
    )
    parser.add_argument("--list", action="store_true", help="Print the datasets and their sequences and exit.")
    parser.add_argument("--score_only", action="store_true", help="Score the stored prediction of a run again.")
    parser.add_argument(
        "--norm_percentiles", type=float, nargs=2, default=None,
        help="Lower and upper percentile of the input normalization instead of 2 and 98, e.g. 1 99.8.",
    )
    args = parser.parse_args()

    if args.list:
        for name in automatic_datasets():
            print(
                name, "3d+t" if is_3d(name) else "2d+t", "trained" if in_training(name) else "held-out",
                " ".join(sequences_of(name)),
            )
        return
    if args.dataset_name not in automatic_datasets():
        parser.error(f"Choose one of the datasets: {automatic_datasets()}")
    if args.sequence not in sequences_of(args.dataset_name):
        parser.error(f"Choose one of the sequences of {args.dataset_name}: {sequences_of(args.dataset_name)}")

    use_tmpdir_for_embeddings()
    if args.norm_percentiles is not None:
        set_norm_percentiles(*args.norm_percentiles)
    tracking_model = args.tracking_model or ("ctc" if is_3d(args.dataset_name) else "general_2d")
    run_tracking(
        args.dataset_name, args.sequence, args.model_type, args.mode, args.output_root, args.linking_mode,
        tracking_model, score_only=args.score_only,
        norm_percentiles=None if args.norm_percentiles is None else tuple(args.norm_percentiles),
    )


if __name__ == "__main__":
    main()
