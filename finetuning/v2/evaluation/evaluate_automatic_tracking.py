"""Automatic tracking with micro-sam2 and Trackastra on the Cell Tracking Challenge (CTC) data, 2d+t and 3d+t.

micro-sam2 segments each frame on its own (AIS or APG, the library defaults of the model), then Trackastra links the
objects over time. The result is written in CTC format and scored against the CTC ground truth: the tracking metrics
of traccuracy (TRA, DET, LNK, divisions) on the TRA markers, and the CTC SEG score on the annotated frames.

Besides the CTC sets it runs on the BacMother test videos and the TOIAM test movie, which are in the CTC format too.
'--method' replaces the micro-sam2 segmentation by a baseline, segmented as in `evaluate_automatic_baselines.py`:
'cpsam' (Cellpose-SAM) or 'microsam_v1' (micro-sam v1 AIS with 'vit_l_lm'). Trackastra links all of them the same way.
The result table marks the datasets that micro-sam2 trained on ('in_training').

A large 3d+t sequence can be split: '--frames START END' segments only these frames and stores them, one file per
frame, as short GPU jobs; '--link_only' then links all stored frames with Trackastra and scores the result, e.g. on a
CPU node with enough memory. The result is the same as of a run in one piece.

'--prepare_data' converts the movies of other sources (E. coli, ALFI, Cardioblast, wing disc, YeaZ, Cell-ACDC and
DynamicNuclearNet, see 'tracking_util.PREPARED_DATASETS') to the CTC format once. '--collect' joins the result tables
of all runs and, with '--figure_folder', writes the central table 'cells_automatic_tracking_results.csv'.

Usage examples:
    python evaluate_automatic_tracking.py -d Fluo-N2DH-GOWT1 -s 01 --mode ais
    python evaluate_automatic_tracking.py -d Fluo-C3DH-A549 -s 02 --mode apg
    python evaluate_automatic_tracking.py -d Fluo-N3DL-DRO -s 01 --mode apg --frames 0 5
    python evaluate_automatic_tracking.py -d Fluo-N3DL-DRO -s 01 --mode apg --link_only
    python evaluate_automatic_tracking.py --list
    python evaluate_automatic_tracking.py --prepare_data ALFI
    python evaluate_automatic_tracking.py --collect --figure_folder ../../../scripts/plotting/v2/for_paper/figures
"""

import os
import time
import shutil
import argparse
import tempfile
import warnings

import tifffile
import numpy as np
import pandas as pd
from tqdm import trange

import torch

from micro_sam.v1.multi_dimensional_segmentation import track_across_frames
from micro_sam.v2.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter

from common import use_tmpdir_for_embeddings
from tracking_util import (
    MICROSAM_TRAINING, PREPARED_DATASETS, TRACKING_DATA, modality, prepare_tracking_data, scores_of, write_tables,
    EXPERIMENT_ROOT, SPARSE_GT_DATASETS, LazyFrames, automatic_datasets, ctc_seg_score, gt_folder_of, in_training,
    is_3d, load_frames, load_sequence, restrict_to_annotated, sequences_of, set_norm_percentiles, tracking_metrics,
)


METHODS = ("microsam2", "cpsam", "microsam_v1")
MICROSAM_V1_MODEL = "vit_l_lm"


def method_model(method, model_type):
    """Get the model that a method segments with."""
    return {"microsam2": model_type, "cpsam": "cpsam", "microsam_v1": MICROSAM_V1_MODEL}[method]


def build_frame_segmenter(method, mode, model_type, dataset_name, ndim, device):
    """Get a function that segments one frame with a method."""
    if method == "cpsam":
        from evaluate_automatic_baselines import _load_cellpose, _segment_cellpose
        model = _load_cellpose("cpsam", device)
        return lambda frame: _segment_cellpose(frame, model, ndim, dataset_name)
    if method == "microsam_v1":
        from evaluate_automatic_baselines import _load_microsam_v1, _segment_microsam_v1
        predictor, segmenter = _load_microsam_v1("microsam_ais", MICROSAM_V1_MODEL, None, device)
        return lambda frame: _segment_microsam_v1(frame, predictor, segmenter, ndim)
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type, segmentation_mode=mode, ndim=ndim, device=device,
    )
    return lambda frame: automatic_instance_segmentation(
        predictor=predictor, segmenter=segmenter, input_path=frame, ndim=ndim, device=device, verbose=False,
    )


def segment_frames(timeseries, segment):
    """Segment every frame on its own and give the objects ids that are unique over all frames."""
    segmentation = np.zeros(timeseries.shape, dtype="uint32")
    offset = 0
    for t in trange(timeseries.shape[0], desc="Segment frames"):
        frame_seg = np.asarray(segment(timeseries[t])).astype("uint32")
        max_id = int(frame_seg.max())
        frame_seg[frame_seg != 0] += offset
        offset += max_id
        segmentation[t] = frame_seg
    return segmentation


def segment_frame_range(dataset_name, sequence, method, model_type, mode, start, end, frames_folder):
    """Segment the frames start to end (exclusive) and store each one, with its time, in 'frames_folder'.

    A frame that is already stored is skipped, so a task that was stopped resumes.
    """
    os.makedirs(frames_folder, exist_ok=True)
    todo = [t for t in range(start, end) if not os.path.exists(os.path.join(frames_folder, f"seg_t{t:04d}.tif"))]
    if not todo:
        print(f"Frames {start} to {end} are already stored in '{frames_folder}'.")
        return
    ndim = 3 if is_3d(dataset_name) else 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segment = build_frame_segmenter(method, mode, model_type, dataset_name, ndim, device)
    for t in todo:
        # Another task may have stored this frame meanwhile, e.g. a smaller shard of the same frames.
        if os.path.exists(os.path.join(frames_folder, f"seg_t{t:04d}.tif")):
            continue
        frame = load_frames(dataset_name, sequence, t, t + 1)[0]
        begin = time.time()
        seg = np.asarray(segment(frame)).astype("uint32")
        seconds = time.time() - begin
        # The time goes first, so a stored frame always has its time.
        with open(os.path.join(frames_folder, "times.txt"), "a") as f:
            f.write(f"{t} {seconds:.1f}\n")
        partial = os.path.join(frames_folder, f"seg_t{t:04d}.partial.tif")
        tifffile.imwrite(partial, seg, compression="zlib")
        os.replace(partial, os.path.join(frames_folder, f"seg_t{t:04d}.tif"))
        print(f"frame {t}: {int(seg.max())} objects in {seconds:.0f} s", flush=True)


def load_stored_frames(frames_folder, n_frames):
    """Load the stored frame segmentations, with ids that are unique over all frames, and their summed time."""
    paths = [os.path.join(frames_folder, f"seg_t{t:04d}.tif") for t in range(n_frames)]
    missing = [t for t, path in enumerate(paths) if not os.path.exists(path)]
    if missing:
        raise RuntimeError(f"{len(missing)} frames are not segmented yet in '{frames_folder}', e.g. {missing[:5]}.")
    first = tifffile.imread(paths[0])
    segmentation = np.zeros((n_frames,) + first.shape, dtype="uint32")
    offset = 0
    for t, path in enumerate(paths):
        frame_seg = first if t == 0 else tifffile.imread(path)
        max_id = int(frame_seg.max())
        frame_seg = frame_seg.astype("uint32")
        frame_seg[frame_seg != 0] += offset
        offset += max_id
        segmentation[t] = frame_seg
    times = {}
    with open(os.path.join(frames_folder, "times.txt")) as f:
        for line in f:
            # Shards append to the same file, so their writes can mix and leave empty or broken lines.
            parts = line.split()
            if len(parts) != 2:
                continue
            times[int(parts[0])] = float(parts[1])
    return segmentation, sum(times.values())


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
    columns = (
        "dataset", "sequence", "ndim", "in_training", "model", "mode", "tracking_model", "linking_mode", "n_frames",
        "n_segmented_objects", "segmentation_time", "tracking_time",
    )
    keep = [c for c in columns if c in old.columns]
    result = pd.DataFrame([{**old.iloc[0][keep].to_dict(), **scores}])
    result.to_csv(result_path, index=False)
    print(result.T.to_string(header=False))
    return result


def run_tracking(
    dataset_name, sequence, model_type, mode, output_root, linking_mode, tracking_model, score_only=False,
    norm_percentiles=None, frames=None, link_only=False, method="microsam2",
):
    """Segment and track one CTC sequence, write the CTC result and return the row of scores.

    With 'score_only', the stored prediction of an earlier run is scored again instead. With 'frames' (start, end),
    only these frames are segmented and stored; with 'link_only', the stored frames are linked and scored.
    """
    volumetric = is_3d(dataset_name)
    if method == "microsam2":
        tag = f"{dataset_name}_{sequence}_{model_type}_{mode}_{tracking_model}_{linking_mode}"
    else:  # A baseline has no AIS / APG mode.
        tag = f"{dataset_name}_{sequence}_{method}_{method_model(method, model_type)}_{tracking_model}_{linking_mode}"
    if norm_percentiles is not None:
        tag += f"_norm{norm_percentiles[0]}-{norm_percentiles[1]}"
    pred_folder = os.path.join(output_root, "predictions", tag)
    result_path = os.path.join(output_root, "results", f"{tag}.csv")
    frames_folder = os.path.join(output_root, "frames", tag)
    if score_only:
        return rescore_tracking(dataset_name, sequence, pred_folder, result_path)
    if os.path.exists(result_path):
        print(f"The result is already stored at '{result_path}'.")
        return pd.read_csv(result_path)
    if frames is not None:
        return segment_frame_range(dataset_name, sequence, method, model_type, mode, *frames, frames_folder)

    timeseries = load_sequence(dataset_name, sequence)
    print(f"{dataset_name} {sequence}: {timeseries.shape} {timeseries.dtype}")
    if link_only:
        segmentation, segmentation_time = load_stored_frames(frames_folder, timeseries.shape[0])
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ndim = 3 if volumetric else 2
        segment = build_frame_segmenter(method, mode, model_type, dataset_name, ndim, device)
        start = time.time()
        segmentation = segment_frames(timeseries, segment)
        segmentation_time = time.time() - start

    # Trackastra with pretrained features writes them to './embeddings', so each run links in a folder of its own.
    work_dir, cwd = tempfile.mkdtemp(dir=os.environ.get("TMPDIR")), os.getcwd()
    os.chdir(work_dir)
    try:
        start = time.time()
        tracked, _ = track_across_frames(
            timeseries=timeseries, segmentation=segmentation, mode=linking_mode, tracking_model=tracking_model,
            output_folder=pred_folder,
        )
        tracking_time = time.time() - start
    finally:
        os.chdir(cwd)
        shutil.rmtree(work_dir, ignore_errors=True)

    scores = score_tracking(dataset_name, sequence, pred_folder, tracked)
    row = {
        "dataset": dataset_name, "sequence": sequence, "ndim": "3d+t" if volumetric else "2d+t",
        "in_training": in_training(dataset_name), "method": method, "model": method_model(method, model_type),
        "mode": mode if method == "microsam2" else "-",
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


# The (method, model) names of the central table, as in the segmentation tables.
TABLE_METHODS = {
    "microsam2": ("micro-sam2", "hvit_l"), "microsam_v1": ("micro-sam", "vit_l_lm"), "cpsam": ("cellpose", "cpsam")
}


def central_automatic(table):
    """The automatic table of the paper: one row per dataset, method, mode and Trackastra model."""
    # Only the default 2/98 percentile normalization; the other normalizations were tests.
    table = table[table["norm_percentiles"].isna() | (table["norm_percentiles"] == "2,98")].copy()
    # The first micro-sam2 runs have no method column.
    table["method"] = table["method"].fillna("microsam2")
    table = table.drop_duplicates(["dataset", "sequence", "method", "mode", "tracking_model"], keep="last")
    table["model"] = [TABLE_METHODS[method][1] for method in table["method"]]
    table["method"] = [TABLE_METHODS[method][0] for method in table["method"]]
    # micro-sam runs AIS; the baselines have no mode.
    table.loc[table["method"] == "micro-sam", "mode"] = "ais"
    table.loc[table["method"] == "cellpose", "mode"] = None
    table["ndim"] = table["ndim"].str[0].astype(int)
    keys = ["dataset", "ndim", "in_training", "method", "model", "mode", "tracking_model", "linking_mode"]
    result = scores_of(table, keys)
    result.insert(1, "category", [modality(d)[0] for d in result["dataset"]])
    result.insert(2, "structure", [modality(d)[1] for d in result["dataset"]])
    # The domain is that of micro-sam2: in domain if micro-sam2 trained on the dataset.
    result.insert(3, "domain", ["ID" if seen else "OOD" for seen in result.pop("in_training")])
    result.insert(5, "in_microsam_training", result["dataset"].isin(MICROSAM_TRAINING))
    result["primary_metric"], result["lower_is_better"] = "tra", False
    return result.sort_values(["ndim", "dataset", "method", "mode", "tracking_model"], na_position="first")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-d", "--dataset_name", type=str, default=None, help="A CTC set like 'Fluo-N2DH-GOWT1', 'BacMother' or 'TOIAM'."
    )
    parser.add_argument("-s", "--sequence", type=str, default="01", help="The sequence, see --list.")
    parser.add_argument("-m", "--model_type", type=str, default="hvit_l_cells")
    parser.add_argument("--mode", type=str, default="ais", choices=("ais", "apg"))
    parser.add_argument(
        "--method", type=str, default="microsam2", choices=METHODS,
        help="The segmentation: micro-sam2, or a baseline ('cpsam', 'microsam_v1').",
    )
    parser.add_argument("-o", "--output_root", type=str, default=os.path.join(EXPERIMENT_ROOT, "automatic"))
    parser.add_argument("--linking_mode", type=str, default="greedy", choices=("greedy", "greedy_nodiv", "ilp"))
    parser.add_argument(
        "--tracking_model", type=str, default=None,
        help="The Trackastra model. By default 'general_2d' for 2d+t and 'ctc' for 3d+t (the only 3d model).",
    )
    parser.add_argument("--list", action="store_true", help="Print the datasets and their sequences and exit.")
    parser.add_argument("--score_only", action="store_true", help="Score the stored prediction of a run again.")
    parser.add_argument(
        "--frames", type=int, nargs=2, default=None, help="Only segment and store the frames START to END (exclusive).",
    )
    parser.add_argument("--link_only", action="store_true", help="Link and score the stored frames of '--frames'.")
    parser.add_argument(
        "--norm_percentiles", type=float, nargs=2, default=None,
        help="Lower and upper percentile of the input normalization instead of 2 and 98, e.g. 1 99.8.",
    )
    parser.add_argument(
        "--prepare_data", nargs="*", default=None, choices=list(PREPARED_DATASETS),
        help="Convert these datasets (all if none is given) to the CTC format of the evaluation and exit.",
    )
    parser.add_argument("--data_root", default=TRACKING_DATA, help="Where --prepare_data writes the movies.")
    parser.add_argument("--collect", action="store_true", help="Collect the result tables of all runs.")
    parser.add_argument("--figure_folder", default=None, help="Where --collect writes the central table.")
    args = parser.parse_args()
    if args.prepare_data is not None:
        prepare_tracking_data(args.prepare_data, args.data_root)
        return
    if args.collect:
        keys = ["dataset", "sequence", "ndim", "in_training", "mode"]
        root = os.path.dirname(args.output_root.rstrip("/"))
        write_tables("automatic", keys, central_automatic, root, args.figure_folder)
        return

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
        frames=None if args.frames is None else tuple(args.frames), link_only=args.link_only, method=args.method,
    )


if __name__ == "__main__":
    main()
