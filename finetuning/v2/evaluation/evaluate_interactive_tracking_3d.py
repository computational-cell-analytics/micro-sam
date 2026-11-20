"""Interactive 3d tracking with micro-sam2: a simulated user tracks every ground-truth track of a 3d+t movie.

The tracking is `micro_sam.v2.prompt_based_tracking.PromptableTracking3D`. For each ground-truth track the user:
1. Draws a box on the largest plane of the object in the first timepoint of the track. Then the user tracks the
   object up to the last timepoint of the track. So the user marks where a track ends or divides.
2. Corrects the track up to 'n_corrections' times. The user finds the timepoint with the lowest 3d IoU below
   'accept_iou'. There the user deletes the prompts and draws the box on the largest plane again.
The script writes the result for each correction budget in CTC format. It scores the result with traccuracy (TRA,
DET, LNK, divisions) and with the SEG score over all timepoints. The lineage of the result is the ground-truth
lineage, because the user starts each daughter track.

The data is a CTC set with dense 3d track masks: Fluo-N3DH-SIM+.

Usage example:
    python evaluate_interactive_tracking_3d.py -d Fluo-N3DH-SIM+ -s 02 --frames 0 20
"""

import os
import glob
import time
import shutil
import argparse
import tempfile

import tifffile
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from micro_sam.sam_annotator._state import AnnotatorState

from interactive_tracking import box_of, iou
from common import use_tmpdir_for_embeddings
from tracking_util import (
    EXPERIMENT_ROOT, dense_seg_score, frame_window, gt_folder_of, load_frames, tracking_metrics, write_ctc_tracks,
)


def load_movie(dataset_name, sequence, first, n_frames):
    """Load a window of a 3d+t movie: the raw frames, the dense track masks and the lineage table."""
    raw = load_frames(dataset_name, sequence, first, first + n_frames)
    folder = os.path.join(gt_folder_of(dataset_name, sequence), "TRA")
    paths = sorted(glob.glob(os.path.join(folder, "man_track*.tif")))[first:first + n_frames]
    masks = np.stack([tifffile.imread(path) for path in paths])
    tracks = pd.read_csv(
        os.path.join(folder, "man_track.txt"), sep=" ", header=None, names=["track", "start", "end", "parent"]
    )
    _, _, tracks = frame_window(raw, masks, tracks, first, len(raw))
    return raw, masks, tracks


def build_tracker(raw, model_type, embedding_path, device):
    """Compute the embeddings of all planes of the movie and build the tracker, as the tracking annotator does."""
    state = AnnotatorState()
    state.reset_state()
    state.initialize_predictor(
        raw, model_type=model_type, save_path=embedding_path, ndim=4, device=device, use_cli=True,
    )
    return state.interactive_segmenter


def add_box(tracker, gt_volume, t):
    """Draw the box of the object on its largest plane."""
    z = int(np.argmax(gt_volume.sum(axis=(1, 2))))
    tracker.add_box_prompts(t, z, [box_of(gt_volume[z])])


def track_ious(track, gt_masks, gt_track, frames):
    """Get the 3d IoU of a predicted track with a ground-truth track on each timepoint."""
    return {t: iou(track[t], gt_masks[t] == gt_track) if t in track else 0.0 for t in frames}


def annotate_track(tracker, gt_masks, gt_track, n_corrections, accept_iou, box_every_timepoint=False):
    """Annotate one ground-truth track and return the predicted track after each correction budget.

    With 'box_every_timepoint' the user draws the box on every timepoint. Then the tracker does not propagate the
    object through time.
    """
    frames = [t for t in range(gt_masks.shape[0]) if np.any(gt_masks[t] == gt_track)]
    t_range = (frames[0], frames[-1])
    tracker.reset_predictor()
    boxed = set(frames) if box_every_timepoint else {frames[0]}
    for t in sorted(boxed):
        add_box(tracker, gt_masks[t] == gt_track, t)
    results = [tracker.predict(t_range=t_range)]
    for _ in range(n_corrections):
        ious = track_ious(results[-1], gt_masks, gt_track, frames)
        candidates = [t for t in frames if t not in boxed]
        if not candidates or min(ious.values()) >= accept_iou:
            break
        t = min(candidates, key=ious.get)
        if ious[t] >= accept_iou:
            break
        tracker.remove_prompts(t)
        add_box(tracker, gt_masks[t] == gt_track, t)
        boxed.add(t)
        results.append(tracker.predict(t_range=t_range))
    # An accepted track stays as it is for the larger budgets.
    results += [results[-1]] * (n_corrections + 1 - len(results))
    return results, [track_ious(track, gt_masks, gt_track, frames) for track in results]


def predicted_tracks(segmentation, gt_tracks):
    """Get the lineage table of a prediction that has the ids and the parents of the ground-truth.

    The function splits a track with a gap at the gap. Each later piece is the child of the piece before it.
    """
    segmentation = segmentation.copy()
    parents = dict(zip(gt_tracks["track"], gt_tracks["parent"]))
    next_id = int(max(parents)) + 1
    rows, last_piece = [], {}
    for track in sorted(parents):
        present = [t for t in range(segmentation.shape[0]) if np.any(segmentation[t] == track)]
        if not present:
            continue
        pieces = np.split(np.array(present), np.where(np.diff(present) > 1)[0] + 1)
        piece_id, parent = track, parents[track]
        for i, piece in enumerate(pieces):
            if i > 0:
                parent, piece_id, next_id = piece_id, next_id, next_id + 1
                for t in piece:
                    segmentation[t][segmentation[t] == track] = piece_id
            rows.append({"track": piece_id, "start": int(piece[0]), "end": int(piece[-1]), "parent": parent})
        last_piece[track] = piece_id
    table = pd.DataFrame(rows)
    # The parent of a child is the last piece of its parent track. A child of a track without pixels is a root.
    is_first_piece = table["track"].isin(parents)
    table.loc[is_first_piece, "parent"] = [last_piece.get(parent, 0) for parent in table.loc[is_first_piece, "parent"]]
    return segmentation, table


def run(
    dataset_name, sequence, frames, model_type, n_corrections, accept_iou, output_root, max_tracks=None,
    box_every_timepoint=False,
):
    """Annotate all tracks of one movie, then write and score the result of each correction budget."""
    tag = f"{dataset_name}_{sequence}_t{frames[0]}-{frames[0] + frames[1]}_{model_type}_c{n_corrections}"
    if max_tracks is not None:
        tag += f"_first{max_tracks}"
    if box_every_timepoint:
        tag += "_boxall"
    result_path = os.path.join(output_root, "results", f"{tag}.csv")
    if os.path.exists(result_path):
        print(f"The result is already stored at '{result_path}'.")
        return pd.read_csv(result_path)

    raw, gt_masks, gt_tracks = load_movie(dataset_name, sequence, *frames)
    if max_tracks is not None:
        gt_tracks = gt_tracks.sort_values(["start", "track"]).head(max_tracks)
        gt_masks = np.where(np.isin(gt_masks, gt_tracks["track"]), gt_masks, 0)
    print(f"{dataset_name} {sequence}: {raw.shape} {raw.dtype}, {len(gt_tracks)} tracks")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_dir = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    budgets = range(n_corrections + 1)
    committed = [np.zeros(gt_masks.shape, dtype="uint16") for _ in budgets]
    iou_rows = []
    try:
        start = time.time()
        tracker = build_tracker(raw, model_type, os.path.join(embedding_dir, "embeddings.zarr"), device)
        embedding_time = time.time() - start

        start = time.time()
        for gt_track in tqdm(sorted(gt_tracks["track"]), desc="Annotate tracks"):
            results, ious = annotate_track(tracker, gt_masks, gt_track, n_corrections, accept_iou, box_every_timepoint)
            for budget in budgets:
                for t, volume_mask in results[budget].items():
                    # A new track does not overwrite a committed object, as in the annotator.
                    committed[budget][t][volume_mask & (committed[budget][t] == 0)] = gt_track
                iou_rows.extend(
                    {"track": gt_track, "t": t, "n_corrections": budget, "iou": value}
                    for t, value in ious[budget].items()
                )
        annotation_time = time.time() - start
    finally:
        shutil.rmtree(embedding_dir, ignore_errors=True)

    pred_root = os.path.join(output_root, "predictions", tag)
    gt_folder = os.path.join(pred_root, "gt", "TRA")
    write_ctc_tracks(gt_masks, gt_tracks, gt_folder)
    iou_table = pd.DataFrame(iou_rows)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    iou_table.to_csv(result_path.replace(".csv", "_ious.csv"), index=False)

    rows = []
    for budget in budgets:
        segmentation, tracks = predicted_tracks(committed[budget], gt_tracks)
        pred_folder = os.path.join(pred_root, f"corrections_{budget}", "TRA")
        write_ctc_tracks(segmentation, tracks, pred_folder)
        scores = tracking_metrics(gt_folder, pred_folder, dense_gt=True)
        scores["SEG"] = dense_seg_score(gt_masks, segmentation)
        scores["track_iou"] = iou_table[iou_table["n_corrections"] == budget].groupby("track")["iou"].mean().mean()
        rows.append({
            "dataset": dataset_name, "sequence": sequence, "first_frame": frames[0], "n_frames": raw.shape[0],
            "model": model_type, "box_every_timepoint": box_every_timepoint, "n_gt_tracks": len(gt_tracks),
            "embedding_time": embedding_time, "annotation_time": annotation_time, "n_corrections": budget, **scores,
        })
    result = pd.DataFrame(rows)
    result.to_csv(result_path, index=False)
    print(result.T.to_string(header=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--dataset_name", type=str, default="Fluo-N3DH-SIM+")
    parser.add_argument("-s", "--sequence", type=str, default="02")
    parser.add_argument(
        "--frames", type=int, nargs=2, default=(0, 20), help="The first timepoint and the number of timepoints.",
    )
    parser.add_argument("-m", "--model_type", type=str, default="hvit_l_cells")
    parser.add_argument("--n_corrections", type=int, default=2, help="The largest number of corrections per track.")
    parser.add_argument("--accept_iou", type=float, default=0.9, help="The user accepts a track above this 3d IoU.")
    parser.add_argument(
        "--box_every_timepoint", action="store_true",
        help="The reference without tracking: the user draws the box on every timepoint.",
    )
    parser.add_argument("--max_tracks", type=int, default=None, help="Only annotate the first tracks, for a test.")
    parser.add_argument("-o", "--output_root", type=str, default=os.path.join(EXPERIMENT_ROOT, "interactive_3d"))
    args = parser.parse_args()

    use_tmpdir_for_embeddings()
    run(
        args.dataset_name, args.sequence, tuple(args.frames), args.model_type, args.n_corrections, args.accept_iou,
        args.output_root, max_tracks=args.max_tracks, box_every_timepoint=args.box_every_timepoint,
    )


if __name__ == "__main__":
    main()
