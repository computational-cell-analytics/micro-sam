"""Shared helpers of the micro-sam2 tracking evaluation: the data, the CTC export and the metrics.

The automatic tracking runs on data in the Cell Tracking Challenge (CTC) format: the CTC training sequences (2d+t and
3d+t), the BacMother test videos and the TOIAM test movie. The interactive tracking needs dense track masks in every
frame to place the prompts of the simulated user, so it runs on DynamicNuclearNet, TOIAM (in-domain), BacMother and the
real movies of 'tracking_fm_data' (out-of-domain).
"""

import os
import re
import glob
import warnings

import zarr
import tifffile
import numpy as np
import pandas as pd

from common import DATA_ROOT

CTC_ROOT = os.path.join(DATA_ROOT, "ctc", "train")
EXPERIMENT_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/tracking"

# The CTC sets of the micro-sam2 generalist training (see 'micro_sam.v2.datasets.generalist_loader.CTC_DATASETS').
CTC_TRAINED = (
    "BF-C2DL-HSC", "BF-C2DL-MuSC", "DIC-C2DH-HeLa", "Fluo-C2DL-Huh7", "Fluo-C2DL-MSC", "Fluo-N2DH-SIM+",
    "PhC-C2DH-U373", "PhC-C2DL-PSC",
)
CTC_SEQUENCES = ("01", "02")

# The other 2D+t data in CTC format and whether micro-sam2 trained on it; TOIAM movie 04 is held out.
TRACKING_DATA = os.path.join(EXPERIMENT_ROOT, "data")
CONVERTED_DATASETS = ("EColi", "ALFI", "Cardioblast", "WingDisc", "YeaZ", "CellACDC", "DynamicNuclearNet")
# Held-out movies from the training sources: the YeaZ phase-contrast movies and the Cell-ACDC test movie.
CONVERTED_IN_DOMAIN = ("YeaZ", "CellACDC", "DynamicNuclearNet")
OTHER_CTC_DATASETS = {
    "BacMother": (os.path.join(DATA_ROOT, "bac_mother", "CTC", "test"), False),
    "TOIAM": (os.path.join(DATA_ROOT, "toiam", "data"), True),
    **{name: (os.path.join(TRACKING_DATA, name), name in CONVERTED_IN_DOMAIN) for name in CONVERTED_DATASETS},
}
TOIAM_TEST_MOVIES = ("04",)
# TOIAM 04 is too large for the simulated user: frames 300 to 399 only.
TOIAM_INTERACTIVE_FRAMES = (300, 100)

DNN_ROOT = os.path.join(DATA_ROOT, "deepcell_tracking_data", "test")
FM_ROOT = os.path.join(DATA_ROOT, "tracking_fm_data", "new_data_131026", "real")

# The interactive datasets and their domain: DynamicNuclearNet and TOIAM were part of the micro-sam2 training.
INTERACTIVE_DATASETS = {
    "dynamicnuclearnet": "in-domain", "toiam": "in-domain", "bacmother": "out-of-domain",
    "tracking_fm_real": "out-of-domain", "ecoli": "out-of-domain", "alfi": "out-of-domain",
    "cardioblast": "out-of-domain", "wingdisc": "out-of-domain", "yeaz": "in-domain", "cellacdc": "in-domain",
}
# The interactive datasets that are read from the CTC folders of an automatic dataset.
INTERACTIVE_CTC = {
    "toiam": "TOIAM", "bacmother": "BacMother", "ecoli": "EColi", "alfi": "ALFI", "cardioblast": "Cardioblast",
    "wingdisc": "WingDisc", "yeaz": "YeaZ", "cellacdc": "CellACDC",
}


def ctc_datasets():
    """Get the names of all CTC sets on disk."""
    return sorted(name for name in os.listdir(CTC_ROOT) if os.path.isdir(os.path.join(CTC_ROOT, name, "01")))


def automatic_datasets():
    """Get the names of all datasets for the automatic tracking: the CTC sets, BacMother and TOIAM."""
    return ctc_datasets() + sorted(OTHER_CTC_DATASETS)


def dataset_root(name):
    return OTHER_CTC_DATASETS[name][0] if name in OTHER_CTC_DATASETS else os.path.join(CTC_ROOT, name)


def sequences_of(name):
    """Get the sequences of a dataset: '01' and '02' for CTC, the test movie of TOIAM, the test videos of BacMother."""
    if name == "TOIAM":
        return TOIAM_TEST_MOVIES
    if name == "BacMother":
        return sorted(folder for folder in os.listdir(dataset_root(name)) if folder.isdigit())
    if name in CONVERTED_DATASETS:
        return sorted(folder for folder in os.listdir(dataset_root(name)) if not folder.endswith("_GT"))
    return CTC_SEQUENCES


def is_3d(name):
    """Check whether a dataset is 3d+t; a CTC set tells it by the second part of its name, e.g. 'C3DH'."""
    return name not in OTHER_CTC_DATASETS and name.split("-")[1][1] == "3"


def in_training(name):
    return OTHER_CTC_DATASETS[name][1] if name in OTHER_CTC_DATASETS else name in CTC_TRAINED


def to_grayscale(frames):
    """Convert RGB frames (T, Y, X, 3), as BacMother stores them, to grayscale."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        return frames
    if all(np.array_equal(frames[..., 0], frames[..., c]) for c in (1, 2)):
        return frames[..., 0]
    return frames.mean(axis=-1).astype(frames.dtype)


def load_sequence(name, sequence):
    """Load the frames of one sequence as a (T, Y, X) or (T, Z, Y, X) array."""
    paths = sorted(glob.glob(os.path.join(dataset_root(name), sequence, "t*.tif")))
    return to_grayscale(np.stack([tifffile.imread(path) for path in paths]))


def gt_folder_of(name, sequence):
    """Get the ground-truth folder of one sequence, with 'TRA' and 'SEG'."""
    return os.path.join(dataset_root(name), f"{sequence}_GT")


def load_ctc_tracks(folder):
    """Load the dense track masks of a CTC folder and its lineage table, columns 'track', 'start', 'end', 'parent'."""
    paths = sorted(glob.glob(os.path.join(folder, "man_track*.tif")))
    masks = np.stack([tifffile.imread(path) for path in paths])
    tracks = pd.read_csv(
        os.path.join(folder, "man_track.txt"), sep=" ", header=None, names=["track", "start", "end", "parent"]
    )
    return masks, tracks


def frame_window(raw, masks, tracks, first, n_frames):
    """Limit a movie and its tracks to a window of frames, numbered from 0. A track whose parent ends before the
    window becomes a root.
    """
    last = first + n_frames - 1
    tracks = tracks[(tracks["start"] <= last) & (tracks["end"] >= first)].copy()
    tracks["parent"] = tracks["parent"].where(tracks["parent"].isin(tracks["track"]), 0)
    tracks["start"] = tracks["start"].clip(lower=first) - first
    tracks["end"] = tracks["end"].clip(upper=last) - first
    return raw[first:last + 1], masks[first:last + 1], tracks


def interactive_samples(dataset_name):
    """Get the names of the movies of an interactive dataset."""
    if dataset_name == "dynamicnuclearnet":
        return sorted(os.listdir(DNN_ROOT))
    if dataset_name == "tracking_fm_real":
        return sorted(os.path.basename(path)[:-len(".ome.zarr")] for path in glob.glob(f"{FM_ROOT}/*.ome.zarr"))
    if dataset_name == "toiam":
        return list(TOIAM_TEST_MOVIES)
    if dataset_name in INTERACTIVE_CTC:
        return list(sequences_of(INTERACTIVE_CTC[dataset_name]))
    raise ValueError(f"Unknown interactive dataset '{dataset_name}'.")


def load_interactive_sample(dataset_name, sample):
    """Load one movie with dense track masks: the raw frames, the track masks and the lineage table."""
    if dataset_name == "dynamicnuclearnet":
        frames = sorted(glob.glob(os.path.join(DNN_ROOT, sample, "01", "*.tif")))
        raw = np.stack([tifffile.imread(path) for path in frames])
        masks, tracks = load_ctc_tracks(os.path.join(DNN_ROOT, sample, "01_GT", "TRA"))
        return raw, masks, tracks
    if dataset_name == "tracking_fm_real":
        f = zarr.open(os.path.join(FM_ROOT, f"{sample}.ome.zarr"), mode="r")
        raw, masks = f["s0"][:], f["labels/cells/s0"][:].astype("uint32")
        return raw, masks, _geff_tracks(f)
    if dataset_name in INTERACTIVE_CTC:
        name = INTERACTIVE_CTC[dataset_name]
        raw = load_sequence(name, sample)
        masks, tracks = load_ctc_tracks(os.path.join(gt_folder_of(name, sample), "TRA"))
        if dataset_name == "toiam":
            raw, masks, tracks = frame_window(raw, masks, tracks, *TOIAM_INTERACTIVE_FRAMES)
        return raw, masks, tracks
    raise ValueError(f"Unknown interactive dataset '{dataset_name}'.")


def _geff_tracks(f):
    """Read the CTC lineage table that the GEFF graph of an OME-Zarr stores; its labels are the CTC track ids."""
    lbep = f["tracks/cells.geff"].attrs["geff"]["extra"]["ctc"]["man_track_lbep"]
    return pd.DataFrame(lbep, columns=["track", "start", "end", "parent"])


def tracking_metrics(gt_folder, pred_folder, gt_track_file=None, dense_gt=False):
    """Compute the tracking metrics of a prediction in CTC format with traccuracy.

    The CTC metrics (TRA, DET, LNK) use the CTC matching, which also works for the marker annotations of the real CTC
    sets. The division metrics need a one-to-one matching: for markers, a linear assignment over any overlap; for
    dense ground-truth, an IoU of at least 0.5.
    """
    from traccuracy import run_metrics
    from traccuracy.loaders import load_ctc_data
    from traccuracy.matchers import CTCMatcher, IOUMatcher
    from traccuracy.metrics import CTCMetrics, DivisionMetrics

    gt = load_ctc_data(gt_folder, track_path=gt_track_file or valid_track_file(gt_folder), name="gt")
    pred_folder = trim_ctc_frames(pred_folder, len(glob.glob(os.path.join(gt_folder, "man_track*.tif"))))
    try:
        pred = load_ctc_data(pred_folder, name="pred")
    except Exception as e:
        # A predicted track can have a gap or an empty parent, which the CTC checks reject.
        warnings.warn(f"The prediction fails the CTC checks ({e}). It is loaded without them.")
        pred = load_ctc_data(pred_folder, name="pred", run_checks=False)
    scores = {}
    results, _ = run_metrics(gt, pred, matcher=CTCMatcher(), metrics=[CTCMetrics()])
    scores.update(results[0]["results"])
    division_matcher = IOUMatcher(iou_threshold=0.5 if dense_gt else 0.0, one_to_one=True)
    try:
        results, _ = run_metrics(gt, pred, matcher=division_matcher, metrics=[DivisionMetrics(max_frame_buffer=2)])
        for frame_buffer, values in results[0]["results"].items():
            for key, value in values.items():
                scores[f"{key} ({frame_buffer})"] = value
    except Exception as e:
        warnings.warn(f"The division metrics failed: {e}")
    return {key: value for key, value in scores.items() if np.isscalar(value)}


def ctc_seg_score(gt_folder, segmentation):
    """Compute the CTC SEG score: the mean IoU of each ground-truth object with the prediction that covers more than
    half of it, 0 without one. The SEG annotations are a subset of the frames; for some 3d sets single slices.
    """
    ious = []
    for path in sorted(glob.glob(os.path.join(gt_folder, "SEG", "man_seg*.tif"))):
        numbers = [int(number) for number in re.findall(r"\d+", os.path.basename(path))]
        gt = tifffile.imread(path)
        pred = segmentation[numbers[0]] if len(numbers) == 1 else segmentation[numbers[0]][numbers[1]]
        if gt.shape != pred.shape:
            raise ValueError(f"The SEG annotation {path} has shape {gt.shape}, the prediction {pred.shape}.")
        ious.extend(_seg_ious(gt, pred))
    return float(np.mean(ious)) if ious else np.nan


def _seg_ious(gt, pred):
    """Get the SEG IoU of each ground-truth object of one frame."""
    gt_ids, gt_sizes = np.unique(gt[gt > 0], return_counts=True)
    pairs, overlaps = np.unique(np.stack([gt[gt > 0], pred[gt > 0]]), axis=1, return_counts=True)
    pred_ids, pred_sizes = np.unique(pred[pred > 0], return_counts=True)
    pred_size = dict(zip(pred_ids, pred_sizes))
    ious = dict.fromkeys(gt_ids.tolist(), 0.0)
    gt_size = dict(zip(gt_ids, gt_sizes))
    for (gt_id, pred_id), overlap in zip(pairs.T, overlaps):
        if pred_id != 0 and overlap > 0.5 * gt_size[gt_id]:
            ious[int(gt_id)] = overlap / (gt_size[gt_id] + pred_size[pred_id] - overlap)
    return list(ious.values())


def write_ctc_tracks(masks, tracks, folder):
    """Write dense track masks and their lineage table in CTC format, for the scoring with traccuracy."""
    os.makedirs(folder, exist_ok=True)
    present = set(np.unique(masks).tolist()) - {0}
    tracks = tracks[tracks["track"].isin(present) & (tracks["start"] <= tracks["end"])]
    tracks[["track", "start", "end", "parent"]].to_csv(
        os.path.join(folder, "man_track.txt"), sep=" ", header=False, index=False
    )
    for t in range(masks.shape[0]):
        tifffile.imwrite(os.path.join(folder, f"man_track{t:04d}.tif"), masks[t].astype("uint32"))


def dense_seg_score(gt_masks, segmentation):
    """Compute the CTC SEG score over all frames of dense ground-truth."""
    ious = []
    for gt, pred in zip(gt_masks, segmentation):
        ious.extend(_seg_ious(gt, pred))
    return float(np.mean(ious)) if ious else np.nan


def split_track_gaps(segmentation, lineages):
    """Split each track at the frames where it is missing, as the CTC format requires.

    Each later piece gets a new id whose parent is the piece before it, and the children of the track move to its last
    piece. The links of a parent without pixels are dropped. The segmentation is changed in place.

    Returns:
        The lineages with one more dict that holds the links of the new pieces.
    """
    frames = {}
    for t in range(segmentation.shape[0]):
        for label_id in np.unique(segmentation[t]):
            if label_id != 0:
                frames.setdefault(int(label_id), []).append(t)
    next_id = int(segmentation.max()) + 1
    last_piece, links = {}, {}
    for label_id, present in frames.items():
        pieces = np.split(np.array(present), np.where(np.diff(present) > 1)[0] + 1)
        previous = label_id
        for piece in pieces[1:]:
            for t in piece:
                segmentation[t][segmentation[t] == label_id] = next_id
            links[previous] = [next_id]
            previous = next_id
            next_id += 1
        last_piece[label_id] = previous
    # A parent without pixels, e.g. a mother track that SAM2 lost, is not written; its children become roots.
    updated = [
        {last_piece.get(parent, parent): children for parent, children in lineage.items() if parent in frames}
        for lineage in lineages
    ]
    return updated + [links]


def valid_track_file(folder):
    """Get the lineage file of a CTC folder, or a copy without the rows that end before they start.

    The official 'Fluo-C3DL-MDA231/02_GT/TRA/man_track.txt' has such a row ('26 4 2 0'), for a label that no mask holds.
    """
    path = os.path.join(folder, "man_track.txt")
    tracks = pd.read_csv(path, sep=" ", header=None, names=["track", "start", "end", "parent"])
    invalid = tracks["start"] > tracks["end"]
    if not invalid.any():
        return path
    warnings.warn(f"Dropping the rows of {path} that end before they start: {tracks[invalid].values.tolist()}")
    fixed_path = os.path.join(EXPERIMENT_ROOT, "gt_fixed", path.replace(os.sep, "_").strip("_"))
    os.makedirs(os.path.dirname(fixed_path), exist_ok=True)
    tracks[~invalid].to_csv(fixed_path, sep=" ", header=False, index=False)
    return fixed_path


def trim_ctc_frames(pred_folder, n_frames):
    """Get a CTC prediction limited to its first 'n_frames', for ground-truth that annotates fewer frames.

    The ground-truth of 'Fluo-N3DH-CE' annotates the first 195 (01) and 190 (02) of 250 frames.
    """
    paths = sorted(glob.glob(os.path.join(pred_folder, "man_track*.tif")))
    if len(paths) <= n_frames:
        return pred_folder
    trimmed = f"{pred_folder.rstrip(os.sep)}_first{n_frames}"
    os.makedirs(trimmed, exist_ok=True)
    for path in paths[:n_frames]:
        link = os.path.join(trimmed, os.path.basename(path))
        if not os.path.exists(link):
            os.symlink(path, link)
    tracks = pd.read_csv(
        os.path.join(pred_folder, "man_track.txt"), sep=" ", header=None, names=["track", "start", "end", "parent"]
    )
    tracks = tracks[tracks["start"] < n_frames].copy()
    tracks["end"] = tracks["end"].clip(upper=n_frames - 1)
    tracks.to_csv(os.path.join(trimmed, "man_track.txt"), sep=" ", header=False, index=False)
    return trimmed


# The CTC sets whose TRA annotation covers only some cells. A prediction of an unannotated cell is no error there.
SPARSE_GT_DATASETS = ("Fluo-N3DL-DRO", "Fluo-N3DL-TRIC")


def restrict_to_annotated(pred_folder, gt_folder, out_folder):
    """Write a CTC prediction limited to the objects that overlap an annotated marker of their frame.

    For ground-truth that annotates only some cells (`SPARSE_GT_DATASETS`): the scoring then ignores the predictions of
    unannotated cells, while missed annotated cells, wrong links, splits and merges still count. A track that loses
    frames is split into linked pieces, as the CTC format requires. The frames are processed one at a time, since the
    3d+t sets do not fit in memory.
    """
    pred_paths = sorted(glob.glob(os.path.join(pred_folder, "man_track*.tif")))
    gt_paths = sorted(glob.glob(os.path.join(gt_folder, "man_track*.tif")))[:len(pred_paths)]
    os.makedirs(out_folder, exist_ok=True)
    frames = {}
    for t, (pred_path, gt_path) in enumerate(zip(pred_paths, gt_paths)):
        pred, gt = tifffile.imread(pred_path), tifffile.imread(gt_path)
        keep = np.setdiff1d(np.unique(pred[gt > 0]), [0])
        pred[~np.isin(pred, keep)] = 0
        tifffile.imwrite(os.path.join(out_folder, f"man_track{t:04d}.tif"), pred.astype("uint32"))
        for label_id in keep:
            frames.setdefault(int(label_id), []).append(t)

    tracks = pd.read_csv(
        os.path.join(pred_folder, "man_track.txt"), sep=" ", header=None, names=["track", "start", "end", "parent"]
    )
    parent_of = dict(zip(tracks["track"].astype(int), tracks["parent"].astype(int)))
    next_id = max(max(frames, default=0), int(tracks["track"].max())) + 1
    relabel, rows, last_piece = {}, [], {}
    for label_id in sorted(frames):
        present = frames[label_id]
        pieces = np.split(np.array(present), np.where(np.diff(present) > 1)[0] + 1)
        previous = None
        for i, piece in enumerate(pieces):
            piece_id = label_id if i == 0 else next_id
            if i > 0:
                next_id += 1
                for t in piece:
                    relabel.setdefault(int(t), {})[label_id] = piece_id
            rows.append([piece_id, int(piece[0]), int(piece[-1]), previous])
            previous = piece_id
        last_piece[label_id] = previous
    for row in rows:
        if row[3] is None:  # The first piece keeps the parent of the track, if that parent was kept.
            parent = parent_of.get(row[0], 0)
            row[3] = last_piece.get(parent, 0)
    for t, mapping in relabel.items():
        path = os.path.join(out_folder, f"man_track{t:04d}.tif")
        frame = tifffile.imread(path)
        for old, new in mapping.items():
            frame[frame == old] = new
        tifffile.imwrite(path, frame)
    pd.DataFrame(rows, columns=["track", "start", "end", "parent"]).to_csv(
        os.path.join(out_folder, "man_track.txt"), sep=" ", header=False, index=False
    )
    return out_folder


class LazyFrames:
    """The frames of a CTC folder, read one at a time, for the SEG score of a prediction that does not fit in memory."""

    def __init__(self, folder):
        self.paths = sorted(glob.glob(os.path.join(folder, "man_track*.tif")))

    def __getitem__(self, t):
        return tifffile.imread(self.paths[t])


def set_norm_percentiles(lower, upper):
    """Replace the 2nd and 98th percentile bounds of the micro-sam2 input normalization."""
    import micro_sam.v2.normalization as normalization
    for function in (normalization.compute_percentile_bounds, normalization.normalize_raw):
        names = function.__code__.co_varnames[:function.__code__.co_argcount]
        defaults = list(function.__defaults__)
        first = len(names) - len(defaults)
        defaults[names.index("lower_percentile") - first] = lower
        defaults[names.index("upper_percentile") - first] = upper
        function.__defaults__ = tuple(defaults)


def frame_paths(name, sequence):
    """Get the paths of the frames of one sequence."""
    return sorted(glob.glob(os.path.join(dataset_root(name), sequence, "t*.tif")))


def load_frames(name, sequence, start, end):
    """Load the frames start to end (exclusive) of one sequence."""
    return to_grayscale(np.stack([tifffile.imread(path) for path in frame_paths(name, sequence)[start:end]]))
