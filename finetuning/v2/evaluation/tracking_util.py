"""Shared helpers of the micro-sam2 tracking evaluation: the data, the CTC export and the metrics.

The automatic tracking runs on data in the Cell Tracking Challenge (CTC) format: the CTC training sequences (2d+t and
3d+t), the BacMother test videos and the TOIAM test movie. The interactive tracking needs dense track masks in every
frame to place the prompts of the simulated user, so it runs on DynamicNuclearNet, TOIAM (in-domain), BacMother and the
real movies of 'tracking_fm_data' (out-of-domain).
"""

import os
import re
import glob
import shutil
import sqlite3
import zipfile
import tempfile
import warnings

import zarr
import tifffile
import numpy as np
import pandas as pd
import scipy.io as sio
import imageio.v3 as imageio

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


# The conversion of the movies of other sources to the CTC format of the tracking evaluation.
WING_DISC_CROP = 160


def split_gaps(masks, parents):
    """Split the tracks with a gap in time; return the new masks and the lineage rows (id, start, end, parent)."""
    masks = masks.copy()
    frames = {}
    for t in range(masks.shape[0]):
        for track in np.unique(masks[t])[1:]:
            frames.setdefault(int(track), []).append(t)
    next_id = max(frames, default=0) + 1
    parts_of = {}
    for track in sorted(frames):
        ts = np.array(frames[track])
        parts_of[track] = []
        for k, part in enumerate(np.split(ts, np.flatnonzero(np.diff(ts) > 1) + 1)):
            new_id = track
            if k > 0:
                new_id, next_id = next_id, next_id + 1
                for t in part:
                    masks[t][masks[t] == track] = new_id
            parts_of[track].append((new_id, int(part[0]), int(part[-1])))
    rows = []
    for track, parts in parts_of.items():
        # A daughter hangs on the last part of its mother; a mother without pixels is dropped.
        mother = parents.get(track, 0)
        parent = parts_of[mother][-1][0] if mother in parts_of else 0
        for new_id, start, end in parts:
            rows.append([new_id, start, end, parent])
            parent = new_id
    return masks, rows


def write_sequence(root, sequence, raw, masks, parents):
    """Write one movie and its dense track masks in the CTC format.

    Frames without any mask at the start or the end are left out: traccuracy shifts the frames of a prediction
    against the ground truth when the first frame is empty, so that nothing matches (E. coli starts so).
    """
    annotated = np.flatnonzero(masks.reshape(masks.shape[0], -1).max(axis=1) > 0)
    raw, masks = raw[annotated[0]:annotated[-1] + 1], masks[annotated[0]:annotated[-1] + 1]
    masks, rows = split_gaps(masks.astype("uint32"), parents)
    digits = max(3, len(str(raw.shape[0] - 1)))
    folders = {name: os.path.join(root, *parts) for name, parts in (
        ("raw", (sequence,)), ("tra", (f"{sequence}_GT", "TRA")), ("seg", (f"{sequence}_GT", "SEG")),
    )}
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    for t in range(raw.shape[0]):
        tifffile.imwrite(os.path.join(folders["raw"], f"t{t:0{digits}d}.tif"), raw[t])
        tifffile.imwrite(os.path.join(folders["tra"], f"man_track{t:0{digits}d}.tif"), masks[t].astype("uint16"))
        tifffile.imwrite(os.path.join(folders["seg"], f"man_seg{t:0{digits}d}.tif"), masks[t].astype("uint16"))
    np.savetxt(os.path.join(folders["tra"], "man_track.txt"), np.array(rows, dtype=int), fmt="%d")
    n_divisions = len({row[3] for row in rows if row[3] > 0})
    print(f"{os.path.basename(root)} {sequence}: {raw.shape}, {len(rows)} tracks, {n_divisions} divisions")


def schnitz_parents(path):
    """The parent position of each schnitz, from 1; MATLAB v7.3 files are HDF5 files and need h5py."""
    try:
        schnitzes = sio.loadmat(path, simplify_cells=True)["schnitzcells"]
        return {k: int(np.atleast_1d(s["P"])[0]) for k, s in enumerate(schnitzes, start=1)}
    except NotImplementedError:
        import h5py
        with h5py.File(path, "r") as f:
            refs = f["schnitzcells"]["P"]
            return {k: int(np.array(f[refs[k - 1, 0]]).ravel()[0]) for k in range(1, refs.shape[0] + 1)}


def prepare_ecoli(output):
    from torch_em.data.datasets.light_microscopy.ecoli_microcolony_lineage import _prepare_colony_labels
    root = os.path.join(DATA_ROOT, "ecoli_microcolony_lineage")
    cache = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    try:
        for lineage_path in sorted(glob.glob(os.path.join(root, "*", "*", "data", "*_lin.mat"))):
            if "__MACOSX" in lineage_path:
                continue
            colony_dir = os.path.dirname(os.path.dirname(lineage_path))
            if os.path.exists(os.path.join(output, f"{os.path.basename(colony_dir)}_GT", "TRA", "man_track.txt")):
                continue
            raw_paths, label_paths = _prepare_colony_labels(colony_dir, cache)
            if not raw_paths:
                continue
            raw = np.stack([tifffile.imread(path) for path in raw_paths])
            masks = np.stack([tifffile.imread(path) for path in label_paths])
            # The track ids are the positions of the schnitzes in the file, from 1; 'P' is the parent position.
            parents = {k: max(p, 0) for k, p in schnitz_parents(lineage_path).items()}
            write_sequence(output, os.path.basename(colony_dir), raw, masks, parents)
    finally:
        shutil.rmtree(cache, ignore_errors=True)


def prepare_alfi(output):
    import pandas as pd
    root = os.path.join(DATA_ROOT, "alfi", "Data&Annotations")
    for movie in sorted(os.listdir(root)):
        table = pd.read_csv(os.path.join(root, movie, f"{movie}_DTLTruth.csv"), dtype={"ID": str})
        names = sorted(table["ID"].unique(), key=lambda name: [int(part) for part in name.split(".")])
        track_of = {name: k for k, name in enumerate(names, start=1)}
        # '3.1' is a daughter of '3'.
        parents = {track_of[name]: track_of.get(name.rsplit(".", 1)[0], 0) if "." in name else 0 for name in names}
        frames = sorted(table["ImNo"].unique())
        raw, masks = [], []
        for number in frames:
            image = imageio.imread(os.path.join(root, movie, "Images", f"I_{movie}_{number:04d}.png"))
            labels = tifffile.imread(os.path.join(root, movie, "instances_labels", f"M_{movie}_{number:04d}.tif"))
            boxes = table[table["ImNo"] == number]
            mask = np.zeros(labels.shape, dtype="uint32")
            for label in np.unique(labels)[1:]:
                ys, xs = np.nonzero(labels == label)
                inside = [
                    ((ys >= b.ymin) & (ys < b.ymin + b.height) & (xs >= b.xmin) & (xs < b.xmin + b.width)).mean()
                    for b in boxes.itertuples()
                ]
                if max(inside) > 0.8:
                    mask[labels == label] = track_of[boxes["ID"].iloc[int(np.argmax(inside))]]
            raw.append(image[..., 0] if image.ndim == 3 else image)
            masks.append(mask)
        write_sequence(output, movie, np.stack(raw), np.stack(masks), parents)


def prepare_cardioblast(output):
    root = os.path.join(DATA_ROOT, "cardioblast_nuclei", "cardioblast_nuclei")
    for folder in sorted(glob.glob(os.path.join(root, "*", "cardioblast_nuclei_*"))):
        name = os.path.basename(folder)
        raw = tifffile.imread(os.path.join(folder, f"{name}.tif"))
        masks = tifffile.imread(os.path.join(folder, f"{name}_mask.tif"))
        # One raw movie has unannotated trailing frames (see the torch_em loader).
        annotated = np.flatnonzero(masks.reshape(masks.shape[0], -1).max(axis=1) > 0)
        n_frames = annotated[-1] + 1
        write_sequence(output, name.replace("cardioblast_nuclei_", ""), raw[:n_frames], masks[:n_frames], {})


def prepare_wing_disc(output):
    root = os.path.join(DATA_ROOT, "wing_disc_timelapse")
    for archive in sorted(glob.glob(os.path.join(root, "Ecd_*.zip"))):
        movie = os.path.basename(archive)[:-len(".zip")]
        if os.path.exists(os.path.join(output, f"{movie}_GT", "TRA", "man_track.txt")):
            continue
        work = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
        try:
            with zipfile.ZipFile(archive) as f:
                # Some movies store the colour images as tif instead of png.
                members = [m for m in f.namelist() if m.endswith(".sqlite") or "tracked_cells_resized." in m]
                f.extractall(work, members)
            database = sqlite3.connect(glob.glob(os.path.join(work, movie, "*.sqlite"))[0])
            histories = database.execute(
                "select cell_id, tissue_analyzer_group_id, first_occ, last_occ, left_daughter_cell_id, "
                "right_daughter_cell_id from cell_histories"
            ).fetchall()
            parents = {}
            for cell, _, _, _, left, right in histories:
                for daughter in (left, right):
                    if daughter is not None:
                        parents[int(daughter)] = int(cell)
            images = sorted(glob.glob(os.path.join(root, "converted", movie, "images", "*.tif")))
            frame_folders = {
                int(re.search(r"_(\d+)$", os.path.dirname(path)).group(1)): path
                for path in glob.glob(os.path.join(work, movie, "Segmentation", "*", "tracked_cells_resized.*"))
            }
            raw, masks = [], []
            for path in images:
                t = int(os.path.basename(path)[:-len(".tif")])
                image = tifffile.imread(path)
                path = frame_folders[t]
                colours = tifffile.imread(path) if path.endswith(".tif") else imageio.imread(path)
                colours = colours[..., :3].astype(np.int64)
                group = colours[..., 0] * 65536 + colours[..., 1] * 256 + colours[..., 2]
                assert group.shape == image.shape[:2], (group.shape, image.shape)
                # The group id of a cell, with the frame, gives its cell id: one cell history between divisions.
                cell_of = {g: c for c, g, first, last, _, _ in histories if first <= t <= last}
                y0, x0 = (np.array(group.shape) - WING_DISC_CROP) // 2
                crop = (slice(y0, y0 + WING_DISC_CROP), slice(x0, x0 + WING_DISC_CROP))
                group = group[crop]
                mask = np.zeros(group.shape, dtype="uint32")
                for g in np.unique(group):
                    if g in cell_of:
                        mask[group == g] = cell_of[g]
                raw.append(image[crop])
                masks.append(mask)
            # The cell ids start at 10000; they fit uint16 masks after renumbering.
            masks = np.stack(masks)
            ids = np.unique(masks)[1:]
            new = {int(old): k for k, old in enumerate(ids, start=1)}
            lookup = np.zeros(int(masks.max()) + 1, dtype="uint32")
            lookup[ids] = np.arange(1, len(ids) + 1)
            parents = {new[d]: new.get(p, 0) for d, p in parents.items() if d in new}
            write_sequence(output, movie, np.stack(raw), lookup[masks], parents)
        finally:
            shutil.rmtree(work, ignore_errors=True)


def prepare_yeaz(output):
    """The YeaZ phase-contrast time-lapse movies of the 'val' and 'test' splits with at least 5 frames.

    micro-sam2 trained on YeaZ bright field only, and micro-sam on the 'train' split of the phase-contrast movies, so
    neither model saw these movies. The masks keep the cell ids over time; buds have no parent link.
    """
    import json
    root = os.path.join(DATA_ROOT, "yeaz")
    splits = json.load(open(os.path.join(root, "yeaz_phc_splits.json")))
    for name in splits["val"] + splits["test"]:
        masks = tifffile.imread(os.path.join(root, "gold-standard-PhC-plus-2", name.replace("_im.tif", "_mask.tif")))
        if masks.ndim != 3 or masks.shape[0] < 5:
            continue
        raw = tifffile.imread(os.path.join(root, "gold-standard-PhC-plus-2", name))
        write_sequence(output, name[:-len("_im.tif")], raw, masks, {})


def prepare_cell_acdc(output):
    """The Cell-ACDC test movie of the micro-sam2 split ('CELL_ACDC_TEST_MOVIES'), with the phase-contrast frames.

    Only the frames listed in the 'acdc_output.csv' table are annotated. A bud keeps growing next to its mother, which
    the CTC lineage cannot express (a mother must end where its daughters start), so the buds have no parent link.
    """
    import pandas as pd
    folder = os.path.join(
        DATA_ROOT, "cell_acdc", "test_data_budding_yeast", "TimeLapse_2D", "MIA_KC_htb1_mCitrine_labeled",
        "Position_8", "Images",
    )
    prefix = os.path.join(folder, "19-03-2021_KCY050_SCGE_s08")
    segmentation = np.load(f"{prefix}_segm.npz")
    masks = segmentation[segmentation.files[0]]
    raw = tifffile.imread(f"{prefix}_phase_contr.tif")
    n_frames = int(pd.read_csv(f"{prefix}_acdc_output.csv")["frame_i"].max()) + 1
    write_sequence(output, "Position_8", raw[:n_frames], masks[:n_frames], {})


def prepare_dynamicnuclearnet(output):
    """The DynamicNuclearNet tracking test movies, the in-domain movies of the interactive tracking.

    micro-sam2 trained on the DynamicNuclearNet segmentation data, not on these movies. They are written again in the
    layout of the automatic tracking, with their lineage table.
    """
    from tracking_util import interactive_samples, load_interactive_sample
    for sample in interactive_samples("dynamicnuclearnet"):
        raw, masks, tracks = load_interactive_sample("dynamicnuclearnet", sample)
        parents = dict(zip(tracks["track"].astype(int), tracks["parent"].astype(int)))
        write_sequence(output, sample, raw, masks, parents)


PREPARED_DATASETS = {
    "EColi": prepare_ecoli, "ALFI": prepare_alfi, "Cardioblast": prepare_cardioblast, "WingDisc": prepare_wing_disc,
    "YeaZ": prepare_yeaz, "CellACDC": prepare_cell_acdc, "DynamicNuclearNet": prepare_dynamicnuclearnet
}


# The collection of the per-run result tables into one table per pipeline.
COLUMNS = ["TRA", "DET", "LNK", "SEG", "Division F1 (Frame Buffer 0)", "Division F1 (Frame Buffer 2)"]

# The scores of the central tables, with their names there.
SCORES = {
    "TRA": "tra", "DET": "det", "LNK": "lnk", "SEG": "seg", "AOGM": "aogm",
    "Division F1 (Frame Buffer 0)": "division_f1", "Division F1 (Frame Buffer 2)": "division_f1_buffer2",
    "Mitotic Branching Correctness (Frame Buffer 0)": "mbc",
}

# The modality of each dataset that its CTC name does not give, as (category, structure).
MODALITIES = {
    "BacMother": ("label-free", "cell"), "TOIAM": ("label-free", "cell"),
    "dynamicnuclearnet": ("fluorescence", "nucleus"), "toiam": ("label-free", "cell"),
    "bacmother": ("label-free", "cell"), "tracking_fm_real": ("mixed", "mixed"),
    # ALFI labels interphase nuclei and whole mitotic cells; it counts as label-free cells.
    "EColi": ("label-free", "cell"), "ecoli": ("label-free", "cell"), "ALFI": ("label-free", "cell"),
    "alfi": ("label-free", "cell"), "Cardioblast": ("fluorescence", "nucleus"),
    "cardioblast": ("fluorescence", "nucleus"),
    "WingDisc": ("fluorescence", "cell"), "wingdisc": ("fluorescence", "cell"),
    "YeaZ": ("label-free", "cell"), "yeaz": ("label-free", "cell"), "CellACDC": ("label-free", "cell"),
    "cellacdc": ("label-free", "cell"), "DynamicNuclearNet": ("fluorescence", "nucleus"),
}

# The datasets that micro-sam (vit_l_lm) trained on, from the v1 training script.
MICROSAM_TRAINING = {
    "BF-C2DL-HSC", "BF-C2DL-MuSC", "DIC-C2DH-HeLa", "Fluo-C2DL-Huh7", "Fluo-C2DL-MSC", "Fluo-C3DH-A549",
    "Fluo-C3DH-A549-SIM", "Fluo-C3DH-H157", "Fluo-C3DL-MDA231", "Fluo-N2DH-SIM+", "Fluo-N3DH-CE", "Fluo-N3DH-CHO",
    "Fluo-N3DH-SIM+", "Fluo-N3DL-DRO", "Fluo-N3DL-TRIC", "PhC-C2DH-U373", "PhC-C2DL-PSC", "dynamicnuclearnet",
}


def collect(folder):
    """Concatenate all result tables of one pipeline, or None without any."""
    paths = sorted(glob.glob(os.path.join(folder, "results", "*.csv")))
    if not paths:
        return None
    return pd.concat([pd.read_csv(path, dtype={"sample": str}) for path in paths], ignore_index=True)


def modality(dataset):
    if dataset in MODALITIES:
        return MODALITIES[dataset]
    prefix, kind = dataset.split("-")[:2]
    return ("fluorescence" if prefix == "Fluo" else "label-free", "nucleus" if kind.startswith("N") else "cell")


def scores_of(table, keys):
    """Average the scores over the sequences or movies of each group, and name them as in the central tables."""
    scores = [column for column in SCORES if column in table.columns]
    grouped = table.groupby(keys, dropna=False)
    result = grouped[scores].mean().rename(columns=SCORES)
    result.insert(0, "n_samples", grouped.size())
    return result.reset_index()


def write_tables(pipeline, keys, central, output_root, figure_folder=None):
    """Join the per-run tables of one pipeline into '<pipeline>_tracking_results.csv' in the output root, print them,
    and, with 'figure_folder', write the central table of the paper ('cells_<pipeline>_tracking_results.csv')."""
    table = collect(os.path.join(output_root, pipeline))
    if table is None:
        print(f"No {pipeline} results yet.")
        return
    save_path = os.path.join(output_root, f"{pipeline}_tracking_results.csv")
    table.to_csv(save_path, index=False)
    columns = keys + [column for column in COLUMNS if column in table.columns]
    print(f"{pipeline} ({save_path}):")
    print(table[columns].round(3).to_string(index=False))
    if figure_folder is not None:
        path = os.path.join(figure_folder, f"cells_{pipeline}_tracking_results.csv")
        central(table).to_csv(path, index=False)
        print(f"Wrote the central table '{path}'.")


def prepare_tracking_data(names=None, output_root=TRACKING_DATA):
    """Convert the datasets of PREPARED_DATASETS (all by default) to the CTC format, see 'write_sequence'."""
    for name in names or list(PREPARED_DATASETS):
        PREPARED_DATASETS[name](os.path.join(output_root, name))
