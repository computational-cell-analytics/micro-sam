"""Convert four out-of-domain tracking datasets of torch_em to the CTC format of the tracking evaluation.

Each movie becomes '<output>/<dataset>/<sequence>/tNNN.tif' with the ground truth in '<sequence>_GT/TRA' (the track
masks 'man_trackNNN.tif' and the lineage table 'man_track.txt') and '<sequence>_GT/SEG' (the same dense masks as
'man_segNNN.tif'). The masks are dense: every cell of every frame has its track id, so the movies serve the automatic
and the interactive tracking. A track with a gap in time is split at the gap, and the later part gets the earlier
part as its parent, as CTC asks.

The datasets (none is in the training data of micro-sam2 or of micro-sam):
- 'EColi': E. coli microcolonies in phase contrast (ecoli_microcolony_lineage), with the full lineage of the
  Schnitzcells files. Only the frames with a segmentation are used (every second image).
- 'ALFI': label-free (DIC) movies of human cells. The masks of a frame are numbered from 1, so each mask gets the
  track of the one tracked box that contains it; a daughter track is named after its mother ('3.1' after '3').
  Interphase cells are labelled by their nucleus, mitotic cells by the whole cell.
- 'Cardioblast': Drosophila cardioblast nuclei in fluorescence; the mask labels are the track ids and the nuclei do
  not divide.
- 'WingDisc': Drosophila wing disc epithelium (E-cadherin), with the lineage of the TissueMiner database. The colour
  of a cell in 'tracked_cells_resized.png' is its 'tissue_analyzer_group_id'. A frame has 1,800 to 3,000 cells, so
  the movies are cropped to WING_DISC_CROP pixels around the image center.

Usage:
    python prepare_tracking_data.py
    python prepare_tracking_data.py --datasets ALFI
"""

import os
import re
import glob
import shutil
import sqlite3
import zipfile
import argparse
import tempfile

import tifffile
import numpy as np
import scipy.io as sio
import imageio.v3 as imageio

from common import DATA_ROOT

OUTPUT_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/tracking/data"
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


DATASETS = {
    "EColi": prepare_ecoli, "ALFI": prepare_alfi, "Cardioblast": prepare_cardioblast, "WingDisc": prepare_wing_disc,
    "YeaZ": prepare_yeaz, "CellACDC": prepare_cell_acdc, "DynamicNuclearNet": prepare_dynamicnuclearnet
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("-o", "--output_root", default=OUTPUT_ROOT)
    args = parser.parse_args()
    for name in args.datasets:
        DATASETS[name](os.path.join(args.output_root, name))


if __name__ == "__main__":
    main()
