"""Evaluation building blocks of the decoder architecture comparison (plan section 6).

Per trained run:
1. `predict-val`: predict the validation data of the manifest (2D images, 3D crops placed once in
   `val_crops_v1.json`) and cache the decoder outputs.
2. `tune`: sweep the AIS post-processing (flow segmentation) on the cached outputs and select one parameter set for 2D
   and one for 3D, shared across datasets, by the mean relative score (a dataset's score over its best score).
3. `test`: score the twelve test sets with the library defaults and with the run's tuned parameters, plus diagnostics
   of the raw decoder outputs and the decode cost.

The test data is the evaluation harness (`finetuning/v2/evaluation/common.py`), so the numbers compare to the
harness numbers. CellPose (`cyto` test split) and Usiigaci have no harness resolver; they are resolved here.
"""

import itertools
import json
import os
import sys
import time
import warnings
from glob import glob
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# The evaluation harness: the copy frozen with a job (see submit_decoder_arch.freeze_code), else the repository's.
HARNESS_FILES = ("common.py", "eval_crops.json", "tuning_crops.json")
EVALUATION_DIR = os.path.join(HERE, "harness")
if not os.path.exists(os.path.join(EVALUATION_DIR, "common.py")):
    EVALUATION_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "evaluation"))
if EVALUATION_DIR not in sys.path:
    sys.path.insert(0, EVALUATION_DIR)

import decoder_arch_lib as lib  # noqa: E402

DATA_ROOT = lib.DATA_ROOT
EVAL_ROOT = os.path.join(lib.EXPERIMENT_ROOT, "evaluation")
VAL_CROPS_PATH = os.path.join(EVAL_ROOT, "val_crops_v1.json")
MODEL_TYPE = lib.MODEL_TYPE

# The validation data. The 3D datasets are scored on fixed crops of the manifest's validation volumes / regions.
VAL_CROP_SHAPE = (32, 512, 512)  # the shape of the harness test crops
MAX_VAL_CROPS_PER_VOLUME = 3
MIN_VAL_CROP_OBJECTS = 3
MAX_VAL_CROP_OVERLAP = 0.1  # the largest share of a crop that another crop of the same volume may overlap
SOURCE_TO_DATASET = {
    "livecell": "livecell",
    "cellpose": "cellpose",
    "tissuenet": "tissuenet",
    "nis3d": "nis3d",
    "plantseg_root": "plantseg_root",
    "embedseg:Mouse-Organoid-Cells-CBG": "embedseg_organoid",
    "embedseg:Platynereis-Nuclei-CBG": "embedseg_platy_nuclei",
    "embedseg:Mouse-Skull-Nuclei-CBG": "embedseg_mouse_skull",
    "embedseg:Platynereis-ISH-Nuclei-CBG": "embedseg_platy_ish",
}
EMBEDSEG_COLLECTIONS = ("embedseg_organoid", "embedseg_platy_nuclei", "embedseg_mouse_skull", "embedseg_platy_ish")
VAL_DATASETS = {
    2: ("livecell", "cellpose", "tissuenet"),
    3: ("nis3d", "plantseg_root", "embedseg_organoid", "embedseg_platy_nuclei", "embedseg_mouse_skull",
        "embedseg_platy_ish"),
}
# The selection weights these groups equally: EmbedSeg counts once, as the mean of its collections (as in the report).
SELECTION_GROUPS = {
    2: {d: (d,) for d in VAL_DATASETS[2]},
    3: {"nis3d": ("nis3d",), "plantseg_root": ("plantseg_root",), "embedseg": EMBEDSEG_COLLECTIONS},
}
# The (raw, label) keys of the h5 / zarr validation data; tif and png data has none.
DATA_KEYS = {"tissuenet": ("raw/rgb", "labels/cell"), "plantseg_root": ("raw", "label")}

# The test sets, as plan section 4.2 / 4.3. EmbedSeg is scored per collection and reported as one mean.
TEST_GROUPS = {
    "2d_in_domain": ("livecell", "cellpose", "tissuenet"),
    "3d_in_domain": ("nis3d", "plantseg_root", "embedseg"),
    "2d_external": ("covid_if_cells", "usiigaci", "dsb"),
    "3d_external": ("plantseg_ovules", "gonuclear", "cshaper"),
}
REPORTED_DATASETS = tuple(name for group in TEST_GROUPS.values() for name in group)
SCORED_TEST_DATASETS = tuple(
    name for group in TEST_GROUPS.values() for dataset in group
    for name in (EMBEDSEG_COLLECTIONS if dataset == "embedseg" else (dataset,))
)
TEST_DATASETS_3D = frozenset(
    ("nis3d", "plantseg_root", "plantseg_ovules", "gonuclear", "cshaper") + EMBEDSEG_COLLECTIONS
)
OWN_RESOLVERS = ("cellpose", "usiigaci")


def ndim_of(dataset: str) -> int:
    return 3 if dataset in TEST_DATASETS_3D or dataset in VAL_DATASETS[3] else 2


#
# The post-processing grid
#

# One flow segmentation sweep per data type. Travel = n_iter * dt; the grid keeps dt fixed and moves n_iter.
# The library defaults of hvit_t (2D: fg 0.5, density 30, min 50, sigma 0.5, travel 50, fw 0.9; 3D: density 80,
# min 500) lie inside both grids, and the defaults are always scored as one extra combination.
GRIDS = {
    2: {
        "foreground_threshold": [0.3, 0.4, 0.5, 0.6, 0.7],
        "sigma": [0.25, 0.5, 1.0],
        "n_iter": [25, 50, 100, 400],
        "dt": [1.0],
        "density_threshold": [5.0, 10.0, 20.0, 30.0, 50.0],
        "foreground_weight": [0.25, 0.5, 0.75, 0.9],
        "min_size": [10, 25, 50, 100],
    },
    3: {
        "foreground_threshold": [0.3, 0.4, 0.5, 0.6, 0.7],
        "sigma": [0.25, 0.5, 1.0],
        "n_iter": [25, 50, 100, 400],
        "dt": [1.0],
        "density_threshold": [20.0, 40.0, 80.0, 160.0],
        "foreground_weight": [0.25, 0.5, 0.75, 0.9],
        "min_size": [100, 250, 500, 1000],
    },
}
# Two values per axis, for checks of the pipeline only (evaluate_decoder_arch.py tune --grid small).
SMALL_GRIDS = {
    ndim: {k: (v if len(v) == 1 else [v[0], v[-1]] if len(v) <= 3 else [v[1], v[-2]]) for k, v in grid.items()}
    for ndim, grid in GRIDS.items()
}
FLOW_KEYS = ("foreground_threshold", "sigma", "n_iter", "dt")
PARAM_KEYS = ("foreground_threshold", "sigma", "n_iter", "dt", "density_threshold", "foreground_weight", "min_size")
INTEGER_KEYS = ("n_iter", "min_size")
MAX_EXTENSION_ROUNDS = 3


def default_params(ndim: int) -> Dict:
    from micro_sam.v2.postprocessing import default_postprocessing
    return dict(default_postprocessing(MODEL_TYPE, "sparse", is_volume=ndim == 3))


def normalize_params(params: Dict) -> Dict:
    """The parameters with the types `flow_instance_segmentation` takes, in a fixed key order."""
    return {k: (int(params[k]) if k in INTEGER_KEYS else float(params[k])) for k in PARAM_KEYS}


def params_key(params: Dict) -> Tuple:
    return tuple(normalize_params(params)[k] for k in PARAM_KEYS)


def grid_params(ndim: int, small: bool = False) -> List[Dict]:
    grid = (SMALL_GRIDS if small else GRIDS)[ndim]
    combos = [normalize_params(dict(zip(grid, values))) for values in itertools.product(*grid.values())]
    default = normalize_params(default_params(ndim))
    if params_key(default) not in {params_key(c) for c in combos}:
        combos.append(default)
    return combos


def step_past(key: str, value):
    """The values one step past 'value' on both sides, for the axes the selection can sit on an edge of."""
    if key == "foreground_threshold":
        return [round(v, 2) for v in (value - 0.1, value + 0.1) if 0.05 <= v <= 0.95]
    if key == "foreground_weight":
        return [round(v, 2) for v in (value - 0.1, min(1.0, value + 0.1)) if 0.0 <= v <= 1.0 and v != value]
    if key == "dt":
        return []
    lower, upper = value / 2, value * 2
    if key in INTEGER_KEYS:
        lower, upper = int(round(lower)), int(round(upper))
    return [v for v in (lower, upper) if v > 0 and v != value]


def edge_extension(winner: Dict, swept: Dict[str, Sequence]) -> List[Dict]:
    """The combinations that step past every grid edge the winner sits on, one axis at a time."""
    extension = []
    for key in PARAM_KEYS:
        values = sorted(set(swept[key]))
        if len(values) < 2 and key == "dt":
            continue
        if winner[key] not in (values[0], values[-1]):
            continue
        for value in step_past(key, winner[key]):
            beyond = value < values[0] or value > values[-1]
            if beyond:
                extension.append(normalize_params({**winner, key: value}))
    return extension


#
# Scoring
#


def mask_segmentation(seg: np.ndarray, valid_roi: Optional[np.ndarray], border_min_size: int) -> np.ndarray:
    """Apply what the harness applies before scoring: the ignore mask and, in 2D, the severed-object filter."""
    from common import drop_severed_objects
    if valid_roi is not None:
        seg = seg.copy()
        seg[~valid_roi] = 0
    if seg.ndim == 2 and border_min_size:
        seg = drop_severed_objects(seg, border_min_size)
    return seg


def msa_scores(seg: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    from elf.evaluation import mean_segmentation_accuracy
    msa, accuracies = mean_segmentation_accuracy(seg, labels, return_accuracies=True)
    return {"msa": float(msa), "sa50": float(accuracies[0]), "sa75": float(accuracies[5])}


def flow_segmentation(prediction: np.ndarray, params: Dict, spacing=None, n_threads: int = 1) -> np.ndarray:
    """The library post-processing of one prediction, exactly as the harness runs it (`postprocess_unisam2`)."""
    from micro_sam.v2.postprocessing import flow_instance_segmentation
    return flow_instance_segmentation(
        prediction[0], prediction[1:4], boundary=prediction[4], model_type=MODEL_TYPE, spacing=spacing,
        n_threads=n_threads, **normalize_params(params),
    ).astype("uint32")


def score_grid(
    prediction: np.ndarray, labels: np.ndarray, params_list: List[Dict], valid_roi: Optional[np.ndarray] = None,
    border_min_size: int = 0, spacing=None, n_threads: int = 1,
) -> np.ndarray:
    """The mSA of every combination on one sample, sharing the expensive steps between combinations.

    Computes the same segmentation as `flow_segmentation` (checked in the tests): the flow density is computed once
    per (foreground_threshold, sigma, n_iter, dt), the seeded watershed once per density threshold and foreground
    weight, and only the size filter varies on top. Unlike the harness sweep it applies the ignore mask.
    """
    from bioimage_cpp.segmentation import label, watershed
    from micro_sam.v2.postprocessing import _compute_flow_density, watershed_heightmap

    params_list = [normalize_params(p) for p in params_list]
    foreground = prediction[0].astype("float32")
    directed = prediction[1:4].astype("float32")
    if foreground.ndim == 2:
        directed = directed[1:]

    scores = np.full(len(params_list), np.nan)
    groups = {}
    for index, params in enumerate(params_list):
        key = tuple(params[k] for k in FLOW_KEYS) + (params["density_threshold"], params["foreground_weight"])
        groups.setdefault(key, []).append(index)

    densities, hmaps = {}, {}
    for key in sorted(groups, key=lambda k: k[:4]):
        ft, sigma, n_iter, dt, density_threshold, fw = key
        if key[:4] not in densities:
            densities.clear()  # the groups are sorted by the flow keys, so a density is never needed again
            densities[key[:4]] = _compute_flow_density(
                directed, foreground > ft, n_iter=int(n_iter), dt=dt, sigma=sigma, spacing=spacing,
                n_threads=n_threads,
            )
        if fw not in hmaps:
            hmaps[fw] = watershed_heightmap(foreground, directed, fw)
        fg_mask, hmap = foreground > ft, hmaps[fw]
        try:
            base = watershed(hmap, markers=label(densities[key[:4]] > density_threshold), mask=fg_mask)
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"Flow segmentation failed for {key}: {e}")
            continue
        for index in groups[key]:
            seg, min_size = base, params_list[index]["min_size"]
            if min_size > 0:
                seg = base.copy()
                ids, sizes = np.unique(seg, return_counts=True)
                seg[np.isin(seg, ids[(sizes < min_size) & (ids > 0)])] = 0
                seg = watershed(hmap, markers=seg, mask=fg_mask)
            seg = mask_segmentation(seg.astype("uint32"), valid_roi, border_min_size)
            scores[index] = msa_scores(seg, labels)["msa"]
    return scores


def object_errors(seg: np.ndarray, labels: np.ndarray, valid_roi: Optional[np.ndarray] = None) -> Dict[str, float]:
    """The share of ground-truth objects that are merged, split or missed by a segmentation.

    For a ground-truth object g, a predicted object p 'covers' it with |g & p| / |g|.
    - merged: the predicted object that covers g most covers at least half of g and of another ground-truth object;
    - split (if not merged): at least two predicted objects cover a fifth of g each;
    - missed: no predicted object covers a fifth of g.
    """
    if valid_roi is not None:
        seg = np.where(valid_roi, seg, 0)
    gt_ids, gt_sizes = np.unique(labels[labels > 0], return_counts=True)
    if gt_ids.size == 0:
        return {"n_gt": 0, "merged": np.nan, "split": np.nan, "missed": np.nan}
    both = (labels > 0) & (seg > 0)
    pairs, counts = np.unique(
        np.stack([labels[both].astype("int64"), seg[both].astype("int64")]), axis=1, return_counts=True,
    )
    size_of = dict(zip(gt_ids.tolist(), gt_sizes.tolist()))
    cover = {}  # gt id -> list of (pred id, coverage)
    for (g, p), c in zip(pairs.T.tolist(), counts.tolist()):
        cover.setdefault(g, []).append((p, c / size_of[g]))
    big_cover = {}  # pred id -> number of gt objects it covers at least half of
    for g, items in cover.items():
        for p, c in items:
            if c >= 0.5:
                big_cover[p] = big_cover.get(p, 0) + 1
    merged = split = missed = 0
    for g in gt_ids.tolist():
        items = cover.get(g, [])
        best_p, best_c = max(items, key=lambda item: item[1]) if items else (0, 0.0)
        if best_c >= 0.5 and big_cover.get(best_p, 0) >= 2:
            merged += 1
        elif sum(c >= 0.2 for _, c in items) >= 2:
            split += 1
        elif not any(c >= 0.2 for _, c in items):
            missed += 1
    n = len(gt_ids)
    return {"n_gt": n, "merged": merged / n, "split": split / n, "missed": missed / n}


def field_diagnostics(prediction: np.ndarray, labels: np.ndarray, valid_roi: Optional[np.ndarray] = None) -> Dict:
    """Diagnostics of the raw decoder outputs: foreground IoU at 0.5 and the recall of the object boundaries by the
    boundary channel at 0.5 (on the inner boundary pixels of the ground-truth objects)."""
    from skimage.segmentation import find_boundaries
    valid = np.ones(labels.shape, dtype=bool) if valid_roi is None else valid_roi
    fg_pred, fg_true = (prediction[0] > 0.5) & valid, (labels > 0) & valid
    union = np.logical_or(fg_pred, fg_true).sum()
    boundaries = find_boundaries(labels, mode="inner") & (labels > 0) & valid
    return {
        "fg_iou": float(np.logical_and(fg_pred, fg_true).sum() / union) if union else np.nan,
        "boundary_recall": float((prediction[4][boundaries] > 0.5).mean()) if boundaries.any() else np.nan,
    }


#
# Data: validation
#


def load_cellpose_2d(raw_path: str, label_path: str, crop: bool = True):
    """One CellPose image prepared as in training (`_prepare_cellpose_raw`: cell channel first, a grayscale image
    triplicated) and then as the harness prepares 2D images (8-bit range, center crop)."""
    from common import CROP_SHAPE_2D, _center_crop_roi, ensure_8bit_range, read_2d
    from bioimage_cpp.segmentation import label as connected_components
    from micro_sam.v2.transforms.raw import _prepare_cellpose_raw
    image = read_2d(raw_path, None)
    image = np.moveaxis(_prepare_cellpose_raw(np.moveaxis(image[..., :3], -1, 0)), 0, -1)
    image = ensure_8bit_range(image)
    roi = _center_crop_roi(image.shape[:2], CROP_SHAPE_2D) if crop else (slice(None), slice(None))
    labels = connected_components(read_2d(label_path, None)[roi]).astype("uint32")
    return image[roi], labels


def load_2d_sample(dataset: str, raw_path: str, label_path: str):
    from common import load_evaluation_sample_2d
    if dataset == "cellpose":
        return load_cellpose_2d(raw_path, label_path)
    raw_key, label_key = DATA_KEYS.get(dataset, (None, None))
    return load_evaluation_sample_2d(raw_path, label_path, raw_key, label_key, dataset)


def _label_volume(dataset: str, label_path: str) -> np.ndarray:
    from elf.io import open_file
    from torch_em.util.image import load_image
    key = DATA_KEYS.get(dataset, (None, None))[1]
    return np.asarray(load_image(label_path) if key is None else open_file(label_path, mode="r")[key][:])


def _tile_starts(start: int, stop: int, size: int) -> List[Tuple[int, int]]:
    """Non-overlapping (start, length) tiles of [start, stop), plus one tile flush with the end."""
    length = stop - start
    if length <= size:
        return [(start, length)]
    starts = list(range(start, stop - size + 1, size))
    if starts[-1] + size < stop:
        starts.append(stop - size)
    return [(s, size) for s in starts]


def place_val_crops(manifest: Dict) -> Dict:
    """Place the 3D validation crops: per validation volume (or region), the tiles of VAL_CROP_SHAPE with the most
    ground-truth objects, at most MAX_VAL_CROPS_PER_VOLUME that overlap each other by at most MAX_VAL_CROP_OVERLAP
    of their volume, each with at least MIN_VAL_CROP_OBJECTS objects."""
    crops = {}
    for unit in manifest["units"]:
        if unit["split"] != "val" or unit["ndim"] != 3:
            continue
        dataset = SOURCE_TO_DATASET[unit["source"]]
        rois = unit.get("rois") or [None] * len(unit["raw_paths"])
        for raw_path, label_path, roi in zip(unit["raw_paths"], unit["label_paths"], rois):
            labels = _label_volume(dataset, label_path)
            background = (0, 1) if dataset == "plantseg_root" else (0,)
            z0, z1 = (0, labels.shape[0]) if roi is None else roi
            candidates = []
            for zs, zl in _tile_starts(z0, z1, VAL_CROP_SHAPE[0]):
                for ys, yl in _tile_starts(0, labels.shape[1], VAL_CROP_SHAPE[1]):
                    for xs, xl in _tile_starts(0, labels.shape[2], VAL_CROP_SHAPE[2]):
                        window = labels[zs:zs + zl, ys:ys + yl, xs:xs + xl]
                        n_objects = int(np.setdiff1d(np.unique(window), background).size)
                        candidates.append(((zs, ys, xs), (zl, yl, xl), n_objects))
            candidates.sort(key=lambda c: -c[2])  # stable: ties keep the tiling order
            chosen = []
            for start, shape, n_objects in candidates:
                if len(chosen) == MAX_VAL_CROPS_PER_VOLUME or n_objects < MIN_VAL_CROP_OBJECTS:
                    break
                overlaps = any(
                    np.prod([max(0, min(s + sl, cs + cl) - max(s, cs))
                             for s, sl, cs, cl in zip(start, shape, c["start"], c["shape"])])
                    > MAX_VAL_CROP_OVERLAP * min(np.prod(shape), np.prod(c["shape"]))
                    for c in chosen
                )
                if not overlaps:
                    chosen.append({"start": list(start), "shape": list(shape), "n_objects": n_objects})
            if not chosen:
                raise RuntimeError(f"No validation crop with {MIN_VAL_CROP_OBJECTS} objects in {label_path}.")
            crops.setdefault(dataset, []).extend(
                {"raw_path": raw_path, "label_path": label_path, **crop} for crop in chosen
            )
    return crops


def val_samples(manifest: Dict, val_crops: Dict, ndim: int) -> Dict[str, List[Dict]]:
    """The validation samples per dataset, as specs that `load_val_sample` reads."""
    samples = {}
    if ndim == 2:
        for unit in manifest["units"]:
            if unit["split"] == "val" and unit["ndim"] == 2:
                dataset = SOURCE_TO_DATASET[unit["source"]]
                samples.setdefault(dataset, []).extend(
                    {"dataset": dataset, "raw_path": r, "label_path": lab}
                    for r, lab in zip(unit["raw_paths"], unit["label_paths"])
                )
    else:
        for dataset in VAL_DATASETS[3]:
            samples[dataset] = [{"dataset": dataset, **crop} for crop in val_crops[dataset]]
    return {d: samples[d] for d in VAL_DATASETS[ndim]}


def load_val_sample(spec: Dict):
    """(raw, labels, valid_roi) of one validation sample, loaded the way the harness loads test samples."""
    from common import GT_MIN_SIZE_2D, drop_severed_objects, load_evaluation_sample_3d
    dataset = spec["dataset"]
    if ndim_of(dataset) == 2:
        raw, labels = load_2d_sample(dataset, spec["raw_path"], spec["label_path"])
        return raw, drop_severed_objects(labels, GT_MIN_SIZE_2D.get(dataset, 0)), None
    raw_key, label_key = DATA_KEYS.get(dataset, (None, None))
    return load_evaluation_sample_3d(
        spec["raw_path"], spec["label_path"], raw_key, label_key, dataset, crop_shape=tuple(spec["shape"]),
        split="val", crop_start=tuple(spec["start"]),
    )


#
# Data: test
#


def _own_test_paths(dataset: str) -> Tuple[List[str], List[str]]:
    if dataset == "cellpose":
        # The official 'cyto' test split (68 images); 'cyto2' has no test split.
        raw = sorted(glob(os.path.join(DATA_ROOT, "cellpose", "cyto", "test", "*_img.png")))
        return raw, [p.replace("_img.png", "_masks.png") for p in raw]
    if dataset == "usiigaci":
        # All labelled images: the Mask R-CNN train / val sets and the T98G example time series. No model of this
        # experiment trains on Usiigaci.
        from torch_em.data.datasets.light_microscopy.usiigaci import get_usiigaci_paths
        raw, labels = [], []
        for split in ("train", "val"):
            r, lab = get_usiigaci_paths(os.path.join(DATA_ROOT, "usiigaci"), split=split)
            raw, labels = raw + r, labels + lab
        return raw, labels
    raise ValueError(dataset)


def n_test_samples(dataset: str) -> int:
    from common import n_samples
    if dataset in OWN_RESOLVERS:
        return len(_own_test_paths(dataset)[0])
    return n_samples(dataset, DATA_ROOT)


def test_samples(dataset: str, sample_ids: Optional[Sequence[int]] = None) -> Iterator[Tuple[int, Tuple]]:
    """(sample id, (raw, labels, valid_roi)) of the test samples of a dataset, in the harness order."""
    from common import GT_MIN_SIZE_2D, drop_severed_objects, load_data
    wanted = None if sample_ids is None else set(sample_ids)
    if dataset in OWN_RESOLVERS:
        for index, (raw_path, label_path) in enumerate(zip(*_own_test_paths(dataset))):
            if wanted is not None and index not in wanted:
                continue
            raw, labels = load_2d_sample(dataset, raw_path, label_path)
            yield index, (raw, drop_severed_objects(labels, GT_MIN_SIZE_2D.get(dataset, 0)), None)
        return
    samples = load_data(dataset, DATA_ROOT, ndim_of(dataset), split="test", sample_ids=wanted)
    ids = sorted(wanted) if wanted is not None else itertools.count()
    yield from zip(ids, samples)


#
# Model
#


def load_model(checkpoint_path: str, device):
    """The trained UniSAM2 of a run, rebuilt from its checkpoint (`decoder_config` selects the decoder)."""
    from common import load_unisam2_model
    return load_unisam2_model(checkpoint_path, device)


def predict(model, raw: np.ndarray, ndim: int, device) -> np.ndarray:
    """The five decoder channels for one sample, exactly as the harness predicts them (tiling, normalization)."""
    from common import predict_unisam2
    return predict_unisam2(model, raw, ndim=ndim, device=device)


def decode_benchmark(model, device, n_z: int, size: int = 1024, repeats: int = 5) -> Dict:
    """Time and memory of the decoder alone, for one input of Z = n_z slices at size x size (float32)."""
    import torch
    x = torch.rand(1, 1, n_z, size, size, device=device)
    with torch.no_grad():
        features, input_shape = model.encode(x)
        original_shape = tuple(x.shape[2:])
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        model.decode(features, input_shape, original_shape)  # warm-up
        torch.cuda.synchronize()
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            model.decode(features, input_shape, original_shape)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        peak = torch.cuda.max_memory_allocated() - base
    return {"n_z": n_z, "size": size, "decode_ms": 1000 * float(np.median(times)), "decode_peak_GiB": peak / 2 ** 30}


#
# Selection and statistics
#


def select_params(scores: Dict[str, np.ndarray], groups: Optional[Dict[str, Sequence[str]]] = None):
    """The combination with the best mean relative score over the dataset groups.

    A group's score is the mean of its datasets' mean mSA; its relative score is that over the group's best score.

    Args:
        scores: Per dataset, the (combinations x samples) mSA matrix; NaN where a combination failed.
        groups: The groups the selection weights equally, as group -> datasets. Default: one group per dataset.

    Returns:
        The index of the best combination, the mean relative score of every combination, the per-dataset mean
        mSA (datasets x combinations, in the order of 'scores') and the per-group mean mSA (groups x combinations).
    """
    groups = groups or {d: (d,) for d in scores}
    n_combos = len(next(iter(scores.values())))
    means = {
        d: np.nan_to_num(np.nanmean(s, axis=1), nan=0.0) if s.shape[1] else np.zeros(n_combos)
        for d, s in scores.items()
    }
    group_means = np.stack([np.mean([means[d] for d in members], axis=0) for members in groups.values()])
    best = group_means.max(axis=1, keepdims=True)
    relative = np.divide(group_means, best, out=np.zeros_like(group_means), where=best > 0).mean(axis=0)
    return int(np.argmax(relative)), relative, np.stack(list(means.values())), group_means


def paired_bootstrap(values: Sequence[np.ndarray], references: Sequence[np.ndarray],
                     weights: Optional[Sequence[float]] = None, n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Paired bootstrap of the difference of a weighted mean of per-group means.

    A group is one scored dataset (or EmbedSeg collection). Samples are resampled within each group, with the same
    resample for the run and the reference, so the interval reflects the variation over images or crops.

    Args:
        values: Per group, the per-sample scores of the run.
        references: Per group, the per-sample scores of the reference, aligned with 'values'.
        weights: The weight of every group in the mean; equal weights by default.
        n_boot: The number of resamples.
        seed: The seed of the resampling.

    Returns:
        The difference, its value relative to the reference and their 95 % intervals, and the two means.
    """
    rng = np.random.default_rng(seed)
    values = [np.asarray(v, dtype="float64") for v in values]
    references = [np.asarray(r, dtype="float64") for r in references]
    if any(len(v) != len(r) or len(v) == 0 for v, r in zip(values, references)):
        raise ValueError("Every group needs the same, non-zero number of samples for the run and the reference.")
    weights = np.full(len(values), 1 / len(values)) if weights is None else np.asarray(weights) / np.sum(weights)
    mean_v = float(np.dot(weights, [v.mean() for v in values]))
    mean_r = float(np.dot(weights, [r.mean() for r in references]))
    boot_v, boot_r = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        means_v, means_r = [], []
        for v, r in zip(values, references):
            index = rng.integers(0, len(v), len(v))
            means_v.append(v[index].mean())
            means_r.append(r[index].mean())
        boot_v[b], boot_r[b] = np.dot(weights, means_v), np.dot(weights, means_r)
    boot_diff = boot_v - boot_r
    boot_rel = np.divide(boot_diff, boot_r, out=np.full(n_boot, np.nan), where=boot_r > 0)
    return {
        "value": mean_v, "reference": mean_r, "diff": mean_v - mean_r,
        "diff_lo": float(np.percentile(boot_diff, 2.5)), "diff_hi": float(np.percentile(boot_diff, 97.5)),
        "rel": (mean_v - mean_r) / mean_r if mean_r > 0 else np.nan,
        "rel_lo": float(np.nanpercentile(boot_rel, 2.5)), "rel_hi": float(np.nanpercentile(boot_rel, 97.5)),
    }


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def load_json(path: str):
    with open(path) as f:
        return json.load(f)
