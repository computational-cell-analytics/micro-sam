"""Cross-dataset phase 1 for APG: does the livecell tuning transfer, and does v2 APG beat v1?

APG was developed and tuned on livecell alone. This asks the same question on other 2d datasets, each
with its own val grid-search and a test evaluation, against micro-sam v1's AIS and APG at their library
defaults.

The data comes from `scripts/apg_experiments/util.py`, which prepares 512x512 crops with val and test
splits for the v1 APG manuscript. The v2 evaluation pipeline (`common.py`) exposes only test splits, so
tuning through it would fit to test. livecell is the exception: it is read through the v2 evaluation's
centre crop, so that its whole 570-image val split is available.

Stages, in order:
    tune      - grid-search plain APG (no box refinement) on val, sharded, then merged.
    ais_tune  - the same for the flow post-processing, so both engines are tuned per dataset.
    box       - box refinement with the two merge filters, on top of the tuned density parameters.
    mcs       - 'min_candidate_size' on top of all of the above.
    test      - all four methods on test, each v2 engine with the configuration its stages chose.
"""

import os
import sys
import json
import time
import argparse
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm
import imageio.v3 as imageio
from skimage.measure import label as connected_components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/mnt/vast-nhr/home/archit/u12090/micro-sam/scripts/apg_experiments")

from elf.evaluation import mean_segmentation_accuracy  # noqa
from elf.evaluation.matching import matching  # noqa

from micro_sam.v2.util import get_sam2_model  # noqa
from micro_sam.v1.evaluation.livecell import _get_livecell_paths  # noqa
from micro_sam.v2.instance_segmentation import get_instance_segmentation_generator  # noqa
from micro_sam.v2.automatic_prompt_generation import (  # noqa
    derive_point_prompts, merge_by_score, refine_with_boxes,
)

from util import get_image_label_paths  # noqa

from common import (  # noqa
    DATA_ROOT, GT_MIN_SIZE_2D, drop_excluded_livecell, export_joint_checkpoint, load_unisam2_model,
)
from baselines_common import (  # noqa
    CROP_SHAPE_2D, _apply_min_size, _center_crop_roi, _ensure_8bit_range, _read_2d,
)

OUT_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/apg_crossdataset"

# Every 2d dataset in results_automatic.csv whose val split is large enough to tune on, plus the three
# small-val ones the earlier rounds already covered. 'cellpose' is that table's 'cellpose_data'.
DATASETS = [
    "livecell", "deepbacs", "dynamicnuclearnet", "dsb",
    "tissuenet", "deepseas", "vicar", "yeaz", "u20s",
    "cellbindb", "cellpose", "omnipose",
]
V2_MODEL_TYPE = "hvit_b"
V1_MODEL_TYPE = "vit_b_lm"

# Cost is linear in this. None means the whole split, which only livecell wants; a missing key must not
# default to it, or a slow dataset overruns the job limit and strands its afterok merge.
DEFAULT_MAX_VAL = 100
MAX_VAL_IMAGES = {"livecell": None}

# Plain APG: the box stage gets its own round rather than being assumed in every cell. These axes change
# the candidates, so each combination costs a fresh round of prompting.
PROMPT_GRID = {
    "candidate_threshold": [1.0, 1.5, 2.0, 2.5, 3.0, 5.0],
    "sigma": [0.25, 0.5, 1.0],
    "foreground_threshold": [0.3, 0.7],
    "n_iter": [25, 50],
}
# Filters over the same records, so these cost only their scoring.
MERGE_GRID = {
    "score_threshold": [0.3, 0.5, 0.7],
    "min_size": [25, 50, 100],
    "max_overlap": [0.15, 0.30],
}

# Round 2, for a dataset whose round-1 winner sat on a grid edge.
PROMPT_GRID_WIDE = {
    "candidate_threshold": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
    "sigma": [0.5, 1.0, 2.0],
    "foreground_threshold": [0.5, 0.7, 0.9],
    "n_iter": [50, 100],
}
MERGE_GRID_WIDE = {
    "score_threshold": [0.0, 0.3, 0.5],
    "min_size": [10, 25, 50],
    "max_overlap": [0.15, 0.30],
}
# 'dt' is fixed because only its product with 'n_iter' matters. 'min_candidate_size' has its own stage.
FIXED = {"dt": 0.25, "min_candidate_size": 1, "multimasking": True}

# A pixel-count filter on density components, pinned by the main grid and swept here on the winner.
MIN_CANDIDATE_SIZES = [1, 2, 4, 8, 16]

# 'box_extension' changes mask sizes, so the two merge filters are re-picked with it. It decides the
# dynamicnuclearnet result outright and nothing predicts its value, so it has to be swept.
BOX_EXTENSIONS = [None, 0, 1, 2, 4, 8]
BOX_MERGE_GRID = {"score_threshold": [0.3, 0.5, 0.7], "min_size": [10, 25, 50, 100]}

# The v2 flow post-processing, tuned per dataset like APG so the two are compared on equal terms. The
# density axes are the expensive ones; the rest are re-runs of the watershed over the same density.
AIS_DENSITY_GRID = {
    "foreground_threshold": [0.3, 0.5, 0.7],
    "sigma": [0.25, 0.5, 1.0, 2.0],
    "n_iter": [50, 100],
}
AIS_WATERSHED_GRID = {
    "density_threshold": [1.0, 2.0, 3.0, 5.0, 10.0, 20.0],
    "min_size": [10, 25, 50, 100],
    "foreground_weight": [0.0, 0.5, 0.75],
}

# The livecell-tuned configuration, evaluated as an extra row because three of its values are off grid.
# Reported next to the per-dataset winner, which separates 'it transfers' from 'retuning is worth it'.
LIVECELL_CONFIG = {
    "candidate_threshold": 2.25, "sigma": 0.5, "foreground_threshold": 0.5, "n_iter": 25,
    "score_threshold": 0.5, "min_size": 50, "max_overlap": 0.20,
}

METRICS = ["mSA", "SA50", "SA75", "precision", "recall", "f1"]


def grids(round_name):
    return (PROMPT_GRID, MERGE_GRID) if round_name == "1" else (PROMPT_GRID_WIDE, MERGE_GRID_WIDE)


def prompt_combos(round_name):
    grid = grids(round_name)[0]
    return [dict(zip(grid, values)) for values in itertools.product(*grid.values())]


def merge_combos(round_name):
    grid = grids(round_name)[1]
    return [dict(zip(grid, values)) for values in itertools.product(*grid.values())]


def combos(round_name):
    """Every grid combination (index = prompt_index * n_merges + merge_index), then the livecell row."""
    grid = [
        {**prompt, **merge, "is_livecell": False}
        for prompt in prompt_combos(round_name) for merge in merge_combos(round_name)
    ]
    return grid + [{**LIVECELL_CONFIG, "is_livecell": True}]


def measure(seg, gt):
    msa, accuracies = mean_segmentation_accuracy(seg, gt, return_accuracies=True)
    stats = matching(seg, gt)
    return (msa, accuracies[0], accuracies[5], stats["precision"], stats["recall"], stats["f1"])


def load_pair(dataset, image_path, label_path):
    """Read one image and its labels, in the preparation that dataset's split uses.

    The size floor matters: relabelling a crop promotes every severed sliver to its own object, and
    those are below the 'min_size' the methods themselves apply, so they would be unmatchable.
    """
    min_size = GT_MIN_SIZE_2D.get(dataset, 0)
    if dataset == "livecell":
        image = _ensure_8bit_range(_read_2d(image_path, None))
        roi = _center_crop_roi(image.shape[:2], CROP_SHAPE_2D)
        gt = connected_components(_read_2d(label_path, None)[roi]).astype("uint32")
        return image[roi], _apply_min_size(gt, min_size, dataset)

    image = _ensure_8bit_range(np.asarray(imageio.imread(image_path)))
    gt = np.asarray(imageio.imread(label_path)).astype("uint32")
    return image, _apply_min_size(gt, min_size, dataset)


def get_paths(dataset, split, limit=None):
    """The image and label paths of one split.

    livecell is resolved directly rather than through the manuscript's prepared crops, because those
    hold only 5 val images per cell type, and read through the v2 evaluation's centre crop instead.
    """
    if dataset == "livecell":
        image_paths, label_paths = _get_livecell_paths(
            input_folder=os.path.join(DATA_ROOT, "livecell"), split=split, n_val_per_cell_type=None,
        )
        if split == "test":
            image_paths, label_paths = drop_excluded_livecell(image_paths, label_paths)
    else:
        image_paths, label_paths = get_image_label_paths(dataset_name=dataset, split=split)

    pairs = sorted((str(image), str(gt)) for image, gt in zip(image_paths, label_paths))
    return pairs if limit is None else pairs[:limit]


def build_v2_segmenter(device):
    interactive_path, decoder_path = export_joint_checkpoint(V2_MODEL_TYPE)
    decoder = load_unisam2_model(decoder_path, device, encoder=V2_MODEL_TYPE)
    return get_instance_segmentation_generator(
        model=get_sam2_model(model_type=V2_MODEL_TYPE, device=device, checkpoint_path=interactive_path),
        decoder=decoder, segmentation_mode="apg", device=device,
    )


def _report(df, grid_keys, dataset):
    pd.set_option("display.width", 240)
    grid = df[~df.is_livecell]
    print(f"\n{dataset}: {int(df.n_images.iloc[0])} val images, {len(grid)} grid combinations\n")
    print(df.head(10).round(4).to_string(index=False))

    best, livecell = grid.iloc[0], df[df.is_livecell].iloc[0]
    print(f"\nbest      {dict((k, best[k]) for k in grid_keys)} -> mSA {best.mSA:.4f}")
    print(f"livecell  {LIVECELL_CONFIG} -> mSA {livecell.mSA:.4f} ({livecell.mSA - best.mSA:+.4f})")
    print("\nbest per axis value (grid rows only):")
    for axis in grid_keys:
        per_axis = grid.groupby(axis).mSA.max().round(4)
        print(f"  {axis:22s} best={per_axis.idxmax():<7} {dict(per_axis)}")


def run_tune(args):
    out_dir = os.path.join(OUT_ROOT, args.dataset, "tune")
    os.makedirs(out_dir, exist_ok=True)
    params_list = combos(args.round)
    stem = f"{args.dataset}_{V2_MODEL_TYPE}_tune_round{args.round}"

    if args.merge:
        totals, n = np.zeros((len(params_list), len(METRICS))), 0
        for shard in range(args.n_shards):
            part = np.load(os.path.join(out_dir, f"{stem}.shard{shard}of{args.n_shards}.npz"))
            totals += part["totals"]
            n += int(part["n_images"])
        df = pd.DataFrame(params_list)
        for index, name in enumerate(METRICS):
            df[name] = totals[:, index] / n
        df["n_images"] = n
        df = df.sort_values("mSA", ascending=False).reset_index(drop=True)
        df.to_csv(os.path.join(out_dir, f"{stem}.csv"), index=False)

        prompt_grid, merge_grid = grids(args.round)
        keys = list(prompt_grid) + list(merge_grid)
        best = df[~df.is_livecell].iloc[0]
        with open(os.path.join(out_dir, "best.json"), "w") as f:
            json.dump({key: float(best[key]) for key in keys}, f)
        _report(df, keys, args.dataset)
        return

    pairs = get_paths(args.dataset, "val", limit=MAX_VAL_IMAGES.get(args.dataset, DEFAULT_MAX_VAL))
    if args.n_shards > 1:
        pairs = pairs[args.shard::args.n_shards]
    print(f"{args.dataset}: {len(pairs)} val images (shard {args.shard}/{args.n_shards}), "
          f"{len(params_list)} combinations", flush=True)

    segmenter = build_v2_segmenter(args.device)
    prompts_grid, merges = prompt_combos(args.round), merge_combos(args.round)

    totals, n_used = np.zeros((len(params_list), len(METRICS))), 0
    t0 = time.perf_counter()

    for image_path, label_path in tqdm(pairs, desc=f"tune {args.dataset}"):
        image, gt = load_pair(args.dataset, image_path, label_path)
        if gt.max() == 0:
            continue
        n_used += 1

        # One encoder and decoder pass per image; every combination reuses this state.
        segmenter.initialize(image, ndim=2)
        prediction = segmenter._prediction
        shape = prediction[0].shape

        def prompt_and_merge(prompt_params, merge_params_list):
            """One prompting round, scored under every merge setting that shares it."""
            prompts = derive_point_prompts(
                prediction[0], prediction[1:], **prompt_params,
                dt=FIXED["dt"], min_candidate_size=FIXED["min_candidate_size"],
            )
            records = [] if prompts is None else segmenter._apply_prompts(
                prompts, multimasking=FIXED["multimasking"], batch_size=64,
            )
            out = []
            for merge_params in merge_params_list:
                kept = [r for r in records if r["predicted_iou"] >= merge_params["score_threshold"]]
                seg = (np.zeros(shape, dtype="uint32") if not kept else merge_by_score(
                    kept, shape, max_overlap=merge_params["max_overlap"],
                    min_size=merge_params["min_size"],
                ))
                out.append(measure(seg, gt))
            del records
            return out

        for prompt_index, prompt_params in enumerate(prompts_grid):
            for merge_index, scores in enumerate(prompt_and_merge(prompt_params, merges)):
                totals[prompt_index * len(merges) + merge_index] += scores

        # The livecell configuration, as the last row. Its prompt values are off the grid, so it needs
        # its own round rather than a lookup.
        prompt_keys, merge_keys = list(PROMPT_GRID), list(MERGE_GRID)  # the livecell row is fixed
        totals[-1] += prompt_and_merge(
            {key: LIVECELL_CONFIG[key] for key in prompt_keys},
            [{key: LIVECELL_CONFIG[key] for key in merge_keys}],
        )[0]

    out = os.path.join(out_dir, f"{stem}.shard{args.shard}of{args.n_shards}.npz")
    np.savez(out, totals=totals, n_images=n_used)
    print(f"shard {args.shard}/{args.n_shards} done in {time.perf_counter() - t0:.0f}s -> {out}")


def run_box(args):
    """Sweep the box stage together with the merge filters, on top of the tuned density parameters."""
    tune_dir = os.path.join(OUT_ROOT, args.dataset, "tune")
    with open(os.path.join(tune_dir, "best.json")) as f:
        best = json.load(f)
    density = {
        "candidate_threshold": best["candidate_threshold"], "sigma": best["sigma"],
        "foreground_threshold": best["foreground_threshold"], "n_iter": int(best["n_iter"]),
    }
    print(f"{args.dataset}: box stage on top of {density}", flush=True)

    merges = [dict(zip(BOX_MERGE_GRID, v)) for v in itertools.product(*BOX_MERGE_GRID.values())]
    rows = [
        {**merge, "box_extension": -1 if extension is None else extension}
        for merge in merges for extension in BOX_EXTENSIONS
    ]
    pairs = get_paths(args.dataset, "val", limit=MAX_VAL_IMAGES.get(args.dataset, DEFAULT_MAX_VAL))
    segmenter = build_v2_segmenter(args.device)
    predictor = segmenter._predictor

    totals, n_used = np.zeros((len(rows), len(METRICS))), 0
    for image_path, label_path in tqdm(pairs, desc=f"box {args.dataset}"):
        image, gt = load_pair(args.dataset, image_path, label_path)
        if gt.max() == 0:
            continue
        n_used += 1
        segmenter.initialize(image, ndim=2)
        prediction = segmenter._prediction
        shape = prediction[0].shape

        prompts = derive_point_prompts(
            prediction[0], prediction[1:], **density,
            dt=FIXED["dt"], min_candidate_size=FIXED["min_candidate_size"],
        )
        records = [] if prompts is None else segmenter._apply_prompts(
            prompts, multimasking=FIXED["multimasking"], batch_size=64,
        )
        index = 0
        for merge in merges:
            kept = [r for r in records if r["predicted_iou"] >= merge["score_threshold"]]
            base = (np.zeros(shape, dtype="uint32") if not kept else merge_by_score(
                kept, shape, max_overlap=best["max_overlap"], min_size=merge["min_size"],
            ))
            for extension in BOX_EXTENSIONS:
                seg = base
                if extension is not None and base.max() > 0:
                    seg = refine_with_boxes(predictor, base, box_extension=extension)
                totals[index] += measure(seg, gt)
                index += 1
        del records

    df = pd.DataFrame(rows)
    for position, name in enumerate(METRICS):
        df[name] = totals[:, position] / n_used
    df["n_images"] = n_used
    df = df.sort_values("mSA", ascending=False).reset_index(drop=True)
    df.to_csv(os.path.join(tune_dir, f"{args.dataset}_{V2_MODEL_TYPE}_box.csv"), index=False)

    winner = df.iloc[0]
    with open(os.path.join(tune_dir, "best.json"), "w") as f:
        json.dump({
            **density, "max_overlap": best["max_overlap"],
            "score_threshold": float(winner["score_threshold"]), "min_size": int(winner["min_size"]),
            "refine_with_box_prompts": bool(winner["box_extension"] >= 0),
            "box_extension": max(0, int(winner["box_extension"])),
        }, f)
    pd.set_option("display.width", 240)
    print(f"\n{args.dataset}: box stage on {n_used} val images, {len(df)} rows")
    print(df.head(10).round(4).to_string(index=False))
    print(f"\ngrid spread: mSA {df.mSA.min():.4f} to {df.mSA.max():.4f} "
          f"(range {df.mSA.max() - df.mSA.min():.4f})")
    print("best per box_extension (-1 is no box stage):")
    print(df.groupby("box_extension").mSA.max().round(4).to_string())


def run_mcs(args):
    """Sweep 'min_candidate_size' on top of the fully tuned configuration."""
    tune_dir = os.path.join(OUT_ROOT, args.dataset, "tune")
    with open(os.path.join(tune_dir, "best.json")) as f:
        best = json.load(f)
    density = {
        "candidate_threshold": best["candidate_threshold"], "sigma": best["sigma"],
        "foreground_threshold": best["foreground_threshold"], "n_iter": int(best["n_iter"]),
    }
    with_box = best.get("refine_with_box_prompts", False)
    extension = int(best.get("box_extension", 0))
    print(f"{args.dataset}: min_candidate_size on top of {density}, box={with_box} ext={extension}",
          flush=True)

    pairs = get_paths(args.dataset, "val", limit=MAX_VAL_IMAGES.get(args.dataset, DEFAULT_MAX_VAL))
    segmenter = build_v2_segmenter(args.device)
    predictor = segmenter._predictor

    totals, n_used = np.zeros((len(MIN_CANDIDATE_SIZES), len(METRICS))), 0
    for image_path, label_path in tqdm(pairs, desc=f"mcs {args.dataset}"):
        image, gt = load_pair(args.dataset, image_path, label_path)
        if gt.max() == 0:
            continue
        n_used += 1
        segmenter.initialize(image, ndim=2)
        prediction = segmenter._prediction
        shape = prediction[0].shape

        for index, min_candidate_size in enumerate(MIN_CANDIDATE_SIZES):
            prompts = derive_point_prompts(
                prediction[0], prediction[1:], **density,
                dt=FIXED["dt"], min_candidate_size=min_candidate_size,
            )
            records = [] if prompts is None else segmenter._apply_prompts(
                prompts, multimasking=FIXED["multimasking"], batch_size=64,
            )
            kept = [r for r in records if r["predicted_iou"] >= best["score_threshold"]]
            seg = (np.zeros(shape, dtype="uint32") if not kept else merge_by_score(
                kept, shape, max_overlap=best["max_overlap"], min_size=int(best["min_size"]),
            ))
            if with_box and seg.max() > 0:
                seg = refine_with_boxes(predictor, seg, box_extension=extension)
            totals[index] += measure(seg, gt)
            del records

    df = pd.DataFrame({"min_candidate_size": MIN_CANDIDATE_SIZES})
    for position, name in enumerate(METRICS):
        df[name] = totals[:, position] / n_used
    df["n_images"] = n_used
    df = df.sort_values("mSA", ascending=False).reset_index(drop=True)
    df.to_csv(os.path.join(tune_dir, f"{args.dataset}_{V2_MODEL_TYPE}_mcs.csv"), index=False)

    winner = int(df.iloc[0].min_candidate_size)
    best["min_candidate_size"] = winner
    with open(os.path.join(tune_dir, "best.json"), "w") as f:
        json.dump(best, f)
    print(f"\n{args.dataset}: min_candidate_size on {n_used} val images")
    print(df.round(4).to_string(index=False))
    baseline = float(df[df.min_candidate_size == 1].iloc[0].mSA)
    print(f"winner {winner}, {df.iloc[0].mSA - baseline:+.4f} against the pinned value of 1")


def ais_combos():
    density = [dict(zip(AIS_DENSITY_GRID, v)) for v in itertools.product(*AIS_DENSITY_GRID.values())]
    watershed_params = [
        dict(zip(AIS_WATERSHED_GRID, v)) for v in itertools.product(*AIS_WATERSHED_GRID.values())
    ]
    return density, watershed_params


def run_ais_tune(args):
    """Grid-search the flow post-processing on val, mirroring the APG tune stage."""
    from bioimage_cpp.segmentation import label, watershed
    from micro_sam.v2.postprocessing import watershed_heightmap, _compute_flow_density

    out_dir = os.path.join(OUT_ROOT, args.dataset, "tune")
    os.makedirs(out_dir, exist_ok=True)
    density_list, watershed_list = ais_combos()
    params_list = [{**d, **w} for d in density_list for w in watershed_list]
    stem = f"{args.dataset}_{V2_MODEL_TYPE}_ais"

    if args.merge:
        totals, n = np.zeros((len(params_list), len(METRICS))), 0
        for shard in range(args.n_shards):
            part = np.load(os.path.join(out_dir, f"{stem}.shard{shard}of{args.n_shards}.npz"))
            totals += part["totals"]
            n += int(part["n_images"])
        df = pd.DataFrame(params_list)
        for index, name in enumerate(METRICS):
            df[name] = totals[:, index] / n
        df["n_images"] = n
        df = df.sort_values("mSA", ascending=False).reset_index(drop=True)
        df.to_csv(os.path.join(out_dir, f"{stem}.csv"), index=False)

        keys = list(AIS_DENSITY_GRID) + list(AIS_WATERSHED_GRID)
        best = df.iloc[0]
        with open(os.path.join(out_dir, "best_ais.json"), "w") as f:
            json.dump({key: float(best[key]) for key in keys}, f)
        pd.set_option("display.width", 240)
        print(f"\n{args.dataset}: AIS on {n} val images, {len(df)} combinations\n")
        print(df.head(10).round(4).to_string(index=False))
        print("\nbest per axis value:")
        for axis in keys:
            per_axis = df.groupby(axis).mSA.max().round(4)
            print(f"  {axis:22s} best={per_axis.idxmax():<7} {dict(per_axis)}")
        return

    pairs = get_paths(args.dataset, "val", limit=MAX_VAL_IMAGES.get(args.dataset, DEFAULT_MAX_VAL))
    if args.n_shards > 1:
        pairs = pairs[args.shard::args.n_shards]
    print(f"{args.dataset}: {len(pairs)} val images (shard {args.shard}/{args.n_shards}), "
          f"{len(params_list)} AIS combinations", flush=True)

    segmenter = build_v2_segmenter(args.device)
    totals, n_used = np.zeros((len(params_list), len(METRICS))), 0
    t0 = time.perf_counter()

    for image_path, label_path in tqdm(pairs, desc=f"ais {args.dataset}"):
        image, gt = load_pair(args.dataset, image_path, label_path)
        if gt.max() == 0:
            continue
        n_used += 1
        segmenter.initialize(image, ndim=2)
        foreground, distances = segmenter._prediction[0], segmenter._prediction[1:]
        if distances.shape[0] > foreground.ndim:
            distances = distances[-foreground.ndim:]

        for density_index, density_params in enumerate(density_list):
            fg_mask = foreground > density_params["foreground_threshold"]
            density = _compute_flow_density(
                distances, fg_mask, n_iter=int(density_params["n_iter"]), dt=FIXED["dt"],
                sigma=density_params["sigma"], backend="cpp",
            )
            heightmaps = {}
            for watershed_index, watershed_params in enumerate(watershed_list):
                weight = watershed_params["foreground_weight"]
                if weight not in heightmaps:
                    heightmaps[weight] = watershed_heightmap(foreground, distances, weight)
                hmap = heightmaps[weight]
                seeds = label(density > watershed_params["density_threshold"])
                seg = watershed(hmap, markers=seeds, mask=fg_mask)
                min_size = watershed_params["min_size"]
                ids, sizes = np.unique(seg, return_counts=True)
                discard = ids[(sizes < min_size) & (ids > 0)]
                if discard.size:
                    seg[np.isin(seg, discard)] = 0
                    seg = watershed(hmap, markers=seg, mask=fg_mask)
                index = density_index * len(watershed_list) + watershed_index
                totals[index] += measure(seg.astype("uint32"), gt)

    out = os.path.join(out_dir, f"{stem}.shard{args.shard}of{args.n_shards}.npz")
    np.savez(out, totals=totals, n_images=n_used)
    print(f"shard {args.shard}/{args.n_shards} done in {time.perf_counter() - t0:.0f}s -> {out}")


def run_test(args):
    out_dir = os.path.join(OUT_ROOT, args.dataset, "test")
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{args.dataset}_{args.method}{args.tag}"

    if args.merge:
        parts = [
            pd.read_csv(os.path.join(out_dir, f"{stem}.shard{i}of{args.n_shards}.csv"))
            for i in range(args.n_shards)
        ]
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(os.path.join(out_dir, f"{stem}.csv"), index=False)
        print(f"{args.dataset} {args.method}: {len(df)} images")
        print(df[METRICS].mean().round(4).to_string())
        return

    pairs = get_paths(args.dataset, "test")
    if args.n_shards > 1:
        pairs = pairs[args.shard::args.n_shards]
    print(f"{args.dataset}: {len(pairs)} test images (shard {args.shard}/{args.n_shards}), "
          f"method {args.method}", flush=True)

    if args.method == "apg_v2":
        if args.params is None:
            with open(os.path.join(OUT_ROOT, args.dataset, "tune", "best.json")) as f:
                best = json.load(f)
        else:
            best = json.loads(args.params)
        params = {
            "candidate_threshold": best["candidate_threshold"], "sigma": best["sigma"],
            "foreground_threshold": best["foreground_threshold"], "n_iter": int(best["n_iter"]),
            "score_threshold": best["score_threshold"], "min_size": int(best["min_size"]),
            "max_overlap": best["max_overlap"],
            "refine_with_box_prompts": best.get("refine_with_box_prompts", False),
            "box_extension": int(best.get("box_extension", 0)),
            **FIXED, "min_candidate_size": int(best.get("min_candidate_size",
                                                        FIXED["min_candidate_size"])),
        }
        print(f"tuned configuration: {params}", flush=True)
        segmenter = build_v2_segmenter(args.device)

        def segment(image):
            segmenter.initialize(image, ndim=2)
            return segmenter.generate(**params)
    elif args.method == "ais_v2":
        from micro_sam.v2.postprocessing import flow_instance_segmentation

        with open(os.path.join(OUT_ROOT, args.dataset, "tune", "best_ais.json")) as f:
            params = json.load(f)
        params["n_iter"] = int(params["n_iter"])
        params["min_size"] = int(params["min_size"])
        params["dt"] = FIXED["dt"]
        print(f"tuned AIS configuration: {params}", flush=True)
        segmenter = build_v2_segmenter(args.device)

        def segment(image):
            segmenter.initialize(image, ndim=2)
            prediction = segmenter._prediction
            return flow_instance_segmentation(prediction[0], prediction[1:], **params)
    else:
        from micro_sam.v1.automatic_segmentation import (
            get_predictor_and_segmenter, automatic_instance_segmentation,
        )
        predictor, v1_segmenter = get_predictor_and_segmenter(
            model_type=V1_MODEL_TYPE, checkpoint=None, device=args.device,
            segmentation_mode="ais" if args.method == "ais_v1" else "apg",
        )

        def segment(image):
            seg = automatic_instance_segmentation(
                predictor=predictor, segmenter=v1_segmenter, input_path=image, ndim=2, verbose=False,
            )
            return np.zeros(image.shape[:2], dtype="uint32") if seg is None else seg.astype("uint32")

    rows = []
    for image_path, label_path in tqdm(pairs, desc=f"{args.method} {args.dataset}"):
        image, gt = load_pair(args.dataset, image_path, label_path)
        if gt.max() == 0:
            continue
        rows.append({"image": os.path.basename(image_path), **dict(zip(METRICS, measure(segment(image), gt)))})

    df = pd.DataFrame(rows)
    stem = f"{args.dataset}_{args.method}{args.tag}"
    suffix = "" if args.n_shards == 1 else f".shard{args.shard}of{args.n_shards}"
    out = os.path.join(out_dir, f"{stem}{suffix}.csv")
    df.to_csv(out, index=False)
    print(f"\n{args.dataset} {args.method}: {len(df)} images")
    print(df[METRICS].mean().round(4).to_string())
    print(f"-> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["tune", "ais_tune", "box", "mcs", "test"])
    parser.add_argument("-d", "--dataset", default=DATASETS[0], choices=DATASETS)
    parser.add_argument("--method", default="apg_v2", choices=["apg_v2", "ais_v2", "ais_v1", "apg_v1"])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n_shards", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--round", default="1", choices=["1", "wide"])
    parser.add_argument("--params", default=None, help="JSON APG config, overriding best.json.")
    parser.add_argument("--tag", default="", help="Suffix for the output file name.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    {"tune": run_tune, "ais_tune": run_ais_tune, "box": run_box, "mcs": run_mcs,
     "test": run_test}[args.stage](args)


if __name__ == "__main__":
    main()
