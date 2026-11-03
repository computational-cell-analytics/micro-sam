"""Parameter search for micro-sam2 automatic segmentation: AIS (sparse and dense post-processing) and APG.

'run' searches the parameters that work best over a set of datasets, and writes them to
'<experiment_folder>/best_parameters.json', one combination per backbone, mode and data type (2d, 3d, dense):
1. A broad grid on the validation split of every dataset. 3d APG screens its prompt axes on a few volumes first,
   and then sweeps its selection axes for the best prompt combinations.
2. Rounds that sweep one step past every grid edge that the best combination sits on.
3. For APG, the best combinations with each refinement mode.
The best combination has the best mean relative score over the datasets: the score of a dataset over its best score.
It runs here, or with '--slurm' as Slurm job arrays. Finished sweeps are kept, so a search resumes where it stopped.
The models are the registry models of `micro_sam.v2.util` with a decoder, e.g. 'hvit_b_cells'. A run downloads
them first.

'sweep' runs one sweep per dataset and writes it to '<tuning_root>/<mode>/<model_type>/<checkpoint_id>/', the layout
that `common.read_tuned_params` reads. 'worker' is the Slurm worker of 'run --slurm'. 'prepare' writes the tuning
crops (tuning_crops.json) and the mock 2d slices of the nucleus volumes that the EM organelle searches tune on.

Usage examples:
    python parameter_search.py run -e <exp> -m hvit_b_cells
    python parameter_search.py run -e <exp> -m hvit_t_cells --mode apg --data 3d --slurm --sbatch "-p gpu -G 1"
    python parameter_search.py sweep -e <exp> -m hvit_b --mode ais -d livecell
    python parameter_search.py run -e <exp> -m hvit_s_em_mito --data 2d -d probtem mitolab_tem --n_samples 10
    python parameter_search.py prepare --mock_nucleus_2d
"""

import os
import sys
import json
import math
import time
import shlex
import argparse
import warnings
import itertools
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from concurrent import futures

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.measure import label as label_components

import torch

from elf.io import open_file
from elf.evaluation import mean_segmentation_accuracy

from torch_em.util.image import load_image

from bioimage_cpp.segmentation import label as connected_components, watershed

from bioimage_py.evaluation import symmetric_best_dice_score

import micro_sam.util
from micro_sam.v2.postprocessing import watershed_heightmap, _compute_flow_density
from micro_sam.v2.util import FINETUNED_MODELS, has_registered_decoder, models as model_registry

from common import (
    DATASETS_2D_EM_NUCLEUS_MOCK, DATASETS_3D, DATASETS_DENSE, DATASET_SPACING, VAL_SPLITS, VAL_Z_RANGE,
    EM_NUCLEUS_MOCK_2D_ROOT, GT_MIN_SIZE_2D, CROP_SHAPE_3D, DATA_ROOT, MODEL_TYPES, MODES, TUNING_CROPS,
    VOLUME_SPEED_OPTIONS, build_model, check_data_download, drop_severed_objects, em_roi, get_data_paths,
    has_val_split, load_data, load_evaluation_sample_2d, load_evaluation_sample_3d, n_samples, predict_unisam2,
    read_tuned_params, resolve_checkpoint_identity, sorted_path_pairs, val_z_range, _read_crops, _read_window,
)

# The temporary embeddings of 3d APG go to the job's node-local TMPDIR. In the micro-sam cache of the home
# directory, the leftovers of jobs killed at their time limit filled the quota (2026-10-22).
if os.environ.get("TMPDIR"):
    import uuid
    import atexit
    import shutil
    import micro_sam.v2.util
    import micro_sam.v2.instance_segmentation
    import micro_sam.v2.automatic_prompt_generation

    def _make_temp_embedding_path():
        parent = Path(os.environ["TMPDIR"], "micro_sam_tmp_embeddings")
        parent.mkdir(parents=True, exist_ok=True)
        path = str(parent / f"{uuid.uuid4().hex}.zarr")
        atexit.register(shutil.rmtree, path, ignore_errors=True)
        return path

    for _module in (micro_sam.util, micro_sam.v2.util, micro_sam.v2.instance_segmentation,
                    micro_sam.v2.automatic_prompt_generation):
        _module.make_temp_embedding_path = _make_temp_embedding_path

# Sparse (flow) grid for LM data. Keys map to flow_instance_segmentation arguments.
LM_GRID = {
    "foreground_threshold": [0.3, 0.4, 0.5, 0.6, 0.7],
    "density_threshold": [5.0, 10.0, 20.0],
    "min_size": [10, 25, 50, 100, 200, 500],
    "sigma": [0.25, 0.5, 1.0, 2.0],
    "n_iter": [50, 100, 200],
    "dt": [0.25, 0.5, 1.0],
    "foreground_weight": [0.0, 0.25, 0.5, 0.65, 0.75, 0.9],
}

# Dense (multicut) grid for EM data. Keys map to run_multicut arguments.
EM_GRID = {
    "beta": [0.5, 0.6, 0.7, 0.8],
    "density_threshold": [3.0, 5.0, 10.0],
    "sigma": [0.5, 1.0, 2.0],
    "n_iter": [25, 50],
}

# Keys map to AutomaticPromptGenerator.generate. Only 'candidate_threshold' and 'sigma' re-prompt
# SAM2; the rest select among proposals that are already there, so they are swept densely for free.
# See APG_PROPOSAL_KEYS and score_sample_apg_cached. 'foreground_threshold' and 'min_size' are pinned
# to their defaults: each moved the result by less than 0.001, which does not pay for a larger sweep.
APG_GRID_2D = {
    "foreground_threshold": [0.7],
    "candidate_threshold": [1.0, 1.5, 2.25, 3.0],
    "n_iter": [50],
    "sigma": [0.5, 1.0, 2.0],
    "min_candidate_size": [1, 4],
    "score_threshold": [0.5, 0.6, 0.7, 0.8],
    "max_overlap": [0.15, 0.3],
    "min_size": [50],
    "refinement": [None, "boxes"],
}

# A volume gates its propagation on the score, so none of its axes are free and this grid stays small.
# A candidate's convergence density scales with the object's size, so small nuclei never clear a single
# rung of the ladder: hence the lower bottom rungs, less smoothing and a smaller component floor.
# 'max_overlap' is swept here and pinned in the 2d grid, because of the failure it causes on a volume:
# a neighbour clips a fifth of an object, the object's own well-propagated mask is then rejected
# wholesale, and the object is left unclaimed.
APG_GRID_3D = {
    "candidate_threshold": [(1.5, 10.0), (1.0, 5.0), (0.5, 5.0), (0.25, 1.0, 5.0)],
    "sigma": [0.25, 0.5, 1.0],
    "min_candidate_size": [1, 4],
    "score_threshold": [0.6],
    "max_overlap": [0.15, 0.5],
    "min_size": [100],
    # Pruning duplicate candidates between propagation rounds is 1.5-2.7x on most volumes at an equal
    # or better score, but it costs cremi 0.03 CREMI, so the sweep decides it per dataset rather than
    # a default doing it for all of them. It roughly halves the propagation, so the extra rung costs
    # far less than a normal one.
    "propagation_waves": [1, 4],
}

# The broad 2d grids of the global search, selected with '--grid_preset broad'. Every axis that the
# v6 per-dataset sweeps chose at a grid edge goes past that edge. The sparse 'n_iter' x 'dt' product
# covers the flow travel from 12.5 to 400. APG sweeps 'foreground_threshold', which APG_GRID_2D pins, and
# keeps 3 values of 'candidate_threshold' and 'sigma', which were flat in the v6 sweeps. Each proposal
# group costs a SAM2 pass, so this is 45 groups. The selection axes are cheap and dense. The refinement
# is off: a refined selection re-prompts SAM2, so it is compared on the best configurations afterwards.
BROAD_GRIDS_2D = {
    "sparse": {
        "foreground_threshold": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "density_threshold": [2.5, 5.0, 10.0, 20.0, 30.0, 40.0],
        "min_size": [10, 25, 50, 100, 200, 500],
        "sigma": [0.25, 0.5, 1.0, 2.0],
        "n_iter": [50, 100, 200],
        "dt": [0.25, 0.5, 1.0, 2.0],
        "foreground_weight": [0.0, 0.25, 0.5, 0.65, 0.75, 0.9, 1.0],
    },
    "apg": {
        "foreground_threshold": [0.3, 0.4, 0.5, 0.6, 0.7],
        "candidate_threshold": [1.0, 2.0, 3.0],
        "n_iter": [50],
        "dt": [0.5],
        "sigma": [0.5, 1.0, 2.0],
        "min_candidate_size": [4],
        "score_threshold": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "max_overlap": [0.1, 0.15, 0.3, 0.5],
        "min_size": [25, 50, 100],
        "refinement": [None],
    },
}
# The broad 3d grids of the global search, keyed like the 2d presets plus '_3d' for volumes. The volumes are
# 32 x 512 x 512 crops of the in-domain validation data, the shape of the test crops (PIPELINE_CROP_3D).
# Sparse goes past every edge the earlier per-dataset sweeps (LM_GRID) chose, and its flow travel covers 25 to 800
# like the 2d grid. 'min_size' counts voxels. Dense goes past the edges of EM_GRID:
# its earlier optimum sat at the lowest 'beta', 'density_threshold' and 'sigma'.
# A volume runs every APG combo in full, so APG sweeps in two stages: 'apg_3d' covers the prompt axes with the
# selection pinned to the middle of its range, and the selection axes are then swept on the best prompt combos
# (see `apg_3d_base_stages`). Its 'foreground_threshold' and 'sigma' keep the values around the earlier
# 3d defaults and the 2d winners, which makes 90 combos. 'propagation_waves' only acts on tiled volumes, so it
# is not swept.
BROAD_GRIDS_3D = {
    "sparse_3d": {
        "foreground_threshold": [0.3, 0.4, 0.5, 0.6, 0.7],
        "density_threshold": [5.0, 10.0, 20.0, 30.0, 50.0, 80.0],
        "min_size": [25, 50, 100, 200, 500, 1000],
        "sigma": [0.15, 0.25, 0.5, 1.0],
        "n_iter": [50, 100, 200, 400],
        "dt": [0.5, 1.0, 2.0],
        "foreground_weight": [0.0, 0.25, 0.5, 0.65, 0.75, 0.9, 1.0],
    },
    "dense": {
        "beta": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "density_threshold": [1.0, 2.0, 3.0, 5.0, 10.0],
        "sigma": [0.25, 0.5, 1.0, 2.0],
        "n_iter": [12, 25, 50, 100],
        "dt": [0.5, 1.0],
    },
    "apg_3d": {
        "foreground_threshold": [0.6, 0.7, 0.8],
        "n_iter": [50],
        "candidate_threshold": [(2.0, 10.0), (1.5, 10.0), (1.0, 5.0), (0.5, 5.0), (0.25, 1.0, 5.0)],
        "sigma": [0.25, 0.5, 1.0],
        "min_candidate_size": [1, 4],
        "score_threshold": [0.6],
        "max_overlap": [0.3],
        "min_size": [100],
    },
}
PRESET_GRIDS_2D = {"broad": BROAD_GRIDS_2D}
PRESET_GRIDS_3D = {"broad": BROAD_GRIDS_3D}
GRID_PRESETS = ("default", *PRESET_GRIDS_2D)

# The APG parameters that decide the mask proposals, which is the half that needs the model. Combos
# that share these reuse one set of proposals, as the sparse grid reuses a cached flow density.
APG_PROPOSAL_KEYS = ("candidate_threshold", "sigma", "foreground_threshold", "n_iter", "dt", "min_candidate_size")

# Sparse-flow parameters that determine the (expensive) convergence-density map. Combos that share
# these reuse a cached density. Only 'density_threshold', 'foreground_weight' and 'min_size' (the cheap
# seed + watershed + size-filter steps) then vary on top. See score_image_sparse_cached.
FLOW_DENSITY_KEYS = ("foreground_threshold", "sigma", "n_iter", "dt")

# Dense-multicut parameters that determine the (expensive) slice-wise oversegmentation and RAG.
# Combos that share these reuse a cached oversegmentation. Only 'beta' (the cheap edge-cost +
# multicut-solve step) then varies on top. See score_image_dense_cached.
OVERSEG_KEYS = ("density_threshold", "sigma", "n_iter", "dt")

# Metric the grid is ranked by, and whether lower is better. 'msa' (mean segmentation accuracy) is
# maximised; 'cremi' (a VI + adapted-Rand combination for neuron segmentation) is minimised.
CRITERION_ASCENDING = {"msa": False, "cremi": True}

POSTPROC_THREADS = 4


def tuning_config(dataset_name, mode, criterion=None, crop_shape=None, grid_override=None, grid_preset="default"):
    """Build the sweep configuration for one dataset and mode.

    Dense-neuron EM datasets (`common.DATASETS_DENSE`) are tuned in dense (multicut) mode and ranked
    by the CREMI score, all others in sparse (flow) mode and ranked by mSA. With mode='apg' the
    postprocessing is replaced by the prompt generation, which is swept over its own grid; the ranking
    metric follows the data either way, so an APG result is directly comparable with the AIS result of
    the same dataset.

    Args:
        dataset_name: The dataset to tune on.
        mode: The segmentation mode, 'ais' or 'apg'.
        criterion: The metric the grid is ranked by. Defaults to the data-specific choice.
        crop_shape: The 3d center crop. Defaults to CROP_SHAPE_3D.
        grid_override: Values replacing the swept values of the named grid axes. Only the named axes
            change, every other axis keeps the values the mode's grid gives it.
        grid_preset: 'default' for the grids above, or 'broad' for BROAD_GRIDS_2D (images) and BROAD_GRIDS_3D
            (volumes).

    Returns:
        The configuration dict.
    """
    is_dense = dataset_name in DATASETS_DENSE
    is_3d = dataset_name in DATASETS_3D

    if mode == "apg":
        postproc_mode, grid = "apg", (APG_GRID_3D if is_3d else APG_GRID_2D)
    else:
        postproc_mode, grid = ("dense", EM_GRID) if is_dense else ("sparse", LM_GRID)

    if grid_preset != "default":
        # A volume takes the '_3d' grid of the preset, and dense data (always volumes) the 'dense' one.
        grids = (PRESET_GRIDS_3D if is_3d else PRESET_GRIDS_2D).get(grid_preset, {})
        key = f"{postproc_mode}_3d" if is_3d and postproc_mode != "dense" else postproc_mode
        if key not in grids:
            raise ValueError(f"The '{grid_preset}' grid does not exist for '{dataset_name}' in {postproc_mode} mode.")
        grid = grids[key]

    if grid_override:
        # A typo would otherwise fail deep inside the sweep.
        unknown = sorted(set(grid_override) - set(grid))
        if unknown:
            raise ValueError(f"{unknown} are not axes of the '{postproc_mode}' grid: {sorted(grid)}.")
        grid = {**grid, **grid_override}

    return {
        "mode": postproc_mode,
        # Which metrics compute_metrics reports. Follows the data, not the mode, so that an APG run
        # on a dense-neuron dataset is still ranked by the CREMI score.
        "metric_mode": "dense" if is_dense else "sparse",
        "ndim": 3 if is_3d else 2,
        "grid": grid,
        "spacing": DATASET_SPACING.get(dataset_name, None),
        "crop": (crop_shape or CROP_SHAPE_3D) if is_3d else None,
        "criterion": criterion or ("cremi" if is_dense else "msa"),
        # Only set for the volumes with no split of their own.
        "z_range": VAL_Z_RANGE.get(dataset_name),
    }


def compute_metrics(seg, labels, metric_mode, border_min_size=0):
    """Compute the evaluation metrics for one segmentation.

    The function always reports mSA. Sparse mode also logs the symmetric best Dice (SBD), which no ranking uses.
    Dense (EM neuron) mode also reports the CREMI score with its VI-split, VI-merge and adapted-Rand components.
    Neuron segmentation is ranked by CREMI (lower is better) instead of mSA.
    """
    if seg.ndim == 2:
        # Symmetric with the ground truth, which `load_evaluation_sample_2d` filtered the same way.
        seg = drop_severed_objects(seg, border_min_size)
    metrics = {"msa": float(mean_segmentation_accuracy(seg, labels))}
    if metric_mode == "sparse":
        metrics["sbd"] = float(symmetric_best_dice_score(seg, labels))
    if metric_mode == "dense":
        from elf.evaluation import cremi_score
        vi_split, vi_merge, adapted_rand, cremi = cremi_score(seg, labels)
        metrics["cremi"] = float(cremi)
        metrics["vi_split"] = float(vi_split)
        metrics["vi_merge"] = float(vi_merge)
        metrics["adapted_rand"] = float(adapted_rand)
    return metrics


def shard_group_keys(postproc_mode, ndim):
    """The parameters whose combos share the expensive step of a sweep, or None if no step is shared."""
    if postproc_mode == "apg" and ndim == 2:
        return APG_PROPOSAL_KEYS
    if postproc_mode == "sparse":
        return FLOW_DENSITY_KEYS
    if postproc_mode == "dense":
        return OVERSEG_KEYS
    return None


def shard_params_list(params_list, shard_index, num_shards, postproc_mode, ndim):
    """Return this shard's slice of 'params_list'.

    2d APG reuses one expensive `propose()` model call across every combo that shares
    APG_PROPOSAL_KEYS (see score_sample_apg_cached), so its cost scales with the number of distinct
    proposal groups touched, not with len(params_list). A plain interleaved slice still touches every
    group in every shard whenever 'num_shards' divides the selection-combo block evenly (it does here:
    16 selections per group), so it would save nothing. Splitting whole groups across shards instead
    makes the split actually proportional. Sparse is split the same way, by FLOW_DENSITY_KEYS, so that
    no two shards integrate the same flow density, and dense by OVERSEG_KEYS, so that no two shards build
    the same oversegmentation. 3d APG runs generate() per combo, which scales with len(params_list)
    directly, so a plain interleaved slice already shares out the work evenly there.
    """
    group_keys = shard_group_keys(postproc_mode, ndim)
    if group_keys is None or num_shards <= 1:
        return params_list[shard_index::num_shards]
    groups = {}
    for params in params_list:
        key = tuple(sorted((k, v) for k, v in params.items() if k in group_keys))
        groups.setdefault(key, []).append(params)
    return [
        params for i, key in enumerate(groups) if i % num_shards == shard_index for params in groups[key]
    ]


def n_shard_groups(params_list, postproc_mode, ndim):
    """The number of groups `shard_params_list` splits 'params_list' into, so the most shards that are not empty."""
    group_keys = shard_group_keys(postproc_mode, ndim)
    if group_keys is None:
        return len(params_list)
    return len({tuple(sorted((k, v) for k, v in p.items() if k in group_keys)) for p in params_list})


def sweep_params_list(config, params_file=None):
    """The combos of a sweep: those of 'params_file' (a JSON list), or else the grid of 'config'."""
    if params_file is not None:
        with open(params_file) as f:
            return json.load(f)
    keys = list(config["grid"])
    params_list = [dict(zip(keys, combo)) for combo in itertools.product(*[config["grid"][k] for k in keys])]
    if config["mode"] == "sparse":
        params_list = deduplicate_flow_travel(params_list)
    return params_list


def deduplicate_flow_travel(params_list):
    """Drop sparse combos whose flow travel duplicates a cheaper one.

    'n_iter' and 'dt' act on the segmentation only through their product, the distance a pixel is
    advected. Keeping the smallest 'n_iter' per product therefore covers the same travel distances
    with fewer combos and fewer integration steps. The equivalence is empirical rather than exact,
    since the integrator's discretization error still depends on 'dt'.
    """
    best, order = {}, []
    for params in params_list:
        if "n_iter" not in params or "dt" not in params:
            return params_list
        others = tuple(sorted((k, v) for k, v in params.items() if k not in ("n_iter", "dt")))
        key = (others, round(params["n_iter"] * params["dt"], 6))
        if key not in best:
            best[key] = params
            order.append(key)
        elif params["n_iter"] < best[key]["n_iter"]:
            best[key] = params
    return [best[key] for key in order]


def score_image_sparse_cached(
    prediction, labels, params_list, n_threads=POSTPROC_THREADS, spacing=None, border_min_size=0,
):
    """Score all sparse combos on one image, caching the flow density across shared combos.

    Faithfully reproduces flow_instance_segmentation: the height map and each convergence-density map
    (the expensive flow integration) are computed once and reused; only the cheap seed / watershed /
    min-size steps vary per (density_threshold, min_size). Returns a per-combo list of metric dicts
    (None where a combo failed), aligned with params_list.
    """
    foreground = prediction[0]
    directed = prediction[1:4]
    ndim = foreground.ndim
    if directed.shape[0] > ndim:
        directed = directed[-ndim:]

    # The convergence densities and the height maps are built up front, so the scoring below only reads them.
    fg_mask_cache, density_cache, hmap_cache = {}, {}, {}
    for params in params_list:
        ft, sigma, n_iter, dt = (params[k] for k in FLOW_DENSITY_KEYS)
        if ft not in fg_mask_cache:
            fg_mask_cache[ft] = foreground > ft
        key = (ft, sigma, n_iter, dt)
        if key not in density_cache:
            density_cache[key] = _compute_flow_density(
                directed, fg_mask_cache[ft], n_iter=int(n_iter), dt=dt, sigma=sigma, spacing=spacing,
                n_threads=n_threads,
            )
        fw = params["foreground_weight"]
        if fw not in hmap_cache:
            hmap_cache[fw] = watershed_heightmap(foreground, directed, fw)

    # The base watershed does not depend on min_size, so the combos that differ only in min_size share it. They
    # are scored together and the base is dropped afterwards: keeping every base of a volume runs out of memory.
    groups = {}
    for index, params in enumerate(params_list):
        key = tuple(params[k] for k in FLOW_DENSITY_KEYS) + (params["density_threshold"], params["foreground_weight"])
        groups.setdefault(key, []).append(index)

    def score_group(key, indices):
        ft, sigma, n_iter, dt, density_threshold, fw = key
        fg_mask, hmap = fg_mask_cache[ft], hmap_cache[fw]
        try:
            base = watershed(hmap, markers=connected_components(density_cache[key[:4]] > density_threshold),
                             mask=fg_mask)
        except Exception as e:
            warnings.warn(f"Sparse postprocessing failed for {dict(zip(FLOW_DENSITY_KEYS, key))}: {e}")
            return [(index, None) for index in indices]
        results = []
        for index in indices:
            try:
                seg, min_size = base, params_list[index]["min_size"]
                if min_size > 0:
                    seg = base.copy()
                    ids, sizes = np.unique(seg, return_counts=True)
                    discard = ids[(sizes < min_size) & (ids > 0)]
                    seg[np.isin(seg, discard)] = 0
                    seg = watershed(hmap, markers=seg, mask=fg_mask)
                results.append((index, compute_metrics(seg.astype("uint32"), labels, "sparse", border_min_size)))
            except Exception as e:
                warnings.warn(f"Sparse postprocessing failed for {params_list[index]}: {e}")
                results.append((index, None))
        return results

    # Scoring a group is independent of the others, and dominates the runtime of the sweep.
    scores = [None] * len(params_list)
    with futures.ThreadPoolExecutor(n_threads) as tp:
        for results in tp.map(lambda item: score_group(*item), groups.items()):
            for index, metrics in results:
                scores[index] = metrics
    return scores


def dense_boundary_and_distances(prediction):
    """Return the (boundary_map, in-plane distances) inputs to run_multicut for a dense prediction."""
    fg = prediction[0]
    boundary_map = fg.max() - fg
    denom = boundary_map.max()
    if denom > 0:
        boundary_map = boundary_map / denom
    distances = np.stack([prediction[2], prediction[3]])
    return boundary_map, distances


def dense_oversegmentation(boundary_map, distances, density_threshold, sigma, n_iter, dt, n_threads):
    """The expensive, beta-independent part of run_multicut: oversegmentation, RAG and edge features."""
    from elf.segmentation.features import compute_rag, compute_boundary_mean_and_length, compute_z_edge_mask
    n_slices = boundary_map.shape[0]
    overseg = np.zeros(boundary_map.shape, dtype="uint64")

    def run_overseg(z):
        bd = boundary_map[z]
        dists = distances[:, z]
        fg_mask = np.ones(bd.shape, dtype="bool")
        # 1 thread per slice: the ThreadPoolExecutor below already parallelizes across slices.
        density = _compute_flow_density(dists, fg_mask, n_iter=int(n_iter), dt=dt, sigma=sigma, n_threads=1)
        seeds = connected_components(density > density_threshold)
        bd = bd if np.issubdtype(bd.dtype, np.floating) else bd.astype("float32")
        wsz = watershed(bd, markers=seeds)
        overseg[z] = wsz
        return int(wsz.max())

    with futures.ThreadPoolExecutor(n_threads) as tp:
        offsets = list(tp.map(run_overseg, range(n_slices)))

    offsets = np.array(offsets, dtype="uint64")
    offsets = np.roll(offsets, 1)
    offsets[0] = 0
    overseg += np.cumsum(offsets)[:, None, None]

    rag = compute_rag(overseg)
    feats = compute_boundary_mean_and_length(rag, overseg, boundary_map)
    z_edges = None if n_slices == 1 else compute_z_edge_mask(rag, overseg)
    return overseg, rag, feats, z_edges, n_slices


def dense_solve(rag, feats, z_edges, overseg, n_slices, beta):
    """Run the beta-dependent edge costs + multicut solve on a cached oversegmentation."""
    from elf.segmentation.features import project_node_labels_to_pixels
    from elf.segmentation.multicut import compute_edge_costs, multicut_decomposition
    if n_slices == 1:
        costs = compute_edge_costs(feats[:, 0], edge_sizes=feats[:, 1], weighting_scheme="all", beta=beta)
    else:
        costs = compute_edge_costs(
            feats[:, 0], edge_sizes=feats[:, 1], weighting_scheme="xyz", z_edge_mask=z_edges, beta=beta,
        )
    node_labels = multicut_decomposition(rag, costs)
    seg = project_node_labels_to_pixels(rag, overseg, node_labels)
    return seg.astype("uint32")


def score_image_dense_cached(prediction, labels, params_list, n_threads=POSTPROC_THREADS, border_min_size=0):
    """Score all dense combos on one image, caching the oversegmentation and RAG across shared combos."""
    boundary_map, distances = dense_boundary_and_distances(prediction)
    overseg_cache = {}
    scores = []
    for params in params_list:
        dt = params.get("dt", 0.5)
        key = tuple(params.get(k, dt if k == "dt" else None) for k in OVERSEG_KEYS)
        if key not in overseg_cache:
            overseg_cache[key] = dense_oversegmentation(
                boundary_map, distances, params["density_threshold"], params["sigma"], params["n_iter"],
                dt=dt, n_threads=n_threads,
            )
        overseg, rag, feats, z_edges, n_slices = overseg_cache[key]
        try:
            seg = dense_solve(rag, feats, z_edges, overseg, n_slices, params["beta"])
            scores.append(compute_metrics(seg, labels, "dense", border_min_size))
        except Exception as e:
            warnings.warn(f"Dense multicut solve failed for {params}: {e}")
            scores.append(None)
    return scores


def score_sample_apg(segmenter, raw, labels, params_list, ndim, metric_mode, spacing=None, border_min_size=0):
    """Return a per-combo list of metric dicts for one sample, aligned with params_list.

    The encoder and the decoder run once per sample. An image then reuses its mask proposals across
    every combo that only selects among them; a volume gates its propagation on the score, so it has
    to run every combo in full.
    """
    segmenter.clear_state()
    # A volume takes the options that stop it re-reading its features on every propagation pass.
    segmenter.initialize(raw, ndim=ndim, **(VOLUME_SPEED_OPTIONS if ndim == 3 else {}))
    if ndim == 2:
        return score_sample_apg_cached(segmenter, labels, params_list, metric_mode, border_min_size)

    scores = []
    for params in params_list:
        try:
            seg = segmenter.generate(spacing=spacing, **params)
            scores.append(compute_metrics(seg.astype("uint32"), labels, metric_mode, border_min_size))
        except Exception as e:
            warnings.warn(f"Prompt generation failed for {params}: {e}")
            scores.append(None)
    return scores


def score_sample_apg_cached(segmenter, labels, params_list, metric_mode, border_min_size=0):
    """Score every combo of an image, prompting SAM2 once per distinct set of proposal parameters."""
    groups = {}
    for index, params in enumerate(params_list):
        key = tuple(sorted((k, v) for k, v in params.items() if k in APG_PROPOSAL_KEYS))
        groups.setdefault(key, []).append(index)

    scores = [None] * len(params_list)
    for key, indices in groups.items():
        try:
            proposals = segmenter.propose(**dict(key))
        except Exception as e:
            warnings.warn(f"Prompt generation failed for {dict(key)}: {e}")
            continue
        for index in indices:
            selection = {k: v for k, v in params_list[index].items() if k not in APG_PROPOSAL_KEYS}
            try:
                seg = segmenter.select(proposals, **selection)
                scores[index] = compute_metrics(seg.astype("uint32"), labels, metric_mode, border_min_size)
            except Exception as e:
                warnings.warn(f"Selection failed for {params_list[index]}: {e}")
    return scores


def cached_prediction(cache_dir, index, model, raw, ndim, device):
    """Return the decoder output of one validation sample, from 'cache_dir' when it holds it.

    Without a cache directory the model always runs. With one, a missing output is predicted and
    written there, through a temporary file so that a concurrent reader never sees half of it.
    """
    if cache_dir is None:
        return predict_unisam2(model, raw, ndim=ndim, device=device)
    path = os.path.join(cache_dir, f"{index}.npy")
    if os.path.exists(path):
        return np.load(path)
    if model is None:
        raise FileNotFoundError(f"'{path}' is not cached and no model was loaded to predict it.")
    prediction = predict_unisam2(model, raw, ndim=ndim, device=device)
    tmp_path = f"{path}.{os.getpid()}.tmp.npy"
    np.save(tmp_path, prediction)
    os.replace(tmp_path, path)
    return prediction


def score_image(
    prediction, labels, mode, params_list, n_threads=POSTPROC_THREADS, spacing=None, border_min_size=0,
):
    """Return a per-combo list of metric dicts for one decoder prediction, aligned with params_list."""
    if mode == "sparse":
        return score_image_sparse_cached(
            prediction, labels, params_list, n_threads=n_threads, spacing=spacing,
            border_min_size=border_min_size,
        )
    return score_image_dense_cached(
        prediction, labels, params_list, n_threads=n_threads, border_min_size=border_min_size
    )


def report_best(df, dataset_name, config):
    """Print the best parameter combination of a finished sweep.

    The values are read back from the CSV for the report only. `common.read_tuned_params` is what
    turns them back into the types `generate` and the postprocessing take.
    """
    criterion = config["criterion"]
    ascending = CRITERION_ASCENDING[criterion]
    best = df.sort_values(f"{criterion}_mean", ascending=ascending).iloc[0]
    params = {key: best[key] for key in config["grid"]}
    direction = "lower is better" if ascending else "higher is better"
    print(f"Best params for {dataset_name} (by {criterion}, {direction}): {params}")
    print(f"{criterion} = {best[f'{criterion}_mean']:.4f} (+/-{best[f'{criterion}_std']:.4f})")


def sweep_cache_is_current(csv_path, config):
    """Return whether a cached sweep contains every parameter in the current grid."""
    columns = set(pd.read_csv(csv_path, nrows=0).columns)
    missing = [key for key in config["grid"] if key not in columns]
    if missing:
        print(f"Ignoring the cached sweep at '{csv_path}'. It lacks parameter columns: {missing}.")
        return False
    return True


def shard_output_path(output_dir, dataset_name, shard_index=0, num_shards=1):
    """The CSV a shard writes to: the canonical path when unsharded, a per-shard path otherwise."""
    if num_shards <= 1:
        return os.path.join(output_dir, f"{dataset_name}.csv")
    return os.path.join(output_dir, f"{dataset_name}.shard{shard_index}of{num_shards}.csv")


def per_image_path(csv_path):
    """The per-image score matrix that is written next to a sweep CSV."""
    return csv_path[:-len(".csv")] + ".per_image.npz"


def save_per_image(csv_path, params_list, sample_index, scores, criterion):
    """Save the criterion of every combo on every scored sample, as a (combos x samples) matrix.

    The CSV keeps only the mean and the standard deviation. The matrix lets a global selection
    bootstrap over the samples, and compare the combos on exactly the same samples.
    """
    np.savez_compressed(
        per_image_path(csv_path), scores=np.asarray(scores, dtype="float64").reshape(len(sample_index), -1).T,
        sample_index=np.asarray(sample_index, dtype="int64"), params=json.dumps(params_list), criterion=criterion,
    )


def merge_per_image(shard_paths, csv_path):
    """Concatenate the per-image matrices of the shards along the combos."""
    shards = [np.load(per_image_path(p)) for p in shard_paths]
    sample_index = shards[0]["sample_index"]
    if any(not np.array_equal(s["sample_index"], sample_index) for s in shards):
        raise ValueError(f"The shards of '{csv_path}' scored different samples.")
    np.savez_compressed(
        per_image_path(csv_path), scores=np.concatenate([s["scores"] for s in shards]), sample_index=sample_index,
        params=json.dumps([p for s in shards for p in json.loads(str(s["params"]))]),
        criterion=str(shards[0]["criterion"]),
    )


def merge_shards(
    output_root, dataset_name, mode, model_type, num_shards, checkpoint_id=None, criterion=None,
    grid_preset="default",
):
    """Combine a dataset's shard CSVs into the canonical ranked file `tune_parameters` would write.

    Each shard scores a disjoint slice of the same grid on the same validation samples, so merging is
    a plain concatenation of the per-combo rows followed by a re-sort; no metric needs recomputing.
    """
    output_dir = os.path.join(output_root, model_type, checkpoint_id) if checkpoint_id else os.path.join(
        output_root, model_type
    )
    shard_paths = [
        shard_output_path(output_dir, dataset_name, shard_index=i, num_shards=num_shards) for i in range(num_shards)
    ]
    missing = [p for p in shard_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing shard(s) for '{dataset_name}': {missing}")

    config = tuning_config(dataset_name, mode, criterion=criterion, grid_preset=grid_preset)
    ascending = CRITERION_ASCENDING[config["criterion"]]
    df = pd.concat([pd.read_csv(p) for p in shard_paths], ignore_index=True)
    df = df.sort_values(f"{config['criterion']}_mean", ascending=ascending).reset_index(drop=True)

    csv_path = os.path.join(output_dir, f"{dataset_name}.csv")
    df.to_csv(csv_path, index=False)
    # The sweeps from before the per-image output have no matrix to merge.
    if all(os.path.exists(per_image_path(p)) for p in shard_paths):
        merge_per_image(shard_paths, csv_path)
    print(f"Merged {num_shards} shards into '{csv_path}'.")
    report_best(df, dataset_name, config)


def tune_parameters(
    model, mode, dataset_name, data_root, model_type, output_root, device,
    n_threads=POSTPROC_THREADS, n_tuning_samples=None, tuning_stride=1, crop_shape=None, criterion=None,
    checkpoint_id=None, shard_index=0, num_shards=1, grid_override=None, grid_preset="default", subsample=None,
    prediction_cache=None, cache_only=False, params_file=None,
):
    """Sweep the grid of a mode on the validation split and return the best parameter combination.

    Inference runs once per sample and every combination is scored on it before moving on, so the
    predictions are never all held in memory at once.

    The result is written below '<output_root>/<model_type>/<checkpoint_id>', which is the layout
    `common.read_tuned_params` reads. An existing file is loaded rather than recomputed, so a
    preempted job resumes at the next dataset instead of sweeping again.

    With 'num_shards' > 1, only every 'num_shards'-th combo (offset by 'shard_index') is swept, and
    the result is written to a per-shard file instead of the canonical one; `merge_shards` combines
    the shards afterwards. This splits a dataset's grid across several jobs without repeating the
    (expensive) per-sample inference in a merge-incompatible way, since every shard still scores its
    slice of the grid against the very same validation samples.

    Args:
        model: The model of the mode, from `common.build_model`.
        mode: The segmentation mode, 'ais' or 'apg'.
        dataset_name: The dataset to tune on.
        data_root: The root the data lives in.
        model_type: The SAM2 backbone, which names the output subdirectory.
        output_root: The root the sweep results are written to.
        device: The torch device.
        n_threads: The threads for the postprocessing.
        n_tuning_samples: Cap the sweep to this many validation samples.
        tuning_stride: Only score every this-th validation sample (after the cell-type-grouped sort
            order in e.g. livecell, a stride spreads the cap evenly across groups instead of taking
            only the first few groups in full).
        crop_shape: The 3d center crop.
        criterion: The metric the grid is ranked by.
        checkpoint_id: The checksum of all model weights used by the mode.
        shard_index: This job's slice of the grid, in [0, num_shards).
        num_shards: How many jobs split the grid. 1 means no sharding.
        grid_override: Values replacing the swept values of the named grid axes, see `tuning_config`.
        grid_preset: The grid to start from, see `tuning_config`.
        subsample: Score this many validation samples, spread evenly over the split. It replaces
            'n_tuning_samples' and 'tuning_stride'.
        prediction_cache: The root of the decoder outputs of the AIS sweep, see `cached_prediction`. A
            sweep that finds every output there needs no model, so it runs on a CPU node.
        cache_only: Only write the decoder outputs to 'prediction_cache', do not sweep.
        params_file: A JSON list of parameter dicts to sweep instead of the grid.

    Returns:
        The best parameter combination, or None when the dataset has nothing held out to tune on, or
        when sharded (a shard's result is partial, so callers should use `merge_shards` afterwards).
    """
    if not has_val_split(dataset_name):
        print(f"'{dataset_name}' has no data held out from the evaluation, so the defaults are used.")
        return None

    config = tuning_config(
        dataset_name, mode, criterion=criterion, crop_shape=crop_shape, grid_override=grid_override,
        grid_preset=grid_preset,
    )
    output_dir = os.path.join(output_root, model_type, checkpoint_id) if checkpoint_id else os.path.join(
        output_root, model_type
    )
    csv_path = shard_output_path(output_dir, dataset_name, shard_index=shard_index, num_shards=num_shards)
    os.makedirs(output_dir, exist_ok=True)

    legacy_path = os.path.join(output_root, model_type, f"{dataset_name}.csv")
    cached_path = None
    if os.path.exists(csv_path):
        cached_path = csv_path
    elif num_shards <= 1 and checkpoint_id is not None and os.path.exists(legacy_path):
        cached_path = legacy_path
    if cached_path is not None and sweep_cache_is_current(cached_path, config):
        if num_shards > 1:
            print(f"Loading the finished shard at '{csv_path}'.")
            return None
        params = read_tuned_params(output_root, dataset_name, model_type, checkpoint_id)
        print(f"Loading the finished sweep at '{cached_path}'.")
        report_best(pd.read_csv(cached_path), dataset_name, config)
        return params

    ndim, postproc_mode = config["ndim"], config["mode"]
    params_list = sweep_params_list(config, params_file)
    if num_shards > 1:
        params_list = shard_params_list(params_list, shard_index, num_shards, postproc_mode, ndim)

    border_min_size = GT_MIN_SIZE_2D.get(dataset_name, 0) if ndim == 2 else 0
    total = n_samples(dataset_name, data_root, split="val")
    if subsample is not None:
        tuning_stride, n_tuning_samples = max(1, total // subsample), subsample
    total = len(range(0, total, tuning_stride))
    if n_tuning_samples is not None:
        total = min(total, n_tuning_samples)
    print(
        f"{dataset_name}: {len(params_list)} combinations over {total} validation sample(s) "
        f"(stride={tuning_stride}), mode='{mode}'."
    )

    cache_dir = None
    if prediction_cache is not None:
        if postproc_mode == "apg":
            raise ValueError("The prediction cache holds decoder outputs, which only the AIS sweep reads.")
        cache_dir = os.path.join(prediction_cache, model_type, checkpoint_id or "default", dataset_name)
        os.makedirs(cache_dir, exist_ok=True)

    samples = load_data(
        dataset_name, data_root, ndim, split="val", crop_shape=config["crop"], z_range=config["z_range"],
    )
    metric_lists = [[] for _ in params_list]
    sample_index, per_image = [], []
    t0 = time.perf_counter()
    n_scored = 0
    for index, (raw, labels, _) in enumerate(tqdm(samples, total=total, desc=f"tune-{mode}")):
        if index % tuning_stride != 0:
            continue
        if n_tuning_samples is not None and n_scored >= n_tuning_samples:
            break
        n_scored += 1
        if labels.max() == 0:  # Nothing to score without ground-truth.
            continue
        try:
            if postproc_mode == "apg":
                scores = score_sample_apg(
                    model, raw, labels, params_list, ndim, config["metric_mode"],
                    spacing=config["spacing"], border_min_size=border_min_size,
                )
            else:
                prediction = cached_prediction(cache_dir, index, model, raw, ndim, device)
                if cache_only:
                    continue
                scores = score_image(
                    prediction, labels, postproc_mode, params_list,
                    n_threads=n_threads, spacing=config["spacing"], border_min_size=border_min_size,
                )
        except Exception as e:
            warnings.warn(f"Skipping validation sample {index} of '{dataset_name}': {e}")
            continue
        for i, metrics in enumerate(scores):
            if metrics is not None:
                metric_lists[i].append(metrics)
        sample_index.append(index)
        per_image.append([np.nan if m is None else m[config["criterion"]] for m in scores])

    if cache_only:
        print(f"Cached the predictions of {n_scored} sample(s) in '{cache_dir}' ({time.perf_counter() - t0:.0f}s).")
        return None

    rows = []
    for params, per_sample in zip(params_list, metric_lists):
        if not per_sample:
            continue
        row = {**params, "n_images": len(per_sample)}
        for metric_key in per_sample[0]:
            values = np.asarray([m[metric_key] for m in per_sample], dtype="float64")
            row[f"{metric_key}_mean"] = float(values.mean())
            row[f"{metric_key}_std"] = float(values.std())
        rows.append(row)

    if not rows:
        warnings.warn(f"The sweep for '{dataset_name}' scored nothing, falling back to the defaults.")
        return None

    ascending = CRITERION_ASCENDING[config["criterion"]]
    df = pd.DataFrame(rows).sort_values(f"{config['criterion']}_mean", ascending=ascending).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    save_per_image(csv_path, params_list, sample_index, per_image, config["criterion"])
    print(f"Saved {csv_path} ({time.perf_counter() - t0:.0f}s).")
    if num_shards > 1:
        return None
    report_best(df, dataset_name, config)
    return read_tuned_params(output_root, dataset_name, model_type, checkpoint_id)


# The registry models that automatic segmentation runs on, which are the ones with a decoder.
REGISTRY_MODELS = [model for model in FINETUNED_MODELS if has_registered_decoder(model)]

# The in-domain data of the micro-sam2 cell models, per data type. Out-of-domain data is never tuned on.
IN_DOMAIN_CELLS = {
    "2d": [
        "bitdepth_nucseg", "bmgd", "cell_acdc", "cellbindb", "cellular", "cpm17", "cvz_fluo_cell", "cvz_fluo_dapi",
        "deepbacs_label_free", "dynamicnuclearnet", "enseg", "flywing", "glysac", "histo_miner", "ifnuclei",
        "livecell", "microbeseg", "monuseg", "neurips_cellseg_fluorescence", "neurips_cellseg_label_free",
        "omnipose_bact_fluor", "omnipose_bact_phase", "pannuke", "puma", "tissuenet", "tsakiroglou", "u20s", "vicar",
        "astih", "tumor_spheroid",
    ],
    "3d": [
        "plantseg_root", "pnas_arabidopsis", "cartocell", "phmamm", "wing_disc", "embedseg_organoid",
        "embedseg_mouse_skull", "embedseg_platy_nuclei", "nis3d", "celegans_atlas", "gonuclear", "nucverse3d",
        "platynereis_cells", "densecell", "synapsenet_compartments",
    ],
    "dense": ["cremi", "fafb", "hemibrain", "zebrafinch_j0126", "liconn", "xpress"],
}
# The in-domain data of the EM mitochondria model: the held-out parts of its training data.
IN_DOMAIN_EM_MITO = {
    "2d": ["probtem"],
    "3d": [
        "densecell_mito", "lucchi", "mitoem_human", "mitoem_rat", "mitoemv2_beta", "mitoemv2_jurkat", "mitoemv2_macro",
        "mitoemv2_podo", "mitoemv2_sperm", "mitoemv2_stem", "mitoemv2_mossy", "asem_mito", "mitonet_kidney",
        "snemi_mito",
    ],
    "dense": [],
}
# The in-domain data of the EM nucleus model: the held-out parts of its training data.
IN_DOMAIN_EM_NUCLEUS = {
    "2d": DATASETS_2D_EM_NUCLEUS_MOCK,
    "3d": ["cellmap_nucleus", "fafb_nuclei", "microns_nuclei", "platynereis_nuclei"],
    "dense": [],
}
# The in-domain data of each registry model family, e.g. 'em_mito' for 'hvit_s_em_mito'.
FAMILY_IN_DOMAIN = {"cells": IN_DOMAIN_CELLS, "em_mito": IN_DOMAIN_EM_MITO, "em_nuclei": IN_DOMAIN_EM_NUCLEUS}
# The modes each data type is tuned for. The neurite EM volumes have no APG.
DATA_MODES = {"2d": ("ais", "apg"), "3d": ("ais", "apg"), "dense": ("ais",)}
# The large 2d validation splits are scored on this many samples, spread evenly.
MAX_SAMPLES = {"tissuenet": 500, "dynamicnuclearnet": 500, "pannuke": 500}
# Volumes are tuned on crops of the shape of the test crops.
PIPELINE_CROP_3D = (32, 512, 512)
# 3d APG runs every combo in full, so it screens its prompt axes on a few volumes first. The datasets with the most
# objects per volume are scored on smaller blocks: (volumes of the screen, crop).
APG_3D_SCREEN_VOLUMES = 2
APG_3D_VOLUMES = 5
APG_3D_BLOCKS = {"pnas_arabidopsis": (3, (32, 256, 256)), "phmamm": (3, (32, 256, 256))}
# The second 3d APG stage crosses the best prompt combos of the screen with these selection axes.
APG_3D_PROMPT_TOP = 2
APG_3D_SELECTION_GRID = {
    "score_threshold": [0.4, 0.5, 0.6, 0.7], "max_overlap": [0.15, 0.3, 0.5], "min_size": [25, 100, 400],
}
# The best APG combos that are swept again with each refinement mode.
REFINEMENT_TOP = {"2d": 10, "3d": 3}
REFINEMENTS = ("boxes", "points+boxes")
# The axes whose values are fractions, and the axes of integers, for the steps past a grid edge.
FRACTION_KEYS = ("foreground_threshold", "score_threshold", "max_overlap", "foreground_weight", "beta")
INTEGER_KEYS = ("n_iter", "min_size", "min_candidate_size")
# The queue and the scheduler are polled this often, in seconds.
POLL_SECONDS = 60
# A worker claims a new task only with this many seconds of its time limit left.
MIN_SECONDS_LEFT = 600
# Asking the scheduler per task would flood it: the running jobs are fetched at most every RUNNING_TTL seconds, or
# every RUNNING_MIN_AGE seconds to judge a claim written after the last fetch.
RUNNING_TTL = 300
RUNNING_MIN_AGE = 30
SCHEDULER = {"end_time": None, "running_time": 0.0, "running_jobs": set()}
SCHEDULER_LOCK = threading.Lock()


def data_type(dataset_name):
    """The data type of a dataset: '2d', '3d' (sparse volumes) or 'dense' (neurite EM volumes)."""
    if dataset_name in DATASETS_DENSE:
        return "dense"
    return "3d" if dataset_name in DATASETS_3D else "2d"


def cap_samples(subsample, n_samples):
    """The smaller of two caps on the number of samples, where None means no cap."""
    if n_samples is None:
        return subsample
    return n_samples if subsample is None else min(subsample, n_samples)


def sample_options(context, data, dataset_name, apg_stage=None):
    """The crop and the number of samples that a sweep of one dataset scores.

    'apg_stage' is 'screen' or 'selection' for 3d APG, which scores fewer volumes, see APG_3D_BLOCKS.
    """
    crop = None if data == "2d" else tuple(context["crop_3d"])
    subsample = MAX_SAMPLES.get(dataset_name)
    if apg_stage is not None:
        volumes = APG_3D_SCREEN_VOLUMES if apg_stage == "screen" else APG_3D_VOLUMES
        if dataset_name in APG_3D_BLOCKS:
            screen_volumes, crop = APG_3D_BLOCKS[dataset_name]
            volumes = screen_volumes if apg_stage == "screen" else APG_3D_VOLUMES
        subsample = volumes
    return {"crop": crop, "subsample": cap_samples(subsample, context["n_samples"])}


def make_sweep(model, mode, data, dataset_name, root, options, params_file=None):
    return {
        "model": model, "model_type": backbone(model), "mode": mode, "data": data, "dataset": dataset_name,
        "root": root, "params_file": params_file, **options,
    }


def sweep_config(sweep):
    return tuning_config(sweep["dataset"], sweep["mode"], crop_shape=sweep["crop"], grid_preset="broad")


def backbone(model):
    """The SAM2 backbone of a registry model, e.g. 'hvit_t' for 'hvit_t_cells'."""
    return model[:6]


def model_family(model):
    """The family of a registry model, e.g. 'em_mito' for 'hvit_s_em_mito'."""
    return model[7:]


def registry_weights(model):
    """Download a registry model, and return the paths of its SAM2 weights and of its decoder."""
    registry = model_registry()
    return {"interactive_checkpoint": registry.fetch(model), "checkpoint": registry.fetch(f"{model}_decoder")}


def checkpoint_identity(context, model):
    """The checkpoint id of the weights of a model, computed once per run."""
    with context["lock"]:
        if model not in context["identities"]:
            weights = context["weights"][model]
            context["identities"][model], _ = resolve_checkpoint_identity(
                "apg", backbone(model), checkpoint_path=weights["checkpoint"],
                interactive_checkpoint_path=weights["interactive_checkpoint"],
            )
        return context["identities"][model]


def sweep_csv(context, sweep, root=None):
    output_dir = os.path.join(
        root or sweep["root"], sweep["mode"], sweep["model_type"], checkpoint_identity(context, sweep["model"])
    )
    return os.path.join(output_dir, f"{sweep['dataset']}.csv")


def sweep_is_done(context, sweep):
    csv_path = sweep_csv(context, sweep)
    return os.path.exists(csv_path) and os.path.exists(per_image_path(csv_path))


def prediction_cache(context, sweep):
    """The decoder outputs of AIS are cached once per dataset and reused by every stage."""
    return os.path.join(context["experiment_folder"], "predictions") if sweep["mode"] == "ais" else None


def write_params(context, data, stage, model_type, mode, combos):
    # The modes of a data type share their stage names, so the mode is part of the name: else the searches of AIS
    # and APG, which run at the same time, overwrite each other's combos.
    path = os.path.join(context["experiment_folder"], "params", data, stage, f"{model_type}_{mode}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(combos, f, indent=1)
    return path


def load_sweep(csv_path):
    """The mean score per combo of one merged sweep, or None if it is missing."""
    if not os.path.exists(per_image_path(csv_path)):
        return None
    data = np.load(per_image_path(csv_path))
    params = [json.dumps(p, sort_keys=True) for p in json.loads(str(data["params"]))]
    return pd.Series(data["scores"].mean(axis=1), index=params)


def load_scores(context, roots, mode, model, datasets):
    """The mean score per combo and dataset over the sweeps below 'roots', which score the same samples."""
    columns = {}
    for dataset_name in datasets:
        sweep = make_sweep(model, mode, None, dataset_name, None, {})
        sweeps = [load_sweep(sweep_csv(context, sweep, root)) for root in roots]
        sweeps = [sweep for sweep in sweeps if sweep is not None]
        if sweeps:
            scores = pd.concat(sweeps)
            columns[dataset_name] = scores[~scores.index.duplicated()]
    return pd.DataFrame(columns)


def grid_edges(combos, params):
    """The parameters of a combo that sit at the edge of the values swept."""
    values = swept_values(json.loads(combo) for combo in combos)
    return [
        f"{key}={value}" for key, value in params.items()
        if len(values.get(key, ())) > 1 and value in (min(values[key]), max(values[key]))
    ]


def select_combos(scores, lower_is_better):
    """Rank the combos swept on every dataset by their mean relative score.

    The relative score of a combo on a dataset is its score over the best score of that dataset (the best one over
    it, when lower is better), so every dataset weighs the same. Leaving one dataset out shows how stable the winner
    is: it is selected again on the other datasets and scored on the one left out.

    Returns:
        The table of the combos, best first, and a summary of the winner.
    """
    n_combos = len(scores)
    scores = scores.dropna()
    if scores.empty:
        raise RuntimeError("No combo was swept on every dataset.")
    relative = scores.min(axis=0) / scores if lower_is_better else scores / scores.max(axis=0)
    table = pd.DataFrame({
        "mean_relative_score": relative.mean(axis=1), "min_relative_score": relative.min(axis=1),
        "mean_score": scores.mean(axis=1),
    }).join(relative.add_prefix("relative_")).sort_values("mean_relative_score", ascending=False)
    winner = table.index[0]
    lodo = {}
    # Leaving one dataset out needs two of them at least.
    for name in relative.columns if relative.shape[1] > 1 else []:
        held_out_winner = relative.drop(columns=name).mean(axis=1).idxmax()
        lodo[name] = (held_out_winner == winner, relative.loc[held_out_winner, name])
    summary = {
        "params": json.loads(winner), "n_datasets": relative.shape[1], "n_combos": len(scores),
        "n_combos_dropped": n_combos - len(scores),
        "mean_relative_score": float(table.loc[winner, "mean_relative_score"]),
        "min_relative_score": float(table.loc[winner, "min_relative_score"]),
        "worst_dataset": relative.loc[winner].idxmin(),
        "lodo_same_winner": int(sum(same for same, _ in lodo.values())),
        "lodo_mean_relative_score": float(np.mean([value for _, value in lodo.values()])) if lodo else None,
        "grid_edges": grid_edges(scores.index, json.loads(winner)),
    }
    return table, summary


def swept_values(combos):
    """The numeric values swept per axis."""
    values = {}
    for combo in combos:
        for key, value in combo.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.setdefault(key, set()).add(value)
    return values


def step_past(key, values, value):
    """The next value past the grid edge that 'value' sits on, or None if it sits inside the grid or at a bound.

    The step continues the spacing of the last two values: geometric where they differ by a factor of 1.5 or more,
    such as 50 and 100, and linear otherwise, such as 0.6 and 0.7.
    """
    values = sorted(values)
    if len(values) < 2 or value not in (values[0], values[-1]):
        return None
    inner, edge = (values[-2], values[-1]) if value == values[-1] else (values[1], values[0])
    if min(inner, edge) > 0 and max(inner, edge) / min(inner, edge) >= 1.5:
        new = edge * edge / inner
    else:
        new = 2 * edge - inner
    if key in FRACTION_KEYS:
        new = min(max(new, 0.0), 1.0)
    elif new <= 0:
        new = edge / 2
    new = max(1, int(round(new))) if key in INTEGER_KEYS else round(float(new), 4)
    return None if new in values else new


def edge_combos(winner, swept):
    """The winner with one axis at a time moved one step past the grid edge that it sits on."""
    combos = []
    for key, value in winner.items():
        if key in swept and value in swept[key]:
            new = step_past(key, swept[key], value)
            if new is not None:
                combos.append({**winner, key: new})
    return combos


def run_params_stage(context, runner, model, mode, data, stage, combos, options):
    """Sweep 'combos' on every dataset of a data type and return the root of the stage."""
    params_file = write_params(context, data, stage, model, mode, combos)
    root = os.path.join(context["experiment_folder"], "tuning", data, stage)
    runner.run([
        make_sweep(model, mode, data, name, root, options[name], params_file)
        for name in context["datasets"][data]
    ])
    return root


def broad_stage(context, runner, model, mode, data):
    """Sweep the broad grid on every dataset of a data type.

    Returns:
        The roots of the stage, the combos it swept, and the sample options of the later stages per dataset.
    """
    datasets = context["datasets"][data]
    options = {name: sample_options(context, data, name) for name in datasets}
    root = os.path.join(context["experiment_folder"], "tuning", data, "broad")
    sweeps = [make_sweep(model, mode, data, name, root, options[name]) for name in datasets]
    runner.run(sweeps)
    return [root], sweep_params_list(sweep_config(sweeps[0])), options


def apg_3d_base_stages(context, runner, model):
    """Screen the prompt axes of 3d APG on a few volumes, then sweep the selection axes of its best prompt combos.

    Returns:
        Like `broad_stage`. The later stages score the volumes of the selection stage.
    """
    datasets = context["datasets"]["3d"]
    screen_options = {name: sample_options(context, "3d", name, apg_stage="screen") for name in datasets}
    root = os.path.join(context["experiment_folder"], "tuning", "3d", "screen")
    sweeps = [make_sweep(model, "apg", "3d", name, root, screen_options[name]) for name in datasets]
    runner.run(sweeps)
    table, _ = select_combos(load_scores(context, [root], "apg", model, datasets), lower_is_better=False)
    prompts = [json.loads(params) for params in table.index[:APG_3D_PROMPT_TOP]]
    keys = list(APG_3D_SELECTION_GRID)
    combos = [
        {**prompt, **dict(zip(keys, values))}
        for prompt in prompts for values in itertools.product(*APG_3D_SELECTION_GRID.values())
    ]
    options = {name: sample_options(context, "3d", name, apg_stage="selection") for name in datasets}
    selection_root = run_params_stage(context, runner, model, "apg", "3d", "selection", combos, options)
    return [selection_root], sweep_params_list(sweep_config(sweeps[0])) + combos, options


def search(context, runner, model, mode, data):
    """Run every stage of the search of one model, mode and data type, and return the summary of its winner."""
    datasets = context["datasets"][data]
    lower_is_better = data == "dense"
    if mode == "apg" and data == "3d":
        roots, combos, options = apg_3d_base_stages(context, runner, model)
    else:
        roots, combos, options = broad_stage(context, runner, model, mode, data)

    for round_index in range(1, context["edge_rounds"] + 1):
        _, summary = select_combos(load_scores(context, roots, mode, model, datasets), lower_is_better)
        new = edge_combos(summary["params"], swept_values(combos))
        if not new:
            break
        stage = f"edge{round_index}"
        roots.append(run_params_stage(context, runner, model, mode, data, stage, new, options))
        combos += new

    if mode == "apg":
        table, _ = select_combos(load_scores(context, roots, mode, model, datasets), lower_is_better)
        best = [json.loads(params) for params in table.index[:REFINEMENT_TOP[data]]]
        refined = [{**p, "refinement": r} for p in best for r in REFINEMENTS if p.get("refinement") != r]
        roots.append(run_params_stage(context, runner, model, mode, data, "refinement", refined, options))

    table, summary = select_combos(load_scores(context, roots, mode, model, datasets), lower_is_better)
    output_dir = os.path.join(context["experiment_folder"], "selection", f"{data}_{mode}")
    os.makedirs(output_dir, exist_ok=True)
    table.head(50).reset_index(names="params").to_csv(os.path.join(output_dir, f"{model}.csv"), index=False)
    with open(os.path.join(output_dir, f"{model}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(
        f"{model} {mode} {data}: mean relative score {summary['mean_relative_score']:.4f} over "
        f"{summary['n_datasets']} datasets, same winner without one dataset for {summary['lodo_same_winner']}, "
        f"{summary['params']}"
    )
    return summary


class LocalRunner:
    """Runs the sweeps one after the other in this process."""

    def __init__(self, context):
        self.context = context
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}

    def model(self, mode, model, ndim):
        key = (mode, model, ndim)
        if key not in self.models:
            # Hold the networks of one model only.
            self.models = {k: v for k, v in self.models.items() if k[1] == model}
            weights = self.context["weights"][model]
            self.models[key] = build_model(
                mode, backbone(model), self.device, ndim, checkpoint_path=weights["checkpoint"],
                interactive_checkpoint_path=weights["interactive_checkpoint"],
            )
        return self.models[key]

    def run(self, sweeps):
        for sweep in sweeps:
            if sweep_is_done(self.context, sweep):
                continue
            mode = sweep["mode"]
            model = self.model(mode, sweep["model"], 2 if sweep["data"] == "2d" else 3)
            tune_parameters(
                model, mode, sweep["dataset"], self.context["data_root"], sweep["model_type"],
                os.path.join(sweep["root"], mode), self.device, n_threads=self.context["n_threads"],
                crop_shape=sweep["crop"], checkpoint_id=checkpoint_identity(self.context, sweep["model"]),
                grid_preset="broad",
                subsample=sweep["subsample"], prediction_cache=prediction_cache(self.context, sweep),
                params_file=sweep["params_file"],
            )


class SlurmRunner:
    """Runs the sweeps as tasks of a queue that Slurm job arrays of `worker` take from.

    Large sweeps are split into shards, which are merged once all of them are done. The searches of all backbones,
    modes and data types share the queue, so they run in parallel.
    """

    def __init__(self, context, sbatch_options, n_workers, threads, slots, max_shards):
        self.context = context
        self.queue_dir = Path(context["experiment_folder"], "queue")
        (self.queue_dir / "claims").mkdir(parents=True, exist_ok=True)
        (self.queue_dir / "logs").mkdir(exist_ok=True)
        self.sbatch_options = shlex.split(sbatch_options)
        self.n_workers, self.threads, self.slots, self.max_shards = n_workers, threads, slots, max_shards
        self.job_name = f"parameter_search_{Path(context['experiment_folder']).name}"
        self.lock = threading.Lock()

    def tasks(self):
        path = self.queue_dir / "tasks.json"
        return json.loads(path.read_text()) if path.exists() else []

    def write_tasks(self, tasks):
        tmp = self.queue_dir / "tasks.json.tmp"
        tmp.write_text(json.dumps(tasks, indent=1))
        os.replace(tmp, self.queue_dir / "tasks.json")

    def num_shards(self, sweep):
        config = sweep_config(sweep)
        params_list = sweep_params_list(config, sweep["params_file"])
        return max(1, min(self.max_shards, n_shard_groups(params_list, config["mode"], config["ndim"])))

    def command(self, sweep, shard_index=0, num_shards=1, cache_only=False):
        weights = self.context["weights"][sweep["model"]]
        command = [
            sys.executable, str(Path(__file__).resolve()), "sweep", "-d", sweep["dataset"],
            "-i", self.context["data_root"], "-e", self.context["experiment_folder"], "-m", sweep["model_type"],
            "--mode", sweep["mode"], "--tuning_root", sweep["root"], "--grid_preset", "broad",
            "-c", weights["checkpoint"], "--interactive_checkpoint", weights["interactive_checkpoint"],
        ]
        if sweep["params_file"]:
            command += ["--params_file", sweep["params_file"]]
        if sweep["crop"]:
            command += ["--crop_3d", *map(str, sweep["crop"])]
        if sweep["subsample"]:
            command += ["--subsample", str(sweep["subsample"])]
        if prediction_cache(self.context, sweep):
            command += ["--prediction_cache", prediction_cache(self.context, sweep)]
        if cache_only:
            command += ["--cache_only"]
        if num_shards > 1:
            command += ["--num_shards", str(num_shards), "--shard_index", str(shard_index)]
        return command

    def add(self, tasks, index_of, sweep, command, requires=None):
        """Add a task, or reuse the task with the same command, and return its index."""
        key = json.dumps(command)
        if key not in index_of:
            tasks.append({"model_type": sweep["model_type"], "command": command, "requires": requires})
            index_of[key] = len(tasks) - 1
        return index_of[key]

    def add_sweeps(self, sweeps):
        """Queue the tasks of the sweeps that are not done yet, and return their indices."""
        with self.lock:
            tasks = self.tasks()
            index_of = {json.dumps(task["command"]): i for i, task in enumerate(tasks)}
            indices = []
            for sweep in sweeps:
                if sweep_is_done(self.context, sweep):
                    continue
                requires = None
                if sweep["mode"] == "ais":
                    # The decoder outputs are cached first, once per dataset, so that all shards and stages share them.
                    root = os.path.join(self.context["experiment_folder"], "tuning", sweep["data"], "cache")
                    cache_sweep = {**sweep, "root": root, "params_file": None}
                    cache = self.add(tasks, index_of, sweep, self.command(cache_sweep, cache_only=True))
                    indices.append(cache)
                    requires = str(self.queue_dir / "claims" / str(cache) / "done")
                num_shards = self.num_shards(sweep)
                for shard_index in range(num_shards):
                    command = self.command(sweep, shard_index, num_shards)
                    indices.append(self.add(tasks, index_of, sweep, command, requires))
            self.write_tasks(tasks)
            return indices

    def submit_workers(self):
        """Submit workers until 'n_workers' of them run or wait, as long as tasks are open."""
        with self.lock:
            tasks = self.tasks()
            claims = self.queue_dir / "claims"
            n_open = sum(1 for i in range(len(tasks)) if not (claims / str(i) / "done").exists())
            jobs = squeue("-u", os.environ.get("USER", ""), "-r", "-n", self.job_name, "-o", "%i").split()
            n_new = min(self.n_workers, n_open) - len(jobs)
            if n_new <= 0:
                return
            script = self.queue_dir / "worker.sh"
            script.write_text(
                f"#!/bin/bash\n{sys.executable} {Path(__file__).resolve()} worker {self.queue_dir} "
                f"--threads {self.threads} --slots {self.slots}\n"
            )
            logs = self.queue_dir / "logs"
            subprocess.run([
                "sbatch", *self.sbatch_options, f"--array=0-{n_new - 1}", f"--job-name={self.job_name}",
                "-c", str(self.threads), "-o", f"{logs}/%A_%a.out", "-e", f"{logs}/%A_%a.err", str(script),
            ], check=True)

    def run(self, sweeps):
        indices = self.add_sweeps(sweeps)
        claims = self.queue_dir / "claims"
        while True:
            failed = [i for i in indices if (claims / str(i) / "failed").exists()]
            if failed:
                raise RuntimeError(f"Tasks {failed} of '{self.queue_dir}' failed, see its logs.")
            if all((claims / str(i) / "done").exists() for i in indices):
                break
            self.submit_workers()
            time.sleep(POLL_SECONDS)
        for sweep in sweeps:
            num_shards = self.num_shards(sweep)
            if num_shards > 1 and not sweep_is_done(self.context, sweep):
                merge_shards(
                    os.path.join(sweep["root"], sweep["mode"]), sweep["dataset"], sweep["mode"], sweep["model_type"],
                    num_shards, checkpoint_id=checkpoint_identity(self.context, sweep["model"]), grid_preset="broad",
                )


def squeue(*args):
    return subprocess.run(["squeue", "-h", *args], capture_output=True, text=True).stdout


def my_job_id():
    """This job as squeue names it: '<array>_<task>' for an array task, whose plain id names the whole array."""
    if "SLURM_ARRAY_JOB_ID" in os.environ:
        return f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ['SLURM_ARRAY_TASK_ID']}"
    return os.environ.get("SLURM_JOB_ID")


def seconds_left():
    """The seconds until the Slurm time limit of this job, or a day outside of Slurm."""
    job_id = my_job_id()
    if job_id is None:
        return 86400
    with SCHEDULER_LOCK:
        if SCHEDULER["end_time"] is None:
            out = squeue("-j", job_id, "-o", "%L").strip().split("\n")[0].strip()
            if not out or out[0] not in "0123456789":
                return 0  # The scheduler did not answer, so claim nothing new until it does.
            days, _, clock = out.rpartition("-")
            parts = [int(p) for p in clock.split(":")]
            while len(parts) < 3:
                parts.insert(0, 0)
            left = (int(days) if days else 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]
            SCHEDULER["end_time"] = time.time() + left
        return SCHEDULER["end_time"] - time.time()


def job_is_running(job_id, since=0.0):
    """Whether a job of this user runs. The list only counts if it is newer than 'since', the time of the claim."""
    with SCHEDULER_LOCK:
        age = time.time() - SCHEDULER["running_time"]
        if age > RUNNING_TTL or (SCHEDULER["running_time"] <= since and age > RUNNING_MIN_AGE):
            out = squeue("-u", os.environ.get("USER", ""), "-t", "RUNNING,COMPLETING", "-o", "%i")
            if not out.strip():
                return True  # No answer: keep the claim rather than run the task twice.
            SCHEDULER["running_jobs"] = set(out.split())
            SCHEDULER["running_time"] = time.time()
        if SCHEDULER["running_time"] <= since:
            return True
        return job_id in SCHEDULER["running_jobs"]


def process_is_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def claim_is_stale(owner_line, written):
    """Whether the worker that wrote a claim ('<job> <host> <pid>') at time 'written' is gone."""
    job, host, pid = owner_line.split()
    if job != (my_job_id() or "local"):
        return not job_is_running(job, since=written)
    return host != os.uname().nodename or not process_is_alive(int(pid))


def try_claim(claim_dir):
    """Claim a task with an atomic mkdir. The claim of a worker that no longer runs is taken over."""
    owner_path = os.path.join(claim_dir, "owner")
    try:
        os.mkdir(claim_dir)
    except FileExistsError:
        if os.path.exists(os.path.join(claim_dir, "done")):
            return False
        try:
            owner_line = Path(owner_path).read_text()
            written = os.path.getmtime(owner_path)
        except FileNotFoundError:
            return False  # Another worker is writing its claim right now.
        if not owner_line.strip() or not claim_is_stale(owner_line, written):
            return False
        # The rename is atomic, so only one worker takes the stale claim over.
        try:
            os.rename(claim_dir, f"{claim_dir}.stale.{os.getpid()}.{time.time()}")
            os.mkdir(claim_dir)
        except OSError:
            return False
    Path(owner_path).write_text(f"{my_job_id() or 'local'} {os.uname().nodename} {os.getpid()}\n")
    return True


def worker_loop(queue_dir, threads, models=None):
    """Claim and run the tasks of a queue until none is open or the time limit is too close.

    A task whose cache is not written yet, or that another running job holds, may still come free, so the worker
    waits for it. A failed task is left for inspection, not retried.
    """
    claims = Path(queue_dir, "claims")
    while True:
        tasks = json.loads(Path(queue_dir, "tasks.json").read_text())
        n_open, claimed = 0, False
        for index, task in enumerate(tasks):
            claim = claims / str(index)
            if (claim / "done").exists() or (claim / "failed").exists():
                continue
            if models is not None and task["model_type"] not in models:
                continue
            n_open += 1
            if task["requires"] is not None and not os.path.exists(task["requires"]):
                continue
            if seconds_left() < MIN_SECONDS_LEFT or not try_claim(str(claim)):
                continue
            claimed = True
            command = task["command"] + ["--n_threads", str(threads)]
            print(f"[{datetime.now():%H:%M:%S}] task {index}: {' '.join(command)}", flush=True)
            t0 = time.time()
            result = subprocess.run(command)
            if result.returncode == 0:
                (claim / "done").write_text(f"{time.time() - t0:.0f}\n")
            else:
                print(f"Task {index} failed with exit code {result.returncode}.", flush=True)
                (claim / "failed").write_text(f"{result.returncode}\n")
        if n_open == 0 or (not claimed and seconds_left() < MIN_SECONDS_LEFT):
            break
        if not claimed:
            time.sleep(POLL_SECONDS)


def run_worker(queue_dir, threads, slots, models=None):
    """Run 'slots' worker loops side by side, each with an equal share of the threads.

    The scoring holds the GIL for much of its time, so several tasks of a few threads use a node better than one.
    """
    if slots <= 1:
        return worker_loop(queue_dir, threads, models)
    with futures.ThreadPoolExecutor(slots) as pool:
        list(pool.map(lambda _: worker_loop(queue_dir, max(1, threads // slots), models), range(slots)))


def search_datasets(dataset_names, data_types, family="cells"):
    """The datasets of every data type to search on: the in-domain ones of a model family, or 'dataset_names' sorted
    by type.
    """
    in_domain = FAMILY_IN_DOMAIN[family]
    datasets = {}
    for data in data_types:
        names = in_domain[data] if dataset_names is None else [n for n in dataset_names if data_type(n) == data]
        datasets[data] = [name for name in names if has_val_split(name)]
    return datasets


def run_command(args):
    # The models of one search share its datasets, so they have to be of one family.
    families = sorted({model_family(model) for model in args.model_type})
    if len(families) > 1:
        raise ValueError(f"The models are of several families {families}. Search one family at a time.")
    datasets = search_datasets(args.dataset_name, args.data, families[0])
    for names in datasets.values():
        for name in names:
            check_data_download(name, args.input_path, split="val")
    context = {
        "experiment_folder": os.path.abspath(args.experiment_folder), "data_root": args.input_path,
        "datasets": datasets, "n_samples": args.n_samples, "edge_rounds": args.edge_rounds,
        "crop_3d": args.crop_3d, "n_threads": args.n_threads, "identities": {}, "lock": threading.Lock(),
        "weights": {model: registry_weights(model) for model in args.model_type},
    }
    searches = [
        (model, mode, data) for model in args.model_type for data in args.data for mode in args.mode
        if mode in DATA_MODES[data] and datasets[data]
    ]
    if args.slurm:
        runner = SlurmRunner(context, args.sbatch, args.workers, args.threads, args.slots, args.max_shards)
        with futures.ThreadPoolExecutor(len(searches)) as pool:
            summaries = list(pool.map(lambda key: search(context, runner, *key), searches))
    else:
        print("Device:", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
        runner = LocalRunner(context)
        summaries = [search(context, runner, *key) for key in searches]

    best = {}
    for (model, mode, data), summary in zip(searches, summaries):
        best.setdefault(model, {}).setdefault(mode, {})[data] = summary["params"]
    path = os.path.join(context["experiment_folder"], "best_parameters.json")
    with open(path, "w") as f:
        json.dump(best, f, indent=1)
    print(f"Wrote the best parameters to '{path}'.")


def sweep_command(args):
    if args.experiment_folder is None:
        raise ValueError("-e/--experiment_folder is required.")
    datasets = args.dataset_name or sorted(VAL_SPLITS)
    grid_override = json.loads(args.grid_override) if args.grid_override else None

    if args.merge_shards:
        for mode in args.mode:
            checkpoint_id, _ = resolve_checkpoint_identity(
                mode, args.model_type, args.joint_checkpoint, args.checkpoint,
                interactive_checkpoint_path=args.interactive_checkpoint,
            )
            tuning_root = os.path.join(args.tuning_root or os.path.join(args.experiment_folder, "tuning"), mode)
            for dataset_name in datasets:
                merge_shards(
                    tuning_root, dataset_name, mode, args.model_type, args.num_shards,
                    checkpoint_id=checkpoint_id, criterion=args.criterion, grid_preset=args.grid_preset,
                )
        return

    for dataset_name in datasets:
        check_data_download(dataset_name, args.input_path)
        # The sweep only reads the held-out 'val' split, which can live in another download than 'test'.
        check_data_download(dataset_name, args.input_path, split="val")

    print("Device:", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    crop_shape = tuple(args.crop_3d) if args.crop_3d else None

    for mode in args.mode:
        checkpoint_id, joint_checksum = resolve_checkpoint_identity(
            mode, args.model_type, args.joint_checkpoint, args.checkpoint,
            interactive_checkpoint_path=args.interactive_checkpoint,
        )
        tuning_root = os.path.join(args.tuning_root or os.path.join(args.experiment_folder, "tuning"), mode)
        # AIS's decoder does not depend on ndim, so one build covers every dataset. APG's video input type does,
        # so it is rebuilt once per dimensionality.
        models = {}
        for dataset_name in datasets:
            ndim = 3 if dataset_name in DATASETS_3D else 2
            model_key = ndim if mode == "apg" else mode
            if mode == "ais" and args.prediction_cache and not args.cache_only:
                models[model_key] = None
            elif model_key not in models:
                models[model_key] = build_model(
                    mode, args.model_type, device, ndim,
                    joint_checkpoint=args.joint_checkpoint, checkpoint_path=args.checkpoint,
                    joint_checksum=joint_checksum, interactive_checkpoint_path=args.interactive_checkpoint,
                )
            tune_parameters(
                models[model_key], mode, dataset_name, args.input_path, args.model_type, tuning_root, device,
                n_threads=args.n_threads, n_tuning_samples=args.n_tuning_samples, tuning_stride=args.tuning_stride,
                crop_shape=crop_shape, criterion=args.criterion, checkpoint_id=checkpoint_id,
                shard_index=args.shard_index, num_shards=args.num_shards, grid_override=grid_override,
                grid_preset=args.grid_preset, subsample=args.subsample,
                prediction_cache=args.prediction_cache, cache_only=args.cache_only, params_file=args.params_file,
            )


def worker_command(args):
    run_worker(args.queue_dir, args.threads, args.slots, models=args.models)


# 'prepare' writes the mock 2d nucleus slices, and as tuning crops the tiles with the most objects.
TUNING_CROPS_PATH = Path(__file__).resolve().parent / "tuning_crops.json"
TUNING_CROP_3D, MAX_TUNING_CROPS_3D = (32, 512, 512), 5
TUNING_CROP_2D, MAX_TUNING_CROPS_2D = (512, 512), 10
MOCK_SLICES, MOCK_MIN_Z_DISTANCE = 10, 4
TUNING_CROP_DATASETS = {
    3: IN_DOMAIN_EM_MITO["3d"] + IN_DOMAIN_EM_NUCLEUS["3d"],
    2: [
        "probtem", "mitolab_tem", "cmito_dauer_manual", "tnbc_mito", "cardiomyocyte_tem_mito", "u2os_fibsem_nucleus",
        *DATASETS_2D_EM_NUCLEUS_MOCK,
    ],
}


def tile_starts(region, crop_shape):
    """The starts of the tiles of a region, a tuple of slices."""
    axes = [
        [sl.start] if sl.stop - sl.start <= size else list(range(sl.start, sl.stop - size + 1, size))
        for sl, size in zip(region, crop_shape)
    ]
    return list(itertools.product(*axes))


def tile_regions(dataset_name, data_root, ndim):
    """The (sample name, loader arguments, tuning region) of every tuning sample of a dataset, in loader order."""
    raw_paths, label_paths, raw_key, label_key = get_data_paths(dataset_name, data_root, split="val")
    regions = []
    for raw_path, label_path in sorted_path_pairs(raw_paths, label_paths):
        paths = (raw_path, label_path, raw_key, label_key, dataset_name)
        if ndim == 3:
            source = load_image(label_path) if label_key is None else open_file(label_path, mode="r")[label_key]
            z_range = val_z_range(dataset_name, raw_path, "val") or VAL_Z_RANGE.get(dataset_name)
            region = _read_window(source.shape, em_roi(dataset_name, label_path, "val"), z_range, source.shape)
        else:
            z_range = None
            image, _ = load_evaluation_sample_2d(*paths)
            region = tuple(slice(0, size) for size in image.shape[:2])
        regions.append((os.path.basename(raw_path), paths, z_range, region))
    names = [name for name, *_ in regions]
    if len(set(names)) != len(names):
        raise RuntimeError(f"The tuning samples of '{dataset_name}' share file names, which name their crops.")
    return regions


def count_tile_objects(paths, ndim, z_range, start):
    if ndim == 3:
        _, labels, _ = load_evaluation_sample_3d(
            *paths, crop_shape=TUNING_CROP_3D, z_range=z_range, split="val", crop_start=start
        )
    else:
        _, labels = load_evaluation_sample_2d(*paths, crop_start=start, crop_shape=TUNING_CROP_2D)
    return len(np.setdiff1d(np.unique(labels), [0]))


def place_tuning_crops(dataset_name, data_root, ndim):
    """The tuning crops of one dataset, as an entry of tuning_crops.json."""
    crop_shape, max_crops = {3: (TUNING_CROP_3D, MAX_TUNING_CROPS_3D), 2: (TUNING_CROP_2D, MAX_TUNING_CROPS_2D)}[ndim]
    regions = tile_regions(dataset_name, data_root, ndim)
    per_sample = max_crops if ndim == 3 else math.ceil(max_crops / len(regions))
    tiles = []
    for name, paths, z_range, region in regions:
        sample_tiles = [
            (n_objects, name, start) for start in tile_starts(region, crop_shape)
            if (n_objects := count_tile_objects(paths, ndim, z_range, start)) > 0
        ]
        tiles += sorted(sample_tiles, key=lambda tile: (-tile[0], tile[2]))[:per_sample]
    kept = sorted(tiles, key=lambda tile: (-tile[0], tile[1], tile[2]))[:max_crops]
    starts = {name: [] for name, *_ in regions}
    for _, name, start in sorted(kept, key=lambda tile: (tile[1], tile[2])):
        starts[name].append(list(start))
    counts = [n for n, _, _ in kept]
    note = f"{len(kept)} crops of {len(regions)} samples, holding {min(counts)}-{max(counts)} objects."
    print(f"{dataset_name}: {note}")
    return {"crop_shape": list(crop_shape), "note": note, "starts": starts}


def write_tuning_crops(placed):
    """Add the placed crops to tuning_crops.json and to the crops this process loads with."""
    with open(TUNING_CROPS_PATH) as f:
        crops = json.load(f)
    crops["datasets"].update(placed)
    tmp_path = f"{TUNING_CROPS_PATH}.tmp{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(crops, f, indent=1)
        f.write("\n")
    os.replace(tmp_path, TUNING_CROPS_PATH)
    TUNING_CROPS.update(_read_crops(TUNING_CROPS_PATH.name))


def mock_slice_labels(labels, min_size):
    """The connected components of a 2d label slice, without the ones below 'min_size' pixels."""
    components = label_components(labels > 0)
    sizes = np.bincount(components.ravel())
    small = np.nonzero(sizes < min_size)[0]
    components[np.isin(components, small[small > 0])] = 0
    return label_components(components > 0).astype("uint32")


def write_mock_slices(dataset_name, data_root, split):
    """Write the mock 2d slices of one split of an in-domain 3d nucleus set, see 'prepare'."""
    source = dataset_name[:-len("_2d")]
    crop = TUNING_CROP_3D if split == "val" else None
    z_range = VAL_Z_RANGE.get(source) if split == "val" else None
    samples = load_data(source, data_root, 3, split=split, crop_shape=crop, z_range=z_range)
    volumes = [(raw, labels) for raw, labels, _ in samples]
    min_size = GT_MIN_SIZE_2D[dataset_name]
    candidates = sorted(
        (
            (n_objects, volume_index, z)
            for volume_index, (_, labels) in enumerate(volumes) for z in range(labels.shape[0])
            if (n_objects := int(mock_slice_labels(labels[z], min_size).max())) > 0
        ),
        key=lambda candidate: (-candidate[0], candidate[1], candidate[2]),
    )
    picked = []
    for _, volume_index, z in candidates:
        if all(v != volume_index or abs(z - pz) >= MOCK_MIN_Z_DISTANCE for v, pz in picked):
            picked.append((volume_index, z))
        if len(picked) == MOCK_SLICES:
            break
    out_dir = os.path.join(data_root, EM_NUCLEUS_MOCK_2D_ROOT, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    for index, (volume_index, z) in enumerate(sorted(picked)):
        raw, labels = volumes[volume_index]
        path = os.path.join(out_dir, f"{split}_{index:02d}.h5")
        with h5py.File(f"{path}.tmp", "w") as f:
            f.create_dataset("raw", data=np.clip(raw[z], 0, 255).astype("uint8"), compression="gzip")
            f.create_dataset("labels", data=mock_slice_labels(labels[z], min_size), compression="gzip")
            f.attrs["source"] = f"{source} {split} volume {volume_index} z {z}"
        os.replace(f"{path}.tmp", path)
    print(f"{dataset_name} {split}: {len(picked)} slices from {len(volumes)} crops.")


def prepare_command(args):
    """The 3d crops come first, since the mock 2d slices are cut from them."""
    if 3 in args.ndim:
        placed = {name: place_tuning_crops(name, args.input_path, 3) for name in TUNING_CROP_DATASETS[3]}
        if not args.dry:
            write_tuning_crops(placed)
    if args.mock_nucleus_2d and not args.dry:
        for dataset_name in DATASETS_2D_EM_NUCLEUS_MOCK:
            for split in ("val", "test"):
                write_mock_slices(dataset_name, args.input_path, split)
    if 2 in args.ndim:
        placed = {name: place_tuning_crops(name, args.input_path, 2) for name in TUNING_CROP_DATASETS[2]}
        if not args.dry:
            write_tuning_crops(placed)


def add_weight_arguments(parser):
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="Weights instead of the joint export.")
    parser.add_argument("--joint_checkpoint", type=str, default="best", help="Joint checkpoint name, without '.pt'.")
    parser.add_argument(
        "--interactive_checkpoint", type=str, default=None,
        help="Standalone interactive weights for APG, bypassing the joint checkpoint. Requires -c for the decoder.",
    )


def add_run_arguments(parser):
    parser.add_argument("-e", "--experiment_folder", type=str, required=True, help="Where the search is written.")
    parser.add_argument("-i", "--input_path", type=str, default=DATA_ROOT, help="The root the data lives in.")
    parser.add_argument(
        "-m", "--model_type", type=str, nargs="+", default=[f"{model}_cells" for model in MODEL_TYPES],
        choices=REGISTRY_MODELS, help="The registry models to search the parameters of.",
    )
    parser.add_argument("--mode", type=str, nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--data", type=str, nargs="+", default=list(DATA_MODES), choices=list(DATA_MODES))
    parser.add_argument(
        "-d", "--dataset_name", type=str, nargs="+", default=None, choices=sorted(VAL_SPLITS), metavar="DATASET",
        help="Datasets to search on instead of the in-domain ones, sorted into the data types by their kind.",
    )
    parser.add_argument("--n_samples", type=int, default=None, help="Score at most this many samples per dataset.")
    parser.add_argument("--edge_rounds", type=int, default=2, help="Rounds that sweep past the grid edges.")
    parser.add_argument("--crop_3d", type=int, nargs=3, default=PIPELINE_CROP_3D, help="The crop of the volumes.")
    parser.add_argument("--n_threads", type=int, default=POSTPROC_THREADS, help="Threads for the postprocessing.")
    parser.add_argument("--slurm", action="store_true", help="Run the sweeps as Slurm job arrays of workers.")
    parser.add_argument("--sbatch", type=str, default="", help="With --slurm: sbatch options, e.g. '-p gpu -G 1'.")
    parser.add_argument("--workers", type=int, default=8, help="With --slurm: the workers to keep submitted.")
    parser.add_argument("--threads", type=int, default=8, help="With --slurm: the CPU threads of a worker.")
    parser.add_argument("--slots", type=int, default=2, help="With --slurm: the tasks a worker runs at once.")
    parser.add_argument("--max_shards", type=int, default=8, help="With --slurm: split a sweep into this many tasks.")


def add_sweep_arguments(parser):
    parser.add_argument(
        "-d", "--dataset_name", type=str, nargs="+", default=None, choices=sorted(VAL_SPLITS), metavar="DATASET",
        help="Datasets to tune. Defaults to every dataset with a validation split (see common.VAL_SPLITS).",
    )
    parser.add_argument("-i", "--input_path", type=str, default=DATA_ROOT, help="The root the data lives in.")
    parser.add_argument("-e", "--experiment_folder", type=str, default=None, help="The folder of the experiment.")
    parser.add_argument("-m", "--model_type", type=str, default="hvit_t", choices=MODEL_TYPES)
    parser.add_argument(
        "--mode", type=str, nargs="+", default=list(MODES), choices=MODES, help="The segmentation modes to tune."
    )
    add_weight_arguments(parser)
    parser.add_argument("--tuning_root", type=str, default=None, help="Where the sweeps are written and read from.")
    parser.add_argument("--n_tuning_samples", type=int, default=None, help="Cap each sweep to this many samples.")
    parser.add_argument(
        "--tuning_stride", type=int, default=1,
        help="Only score every this-th validation sample, so that a cap spreads over a sort order by cell type.",
    )
    parser.add_argument("--criterion", type=str, default=None, choices=sorted(CRITERION_ASCENDING))
    parser.add_argument(
        "--grid_override", type=str, default=None,
        help="JSON object replacing the swept values of the named grid axes; the other axes keep their values.",
    )
    parser.add_argument(
        "--grid_preset", type=str, default="default", choices=GRID_PRESETS,
        help="The grid to sweep: 'default', or 'broad' (BROAD_GRIDS_2D and BROAD_GRIDS_3D).",
    )
    parser.add_argument(
        "--params_file", type=str, default=None,
        help="A JSON list of parameter dicts to sweep instead of the grid. The grid preset still sets the config.",
    )
    parser.add_argument("--subsample", type=int, default=None, help="Score this many samples, spread evenly.")
    parser.add_argument(
        "--prediction_cache", type=str, default=None,
        help="Root of the cached decoder outputs of AIS. With all of them cached, AIS loads no model.",
    )
    parser.add_argument("--cache_only", action="store_true", help="Only fill --prediction_cache, do not sweep.")
    parser.add_argument("--n_threads", type=int, default=POSTPROC_THREADS, help="Threads for the postprocessing.")
    parser.add_argument("--crop_3d", type=int, nargs=3, default=None, help="Override the 3d crop (Z Y X).")
    parser.add_argument("--num_shards", type=int, default=1, help="Split the grid of a dataset across this many jobs.")
    parser.add_argument("--shard_index", type=int, default=0, help="This job's slice of the grid, in [0, num_shards).")
    parser.add_argument("--merge_shards", action="store_true", help="Merge the shards of a sharded sweep.")


def add_worker_arguments(parser):
    parser.add_argument("queue_dir", type=str, help="The queue of the search, '<experiment_folder>/queue'.")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--slots", type=int, default=1, help="The tasks the worker runs at once.")
    parser.add_argument("--models", nargs="+", default=None, help="Run only the tasks of these backbones.")


def add_prepare_arguments(parser):
    parser.add_argument("-i", "--input_path", type=str, default=DATA_ROOT, help="The root the data lives in.")
    parser.add_argument("--ndim", type=int, nargs="+", default=[3, 2], choices=(2, 3), help="The crops to place.")
    parser.add_argument("--mock_nucleus_2d", action="store_true", help="Write the mock 2d nucleus slices too.")
    parser.add_argument("--dry", action="store_true", help="Print the crops without writing anything.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    add_run_arguments(commands.add_parser("run", help="Search the best parameters over a set of datasets."))
    add_sweep_arguments(commands.add_parser("sweep", help="Sweep one grid or list of combos per dataset."))
    add_worker_arguments(commands.add_parser("worker", help="The Slurm worker of 'run --slurm'."))
    add_prepare_arguments(commands.add_parser("prepare", help="Write the mock 2d slices and the tuning crops."))
    args = parser.parse_args()
    commands = {"run": run_command, "sweep": sweep_command, "worker": worker_command, "prepare": prepare_command}
    commands[args.command](args)


if __name__ == "__main__":
    main()
