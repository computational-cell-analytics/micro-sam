"""Parameter search for micro-sam2 automatic segmentation, AISv2 and APGv2.

Sweeps the postprocessing / prompt-generation grid on the validation split of every dataset that
has one held out (see `common.VAL_SPLITS` and `common.VAL_Z_RANGE`), once per requested mode, and
writes each ranked sweep to '<tuning_root>/<mode>/<model_type>/<checkpoint_id>/<dataset_name>.csv'.
That is the layout `common.read_tuned_params` reads, so `evaluate_automatic_segmentation.py` only
has to load the result instead of repeating the sweep for every evaluation job.

EM datasets are tuned in dense (multicut) mode and ranked by the CREMI score, all others in sparse
(flow) mode and ranked by mSA.

Usage examples:
    python parameter_search.py -m hvit_b -e <exp> --mode ais
    python parameter_search.py -m hvit_b -e <exp> --mode apg -d livecell gonuclear
"""

import os
import time
import argparse
import warnings
import itertools
import threading
from concurrent import futures

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from elf.evaluation import mean_segmentation_accuracy

from bioimage_cpp.segmentation import label as connected_components, watershed

from micro_sam.v2.postprocessing import watershed_heightmap, _compute_flow_density

from common import (
    DATASETS_3D, DATASETS_3D_EM, DATASET_SPACING, VAL_SPLITS, VAL_Z_RANGE,
    GT_MIN_SIZE_2D, CROP_SHAPE_3D, DATA_ROOT, MODEL_TYPES, MODES, VOLUME_SPEED_OPTIONS,
    build_model, check_data_download, drop_severed_objects, has_val_split, load_data,
    n_samples, predict_unisam2, read_tuned_params, resolve_checkpoint_identity,
)

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
}

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


def tuning_config(dataset_name, mode, criterion=None, crop_shape=None):
    """Build the sweep configuration for one dataset and mode.

    EM datasets are tuned in dense (multicut) mode and ranked by the CREMI score, all others in
    sparse (flow) mode and ranked by mSA. With mode='apg' the postprocessing is replaced by the
    prompt generation, which is swept over its own grid; the ranking metric follows the data either
    way, so an APG result is directly comparable with the AIS result of the same dataset.

    Args:
        dataset_name: The dataset to tune on.
        mode: The segmentation mode, 'ais' or 'apg'.
        criterion: The metric the grid is ranked by. Defaults to the data-specific choice.
        crop_shape: The 3d center crop. Defaults to CROP_SHAPE_3D.

    Returns:
        The configuration dict.
    """
    is_em = dataset_name in DATASETS_3D_EM
    is_3d = dataset_name in DATASETS_3D

    if mode == "apg":
        postproc_mode, grid = "apg", (APG_GRID_3D if is_3d else APG_GRID_2D)
    else:
        postproc_mode, grid = ("dense", EM_GRID) if is_em else ("sparse", LM_GRID)

    return {
        "mode": postproc_mode,
        # Which metrics compute_metrics reports. Follows the data, not the mode, so that an APG run
        # on EM is still ranked by the CREMI score.
        "metric_mode": "dense" if is_em else "sparse",
        "ndim": 3 if is_3d else 2,
        "grid": grid,
        "spacing": DATASET_SPACING.get(dataset_name, None),
        "crop": (crop_shape or CROP_SHAPE_3D) if is_3d else None,
        "criterion": criterion or ("cremi" if is_em else "msa"),
        # Only set for the volumes with no split of their own.
        "z_range": VAL_Z_RANGE.get(dataset_name),
    }


def compute_metrics(seg, labels, metric_mode, border_min_size=0):
    """Compute the evaluation metrics for one segmentation.

    Always reports mSA. For dense (EM neuron) mode it additionally reports the CREMI score and its
    VI-split / VI-merge / adapted-Rand components, since neuron segmentation is ranked by CREMI
    (lower is better) rather than mSA.
    """
    if seg.ndim == 2:
        # Symmetric with the ground truth, which `load_evaluation_sample_2d` filtered the same way.
        seg = drop_severed_objects(seg, border_min_size)
    metrics = {"msa": float(mean_segmentation_accuracy(seg, labels))}
    if metric_mode == "dense":
        from elf.evaluation import cremi_score
        vi_split, vi_merge, adapted_rand, cremi = cremi_score(seg, labels)
        metrics["cremi"] = float(cremi)
        metrics["vi_split"] = float(vi_split)
        metrics["vi_merge"] = float(vi_merge)
        metrics["adapted_rand"] = float(adapted_rand)
    return metrics


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
    prediction, labels, params_list, backend, n_threads=POSTPROC_THREADS, spacing=None, border_min_size=0,
):
    """Score all sparse combos on one image, caching the flow density across shared combos.

    Faithfully reproduces flow_instance_segmentation: the height map and each convergence-density map
    (the expensive flow integration) are computed once and reused; only the cheap seed / watershed /
    min-size steps vary per (density_threshold, min_size). Returns a per-combo list of metric dicts
    (None where a combo failed), aligned with params_list.
    """
    foreground = prediction[0]
    directed = prediction[1:]
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
                backend=backend, n_threads=n_threads,
            )
        fw = params["foreground_weight"]
        if fw not in hmap_cache:
            hmap_cache[fw] = watershed_heightmap(foreground, directed, fw)

    # The base watershed does not depend on min_size, so all min_size values of a combo reuse it.
    base_cache, base_lock = {}, threading.Lock()

    def base_segmentation(key, fg_mask, density, density_threshold, hmap):
        with base_lock:
            cached = base_cache.get(key)
        if cached is None:
            seeds = connected_components(density > density_threshold)
            cached = watershed(hmap, markers=seeds, mask=fg_mask)
            with base_lock:
                base_cache[key] = cached
        return cached

    def score(params):
        ft, sigma, n_iter, dt = (params[k] for k in FLOW_DENSITY_KEYS)
        fw, density_threshold = params["foreground_weight"], params["density_threshold"]
        fg_mask = fg_mask_cache[ft]
        hmap = hmap_cache[fw]
        try:
            key = (ft, sigma, n_iter, dt, density_threshold, fw)
            seg = base_segmentation(key, fg_mask, density_cache[(ft, sigma, n_iter, dt)], density_threshold, hmap)
            min_size = params["min_size"]
            if min_size > 0:
                seg = seg.copy()
                ids, sizes = np.unique(seg, return_counts=True)
                discard = ids[(sizes < min_size) & (ids > 0)]
                seg[np.isin(seg, discard)] = 0
                seg = watershed(hmap, markers=seg, mask=fg_mask)
            return compute_metrics(seg.astype("uint32"), labels, "sparse", border_min_size)
        except Exception as e:
            warnings.warn(f"Sparse postprocessing failed for {params}: {e}")
            return None

    # Scoring a combination is independent of the others, and dominates the runtime of the sweep.
    with futures.ThreadPoolExecutor(n_threads) as tp:
        return list(tp.map(score, params_list))


def dense_boundary_and_distances(prediction):
    """Return the (boundary_map, in-plane distances) inputs to run_multicut for a dense prediction."""
    fg = prediction[0]
    boundary_map = fg.max() - fg
    denom = boundary_map.max()
    if denom > 0:
        boundary_map = boundary_map / denom
    distances = np.stack([prediction[2], prediction[3]])
    return boundary_map, distances


def dense_oversegmentation(boundary_map, distances, density_threshold, sigma, n_iter, dt, backend, n_threads):
    """The expensive, beta-independent part of run_multicut: oversegmentation, RAG and edge features."""
    from elf.segmentation.features import compute_rag, compute_boundary_mean_and_length, compute_z_edge_mask
    n_slices = boundary_map.shape[0]
    overseg = np.zeros(boundary_map.shape, dtype="uint64")

    def run_overseg(z):
        bd = boundary_map[z]
        dists = distances[:, z]
        fg_mask = np.ones(bd.shape, dtype="bool")
        density = _compute_flow_density(
            dists, fg_mask, n_iter=int(n_iter), dt=dt, sigma=sigma, verbose=False, backend=backend, n_threads=1,
        )
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


def score_image_dense_cached(prediction, labels, params_list, backend, n_threads=POSTPROC_THREADS, border_min_size=0):
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
                dt=dt, backend=backend, n_threads=n_threads,
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


def score_image(
    prediction, labels, mode, params_list, backend, n_threads=POSTPROC_THREADS, spacing=None, border_min_size=0,
):
    """Return a per-combo list of metric dicts for one decoder prediction, aligned with params_list."""
    if mode == "sparse":
        return score_image_sparse_cached(
            prediction, labels, params_list, backend, n_threads=n_threads, spacing=spacing,
            border_min_size=border_min_size,
        )
    return score_image_dense_cached(
        prediction, labels, params_list, backend, n_threads=n_threads, border_min_size=border_min_size
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


def tune_parameters(
    model, mode, dataset_name, data_root, model_type, output_root, device,
    backend="cpp", n_threads=POSTPROC_THREADS, n_tuning_samples=None, crop_shape=None, criterion=None,
    checkpoint_id=None,
):
    """Sweep the grid of a mode on the validation split and return the best parameter combination.

    Inference runs once per sample and every combination is scored on it before moving on, so the
    predictions are never all held in memory at once.

    The result is written below '<output_root>/<model_type>/<checkpoint_id>', which is the layout
    `common.read_tuned_params` reads. An existing file is loaded rather than recomputed, so a
    preempted job resumes at the next dataset instead of sweeping again.

    Args:
        model: The model of the mode, from `common.build_model`.
        mode: The segmentation mode, 'ais' or 'apg'.
        dataset_name: The dataset to tune on.
        data_root: The root the data lives in.
        model_type: The SAM2 backbone, which names the output subdirectory.
        output_root: The root the sweep results are written to.
        device: The torch device.
        backend: The backend for the flow computation.
        n_threads: The threads for the postprocessing.
        n_tuning_samples: Cap the sweep to this many validation samples.
        crop_shape: The 3d center crop.
        criterion: The metric the grid is ranked by.
        checkpoint_id: The checksum of all model weights used by the mode.

    Returns:
        The best parameter combination, or None when the dataset has nothing held out to tune on.
    """
    if not has_val_split(dataset_name):
        print(f"'{dataset_name}' has no data held out from the evaluation, so the defaults are used.")
        return None

    config = tuning_config(dataset_name, mode, criterion=criterion, crop_shape=crop_shape)
    output_dir = os.path.join(output_root, model_type, checkpoint_id) if checkpoint_id else os.path.join(
        output_root, model_type
    )
    csv_path = os.path.join(output_dir, f"{dataset_name}.csv")
    os.makedirs(output_dir, exist_ok=True)

    legacy_path = os.path.join(output_root, model_type, f"{dataset_name}.csv")
    if os.path.exists(csv_path) or (checkpoint_id is not None and os.path.exists(legacy_path)):
        params = read_tuned_params(output_root, dataset_name, model_type, checkpoint_id)
        cached_path = csv_path if os.path.exists(csv_path) else legacy_path
        print(f"Loading the finished sweep at '{cached_path}'.")
        report_best(pd.read_csv(cached_path), dataset_name, config)
        return params

    ndim, postproc_mode = config["ndim"], config["mode"]
    keys = list(config["grid"])
    params_list = [dict(zip(keys, combo)) for combo in itertools.product(*[config["grid"][k] for k in keys])]
    if postproc_mode == "sparse":
        n_full = len(params_list)
        params_list = deduplicate_flow_travel(params_list)
        if len(params_list) < n_full:
            print(f"Deduplicated {n_full} combos to {len(params_list)} by flow travel (n_iter * dt).")

    border_min_size = GT_MIN_SIZE_2D.get(dataset_name, 0) if ndim == 2 else 0
    total = n_samples(dataset_name, data_root, split="val")
    if n_tuning_samples is not None:
        total = min(total, n_tuning_samples)
    print(f"{dataset_name}: {len(params_list)} combinations over {total} validation sample(s), mode='{mode}'.")

    samples = load_data(
        dataset_name, data_root, ndim, split="val", crop_shape=config["crop"], z_range=config["z_range"],
    )
    metric_lists = [[] for _ in params_list]
    t0 = time.perf_counter()
    for index, (raw, labels, _) in enumerate(tqdm(samples, total=total, desc=f"tune-{mode}")):
        if n_tuning_samples is not None and index >= n_tuning_samples:
            break
        if labels.max() == 0:  # Nothing to score without ground-truth.
            continue
        try:
            if postproc_mode == "apg":
                scores = score_sample_apg(
                    model, raw, labels, params_list, ndim, config["metric_mode"],
                    spacing=config["spacing"], border_min_size=border_min_size,
                )
            else:
                prediction = predict_unisam2(model, raw, ndim=ndim, device=device)
                scores = score_image(
                    prediction, labels, postproc_mode, params_list, backend,
                    n_threads=n_threads, spacing=config["spacing"], border_min_size=border_min_size,
                )
        except Exception as e:
            warnings.warn(f"Skipping validation sample {index} of '{dataset_name}': {e}")
            continue
        for i, metrics in enumerate(scores):
            if metrics is not None:
                metric_lists[i].append(metrics)

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
    print(f"Saved {csv_path} ({time.perf_counter() - t0:.0f}s).")
    report_best(df, dataset_name, config)
    return read_tuned_params(output_root, dataset_name, model_type, checkpoint_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-d", "--dataset_name", type=str, nargs="+", default=None, choices=sorted(VAL_SPLITS),
        help="Datasets to tune. Defaults to every dataset with a validation split (see common.VAL_SPLITS).",
    )
    parser.add_argument("-i", "--input_path", type=str, default=DATA_ROOT, help="The root the data lives in.")
    parser.add_argument("-e", "--experiment_folder", type=str, required=True)
    parser.add_argument("-m", "--model_type", type=str, default="hvit_t", choices=MODEL_TYPES)
    parser.add_argument(
        "--mode", type=str, nargs="+", default=list(MODES), choices=MODES, help="The segmentation modes to tune."
    )
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="Weights instead of the joint export.")
    parser.add_argument("--joint_checkpoint", type=str, default="best", help="Joint checkpoint name, without '.pt'.")
    parser.add_argument(
        "--interactive_checkpoint", type=str, default=None,
        help="Standalone interactive weights for apg mode, bypassing the joint checkpoint entirely. "
             "Requires -c/--checkpoint for the decoder half too.",
    )
    parser.add_argument("--tuning_root", type=str, default=None, help="Where the sweeps are written and read from.")
    parser.add_argument("--n_tuning_samples", type=int, default=None, help="Cap each sweep to this many samples.")
    parser.add_argument("--criterion", type=str, default=None, choices=sorted(CRITERION_ASCENDING))
    parser.add_argument("--backend", type=str, default="cpp", choices=("cpp", "python"), help="Flow backend.")
    parser.add_argument("--n_threads", type=int, default=POSTPROC_THREADS, help="Threads for the postprocessing.")
    parser.add_argument("--crop_3d", type=int, nargs=3, default=None, help="Override the 3d crop (Z Y X).")
    args = parser.parse_args()

    datasets = args.dataset_name or sorted(VAL_SPLITS)
    for dataset_name in datasets:
        check_data_download(dataset_name, args.input_path)

    print("Device:", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    crop_shape = tuple(args.crop_3d) if args.crop_3d else None

    for mode in args.mode:
        checkpoint_id, joint_checksum = resolve_checkpoint_identity(
            mode, args.model_type, args.joint_checkpoint, args.checkpoint,
            interactive_checkpoint_path=args.interactive_checkpoint,
        )
        tuning_root = args.tuning_root or os.path.join(args.experiment_folder, "tuning")
        tuning_root = os.path.join(tuning_root, mode)

        # AIS's decoder does not depend on ndim, so one build covers every dataset. APG's video
        # input type does, so it is rebuilt once per dimensionality group instead.
        models = {}
        for dataset_name in datasets:
            ndim = 3 if dataset_name in DATASETS_3D else 2
            model_key = ndim if mode == "apg" else mode
            if model_key not in models:
                models[model_key] = build_model(
                    mode, args.model_type, device, ndim,
                    joint_checkpoint=args.joint_checkpoint, checkpoint_path=args.checkpoint,
                    joint_checksum=joint_checksum, interactive_checkpoint_path=args.interactive_checkpoint,
                )
            tune_parameters(
                models[model_key], mode, dataset_name, args.input_path, args.model_type, tuning_root, device,
                backend=args.backend, n_threads=args.n_threads, n_tuning_samples=args.n_tuning_samples,
                crop_shape=crop_shape, criterion=args.criterion, checkpoint_id=checkpoint_id,
            )


if __name__ == "__main__":
    main()
