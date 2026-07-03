"""Grid search over automatic-segmentation postprocessing parameters for the hvit_t_cells model.

Loads the finetuned SAM2 'hvit_t_cells' model from the micro-sam v2 download registry (the same
model the annotator uses by default), runs the UniSAM2 decoder once per image, then sweeps the
postprocessing hyperparameters and averages the mean segmentation accuracy (mSA) over the images
of each track. Three tracks are supported:

    lm_cell     LM cell segmentation, sparse (flow) mode, 2d (livecell).
    lm_nucleus  LM nucleus segmentation, sparse (flow) mode, 2d (dsb).
    em_neurons  EM neuron segmentation, dense (multicut) mode, 3d (cremi + snemi + humanneurons).

For each track the best parameter combination is written to '<output_dir>/<track>.csv' together with
the full sweep, and the best row is printed.

Usage:
    python grid_search_automatic_cells.py --track lm_cell
    python grid_search_automatic_cells.py --track em_neurons --em_crop 16 512 512
    python grid_search_automatic_cells.py --track all -o /path/to/results
"""

import os
import sys
import time
import argparse
import warnings
import itertools
from concurrent import futures

import numpy as np
import pandas as pd
import imageio.v3 as imageio
from tqdm import tqdm

from elf.io import open_file
from elf.evaluation import mean_segmentation_accuracy
from bioimage_cpp.segmentation import label as connected_components, watershed

from micro_sam.v2.postprocessing import flow_instance_segmentation, run_multicut, _compute_flow_density

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_ROOT, get_data_paths, load_volume  # noqa


OUTPUT_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/grid-search-hvit-t-cells"
DEVICE = "cuda"
MODEL_NAME = "hvit_t_cells"

CROP_SHAPE_2D = (512, 512)

# Sparse (flow) grid for LM data. Keys map to flow_instance_segmentation arguments.
LM_GRID = {
    "foreground_threshold": [0.3, 0.5, 0.7],
    "density_threshold": [5.0, 10.0, 20.0],
    "min_size": [10, 100, 500],
    "sigma": [0.5, 1.0, 2.0],
    "n_iter": [50, 100, 200],
    "dt": [0.25, 0.5, 1.0],
}

# Dense (multicut) grid for EM data. Keys map to run_multicut arguments.
EM_GRID = {
    "beta": [0.5, 0.6, 0.7, 0.8],
    "density_threshold": [3.0, 5.0, 10.0],
    "sigma": [0.5, 1.0, 2.0],
    "n_iter": [25, 50],
}

# Sparse-flow parameters that determine the (expensive) convergence-density map. Combos that share
# these reuse a cached density; only 'density_threshold' and 'min_size' (the cheap seed + watershed +
# size-filter steps) then vary on top. See score_image_sparse_cached.
FLOW_DENSITY_KEYS = ("foreground_threshold", "sigma", "n_iter", "dt")

# Dense-multicut parameters that determine the (expensive) slice-wise oversegmentation and RAG.
# Combos that share these reuse a cached oversegmentation; only 'beta' (the cheap edge-cost +
# multicut-solve step) then varies on top. See score_image_dense_cached.
OVERSEG_KEYS = ("density_threshold", "sigma", "n_iter", "dt")

# Metric used to rank each track's grid, and whether lower is better. 'msa' (mean segmentation accuracy)
# is maximised; 'cremi' (the CREMI score, a VI + adapted-Rand combination for neuron segmentation) is
# minimised.
CRITERION_ASCENDING = {"msa": False, "cremi": True}

# Each track: the datasets to evaluate, the postprocessing mode, the spatial dimensionality, the grid,
# an optional anisotropic voxel 'spacing' for 3d flow smoothing (matching common._DATASET_SPACING), the
# 3d center-crop 'crop' (None for 2d), and the 'criterion' used to pick the best combo. Crops match the
# micro-sam v2 eval (evaluate_3d uses (8,512,512) for LM; the EM neuron grid search uses (32,512,512)).
TRACKS = {
    "lm_cell": {
        "datasets": ["livecell"], "mode": "sparse", "ndim": 2, "grid": LM_GRID,
        "spacing": None, "crop": None, "criterion": "msa",
    },
    "lm_nucleus": {
        "datasets": ["dsb"], "mode": "sparse", "ndim": 2, "grid": LM_GRID,
        "spacing": None, "crop": None, "criterion": "msa",
    },
    "skull_nuclei": {
        "datasets": ["embedseg"], "mode": "sparse", "ndim": 3, "grid": LM_GRID,
        "spacing": (4, 1, 1), "crop": (8, 512, 512), "criterion": "msa",
    },
    "em_neurons": {
        "datasets": ["cremi", "snemi", "humanneurons"], "mode": "dense", "ndim": 3, "grid": EM_GRID,
        "spacing": None, "crop": (32, 512, 512), "criterion": "cremi",
    },
}

POSTPROC_THREADS = 4


def load_model(device, checkpoint_path=None, model_name=MODEL_NAME):
    """Load the UniSAM2 model for the finetuned hvit_t_cells registry model.

    Mirrors the annotator's decoder-loading path: the finetuned decoder is fetched from the
    micro-sam v2 download registry and loaded via get_unisam2_model, which rebuilds the matching
    SAM2 encoder and (re)defines all weights from the decoder checkpoint.

    Args:
        device: The device to load the model onto.
        checkpoint_path: Optional path to a custom UniSAM2 / joint checkpoint. If given, it is used
            instead of the registry model.
        model_name: The registry model name, e.g. 'hvit_t_cells'.

    Returns:
        The UniSAM2 model in eval mode.
    """
    from micro_sam.v2.automatic_segmentation import get_unisam2_model

    if checkpoint_path is not None:
        print(f"Loading UniSAM2 model from custom checkpoint '{checkpoint_path}'.")
        return get_unisam2_model(checkpoint_path, device=device, encoder=model_name[:6])

    from micro_sam.v2.util import FINETUNED_MODELS, _download_finetuned_sam2_model
    assert model_name in FINETUNED_MODELS, f"'{model_name}' is not a registered model: {FINETUNED_MODELS}."
    print(f"Fetching finetuned decoder for '{model_name}' from the micro-sam v2 registry.")
    _, _, decoder_source = _download_finetuned_sam2_model(model_name)
    if decoder_source is None:
        raise RuntimeError(f"The registry model '{model_name}' has no registered decoder.")
    return get_unisam2_model(decoder_source, device=device, encoder=model_name[:6])


def predict(model, raw, ndim, device):
    """Run the UniSAM2 model to predict foreground + directed distances, shape (4, *spatial).

    Uses the same tiled inference path as the micro-sam v2 evaluation (common.predict_unisam2), so
    the tuned parameters transfer directly to the evaluation scripts.
    """
    from common import predict_unisam2
    return predict_unisam2(model, raw, ndim=ndim, device=device)


def read_image_2d(path, key):
    """Read a 2d (grayscale) image from a plain image file or an H5/zarr key."""
    if key is not None:
        arr = np.asarray(open_file(path, mode="r")[key][:])
    else:
        arr = np.asarray(imageio.imread(path))
    if arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[1] > arr.shape[0] and arr.shape[2] > arr.shape[0]:
        arr = arr.transpose(1, 2, 0)
    # The UniSAM2 2d inference path expects single-channel input; reduce a trailing channel axis.
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    return arr


def center_crop_2d(arr, crop_shape):
    """Return a center crop of a 2d array along its first two axes."""
    roi = []
    for size, crop in zip(arr.shape[:2], crop_shape):
        crop = min(crop, size)
        start = (size - crop) // 2
        roi.append(slice(start, start + crop))
    return arr[tuple(roi)]


def resolve_data_paths(dataset_name, livecell_per_celltype=None):
    """Return (raw_paths, label_paths, raw_key, label_key) for a dataset's evaluation split.

    Special-cases dsb to use the smaller 'reduced' fluorescence test split (50 images) and livecell to
    use the built-in per-cell-type stratification ('n_val_per_cell_type' in _get_livecell_paths), rather
    than the full sets that common.get_data_paths returns; all other datasets defer to common.
    """
    if dataset_name == "dsb":
        from torch_em.data import datasets
        img, gt = datasets.dsb.get_dsb_paths(
            path=os.path.join(DATA_ROOT, "dsb"), source="reduced", split="test",
        )
        return img, gt, None, None
    if dataset_name == "livecell":
        from micro_sam.v1.evaluation.livecell import _get_livecell_paths
        img, gt = _get_livecell_paths(
            input_folder=os.path.join(DATA_ROOT, "livecell"), split="test",
            n_val_per_cell_type=livecell_per_celltype,
        )
        return sorted(img), sorted(gt), None, None
    return get_data_paths(dataset_name, DATA_ROOT)


def build_work_items(track_cfg, n_images, livecell_per_celltype):
    """Resolve a track's datasets into a flat list of (dataset, raw_path, label_path, raw_key, label_key).

    Datasets whose paths cannot be resolved (e.g. a loader missing from the installed torch-em) are
    skipped with a warning rather than aborting the whole track. livecell is stratified to
    'livecell_per_celltype' images per cell type (built into _get_livecell_paths); other 2d datasets use
    every test image (optionally capped to the first 'n_images'); 3d tracks use every available volume.
    """
    items = []
    for dataset_name in track_cfg["datasets"]:
        try:
            raw_paths, label_paths, raw_key, label_key = resolve_data_paths(dataset_name, livecell_per_celltype)
        except Exception as e:
            warnings.warn(f"Skipping dataset '{dataset_name}': {e}")
            continue

        pairs = list(zip(raw_paths, label_paths))
        if track_cfg["ndim"] == 2 and dataset_name != "livecell" and n_images is not None:
            pairs = pairs[:n_images]
        for raw_path, label_path in pairs:
            items.append((dataset_name, raw_path, label_path, raw_key, label_key))
    return items


def load_sample(item, ndim, em_crop):
    """Load and preprocess one work item into (raw, labels), matching the micro-sam v2 eval crops."""
    dataset_name, raw_path, label_path, raw_key, label_key = item
    if ndim == 2:
        raw = center_crop_2d(read_image_2d(raw_path, raw_key), CROP_SHAPE_2D).astype("float32")
        labels = center_crop_2d(read_image_2d(label_path, label_key), CROP_SHAPE_2D)
        labels = connected_components(labels).astype("uint32")
    else:
        raw, labels, _ = load_volume(
            raw_path=raw_path, label_path=label_path, raw_key=raw_key, label_key=label_key,
            dataset_name=dataset_name, crop_shape=tuple(em_crop),
        )
    return raw, labels


def compute_metrics(seg, labels, mode):
    """Compute the evaluation metrics for one segmentation.

    Always reports mSA. For dense (EM neuron) mode it additionally reports the CREMI score and its
    VI-split / VI-merge / adapted-Rand components, since neuron segmentation is ranked by CREMI
    (lower is better) rather than mSA.
    """
    metrics = {"msa": float(mean_segmentation_accuracy(seg, labels))}
    if mode == "dense":
        from elf.evaluation import cremi_score
        vi_split, vi_merge, adapted_rand, cremi = cremi_score(seg, labels)
        metrics["cremi"] = float(cremi)
        metrics["vi_split"] = float(vi_split)
        metrics["vi_merge"] = float(vi_merge)
        metrics["adapted_rand"] = float(adapted_rand)
    return metrics


def postprocess(prediction, mode, params, backend="cpp", n_threads=POSTPROC_THREADS, spacing=None):
    """Convert a (4, *spatial) prediction into an instance segmentation for the given mode."""
    if mode == "dense":
        fg = prediction[0]
        boundary_map = fg.max() - fg
        denom = boundary_map.max()
        if denom > 0:
            boundary_map = boundary_map / denom
        distances = np.stack([prediction[2], prediction[3]])
        seg = run_multicut(boundary_map, distances, backend=backend, n_threads=n_threads, **params)
    else:
        seg = flow_instance_segmentation(
            prediction[0], prediction[1:], backend=backend, n_threads=n_threads, spacing=spacing, **params,
        )
    return seg.astype("uint32")


def score_image_sparse_cached(prediction, labels, params_list, backend, n_threads=POSTPROC_THREADS, spacing=None):
    """Score all sparse combos on one image, caching the flow density across shared combos.

    Faithfully reproduces flow_instance_segmentation: the height map and each convergence-density map
    (the expensive flow integration) are computed once and reused; only the cheap seed / watershed /
    min-size steps vary per (density_threshold, min_size). 'spacing' is the anisotropic voxel spacing
    for 3d flow smoothing. Returns a per-combo list of metric dicts (None where a combo failed), aligned
    with params_list.
    """
    foreground = prediction[0]
    directed = prediction[1:]
    ndim = foreground.ndim
    if directed.shape[0] > ndim:
        directed = directed[-ndim:]
    hmap = np.linalg.norm(directed, axis=0)
    hmap = hmap.max() - hmap

    fg_mask_cache, density_cache = {}, {}
    scores = []
    for params in params_list:
        ft, sigma, n_iter, dt = (params[k] for k in FLOW_DENSITY_KEYS)
        if ft not in fg_mask_cache:
            fg_mask_cache[ft] = foreground > ft
        fg_mask = fg_mask_cache[ft]

        key = (ft, sigma, n_iter, dt)
        if key not in density_cache:
            density_cache[key] = _compute_flow_density(
                directed, fg_mask, n_iter=int(n_iter), dt=dt, sigma=sigma, spacing=spacing,
                backend=backend, n_threads=n_threads,
            )
        density = density_cache[key]

        try:
            seeds = connected_components(density > params["density_threshold"])
            seg = watershed(hmap, markers=seeds, mask=fg_mask)
            min_size = params["min_size"]
            if min_size > 0:
                ids, sizes = np.unique(seg, return_counts=True)
                discard = ids[(sizes < min_size) & (ids > 0)]
                seg[np.isin(seg, discard)] = 0
                seg = watershed(hmap, markers=seg, mask=fg_mask)
            scores.append(compute_metrics(seg.astype("uint32"), labels, "sparse"))
        except Exception as e:
            warnings.warn(f"Sparse postprocessing failed for {params}: {e}")
            scores.append(None)
    return scores


def _dense_boundary_and_distances(prediction):
    """Return the (boundary_map, in-plane distances) inputs to run_multicut for a dense prediction."""
    fg = prediction[0]
    boundary_map = fg.max() - fg
    denom = boundary_map.max()
    if denom > 0:
        boundary_map = boundary_map / denom
    distances = np.stack([prediction[2], prediction[3]])
    return boundary_map, distances


def _dense_oversegmentation(boundary_map, distances, density_threshold, sigma, n_iter, dt, backend, n_threads):
    """Slice-wise oversegmentation + RAG + boundary features: the expensive, beta-independent part of run_multicut.

    Replicates run_multicut up to (but not including) the beta-dependent edge costs, so the result can be
    cached and reused across all beta values.
    """
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


def _dense_solve(rag, feats, z_edges, overseg, n_slices, beta):
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


def score_image_dense_cached(prediction, labels, params_list, backend, n_threads=POSTPROC_THREADS):
    """Score all dense combos on one image, caching the oversegmentation + RAG across shared combos.

    Faithfully reproduces run_multicut: the slice-wise oversegmentation and region-adjacency graph (the
    expensive part) are computed once per (density_threshold, sigma, n_iter, dt) and reused; only the
    cheap edge-cost + multicut-solve step varies per beta. Returns a per-combo list of metric dicts
    (None where a combo failed), aligned with params_list.
    """
    boundary_map, distances = _dense_boundary_and_distances(prediction)
    overseg_cache = {}
    scores = []
    for params in params_list:
        dt = params.get("dt", 0.5)
        key = tuple(params.get(k, dt if k == "dt" else None) for k in OVERSEG_KEYS)
        if key not in overseg_cache:
            overseg_cache[key] = _dense_oversegmentation(
                boundary_map, distances, params["density_threshold"], params["sigma"], params["n_iter"],
                dt=dt, backend=backend, n_threads=n_threads,
            )
        overseg, rag, feats, z_edges, n_slices = overseg_cache[key]
        try:
            seg = _dense_solve(rag, feats, z_edges, overseg, n_slices, params["beta"])
            scores.append(compute_metrics(seg, labels, "dense"))
        except Exception as e:
            warnings.warn(f"Dense multicut solve failed for {params}: {e}")
            scores.append(None)
    return scores


def score_image(
    prediction, labels, mode, params_list, backend, use_flow_cache=True, n_threads=POSTPROC_THREADS, spacing=None,
):
    """Return a per-combo list of metric dicts for one image (None where a combo failed), aligned with params_list."""
    if use_flow_cache and mode == "sparse":
        return score_image_sparse_cached(
            prediction, labels, params_list, backend, n_threads=n_threads, spacing=spacing,
        )
    if use_flow_cache and mode == "dense":
        return score_image_dense_cached(prediction, labels, params_list, backend, n_threads=n_threads)
    scores = []
    for params in params_list:
        try:
            seg = postprocess(prediction, mode, params, backend=backend, n_threads=n_threads, spacing=spacing)
            scores.append(compute_metrics(seg, labels, mode))
        except Exception as e:
            warnings.warn(f"Postprocessing failed for {params}: {e}")
            scores.append(None)
    return scores


def run_track(
    track_name, model, n_images, livecell_per_celltype, output_dir, backend, device,
    crop_override=None, use_flow_cache=True, n_threads=POSTPROC_THREADS,
):
    """Run the full grid search for one track and save the results CSV.

    Inference is run once per image (streaming, so predictions are never all held in memory at once)
    and every parameter combination is scored on that image before moving on. The per-image mSA is
    then averaged over all images of the track. Datasets or images that fail to load are skipped with
    a warning. 'crop_override' replaces the track's default 3d crop when given.
    """
    track_cfg = TRACKS[track_name]
    ndim = track_cfg["ndim"]
    mode = track_cfg["mode"]
    spacing = track_cfg.get("spacing")
    crop = crop_override if crop_override is not None else track_cfg.get("crop")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{track_name}.csv")

    if os.path.exists(csv_path):
        print(f"Results already exist at '{csv_path}', loading.")
        df = pd.read_csv(csv_path)
    else:
        items = build_work_items(track_cfg, n_images, livecell_per_celltype)
        if not items:
            warnings.warn(f"No usable data for track '{track_name}', skipping.")
            return None

        keys = list(track_cfg["grid"].keys())
        combos = list(itertools.product(*[track_cfg["grid"][k] for k in keys]))
        params_list = [dict(zip(keys, combo)) for combo in combos]
        metric_lists = [[] for _ in combos]  # per combo: a list of per-image metric dicts
        criterion = track_cfg.get("criterion", "msa")
        ascending = CRITERION_ASCENDING[criterion]
        print(f"{track_name}: {len(combos)} combinations over {len(items)} sample(s), mode='{mode}'.")

        t0 = time.perf_counter()
        for item in tqdm(items, desc=f"{track_name} samples"):
            try:
                raw, labels = load_sample(item, ndim, crop)
                prediction = predict(model, raw, ndim=ndim, device=device)
            except Exception as e:
                warnings.warn(f"Skipping sample '{item[1]}': {e}")
                continue
            scores = score_image(
                prediction, labels, mode, params_list, backend,
                use_flow_cache=use_flow_cache, n_threads=n_threads, spacing=spacing,
            )
            for i, metrics in enumerate(scores):
                if metrics is not None:
                    metric_lists[i].append(metrics)

        rows = []
        for params, per_image in zip(params_list, metric_lists):
            if not per_image:
                continue
            row = {**params, "n_images": len(per_image)}
            for metric_key in per_image[0]:
                values = [m[metric_key] for m in per_image]
                row[f"{metric_key}_mean"] = float(np.mean(values))
                row[f"{metric_key}_std"] = float(np.std(values))
            rows.append(row)
        sort_col = f"{criterion}_mean"
        df = pd.DataFrame(rows).sort_values(sort_col, ascending=ascending).reset_index(drop=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path} ({time.perf_counter() - t0:.0f}s).")

    criterion = track_cfg.get("criterion", "msa")
    ascending = CRITERION_ASCENDING[criterion]
    best = df.sort_values(f"{criterion}_mean", ascending=ascending).iloc[0]
    best_params = {k: best[k] for k in track_cfg["grid"].keys()}
    crit_mean, crit_std = best[f"{criterion}_mean"], best[f"{criterion}_std"]
    direction = "lower is better" if ascending else "higher is better"
    print(f"Best params for {track_name} (by {criterion}, {direction}): {best_params}")
    print(f"  {criterion} = {crit_mean:.4f} (+/-{crit_std:.4f})")
    return best_params


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track", default="all", choices=list(TRACKS) + ["all"], help="Which track to run.")
    parser.add_argument("-o", "--output_dir", default=OUTPUT_ROOT, help="Directory to write result CSVs.")
    parser.add_argument("-n", "--n_images", type=int, default=None, help="Cap images per 2d dataset (default: all).")
    parser.add_argument("--livecell_per_celltype", type=int, default=50, help="Images per livecell cell type.")
    parser.add_argument("--crop_3d", type=int, nargs=3, default=None, help="Override the 3d crop (Z Y X).")
    parser.add_argument("-c", "--checkpoint_path", default=None, help="Custom checkpoint instead of registry model.")
    parser.add_argument("--backend", default="cpp", choices=["cpp", "python"], help="Flow computation backend.")
    parser.add_argument("--no_flow_cache", action="store_true", help="Disable the lazy postprocessing caching.")
    parser.add_argument("--n_threads", type=int, default=POSTPROC_THREADS, help="Threads for postprocessing.")
    args = parser.parse_args()

    import torch
    device = DEVICE if torch.cuda.is_available() else "cpu"
    print("Device:", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")

    model = load_model(device, checkpoint_path=args.checkpoint_path)

    crop_override = tuple(args.crop_3d) if args.crop_3d is not None else None
    tracks = list(TRACKS) if args.track == "all" else [args.track]
    summary = {}
    for track_name in tracks:
        summary[track_name] = run_track(
            track_name, model, args.n_images, args.livecell_per_celltype,
            args.output_dir, args.backend, device, crop_override=crop_override,
            use_flow_cache=(not args.no_flow_cache), n_threads=args.n_threads,
        )

    print("\nBest parameters per track:")
    for track_name, params in summary.items():
        print(f"{track_name}: {params}")


if __name__ == "__main__":
    main()
