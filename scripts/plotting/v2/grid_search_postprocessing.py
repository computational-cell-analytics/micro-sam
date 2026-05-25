"""Grid search over postprocessing hyperparameters for UniSAM2 predictions.

Runs the UniSAM2 model live on one sample per dataset, sweeps postprocessing
parameters for the cpp backend, evaluates mSA against ground truth, and saves
combined results to CSV.

LM datasets (nis3d, plantseg_ovules) use flow_instance_segmentation.
EM datasets (humanneurons, snemi, cremi, liconn) use run_multicut.
liconn is LM data but is treated as EM (dense connectomic reconstruction).

Results are stored one CSV per (dataset, model) pair:
    <output_dir>/<dataset>/<model>.csv

Usage:
    python grid_search_postprocessing.py -d humanneurons -m automatic
    python grid_search_postprocessing.py -d nis3d -m automatic
    python grid_search_postprocessing.py -d plantseg_ovules -m automatic
    python grid_search_postprocessing.py -d liconn -m automatic
"""

import argparse
import itertools
import os
import sys
import time

sys.path.insert(0, "/mnt/vast-kisski/home/archit/u28048/micro-sam/finetuning/v2/evaluation")

import h5py  # noqa
import pandas as pd  # noqa
from tqdm import tqdm  # noqa

from elf.evaluation import mean_segmentation_accuracy  # noqa
from micro_sam.v2.postprocessing import flow_instance_segmentation, run_multicut  # noqa
from common import (  # noqa
    DATA_ROOT, UNISAM2_CHECKPOINT, load_unisam2_model, load_volume, predict_unisam2,
)


PREDICTIONS_ROOT = "/mnt/vast-nhr/home/archit/u12090/micro-sam/finetuning/v2/generalist/predictions"
OUTPUT_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/grid-search-experiments"
DEVICE = "cuda"

DATASETS = [
    "snemi", "nis3d", "plantseg_root", "cremi", "humanneurons",
    "plantseg_ovules", "pnas_arabidopsis", "celegans_atlas", "mitoem", "liconn",
    "cremi_padded", "microns_minnie65",
]
EM_DATASETS = {"snemi", "cremi", "humanneurons", "mitoem", "liconn", "cremi_padded", "microns_minnie65"}

# Isotropic 700 x 700 x 700 sub-volume for liconn evaluation (5.5 GB distances).
# Centered at Z=400, Y=2412, X=1912 in the full 795 x 4870 x 3825 zarr.
LICONN_EVAL_ROI = (slice(50, 750), slice(2062, 2762), slice(1562, 2262))

# Fixed 100 x 1024 x 1024 sub-volume for mitoem evaluation.
# The val volume is 100 x 4096 x 4096; global multicut on the full XY is not tractable.
# Take all 100 Z slices but crop to a centre 1024 x 1024 XY region.
MITOEM_EVAL_ROI = (slice(None), slice(1536, 2560), slice(1536, 2560))

# Crop that maps the 200 x 3072 x 3072 padded sampleB raw back to the 125 x 1250 x 1250 labeled region.
# Derived from the labels offset attribute [1480, 3644, 3644] divided by resolution [40, 4, 4].
CREMI_PADDED_LABEL_ROI = (slice(37, 162), slice(911, 2161), slice(911, 2161))

# Fixed 100 x 1024 x 1024 sub-volume for microns_minnie65 grid-search evaluation.
# Each zarr box is 512 x 4096 x 4096; take 100 central Z-slices and centre 1024 x 1024 XY.
MICRONS_MINNIE65_EVAL_ROI = (slice(206, 306), slice(1536, 2560), slice(1536, 2560))
# 200 x 2048 x 2048 central sub-volume for full-volume evaluation (~35 min total).
MICRONS_MINNIE65_FULL_ROI = (slice(156, 356), slice(1024, 3072), slice(1024, 3072))
# 69882d21db4e has 261 instances in the 32x512x512 grid-search crop (best among all boxes).
MICRONS_MINNIE65_GRID_SEARCH_ZARR = "69882d21db4e.zarr"

LM_GRID = {
    "foreground_threshold": [0.3, 0.5, 0.7],
    "density_threshold": [5.0, 10.0, 20.0],
    "min_size": [10, 100, 500],
    "sigma": [0.5, 1.0, 2.0],
}

EM_GRID = {
    "beta": [0.5, 0.6, 0.7, 0.8],
    "density_threshold": [3.0, 5.0, 10.0],
    "sigma": [0.5, 1.0, 2.0],
}


GRID_SEARCH_CROP = {
    "snemi": (32, 512, 512),
    "nis3d": (32, 512, 512),
    "plantseg_root": (16, 256, 256),
    "cremi": (32, 512, 512),
    "humanneurons": (32, 512, 512),
    "plantseg_ovules": (16, 256, 256),
    "pnas_arabidopsis": (32, 256, 256),
    "celegans_atlas": (64, 128, 512),
    "mitoem": (32, 512, 512),
    "liconn": (32, 512, 512),
    "cremi_padded": (32, 512, 512),
    "microns_minnie65": (32, 512, 512),
}


def _center_crop(arr, crop_shape):
    """Return a center crop of arr along its last len(crop_shape) axes."""
    roi = []
    for size, crop in zip(arr.shape[-len(crop_shape):], crop_shape):
        crop = min(crop, size)
        start = (size - crop) // 2
        roi.append(slice(start, start + crop))
    full_roi = (slice(None),) * (arr.ndim - len(crop_shape)) + tuple(roi)
    return arr[full_roi]


def _get_data_paths_grid_search(dataset_name):
    """Return (raw_paths, label_paths, raw_key, label_key) for grid-search datasets.

    Handles plantseg sub-variants (ovules, root) that are not top-level entries
    in common.get_data_paths. For liconn, returns the zarr path; actual loading
    is handled separately in _generate_live_predictions via LICONN_EVAL_ROI.
    """
    import torch_em.data.datasets as datasets
    p = DATA_ROOT
    if dataset_name == "plantseg_ovules":
        paths = datasets.plantseg.get_plantseg_paths(
            path=os.path.join(p, "plantseg_ovules"), name="ovules", split="test",
        )
        return sorted(paths), sorted(paths), "raw", "label"
    if dataset_name == "plantseg_root":
        paths = datasets.plantseg.get_plantseg_paths(
            path=os.path.join(p, "plantseg_root"), name="root", split="test",
        )
        return sorted(paths), sorted(paths), "raw", "label"
    if dataset_name == "liconn":
        zarr_path = os.path.join(p, "liconn", "liconn.zarr")
        return [zarr_path], [zarr_path], "raw", "seg_proofread"
    if dataset_name == "mitoem":
        n5_path = os.path.join(p, "mitoem", "human_val.n5")
        return [n5_path], [n5_path], "raw", "labels"
    if dataset_name == "cremi_padded":
        # Returns padded sampleB path; _generate_live_predictions switches to cropped for grid search.
        import torch_em.data.datasets as datasets
        paths = datasets.cremi.get_cremi_paths(
            path=os.path.join(p, "cremi"), samples=("B",), version="padded",
        )
        return paths, paths, "volumes/raw", "volumes/labels/neuron_ids"
    if dataset_name == "microns_minnie65":
        import glob as glob_mod
        zarr_root = os.path.join(p, "microns-minnie65")
        paths = sorted(glob_mod.glob(os.path.join(zarr_root, "*.zarr")))
        return paths, paths, "raw", "labels"
    from common import get_data_paths
    return get_data_paths(dataset_name, p)


def _generate_live_predictions(dataset_name, model, crop_shape=None):
    """Run model inference on the first test sample and return (raw, distances, labels, valid_roi).

    If crop_shape is None, the full volume is loaded without cropping.
    For liconn and mitoem, always loads from a fixed eval ROI (full volumes are too large).
    """
    from elf.io import open_file
    from skimage.measure import label as connected_components

    raw_paths, label_paths, raw_key, label_key = _get_data_paths_grid_search(dataset_name)
    if dataset_name == "liconn":
        import zarr as zarr_mod
        store = zarr_mod.open(raw_paths[0])
        raw = store["raw"][LICONN_EVAL_ROI].astype("float32")
        labels = store["seg_proofread"][LICONN_EVAL_ROI]
        if crop_shape is not None:
            raw = _center_crop(raw, crop_shape)
            labels = _center_crop(labels, crop_shape)
        labels = connected_components(labels).astype("uint32")
        valid_roi = None
    elif dataset_name == "mitoem":
        roi = MITOEM_EVAL_ROI if crop_shape is not None else (slice(None), slice(None), slice(None))
        with open_file(raw_paths[0], mode="r") as f:
            raw = f[raw_key][roi].astype("float32")
        with open_file(label_paths[0], mode="r") as f:
            labels = f[label_key][roi]
        if crop_shape is not None:
            raw = _center_crop(raw, crop_shape)
            labels = _center_crop(labels, crop_shape)
        labels = connected_components(labels).astype("uint32")
        valid_roi = None
    elif dataset_name == "cremi_padded":
        # Grid search runs on cropped sampleB; full-volume run uses padded sampleB.
        # For the full-volume run, raw is 200x3072x3072 but labels are 125x1250x1250
        # (the inner annotated region). Postprocessing runs on the full raw; the
        # segmentation is cropped to CREMI_PADDED_LABEL_ROI before evaluation.
        cremi_dir = os.path.join(DATA_ROOT, "cremi")
        if crop_shape is not None:
            raw, labels, valid_roi = load_volume(
                raw_path=os.path.join(cremi_dir, "sampleB.h5"),
                label_path=os.path.join(cremi_dir, "sampleB.h5"),
                raw_key="volumes/raw",
                label_key="volumes/labels/neuron_ids",
                dataset_name="cremi_padded",
                crop_shape=crop_shape,
            )
        else:
            padded_path = os.path.join(cremi_dir, "sampleB_padded.h5")
            with open_file(padded_path, mode="r") as f:
                raw = f["volumes/raw"][:].astype("float32")
                labels = connected_components(f["volumes/labels/neuron_ids"][:]).astype("uint32")
            valid_roi = None
    elif dataset_name == "microns_minnie65":
        import zarr as zarr_mod
        zarr_path = os.path.join(DATA_ROOT, "microns-minnie65", MICRONS_MINNIE65_GRID_SEARCH_ZARR)
        store = zarr_mod.open(zarr_path, mode="r")
        if crop_shape is not None:
            raw = store["raw"][MICRONS_MINNIE65_EVAL_ROI].astype("float32")
            labels = store["labels"][MICRONS_MINNIE65_EVAL_ROI]
            raw = _center_crop(raw, crop_shape)
            labels = _center_crop(labels, crop_shape)
        else:
            raw = store["raw"][MICRONS_MINNIE65_FULL_ROI].astype("float32")
            labels = store["labels"][MICRONS_MINNIE65_FULL_ROI]
        labels = connected_components(labels).astype("uint32")
        valid_roi = None
    elif crop_shape is not None:
        raw, labels, valid_roi = load_volume(
            raw_path=raw_paths[0],
            label_path=label_paths[0],
            raw_key=raw_key,
            label_key=label_key,
            dataset_name=dataset_name,
            crop_shape=crop_shape,
        )
    else:
        from torch_em.transform.raw import normalize
        from torch_em.util.image import load_image
        if raw_key is None:
            raw = load_image(raw_paths[0]).astype("float32")
            labels = connected_components(load_image(label_paths[0])).astype("uint32")
        else:
            with open_file(raw_paths[0], mode="r") as f:
                raw = f[raw_key][:].astype("float32")
            with open_file(label_paths[0], mode="r") as f:
                labels = connected_components(f[label_key][:]).astype("uint32")
        if raw.max() > 255:
            raw = normalize(raw) * 255
        valid_roi = None
    print(f"Running inference on {raw_paths[0]} (shape={raw.shape}) ...")
    distances = predict_unisam2(model, raw, ndim=3, device=DEVICE)
    return raw, distances, labels, valid_roi


def _load_predictions(dataset_name, model_name, predictions_root, crop_shape=None):
    """Load distances and labels from a cached H5 prediction file.

    Args:
        dataset_name: one of DATASETS.
        model_name: 'automatic' or 'joint'.
        predictions_root: root directory of H5 prediction files.
        crop_shape: if given, center-crop to this (Z, Y, X) shape before returning.

    Returns:
        (raw, distances, labels): (Z, Y, X) float32, (4, Z, Y, X) float32, (Z, Y, X) uint32.
    """
    path = os.path.join(predictions_root, dataset_name, model_name, "sample_000.h5")
    with h5py.File(path, "r") as f:
        raw = f["raw"][:]
        distances = f["distances"][:]
        labels = f["labels"][:]
    if crop_shape is not None:
        raw = _center_crop(raw, crop_shape)
        distances = _center_crop(distances, crop_shape)
        labels = _center_crop(labels, crop_shape)
    return raw, distances, labels


def _postprocess_lm(
    distances, dataset_name, foreground_threshold, density_threshold, min_size, sigma,
    backend="cpp", n_threads=8,
):
    return flow_instance_segmentation(
        foreground=distances[0],
        directed_distances=distances[1:],
        foreground_threshold=foreground_threshold,
        density_threshold=density_threshold,
        min_size=min_size,
        sigma=sigma,
        backend=backend,
        n_threads=n_threads,
    ).astype("uint32")


def _postprocess_em(distances, beta, density_threshold, sigma, backend="cpp", n_threads=8):
    fg = distances[0]
    boundary_map = fg.max() - fg
    boundary_map /= boundary_map.max()
    return run_multicut(
        boundary_map, distances[2:],
        beta=beta,
        density_threshold=density_threshold,
        sigma=sigma,
        backend=backend,
        n_threads=n_threads,
    ).astype("uint32")


def _postprocess_em_blockwise(
    distances, beta, density_threshold, sigma, block_shape=(10, 512, 512), halo=(2, 32, 32), n_levels=1,
    backend="cpp",
):
    """Slice-wise oversegmentation + blockwise multicut for large EM volumes.

    Replicates the run_multicut pipeline but replaces global multicut with
    elf's blockwise_mc_impl, which solves the problem hierarchically over
    spatial blocks and avoids OOM on large volumes.

    Args:
        distances: (4, Z, Y, X) float32 model output.
        beta: Multicut boundary bias.
        density_threshold: Convergence-density threshold for slice-wise seeding.
        sigma: Gaussian sigma for smoothing the convergence-density map.
        block_shape: Block shape (Z, Y, X) for blockwise multicut.
        halo: Halo (Z, Y, X) added around each block for overlap.
        n_levels: Number of hierarchy levels in blockwise_mc_impl.
    """
    from concurrent import futures as cf

    import numpy as np
    from skimage.measure import label
    from skimage.segmentation import watershed

    from elf.segmentation.blockwise_mc_impl import blockwise_mc_impl
    from elf.segmentation.features import (
        compute_boundary_mean_and_length, compute_rag,
        compute_z_edge_mask, project_node_labels_to_pixels,
    )
    from elf.segmentation.multicut import compute_edge_costs, multicut_decomposition
    from micro_sam.v2.postprocessing import _compute_flow_density

    fg = distances[0]
    boundary_map = fg.max() - fg
    boundary_map /= boundary_map.max()
    dist_2d = distances[2:]

    n_slices = boundary_map.shape[0]
    overseg = np.zeros(boundary_map.shape, dtype="uint64")

    def _run_overseg(z):
        bd = boundary_map[z]
        dists = dist_2d[:, z]
        fg_mask = np.ones(bd.shape, dtype="bool")
        density = _compute_flow_density(
            dists, fg_mask, n_iter=50, dt=0.5, sigma=sigma, verbose=False, backend=backend, n_threads=1,
        )
        seeds = label(density > density_threshold)
        wsz = watershed(bd, markers=seeds)
        overseg[z] = wsz
        return int(wsz.max())

    n_threads = 8
    with cf.ThreadPoolExecutor(n_threads) as tp:
        offsets = list(tqdm(tp.map(_run_overseg, range(n_slices)), total=n_slices, desc="Slice-wise oversegmentation"))

    offsets = np.array(offsets, dtype="uint64")
    offsets = np.roll(offsets, 1)
    offsets[0] = 0
    overseg += np.cumsum(offsets)[:, None, None]

    print("Building RAG ...")
    rag = compute_rag(overseg, n_threads=n_threads)
    feats = compute_boundary_mean_and_length(rag, boundary_map)
    z_edges = compute_z_edge_mask(rag, overseg)
    costs = compute_edge_costs(
        feats[:, 0], edge_sizes=feats[:, 1],
        weighting_scheme="xyz", z_edge_mask=z_edges, beta=beta,
    )

    def _solver(graph, edge_costs):
        return multicut_decomposition(graph, edge_costs, internal_solver="kernighan-lin")

    print(f"Running blockwise multicut (block_shape={block_shape}, n_levels={n_levels}) ...")
    node_labels = blockwise_mc_impl(
        rag, costs, overseg,
        internal_solver=_solver,
        block_shape=list(block_shape),
        n_threads=n_threads,
        n_levels=n_levels,
        halo=list(halo),
    )
    seg = project_node_labels_to_pixels(rag, node_labels)
    return seg.astype("uint32")


def run_grid_search(dataset_name, model_name, output_dir):
    """Run hyperparameter grid search for one (dataset, model) pair using the cpp backend.

    Runs model inference once, sweeps all postprocessing combos, and saves
    mSA and wall-clock time per combo to CSV.

    Args:
        dataset_name: one of DATASETS.
        model_name: 'automatic' or 'joint'.
        output_dir: root directory to write CSV result files.
    """
    save_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{model_name}.csv")

    if os.path.exists(csv_path):
        print(f"CSV already exists at {csv_path!r}, skipping.")
        df = pd.read_csv(csv_path)
    else:
        crop = GRID_SEARCH_CROP[dataset_name]
        print(f"Loading model and generating predictions for {dataset_name}/{model_name} ...")
        model = load_unisam2_model(UNISAM2_CHECKPOINT, DEVICE)
        raw, distances, labels, _ = _generate_live_predictions(dataset_name, model, crop)

        is_em = dataset_name in EM_DATASETS
        grid = EM_GRID if is_em else LM_GRID
        keys = list(grid.keys())
        combos = list(itertools.product(*[grid[k] for k in keys]))
        print(f"  {len(combos)} combinations.")

        rows = []
        for combo in tqdm(combos, desc=f"{dataset_name}/{model_name}"):
            params = dict(zip(keys, combo))
            t0 = time.perf_counter()
            if is_em:
                seg = _postprocess_em(distances, backend="cpp", **params)
            else:
                seg = _postprocess_lm(distances, dataset_name, backend="cpp", **params)
            elapsed = time.perf_counter() - t0
            msa = mean_segmentation_accuracy(seg, labels)
            rows.append({**params, "msa": msa, "time_s": elapsed})

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"  Saved {csv_path}.")

    best = df.sort_values("msa", ascending=False).iloc[0]
    best_params = {k: best[k] for k in df.columns if k not in ("msa", "time_s")}
    print(f"  Best params: {best_params} -> mSA={best['msa']:.4f}, mean time/combo={df['time_s'].mean():.2f}s")


def run_best_on_full_volume(dataset_name, model_name, output_dir, source_csv=None, force=False):
    """Run postprocessing with best grid-search params on the full uncropped first test sample.

    Args:
        dataset_name: one of DATASETS.
        model_name: 'automatic' or 'joint'.
        output_dir: root directory containing CSV results and where H5 will be written.
        source_csv: path to a CSV from another dataset to copy params from.
            If None, uses the dataset's own CSV.
        force: overwrite existing H5 and re-run even if already completed.
    """
    csv_path = source_csv if source_csv is not None else os.path.join(output_dir, dataset_name, f"{model_name}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Run the grid search first: {csv_path!r}")

    h5_path = os.path.join(output_dir, dataset_name, f"{model_name}_best_full.h5")
    if os.path.exists(h5_path) and not force:
        print(f"Already exists: {h5_path!r}")
        return

    df = pd.read_csv(csv_path)
    best = df.sort_values("msa", ascending=False).iloc[0]
    best_params = {k: best[k] for k in df.columns if k not in ("msa", "time_s")}
    print(f"Best params for {dataset_name}/{model_name}: {best_params} (mSA={best['msa']:.4f})")

    print("Loading model and generating full-volume predictions ...")
    model = load_unisam2_model(UNISAM2_CHECKPOINT, DEVICE)
    raw, distances, labels, _ = _generate_live_predictions(dataset_name, model, crop_shape=None)
    print(f"  Volume shape: {raw.shape}")

    is_em = dataset_name in EM_DATASETS
    print("Running postprocessing ...")
    t0 = time.perf_counter()
    if dataset_name == "cremi_padded":
        # Full raw is 200x3072x3072; use blockwise multicut on the full volume.
        # seg_full is saved; the labeled-region crop is used only for scoring.
        seg_full = _postprocess_em_blockwise(distances, backend="cpp", **best_params)
        seg = seg_full[CREMI_PADDED_LABEL_ROI]
    elif dataset_name in ("liconn", "microns_minnie65"):
        seg = _postprocess_em_blockwise(distances, backend="cpp", **best_params)
    elif is_em:
        seg = _postprocess_em(distances, backend="cpp", **best_params)
    else:
        seg = _postprocess_lm(distances, dataset_name, backend="cpp", **best_params)
    elapsed = time.perf_counter() - t0

    msa = mean_segmentation_accuracy(seg, labels)
    if is_em:
        from elf.evaluation import cremi_score
        vi_split, vi_merge, _, cremi = cremi_score(seg, labels)
        print(
            f"  mSA={msa:.4f}, VI-split={vi_split:.4f}, VI-merge={vi_merge:.4f}, "
            f"CREMI={cremi:.4f}, time={elapsed:.1f}s"
        )
    else:
        print(f"  mSA={msa:.4f}, time={elapsed:.1f}s")

    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    save_seg = seg_full if dataset_name == "cremi_padded" else seg
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("labels", data=labels, compression="gzip")
        f.create_dataset("predicted_instances", data=save_seg, compression="gzip")
    print(f"Saved {h5_path}.")


def run_with_fixed_params(dataset_name, model_name, params, predictions_root, output_dir, block_z=None):
    """Apply fixed postprocessing params to a full volume without grid search or mSA evaluation.

    Args:
        dataset_name: one of DATASETS.
        model_name: 'automatic' or 'joint'.
        params: dict of postprocessing parameters.
        predictions_root: root directory of H5 prediction files.
        output_dir: root directory to write the output H5.
        block_z: if set, use blockwise multicut with this Z block size (EM only).
    """
    save_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    h5_path = os.path.join(save_dir, f"{model_name}_segmentation.h5")

    if os.path.exists(h5_path):
        print(f"Already exists: {h5_path!r}")
        return

    print(f"Loading predictions for {dataset_name}/{model_name} ...")
    raw, distances, labels = _load_predictions(dataset_name, model_name, predictions_root, crop_shape=None)
    print(f"  Volume shape: {raw.shape}")

    is_em = dataset_name in EM_DATASETS
    print(f"Running postprocessing with params: {params}")
    if is_em and block_z is not None:
        seg = _postprocess_em_blockwise(distances, block_shape=(block_z, 512, 512), **params)
    elif is_em:
        seg = _postprocess_em(distances, **params)
    else:
        seg = _postprocess_lm(distances, dataset_name, **params)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("labels", data=labels, compression="gzip")
        f.create_dataset("predicted_instances", data=seg, compression="gzip")
    print(f"Saved {h5_path}.")


def main():
    parser = argparse.ArgumentParser(
        description="Grid search over postprocessing hyperparameters for UniSAM2 predictions."
    )
    parser.add_argument(
        "-d", "--dataset", required=True, choices=DATASETS,
        help="Dataset to run grid search on.",
    )
    parser.add_argument(
        "-m", "--model", required=True, choices=["automatic", "joint"],
        help="Model variant.",
    )
    parser.add_argument(
        "-p", "--predictions_dir", type=str, default=PREDICTIONS_ROOT,
        help="Root directory of H5 prediction files (used by --copy_params_from / --full_volume).",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default=OUTPUT_ROOT,
        help="Root directory to write CSV result files.",
    )
    parser.add_argument(
        "--full_volume", action="store_true",
        help="Run best params on the full uncropped volume instead of grid search.",
    )
    parser.add_argument(
        "--copy_params_from", type=str, default=None,
        help="Apply best params from this dataset's CSV without grid search or mSA evaluation.",
    )
    parser.add_argument(
        "--block_z", type=int, default=None,
        help="Use blockwise multicut with this Z block size (requires --copy_params_from, EM only).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing H5 and re-run full-volume evaluation even if already completed.",
    )
    args = parser.parse_args()

    if args.full_volume and args.copy_params_from:
        source_csv = os.path.join(args.output_dir, args.copy_params_from, f"{args.model}.csv")
        if not os.path.exists(source_csv):
            raise FileNotFoundError(f"Source CSV not found: {source_csv!r}")
        run_best_on_full_volume(args.dataset, args.model, args.output_dir, source_csv=source_csv, force=args.force)
    elif args.copy_params_from:
        source_csv = os.path.join(args.output_dir, args.copy_params_from, f"{args.model}.csv")
        if not os.path.exists(source_csv):
            raise FileNotFoundError(f"Source CSV not found: {source_csv!r}")
        df = pd.read_csv(source_csv)
        best = df.sort_values("msa", ascending=False).iloc[0]
        params = {k: best[k] for k in df.columns if k not in ("msa", "time_s")}
        print(f"Using params from {args.copy_params_from}: {params}")
        run_with_fixed_params(
            args.dataset, args.model, params, args.predictions_dir, args.output_dir,
            block_z=args.block_z,
        )
    elif args.full_volume:
        run_best_on_full_volume(args.dataset, args.model, args.output_dir, force=args.force)
    else:
        run_grid_search(args.dataset, args.model, args.output_dir)


if __name__ == "__main__":
    main()
