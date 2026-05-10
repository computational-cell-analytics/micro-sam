"""Grid search over postprocessing hyperparameters for UniSAM2 cached predictions.

Loads cached H5 prediction files, sweeps postprocessing parameters, evaluates
mSA against ground truth, and saves results to CSV.

LM datasets (nis3d, plantseg_root) use flow_instance_segmentation.
EM datasets (snemi) use run_multicut.

Results are stored one CSV per (dataset, model) pair:
    <output_dir>/<dataset>/<model>.csv

Usage:
    python grid_search_postprocessing.py -d snemi -m automatic
    python grid_search_postprocessing.py -d nis3d -m joint
    python grid_search_postprocessing.py -d plantseg_root -m automatic
"""

import argparse
import itertools
import os

import h5py
import pandas as pd
from tqdm import tqdm

from elf.evaluation import mean_segmentation_accuracy
from micro_sam.v2.postprocessing import flow_instance_segmentation, run_multicut


PREDICTIONS_ROOT = "/mnt/vast-nhr/home/archit/u12090/micro-sam/finetuning/v2/generalist/predictions"
OUTPUT_ROOT = "/mnt/vast-nhr/home/archit/u12090/micro-sam/finetuning/v2/generalist/grid_search_results"

DATASETS = [
    "snemi", "nis3d", "plantseg_root", "cremi", "humanneurons",
    "plantseg_ovules", "pnas_arabidopsis", "celegans_atlas", "mitoem",
]
EM_DATASETS = {"snemi", "cremi", "humanneurons", "mitoem"}

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


def _postprocess_lm(distances, dataset_name, foreground_threshold, density_threshold, min_size, sigma):
    return flow_instance_segmentation(
        foreground=distances[0],
        directed_distances=distances[1:],
        foreground_threshold=foreground_threshold,
        density_threshold=density_threshold,
        min_size=min_size,
        sigma=sigma,
    ).astype("uint32")


def _postprocess_em(distances, beta, density_threshold, sigma):
    fg = distances[0]
    boundary_map = fg.max() - fg
    boundary_map /= boundary_map.max()
    return run_multicut(
        boundary_map, distances[2:],
        beta=beta,
        density_threshold=density_threshold,
        sigma=sigma,
    ).astype("uint32")


def _postprocess_em_blockwise(
    distances, beta, density_threshold, sigma, block_shape=(10, 512, 512), halo=(2, 32, 32), n_levels=2
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
        density = _compute_flow_density(dists, fg_mask, n_iter=50, dt=0.5, sigma=sigma, verbose=False)
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
    rag = compute_rag(overseg)
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


def run_grid_search(dataset_name, model_name, predictions_root, output_dir):
    """Run hyperparameter grid search for one (dataset, model) pair.

    Args:
        dataset_name: one of DATASETS.
        model_name: 'automatic' or 'joint'.
        predictions_root: root directory of H5 prediction files.
        output_dir: root directory to write CSV result files.
    """
    save_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{model_name}.csv")
    h5_path = os.path.join(save_dir, f"{model_name}_best.h5")

    crop = GRID_SEARCH_CROP[dataset_name]
    print(f"Loading predictions for {dataset_name}/{model_name} (crop={crop}) ...")
    raw, distances, labels = _load_predictions(dataset_name, model_name, predictions_root, crop_shape=crop)

    is_em = dataset_name in EM_DATASETS

    if os.path.exists(csv_path):
        print(f"CSV already exists at {csv_path!r}, loading best params ...")
        df = pd.read_csv(csv_path)
    else:
        grid = EM_GRID if is_em else LM_GRID
        keys = list(grid.keys())
        combos = list(itertools.product(*[grid[k] for k in keys]))
        print(f"  {len(combos)} combinations.")

        rows = []
        for combo in tqdm(combos, desc=f"{dataset_name}/{model_name}"):
            params = dict(zip(keys, combo))
            if is_em:
                seg = _postprocess_em(distances, **params)
            else:
                seg = _postprocess_lm(distances, dataset_name, **params)
            msa = mean_segmentation_accuracy(seg, labels)
            rows.append({**params, "msa": msa})

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"  Saved {csv_path}.")
        print(df.sort_values("msa", ascending=False).head(5).to_string(index=False))

    best = df.sort_values("msa", ascending=False).iloc[0]
    best_params = best.drop("msa").to_dict()
    print(f"  Best params: {best_params} -> mSA={best['msa']:.4f}")

    if os.path.exists(h5_path):
        print(f"  Best segmentation already saved at {h5_path!r}.")
    else:
        if is_em:
            best_seg = _postprocess_em(distances, **best_params)
        else:
            best_seg = _postprocess_lm(distances, dataset_name, **best_params)
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")
            f.create_dataset("predicted_instances", data=best_seg, compression="gzip")
        print(f"  Saved best segmentation to {h5_path}.")


def run_best_on_full_volume(dataset_name, model_name, predictions_root, output_dir):
    """Run postprocessing with best grid-search params on the full uncropped volume.

    Args:
        dataset_name: one of DATASETS.
        model_name: 'automatic' or 'joint'.
        predictions_root: root directory of H5 prediction files.
        output_dir: root directory containing CSV results and where H5 will be written.
    """
    csv_path = os.path.join(output_dir, dataset_name, f"{model_name}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Run the grid search first: {csv_path!r}")

    h5_path = os.path.join(output_dir, dataset_name, f"{model_name}_best_full.h5")
    if os.path.exists(h5_path):
        print(f"Already exists: {h5_path!r}")
        return

    df = pd.read_csv(csv_path)
    best = df.sort_values("msa", ascending=False).iloc[0]
    best_params = best.drop("msa").to_dict()
    print(f"Best params for {dataset_name}/{model_name}: {best_params} (mSA={best['msa']:.4f})")

    print("Loading full volume ...")
    raw, distances, labels = _load_predictions(dataset_name, model_name, predictions_root, crop_shape=None)
    print(f"  Volume shape: {raw.shape}")

    is_em = dataset_name in EM_DATASETS
    print("Running postprocessing ...")
    if is_em:
        seg = _postprocess_em(distances, **best_params)
    else:
        seg = _postprocess_lm(distances, dataset_name, **best_params)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("labels", data=labels, compression="gzip")
        f.create_dataset("predicted_instances", data=seg, compression="gzip")
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
        help="Root directory of H5 prediction files.",
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
    args = parser.parse_args()

    if args.copy_params_from:
        source_csv = os.path.join(args.output_dir, args.copy_params_from, f"{args.model}.csv")
        if not os.path.exists(source_csv):
            raise FileNotFoundError(f"Source CSV not found: {source_csv!r}")
        df = pd.read_csv(source_csv)
        best = df.sort_values("msa", ascending=False).iloc[0]
        params = best.drop("msa").to_dict()
        print(f"Using params from {args.copy_params_from}: {params}")
        run_with_fixed_params(
            args.dataset, args.model, params, args.predictions_dir, args.output_dir,
            block_z=args.block_z,
        )
    elif args.full_volume:
        run_best_on_full_volume(args.dataset, args.model, args.predictions_dir, args.output_dir)
    else:
        run_grid_search(args.dataset, args.model, args.predictions_dir, args.output_dir)


if __name__ == "__main__":
    main()
