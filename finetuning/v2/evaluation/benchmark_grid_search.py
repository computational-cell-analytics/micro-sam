"""Benchmark the lazy postprocessing caching in grid_search_automatic_cells.

For one representative image per mode, runs the full parameter grid twice - once with the lazy caching
(sparse: cache the flow density per (fg, sigma, n_iter, dt); dense: cache the oversegmentation + RAG
per (density, sigma, n_iter, dt)) and once with the plain per-combo path - and reports the wall-clock
time, speedup, and the maximum mSA difference between the two (which should be ~0, confirming the
caching does not change results).

Usage:
    python benchmark_grid_search.py                 # both modes
    python benchmark_grid_search.py --mode sparse
    python benchmark_grid_search.py --mode dense --em_crop 32 512 512 --n_threads 4
"""

import os
import sys
import time
import argparse
import itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grid_search_automatic_cells as gs  # noqa


def benchmark_mode(mode, model, device, em_crop, n_threads, backend):
    """Time the cached vs uncached sweep for one mode on a single image and check equivalence."""
    if mode == "sparse":
        track_cfg = gs.TRACKS["lm_cell"]
        items = gs.build_work_items(track_cfg, n_images=1, livecell_per_celltype=1)
    else:
        track_cfg = gs.TRACKS["em_neurons"]
        items = gs.build_work_items(track_cfg, n_images=1, livecell_per_celltype=1)
    if not items:
        print(f"No data available for mode '{mode}', skipping.")
        return

    ndim = track_cfg["ndim"]
    raw, labels = gs.load_sample(items[0], ndim, em_crop)
    prediction = gs.predict(model, raw, ndim=ndim, device=device)

    keys = list(track_cfg["grid"].keys())
    combos = list(itertools.product(*[track_cfg["grid"][k] for k in keys]))
    params_list = [dict(zip(keys, combo)) for combo in combos]
    print(f"\n=== {mode} ({items[0][0]}, {len(params_list)} combos, image shape {raw.shape}) ===")

    t0 = time.perf_counter()
    cached = gs.score_image(prediction, labels, mode, params_list, backend, use_flow_cache=True, n_threads=n_threads)
    t_cached = time.perf_counter() - t0

    t0 = time.perf_counter()
    plain = gs.score_image(prediction, labels, mode, params_list, backend, use_flow_cache=False, n_threads=n_threads)
    t_plain = time.perf_counter() - t0

    # score_image returns a per-combo metric dict; compare the primary criterion metric.
    criterion = track_cfg.get("criterion", "msa")
    diffs = [
        abs(c[criterion] - p[criterion]) for c, p in zip(cached, plain) if c is not None and p is not None
    ]
    max_diff = max(diffs) if diffs else float("nan")
    speedup = t_plain / t_cached if t_cached > 0 else float("nan")
    print(f"cached:  {t_cached:8.1f}s ({t_cached / len(params_list) * 1000:.1f} ms/combo)")
    print(f"plain:   {t_plain:8.1f}s ({t_plain / len(params_list) * 1000:.1f} ms/combo)")
    print(f"speedup: {speedup:8.2f}x")
    print(f"max mSA difference (cached vs plain): {max_diff:.2e}")
    print("EQUIVALENT" if max_diff < 1e-6 else ("CLOSE" if max_diff < 1e-3 else "MISMATCH - investigate"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", default="both", choices=["both", "sparse", "dense"], help="Which mode to benchmark.")
    parser.add_argument("--em_crop", type=int, nargs=3, default=[32, 512, 512], help="EM volume crop (Z Y X).")
    parser.add_argument("--n_threads", type=int, default=gs.POSTPROC_THREADS, help="Threads for postprocessing.")
    parser.add_argument("--backend", default="cpp", choices=["cpp", "python"], help="Flow computation backend.")
    parser.add_argument("-c", "--checkpoint_path", default=None, help="Custom checkpoint instead of registry model.")
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
    model = gs.load_model(device, checkpoint_path=args.checkpoint_path)

    modes = ["sparse", "dense"] if args.mode == "both" else [args.mode]
    for mode in modes:
        benchmark_mode(mode, model, device, args.em_crop, args.n_threads, args.backend)


if __name__ == "__main__":
    main()
