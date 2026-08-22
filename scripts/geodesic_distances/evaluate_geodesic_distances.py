"""Oracle comparison of the euclidean and geodesic distance transforms for AIS v2.

Builds the directed distance fields from the *ground truth* labels and runs the AIS v2
post-processing on them (``flow_instance_segmentation`` for sparse LM data,
``run_multicut`` for dense EM data). This gives the upper bound that a perfectly trained
model could reach with each representation, so the representations can be compared
without training anything.
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
from skimage.filters import gaussian

import napari

from elf.evaluation import mean_segmentation_accuracy, cremi_score

from micro_sam.v2.postprocessing import DEFAULT_POSTPROCESSING, flow_instance_segmentation, run_multicut

from common import (
    VARIANTS, VARIANT_LABELS, DENSITY_GRID, ITER_GRID, ITER_GRID_3D, SIGMA_GRID,
    compute_distance_variants, compute_slicewise_variants,
    foreground_target, load_cremi, load_dsb, load_gonuclear, load_livecell, load_snemi,
)


def segment(sample, directed, setting, n_threads, foreground_sigma=0.0):
    """Run the AIS v2 post-processing on distance fields derived from the ground truth."""
    labels, mode = sample["labels"], sample["mode"]
    foreground = foreground_target(labels, mode)
    # A trained model predicts a smooth probability map. A binary one leaves the multicut
    # watershed with flat plateaus, which has nothing to do with the distance representation.
    if foreground_sigma > 0:
        foreground = gaussian(foreground, sigma=foreground_sigma)
    params = {k: v for k, v in DEFAULT_POSTPROCESSING[mode].items() if k not in setting}
    params.update(setting)

    if mode == "dense":
        # The multicut path integrates the flow slice-wise, so it only uses the in-plane channels.
        return run_multicut(1.0 - foreground, directed[1:], n_threads=n_threads, **params)

    return flow_instance_segmentation(
        foreground, directed, spacing=sample["sampling"], n_threads=n_threads, **params
    )


def evaluate(segmentation, labels):
    """Compute the LM and EM metrics for one segmentation."""
    segmentation = np.asarray(segmentation).astype("uint32")
    msa, accuracies = mean_segmentation_accuracy(segmentation, labels, return_accuracies=True)
    vi_split, vi_merge, are, score = cremi_score(segmentation, labels, ignore_gt=[0])
    return {
        "mSA": msa, "SA50": accuracies[0], "VOI_split": vi_split, "VOI_merge": vi_merge,
        "ARE": are, "CREMI": score,
        "n_pred": int(len(np.unique(segmentation)) - 1), "n_gt": int(len(np.unique(labels)) - 1),
    }


def run_sample(sample, grid, n_threads, keep_variants, verbose, foreground_sigma=0.0, slicewise=False):
    """Segment and score one sample with every distance variant and every post-processing setting."""
    labels = sample["labels"]
    if slicewise:
        variants = compute_slicewise_variants(labels, sampling=sample["sampling"], verbose=verbose)
    else:
        variants, _, _ = compute_distance_variants(labels, sampling=sample["sampling"], verbose=verbose)

    rows = []
    for variant in VARIANTS:
        for setting in grid:
            segmentation = segment(sample, variants[variant], setting, n_threads, foreground_sigma)
            row = {"sample": sample["name"], "variant": variant, **setting}
            row.update(evaluate(segmentation, labels))
            rows.append(row)
            print(
                f"{sample['name']} [{VARIANT_LABELS[variant]} {setting}]: "
                f"mSA {row['mSA']:.4f}, CREMI {row['CREMI']:.4f}"
            )

    return rows, (variants if keep_variants else None)


def select_settings(table, grid, metric, maximize):
    """Pick the post-processing setting that optimizes the mean metric over all samples, per variant."""
    keys = list(grid[0])
    best = {}
    for variant in VARIANTS:
        per_setting = table[table["variant"] == variant].groupby(keys)[metric].mean()
        values = per_setting.idxmax() if maximize else per_setting.idxmin()
        values = values if isinstance(values, tuple) else (values,)
        best[variant] = dict(zip(keys, [v.item() if hasattr(v, "item") else v for v in values]))
    return best


def summarize(table, settings, columns, defaults):
    """Average the metrics over the samples, for the selected setting of each variant."""
    parts = []
    for variant in VARIANTS:
        mask = table["variant"] == variant
        for key, value in settings[variant].items():
            mask &= table[key] == value
        parts.append(table[mask])
    summary = pd.concat(parts).groupby("variant")[columns].mean().reindex(VARIANTS)
    for key in defaults:
        summary.insert(0, key, [settings[variant][key] for variant in VARIANTS])
    return summary.rename(index=VARIANT_LABELS)


def show(samples, all_segmentations, title):
    """@private"""
    viewer = napari.Viewer(title=title)
    for sample, segmentations in zip(samples, all_segmentations):
        scale = (1.0,) * sample["labels"].ndim if sample["sampling"] is None else sample["sampling"]
        name = sample["name"]
        viewer.add_image(sample["image"], name=f"{name}: raw", scale=scale, visible=False)
        viewer.add_labels(sample["labels"], name=f"{name}: ground truth", scale=scale, visible=False)
        for variant, segmentation in segmentations.items():
            viewer.add_labels(
                np.asarray(segmentation).astype("uint32"), name=f"{name}: {VARIANT_LABELS[variant]}",
                scale=scale, visible=False,
            )
    # Only the layers of the first sample start out visible, the rest would just stack on top.
    for layer in viewer.layers[:len(VARIANTS) + 2]:
        layer.visible = True
    napari.run()


def get_samples(args):
    """@private"""
    if args.dataset == "cremi":
        return load_cremi(
            args.cremi_root, args.cremi_samples, args.offset, args.shape, args.min_size, args.sampling
        )
    if args.dataset == "snemi":
        return load_snemi(args.snemi_root, args.snemi_sample, args.offset, args.shape, args.min_size, args.sampling)
    if args.dataset == "livecell":
        return load_livecell(args.livecell_root, args.cell_types, args.n_images, args.min_size)
    if args.dataset == "dsb":
        return load_dsb(args.dsb_root, args.n_images, args.min_size)
    return load_gonuclear(
        args.gonuclear_root, args.gonuclear_samples, args.gonuclear_shape, args.min_size, args.gonuclear_sampling
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset", choices=["cremi", "snemi", "livecell", "dsb", "gonuclear"], help="Dataset to evaluate."
    )
    parser.add_argument("--cremi_root", default="/home/anwai/data/cremi", help="The CREMI data folder.")
    parser.add_argument("--cremi_samples", nargs="+", default=["A"], help="The CREMI samples.")
    parser.add_argument("--offset", type=int, nargs=3, default=[40, 400, 400], help="The CREMI roi offset.")
    parser.add_argument("--shape", type=int, nargs=3, default=[25, 512, 512], help="The CREMI roi shape.")
    parser.add_argument("--sampling", type=float, nargs=3, default=[40.0, 4.0, 4.0], help="The CREMI voxel size.")
    parser.add_argument("--snemi_root", default="/home/anwai/data/snemi", help="The SNEMI data folder.")
    parser.add_argument("--snemi_sample", default="train", help="The SNEMI sample.")
    parser.add_argument("--livecell_root", default="/home/anwai/data/livecell", help="The LIVECell data folder.")
    parser.add_argument("--cell_types", nargs="+", default=["A172", "SHSY5Y"], help="The LIVECell cell types.")
    parser.add_argument("--dsb_root", default="/home/anwai/data/dsb", help="The DSB data folder.")
    parser.add_argument("--gonuclear_root", default="/home/anwai/data/gonuclear", help="The GoNuclear data folder.")
    parser.add_argument("--gonuclear_samples", type=int, nargs="+", default=None, help="The GoNuclear sample ids.")
    parser.add_argument("--gonuclear_shape", type=int, nargs=3, default=[64, 256, 256], help="GoNuclear crop shape.")
    parser.add_argument("--gonuclear_sampling", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="Voxel size.")
    parser.add_argument("--n_images", type=int, default=5, help="Number of 2d images per cell type / dataset.")
    parser.add_argument("--min_size", type=int, default=50, help="Objects below this size are discarded.")
    parser.add_argument("--n_threads", type=int, default=8, help="Threads for the flow integration.")
    parser.add_argument("--foreground_sigma", type=float, default=0.0, help="Smooth the ground truth foreground.")
    parser.add_argument("--slicewise", action="store_true", help="Solve the distances per z slice, for dense data.")
    parser.add_argument("--sweep", action="store_true", help="Sweep the post-processing settings per variant.")
    parser.add_argument("--settings_from", default=None, help="Re-use the settings selected by an earlier run.")
    parser.add_argument("--settings_metric", choices=["msa", "cremi"], default=None, help="Which selection to reuse.")
    parser.add_argument("--view", action="store_true", help="Show the segmentations in napari.")
    parser.add_argument("--result_path", default=None, help="Write the per sample results to this json file.")
    args = parser.parse_args()

    samples = get_samples(args)
    mode = samples[0]["mode"]
    defaults = {key: DEFAULT_POSTPROCESSING[mode][key] for key in ("n_iter", "density_threshold", "sigma")}
    if args.settings_from is not None:
        # Re-use the settings that an earlier sweep selected, so the run reproduces its table cheaply.
        with open(args.settings_from) as f:
            previous = json.load(f)
        if args.settings_metric is None:
            key = "best_cremi" if mode == "dense" else "best_msa"
        else:
            key = f"best_{args.settings_metric}"
        selected = previous[key]
        grid = [dict(setting) for setting in {tuple(sorted(s.items())) for s in selected.values()}]
    elif args.sweep:
        iter_grid = ITER_GRID_3D[mode] if samples[0]["labels"].ndim == 3 else ITER_GRID[mode]
        grid = [{"n_iter": n_iter, "density_threshold": threshold, "sigma": sigma}
                for n_iter in iter_grid for threshold in DENSITY_GRID[mode] for sigma in SIGMA_GRID[mode]]
    else:
        grid = [defaults]
    print(f"Evaluating {len(samples)} {args.dataset} samples over {len(grid)} settings.")

    rows, cached = [], []
    for sample in samples:
        sample_rows, variants = run_sample(
            sample, grid, args.n_threads, args.view, args.view, args.foreground_sigma, args.slicewise
        )
        rows.extend(sample_rows)
        cached.append(variants)

    table = pd.DataFrame(rows)
    columns = ["mSA", "SA50", "VOI_split", "VOI_merge", "ARE", "CREMI", "n_pred", "n_gt"]

    by_msa = select_settings(table, grid, "mSA", maximize=True)
    print(f"\n{args.dataset} ({len(samples)} samples), setting with the best mean mSA per variant:")
    print(summarize(table, by_msa, columns, defaults).round(4).to_string())

    by_cremi = select_settings(table, grid, "CREMI", maximize=False)
    print(f"\n{args.dataset}, setting with the best (lowest) mean CREMI score per variant:")
    print(summarize(table, by_cremi, columns, defaults).round(4).to_string())

    if args.sweep:
        mask = np.ones(len(table), dtype="bool")
        for key, value in defaults.items():
            mask &= (table[key] == value).to_numpy()
        default_summary = table[mask].groupby("variant")[columns].mean().reindex(VARIANTS)
        print(f"\n{args.dataset}, at the pipeline defaults {defaults}:")
        print(default_summary.rename(index=VARIANT_LABELS).round(4).to_string())

    best_settings = by_cremi if samples[0]["mode"] == "dense" else by_msa

    if args.result_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.result_path)), exist_ok=True)
        with open(args.result_path, "w") as f:
            json.dump(
                {"dataset": args.dataset, "best_msa": by_msa, "best_cremi": by_cremi, "rows": rows}, f, indent=2
            )
        print(f"Wrote the per sample results to {args.result_path}.")

    if args.view:
        all_segmentations = [
            {
                variant: segment(
                    sample, variants[variant], best_settings[variant], args.n_threads, args.foreground_sigma
                ) for variant in VARIANTS
            }
            for sample, variants in zip(samples, cached)
        ]
        show(samples, all_segmentations, f"{args.dataset}: AIS v2 from ground truth distances")


if __name__ == "__main__":
    main()
