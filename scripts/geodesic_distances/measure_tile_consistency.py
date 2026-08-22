"""Measure how much the distance targets change when the crop window moves.

The euclidean target is local: a pixel's value depends only on the membrane near it, so cropping
does not change it. The hybrid target references the object's center, which is global: an object
that a tile cuts gets the center of its *visible* part instead. That predicts seam artifacts in
tiled inference, and is the one property that could make the hybrid unusable in production.

For a fixed region of the image, the same fields are recomputed from many differently offset crops
that all contain the region, and each is compared against the field computed on the full image.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import (
    VARIANTS, VARIANT_LABELS, compute_distance_variants, to_consecutive_labels, load_dsb, load_livecell
)

EPS = 1e-7


def centered_region(shape, size):
    """The block in the middle of the image that every tile has to contain."""
    return tuple(slice((extent - size) // 2, (extent - size) // 2 + size) for extent in shape)


def tile_offsets(shape, tile, region, n_offsets):
    """Offsets of all tiles of the given size that fully contain the region."""
    ranges = []
    for axis, part in enumerate(region):
        low = max(part.stop - tile, 0)
        high = min(part.start, shape[axis] - tile)
        if high < low:
            raise ValueError(f"A tile of {tile} cannot contain the region along axis {axis}.")
        ranges.append(np.unique(np.linspace(low, high, n_offsets).astype(int)))
    return [(y, x) for y in ranges[0] for x in ranges[1]]


def inside_fraction_map(labels, tile_slices):
    """For every pixel, which fraction of its object lies inside the tile."""
    total = np.bincount(labels.ravel()).astype("float32")
    within = np.bincount(labels[tile_slices].ravel(), minlength=len(total)).astype("float32")
    fraction = np.divide(within, total, out=np.ones_like(total), where=total > 0)
    fraction[0] = 1.0
    return fraction[labels]


INSIDE_BINS = [(0.0, 0.5, "<50% inside"), (0.5, 0.75, "50-75%"), (0.75, 0.9, "75-90%"),
               (0.9, 0.99, "90-99%"), (0.99, 1.01, "whole")]


def region_fields(task):
    """@private"""
    index, offset, crop_labels, region_local = task
    crop_labels = to_consecutive_labels(crop_labels, min_size=0, apply_label=True)
    variants, _, _ = compute_distance_variants(crop_labels, sampling=None, verbose=False)
    extracted = {name: field[(slice(None),) + region_local].copy() for name, field in variants.items()}
    return index, offset, extracted


def angular_deviation(field, reference, min_magnitude):
    """Angle in degrees between two vector fields, and where both are long enough to have a direction.

    Vectors near a membrane (or at the center, for the pure center field) have almost no length, so
    their direction is numerical noise and would dominate the tail of the statistic.
    """
    field_norm = np.linalg.norm(field, axis=0, keepdims=True)
    reference_norm = np.linalg.norm(reference, axis=0, keepdims=True)
    cosine = ((field / (field_norm + EPS)) * (reference / (reference_norm + EPS))).sum(axis=0)
    usable = (field_norm[0] > min_magnitude) & (reference_norm[0] > min_magnitude)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))), usable


def build_tasks(samples, tile, region_size, n_offsets):
    """@private"""
    tasks, layout = [], {}
    for index, sample in enumerate(samples):
        labels = sample["labels"]
        region = centered_region(labels.shape, region_size)
        offsets = tile_offsets(labels.shape, tile, region, n_offsets)
        layout[index] = (labels, region, offsets)

        full_local = region
        tasks.append((index, None, labels, full_local))
        for offset in offsets:
            tile_slices = tuple(slice(o, o + tile) for o in offset)
            local = tuple(slice(part.start - o, part.stop - o) for part, o in zip(region, offset))
            tasks.append((index, offset, labels[tile_slices], local))
    return tasks, layout


def summarize(results, layout, tile, region_size, min_magnitude):
    """@private"""
    references = {index: fields for index, offset, fields in results if offset is None}

    rows = []
    for index, offset, fields in results:
        if offset is None:
            continue
        labels, region, _ = layout[index]
        tile_slices = tuple(slice(o, o + tile) for o in offset)
        inside = inside_fraction_map(labels, tile_slices)[region]
        foreground = labels[region] > 0

        for variant in VARIANTS:
            deviation, usable = angular_deviation(fields[variant], references[index][variant], min_magnitude)
            groups = [(name, (inside >= low) & (inside < high)) for low, high, name in INSIDE_BINS]
            for group, mask in groups:
                mask = mask & foreground & usable
                if not mask.any():
                    continue
                values = deviation[mask]
                rows.append({
                    "variant": variant, "inside": group, "n_pixels": int(mask.sum()),
                    "median_deg": float(np.median(values)),
                    "p90_deg": float(np.percentile(values, 90)),
                    "frac_over_30": float(np.mean(values > 30)),
                    "frac_over_90": float(np.mean(values > 90)),
                })

    table = pd.DataFrame(rows)
    weighted = table.groupby(["variant", "inside"]).apply(
        lambda g: pd.Series({
            column: np.average(g[column], weights=g["n_pixels"])
            for column in ["median_deg", "p90_deg", "frac_over_30", "frac_over_90"]
        }), include_groups=False
    )
    cut_fraction = table[table["inside"] != "whole"]["n_pixels"].sum() / max(table["n_pixels"].sum(), 1)
    print(
        f"Tile {tile}, region {region_size}: {100 * cut_fraction:.1f}% of the compared foreground "
        f"belongs to objects that the tile cuts."
    )
    order = [name for _, _, name in INSIDE_BINS]
    weighted = weighted.reindex(
        pd.MultiIndex.from_product([list(VARIANTS), order], names=["variant", "inside"])
    ).dropna(how="all")
    return weighted.rename(index=VARIANT_LABELS).round(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["livecell", "dsb"], help="Dataset to measure on.")
    parser.add_argument("--livecell_root", default="/home/anwai/data/livecell", help="The LIVECell data folder.")
    parser.add_argument("--cell_types", nargs="+", default=["A172", "SHSY5Y"], help="The LIVECell cell types.")
    parser.add_argument("--dsb_root", default="/home/anwai/data/dsb", help="The DSB data folder.")
    parser.add_argument("--n_images", type=int, default=4, help="Number of images per cell type / dataset.")
    parser.add_argument("--tile", type=int, default=256, help="The tile size.")
    parser.add_argument("--region", type=int, default=96, help="The region every tile has to contain.")
    parser.add_argument("--n_offsets", type=int, default=4, help="Offsets per axis, giving n_offsets^2 tiles.")
    parser.add_argument("--min_size", type=int, default=50, help="Objects below this size are discarded.")
    parser.add_argument("--min_magnitude", type=float, default=0.05, help="Ignore vectors shorter than this.")
    parser.add_argument("--n_workers", type=int, default=14, help="Parallel worker processes.")
    args = parser.parse_args()

    if args.dataset == "livecell":
        samples = load_livecell(args.livecell_root, args.cell_types, args.n_images, args.min_size)
    else:
        samples = load_dsb(args.dsb_root, args.n_images, args.min_size)

    tasks, layout = build_tasks(samples, args.tile, args.region, args.n_offsets)
    print(f"Measuring {len(samples)} {args.dataset} images over {len(tasks)} crops on {args.n_workers} workers.")

    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        results = list(tqdm(pool.map(region_fields, tasks), total=len(tasks), desc="Computing fields"))

    print(summarize(results, layout, args.tile, args.region, args.min_magnitude).to_string())


if __name__ == "__main__":
    main()
