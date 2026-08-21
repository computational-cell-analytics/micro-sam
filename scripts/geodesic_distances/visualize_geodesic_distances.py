"""Compare the current directed distance transform with geodesic distances in napari.

The automatic branch of the SAM2 training currently regresses the *directed euclidean*
distance to the nearest object boundary
(:class:`micro_sam.v2.transforms.labels.DirectedPerObjectBoundaryDistanceTransform`,
built on ``bioimage_cpp.distance.vector_difference_transform``).

This script computes the geodesic counterparts with
``bioimage_cpp.distance.geodesic_distance_field``, which constrains all paths to stay
inside the object, and shows both side by side for 2d LIVECell and 3d CREMI data.
"""

import time
import argparse

import h5py
import numpy as np

import napari

from common import (
    EPS, compute_geodesic_fields, compute_current_distances, compute_distance_variants,
    load_cremi, load_livecell
)


def report_differences(labels, fields, directed_boundary, current, sampling):
    """Quantify how much the geodesic fields deviate from the euclidean ones."""
    ndim = labels.ndim
    foreground = labels > 0
    voxel = 1.0 if sampling is None else min(sampling)

    excess = fields["center_distance_excess"][foreground]
    ratio = fields["center_distance_ratio"][foreground]
    print(f"Center distance: {100 * np.mean(excess > voxel):.1f}% of the foreground is more than one voxel "
          f"further along the object than in a straight line (median excess {np.median(excess):.2f}, "
          f"p95 {np.percentile(excess, 95):.2f}, max {excess.max():.2f}).")
    print(f"Detour factor: median {np.median(ratio):.2f}x, p95 {np.percentile(ratio, 95):.2f}x, "
          f"max {ratio.max():.2f}x; {100 * np.mean(ratio > 1.5):.1f}% of the foreground detours by more "
          f"than 1.5x and {100 * np.mean(ratio > 2.0):.2f}% by more than 2x.")

    # Compare the direction of the current transform with its geodesic counterpart.
    reference = current[1:][-ndim:][:, foreground]
    geodesic = directed_boundary[:, foreground]
    reference = reference / (np.linalg.norm(reference, axis=0, keepdims=True) + EPS)
    geodesic = geodesic / (np.linalg.norm(geodesic, axis=0, keepdims=True) + EPS)
    angles = np.degrees(np.arccos(np.clip((reference * geodesic).sum(axis=0), -1.0, 1.0)))
    print(f"Directed boundary distances: median angle {np.median(angles):.1f} degrees, "
          f"{100 * np.mean(angles > 30):.1f}% of the foreground deviates by more than 30 degrees.")


def add_distance_layers(viewer, fields, directed, current, ndim, scale):
    """@private"""
    axis_names = ("z", "y", "x")[-ndim:]

    # The current transform pads 2d inputs to 3d, so its channels are always [fg, d_z, d_y, d_x].
    for i, axis in enumerate(("z", "y", "x")):
        channel = current[i + 1]
        if not channel.any():
            continue
        viewer.add_image(
            channel, name=f"current euclidean: boundary d{axis}", colormap="PiYG",
            contrast_limits=(-1, 1), visible=False, scale=scale,
        )

    for name, field in zip(("boundary", "center flow", "hybrid"), directed):
        for i, axis in enumerate(axis_names):
            viewer.add_image(
                field[i], name=f"geodesic: {name} d{axis}", colormap="PiYG",
                contrast_limits=(-1, 1), visible=False, scale=scale,
            )

    scalar_layers = [
        ("euclidean_boundary_distance", "euclidean boundary distance", "viridis", False),
        ("geodesic_boundary_distance", "geodesic boundary distance", "viridis", False),
        ("euclidean_center_distance", "euclidean center distance", "magma", False),
        ("geodesic_center_distance", "geodesic center distance", "magma", True),
    ]
    for key, name, colormap, visible in scalar_layers:
        viewer.add_image(
            fields[key], name=name, colormap=colormap, contrast_limits=(0, 1), visible=visible, scale=scale,
        )

    excess = fields["center_distance_excess"]
    viewer.add_image(
        excess, name="center distance excess (geodesic - euclidean)", colormap="inferno",
        contrast_limits=(0, float(excess.max()) + EPS), visible=False, scale=scale,
    )
    ratio = fields["center_distance_ratio"]
    viewer.add_image(
        ratio, name="center distance detour factor", colormap="inferno",
        contrast_limits=(1, float(ratio.max()) + EPS), visible=False, scale=scale,
    )


def add_flow_vectors(viewer, directed, labels, step, scale, name):
    """Show a directed distance field as a napari vectors layer, for 2d data only."""
    coords = np.stack(np.meshgrid(*[np.arange(0, s, step) for s in labels.shape], indexing="ij"), axis=-1)
    coords = coords.reshape(-1, labels.ndim)
    coords = coords[labels[tuple(coords.T)] > 0]

    vectors = np.stack([directed[i][tuple(coords.T)] for i in range(labels.ndim)], axis=-1)
    viewer.add_vectors(
        np.stack([coords.astype("float32"), vectors * step], axis=1),
        name=name, edge_width=0.5, length=1.0, visible=False, scale=scale,
    )


def save_fields(save_path, image, labels, fields, directed, current):
    """Store the computed fields so they can be inspected without recomputing them."""
    names = ("geodesic_directed_boundary", "geodesic_directed_center", "geodesic_directed_hybrid")
    data = [
        ("raw", image), ("instances", labels), ("current_directed", current),
        *zip(names, directed), *fields.items(),
    ]
    with h5py.File(save_path, "a") as f:
        for name, values in data:
            if name in f:
                del f[name]
            f.create_dataset(name, data=values, compression="gzip")
    print(f"Saved the distance fields to {save_path}.")


def visualize(sample, title, vector_step=0, save_path=None):
    """@private"""
    image, labels, sampling = sample["image"], sample["labels"], sample["sampling"]

    start = time.perf_counter()
    fields, (directed_boundary, directed_center) = compute_geodesic_fields(labels, sampling=sampling)
    directed_hybrid = compute_distance_variants(labels, sampling=sampling, verbose=False)[0]["geodesic_hybrid"]
    directed = (directed_boundary, directed_center, directed_hybrid)
    geodesic_time = time.perf_counter() - start

    start = time.perf_counter()
    current = compute_current_distances(labels, sampling=sampling)
    print(f"Current transform: {time.perf_counter() - start:.2f} s, geodesic transform: {geodesic_time:.2f} s.")
    report_differences(labels, fields, directed_boundary, current, sampling)

    if save_path is not None:
        save_fields(save_path, image, labels, fields, directed, current)
        return

    scale = (1.0,) * labels.ndim if sampling is None else sampling
    viewer = napari.Viewer(title=title)
    viewer.add_image(image, name="raw", scale=scale)
    viewer.add_labels(labels, name="instances", scale=scale, visible=False)
    add_distance_layers(viewer, fields, directed, current, labels.ndim, scale)
    if vector_step > 0 and labels.ndim == 2:
        add_flow_vectors(viewer, directed_boundary, labels, vector_step, scale, "geodesic: boundary flow")
        add_flow_vectors(viewer, directed_center, labels, vector_step, scale, "geodesic: center flow")
        add_flow_vectors(viewer, directed_hybrid, labels, vector_step, scale, "geodesic: hybrid flow")

    napari.run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["livecell", "cremi"], help="Which dataset to visualize.")
    parser.add_argument("--livecell_root", default="/home/anwai/data/livecell", help="The LIVECell data folder.")
    parser.add_argument("--cell_type", default="A172", help="The LIVECell cell type.")
    parser.add_argument("--index", type=int, default=0, help="The index of the image to visualize.")
    parser.add_argument("--cremi_root", default="/home/anwai/data/cremi", help="The CREMI data folder.")
    parser.add_argument("--sample", default="A", help="The CREMI sample.")
    parser.add_argument("--offset", type=int, nargs=3, default=[40, 400, 400], help="The CREMI roi offset.")
    parser.add_argument("--shape", type=int, nargs=3, default=[16, 384, 384], help="The CREMI roi shape.")
    parser.add_argument("--sampling", type=float, nargs=3, default=[40.0, 4.0, 4.0], help="The CREMI voxel size.")
    parser.add_argument("--min_size", type=int, default=50, help="Objects below this size are discarded.")
    parser.add_argument("--vector_step", type=int, default=8, help="Subsampling for the 2d flow vectors.")
    parser.add_argument("--save_path", default=None, help="Save the fields to this h5 file instead of viewing them.")
    args = parser.parse_args()

    if args.dataset == "livecell":
        samples = load_livecell(args.livecell_root, [args.cell_type], args.index + 1, args.min_size)
        sample = samples[args.index]
        title = "LIVECell: geodesic vs. euclidean distances"
    else:
        samples = load_cremi(
            args.cremi_root, [args.sample], args.offset, args.shape, args.min_size, args.sampling
        )
        sample = samples[0]
        title = "CREMI: geodesic vs. euclidean distances"

    print(f"{sample['name']} {sample['labels'].shape}: {sample['labels'].max()} objects.")
    visualize(sample, title, args.vector_step, args.save_path)


if __name__ == "__main__":
    main()
