"""Post-processing for UniSAM2 automatic segmentation predictions.

Converts the model's raw outputs (foreground probability + directed distance channels)
into instance segmentation maps. Two strategies are provided:

- ``flow_instance_segmentation``: CellPose-style flow following. Suitable for LM data
  (2D and 3D).
- ``run_multicut``: Slice-wise oversegmentation + graph multicut. Suitable for EM data
  with large, densely-packed objects (3D only).
"""

from concurrent import futures
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm

from bioimage_cpp.flow import compute_flow_density
from bioimage_cpp.segmentation import label, watershed

from .util import DEFAULT_MODEL

# Per (model_type, mode) defaults from the registry parameter search: the best-average-rank
# combination across every dataset that shares that mode's grid, computed separately for each of the
# 4 registry backbones.
# 'boundary_magnitude_max' is the instance filter of `flow_instance_segmentation`; None keeps it off.
# 'sparse_volume' holds the keys whose default differs for a volume (a size floor counts voxels, not
# pixels); it is layered over 'sparse' by `default_postprocessing(..., ndim=3)`.
#
# The hvit_t entry is the result of the 2026-09 AIS optimization on the joint/v4 geodesic checkpoint
# (finetuning/v2/evaluation/optimization/notes/AIS_V4_OPTIMIZATION.md): against the registry values
# (min_size 100, sigma 0.5, no filter) it gains +2.4 % balanced mSA on eleven 2d development datasets
# (9 up, worst -0.8 %), +4.3 % on the 2d holdout and +22 % on the 3d LM crops, with the wider density
# smoothing merging the jittering sinks of large cells and the boundary filter removing false regions.
DEFAULT_POSTPROCESSING = {
    "hvit_t": {
        "sparse": {
            "foreground_threshold": 0.5, "density_threshold": 10.0, "min_size": 50,
            "sigma": 1.0, "n_iter": 50, "dt": 0.5, "foreground_weight": 0.5, "boundary_magnitude_max": 0.4,
        },
        "sparse_volume": {"min_size": 200, "foreground_threshold": 0.6},
        "dense": {"beta": 0.5, "density_threshold": 5.0, "sigma": 0.5, "n_iter": 50, "dt": 0.5},
    },
    "hvit_s": {
        "sparse": {
            "foreground_threshold": 0.5, "density_threshold": 20.0, "min_size": 100,
            "sigma": 0.25, "n_iter": 50, "dt": 0.5, "foreground_weight": 0.75, "boundary_magnitude_max": None,
        },
        "sparse_volume": {},
        "dense": {"beta": 0.5, "density_threshold": 3.0, "sigma": 0.5, "n_iter": 25, "dt": 0.5},
    },
    "hvit_b": {
        "sparse": {
            "foreground_threshold": 0.5, "density_threshold": 20.0, "min_size": 100,
            "sigma": 0.25, "n_iter": 50, "dt": 0.5, "foreground_weight": 0.65, "boundary_magnitude_max": None,
        },
        "sparse_volume": {},
        "dense": {"beta": 0.5, "density_threshold": 5.0, "sigma": 0.5, "n_iter": 50, "dt": 0.5},
    },
    "hvit_l": {
        "sparse": {
            "foreground_threshold": 0.4, "density_threshold": 10.0, "min_size": 50,
            "sigma": 0.5, "n_iter": 50, "dt": 0.25, "foreground_weight": 0.65, "boundary_magnitude_max": None,
        },
        "sparse_volume": {},
        "dense": {"beta": 0.5, "density_threshold": 5.0, "sigma": 1.0, "n_iter": 50, "dt": 0.5},
    },
}


def default_postprocessing(model_type: str = DEFAULT_MODEL, mode: str = "sparse", ndim: int = 2) -> dict:
    """The default postprocessing parameters for one model type, mode and dimensionality.

    Args:
        model_type: The SAM2 backbone, e.g. 'hvit_t', or a finetuned model built on one, e.g.
            'hvit_t_cells' (only the backbone prefix is used to look up the table). Must be one of the
            4 registry backbones.
        mode: 'sparse' (`flow_instance_segmentation`) or 'dense' (`run_multicut`).
        ndim: The number of spatial dimensions of the data, 2 or 3. A volume takes the
            '<mode>_volume' overrides of the table on top of the mode's defaults.

    Returns:
        The default parameter dict for that model type, mode and dimensionality.
    """
    backbone = model_type[:6]
    if backbone not in DEFAULT_POSTPROCESSING:
        raise ValueError(
            f"No default postprocessing parameters for model type '{model_type}'. "
            f"Choose one built on a backbone in {sorted(DEFAULT_POSTPROCESSING)}."
        )
    table = DEFAULT_POSTPROCESSING[backbone]
    defaults = dict(table[mode])
    if ndim == 3:
        defaults.update(table.get(f"{mode}_volume", {}))
    return defaults


def _compute_flow_density(
    directed_distances: np.ndarray,
    fg_mask: np.ndarray,
    n_iter: int = 100,
    dt: float = 0.5,
    sigma: float = 1.0,
    spacing: Optional[Tuple] = None,
    n_threads: int = 8,
) -> np.ndarray:
    """Integrate a flow field and return a convergence-density map.

    Pixels in the foreground are advected along the (negated) directed-distance
    field. The density of where they converge gives one peak per object, which is
    used downstream as seeds for a seeded watershed.

    Args:
        directed_distances: Distance channels stacked along axis 0,
            shape (ndim, *spatial).
        fg_mask: Boolean foreground mask, shape (*spatial).
        n_iter: Number of integration steps.
        dt: Step size per integration step.
        sigma: Gaussian smoothing sigma applied to the density map.
        spacing: Anisotropic voxel spacing for 3D data, e.g. (4, 1, 1).
            Used for physically-isotropic Gaussian smoothing.
        n_threads: Number of threads for ``bioimage_cpp.flow.compute_flow_density``.

    Returns:
        Smoothed convergence-density map, same spatial shape as fg_mask.
    """
    return compute_flow_density(
        -directed_distances, fg_mask,
        n_iter=n_iter, dt=dt, sigma=sigma, spacing=spacing, number_of_threads=n_threads,
    )


def watershed_heightmap(
    foreground: np.ndarray, directed_distances: np.ndarray, foreground_weight: float
) -> np.ndarray:
    """Build the heightmap that separates touching objects in the seeded watershed.

    The inverted distance magnitude puts object boundaries at its local minima, which is a weak edge
    signal. The foreground probability carries the actual object edge, so mixing the two in sharpens
    the boundaries. Both terms are scaled to [0, 1] to make the weight meaningful.

    Args:
        foreground: Foreground probability map, shape (*spatial).
        directed_distances: Distance channels stacked along axis 0, shape (ndim, *spatial).
        foreground_weight: Weight of the foreground term. 0 uses the distances only.

    Returns:
        The watershed heightmap, float32 array of the same spatial shape as foreground.
    """
    distances = np.linalg.norm(directed_distances, axis=0)
    distances = distances.max() - distances
    distances /= (distances.max() + 1e-9)
    hmap = foreground_weight * (1.0 - np.clip(foreground, 0, 1)) + (1.0 - foreground_weight) * distances
    return np.ascontiguousarray(hmap, dtype="float32")


def drop_instances_without_boundary_dip(
    segmentation: np.ndarray, directed_distances: np.ndarray, max_median: float
) -> np.ndarray:
    """Drop the instances whose boundary shows no dip of the distance magnitude.

    The magnitude of the directed distances falls to (almost) zero along the boundary of every object
    the decoder recognised, because the distance to the object's boundary is what it predicts. A false
    foreground region carries no such structure: its boundary runs through the decoder's background
    output (magnitude about one) or through the interior of a field that belongs to something else. An
    instance whose median boundary magnitude exceeds 'max_median' is therefore removed. The rule is
    label-free and scale-free, and a real object passes it at any size.

    Args:
        segmentation: The instance segmentation, shape (*spatial).
        directed_distances: Distance channels stacked along axis 0, shape (ndim, *spatial).
        max_median: Instances whose median boundary magnitude exceeds this value are dropped.

    Returns:
        The filtered segmentation, same dtype and shape.
    """
    # The inner boundary: instance pixels with an axis neighbour of another label (or background).
    boundary = np.zeros(segmentation.shape, dtype=bool)
    for axis in range(segmentation.ndim):
        lower = [slice(None)] * segmentation.ndim
        upper = [slice(None)] * segmentation.ndim
        lower[axis], upper[axis] = slice(None, -1), slice(1, None)
        differs = segmentation[tuple(lower)] != segmentation[tuple(upper)]
        boundary[tuple(lower)] |= differs
        boundary[tuple(upper)] |= differs
    boundary &= segmentation != 0
    if not boundary.any():
        return segmentation
    labels = segmentation[boundary]
    values = np.linalg.norm(directed_distances[(slice(None),) + np.nonzero(boundary)], axis=0)
    # One sort over the boundary pixels gives every instance's median (the mean of the two middle values
    # for an even count, like `scipy.ndimage.median`).
    order = np.lexsort((values, labels))
    labels, values = labels[order], values[order]
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    counts = np.diff(np.r_[starts, len(labels)])
    upper_middle = values[starts + counts // 2]
    lower_middle = values[starts + (counts - 1) // 2]
    medians = 0.5 * (upper_middle + lower_middle)
    drop = labels[starts][medians > max_median]
    if drop.size == 0:
        return segmentation
    return np.where(np.isin(segmentation, drop), 0, segmentation).astype(segmentation.dtype)


def flow_instance_segmentation(
    foreground: np.ndarray,
    directed_distances: np.ndarray,
    model_type: str = DEFAULT_MODEL,
    foreground_threshold: Optional[float] = None,
    n_iter: Optional[int] = None,
    dt: Optional[float] = None,
    sigma: Optional[float] = None,
    spacing: Optional[Tuple] = None,
    density_threshold: Optional[float] = None,
    min_size: Optional[int] = None,
    foreground_weight: Optional[float] = None,
    n_threads: int = 8,
    boundary_magnitude_max: Optional[float] = None,
) -> np.ndarray:
    """Instance segmentation from directed-distance predictions via flow following.

    Integrates a CellPose-style flow field derived from the directed distances,
    extracts convergence-density seeds, and finalises instances with a seeded
    watershed. Works for both 2D and 3D inputs.

    If 3 distance channels are supplied for a 2D foreground map the leading
    z-channel is automatically dropped, so you can always pass ``out[1:]``
    regardless of dimensionality.

    Args:
        foreground: Foreground probability map, shape (Y, X) or (Z, Y, X).
        directed_distances: Distance channels stacked along axis 0,
            shape (ndim, *spatial) or (3, *spatial) for 2D input.
        model_type: The SAM2 backbone the predictions came from, e.g. 'hvit_t'. Selects the default
            for any of the tunable parameters below left as None, see `default_postprocessing`.
        foreground_threshold: Foreground binarisation threshold.
        n_iter: Number of flow-integration steps.
        dt: Integration step size.
        sigma: Gaussian sigma for smoothing the convergence-density map.
        spacing: Anisotropic voxel spacing for 3D inputs, e.g. (4, 1, 1).
        density_threshold: Convergence-density threshold for seed extraction.
        min_size: Minimum object size (pixels/voxels) to keep.
        foreground_weight: Weight of the foreground term in the watershed heightmap, see
            `watershed_heightmap`.
        n_threads: Number of threads for the flow computation.
        boundary_magnitude_max: Drop instances whose median boundary magnitude exceeds this value, see
            `drop_instances_without_boundary_dip`. None takes the per-model default, which may itself be
            None (no filtering); pass ``float("inf")`` to disable a default filter explicitly.

    Returns:
        Instance segmentation, uint32 array, same spatial shape as foreground.
    """
    defaults = default_postprocessing(model_type, "sparse", ndim=foreground.ndim)
    if foreground_threshold is None:
        foreground_threshold = defaults["foreground_threshold"]
    if boundary_magnitude_max is None:
        boundary_magnitude_max = defaults.get("boundary_magnitude_max")
    if n_iter is None:
        n_iter = defaults["n_iter"]
    if dt is None:
        dt = defaults["dt"]
    if sigma is None:
        sigma = defaults["sigma"]
    if density_threshold is None:
        density_threshold = defaults["density_threshold"]
    if min_size is None:
        min_size = defaults["min_size"]
    if foreground_weight is None:
        foreground_weight = defaults["foreground_weight"]

    ndim = foreground.ndim
    if directed_distances.shape[0] > ndim:
        directed_distances = directed_distances[-ndim:]
    assert directed_distances.shape[0] == ndim, (
        f"Expected {ndim} distance channels, got {directed_distances.shape[0]}."
    )

    fg_mask = foreground > foreground_threshold

    density = _compute_flow_density(
        directed_distances, fg_mask, n_iter=n_iter, dt=dt, sigma=sigma, spacing=spacing, n_threads=n_threads,
    )

    seeds = label(density > density_threshold)
    hmap = watershed_heightmap(foreground, directed_distances, foreground_weight)
    seg = watershed(hmap, markers=seeds, mask=fg_mask)

    if min_size > 0:
        ids, sizes = np.unique(seg, return_counts=True)
        discard = ids[(sizes < min_size) & (ids > 0)]
        seg[np.isin(seg, discard)] = 0
        seg = watershed(hmap, markers=seg, mask=fg_mask)

    # After the size filter, so that a dropped region is not refilled by its neighbours.
    if boundary_magnitude_max is not None and np.isfinite(boundary_magnitude_max):
        seg = drop_instances_without_boundary_dip(seg, directed_distances, boundary_magnitude_max)

    return seg.astype("uint32")


def run_multicut(
    boundary_map: np.ndarray,
    distances: np.ndarray,
    model_type: str = DEFAULT_MODEL,
    beta: Optional[float] = None,
    density_threshold: Optional[float] = None,
    n_iter: Optional[int] = None,
    dt: Optional[float] = None,
    sigma: Optional[float] = None,
    n_threads: int = 8,
) -> np.ndarray:
    """Instance segmentation for 3D EM data via slice-wise oversegmentation + multicut.

    For each z-slice a convergence-density seeded watershed produces an
    oversegmentation. A region-adjacency graph is then lifted across all
    slices and a multicut optimisation yields the final 3D instances.

    Args:
        boundary_map: Boundary probability map, shape (Z, Y, X).
            Typically ``1 - foreground`` or ``fg.max() - fg``.
        distances: In-plane distance channels (ydist, xdist), shape (2, Z, Y, X).
        model_type: The SAM2 backbone the predictions came from, e.g. 'hvit_t'. Selects the default
            for any of the tunable parameters below left as None, see `default_postprocessing`.
        beta: Multicut boundary bias; higher values favour more merging.
        density_threshold: Convergence-density threshold for seed extraction
            in the slice-wise oversegmentation.
        n_iter: Flow integration steps for the oversegmentation seeding.
        dt: Flow integration step size.
        sigma: Gaussian sigma for smoothing the convergence-density map.
        n_threads: Number of threads for the parallel slice-wise oversegmentation.

    Returns:
        Instance segmentation, uint64 array, shape (Z, Y, X).
    """
    defaults = default_postprocessing(model_type, "dense")
    if beta is None:
        beta = defaults["beta"]
    if density_threshold is None:
        density_threshold = defaults["density_threshold"]
    if n_iter is None:
        n_iter = defaults["n_iter"]
    if dt is None:
        dt = defaults["dt"]
    if sigma is None:
        sigma = defaults["sigma"]

    from elf.segmentation.features import (
        compute_rag, compute_boundary_mean_and_length,
        compute_z_edge_mask, project_node_labels_to_pixels,
    )
    from elf.segmentation.multicut import compute_edge_costs, multicut_decomposition

    n_slices = boundary_map.shape[0]
    overseg = np.zeros(boundary_map.shape, dtype="uint64")

    def _run_overseg(z):
        bd = boundary_map[z]
        dists = distances[:, z]
        fg_mask = np.ones(bd.shape, dtype="bool")
        # 1 thread per slice: the ThreadPoolExecutor above already parallelizes across slices, so a
        # higher value here would oversubscribe instead of adding throughput.
        density = _compute_flow_density(dists, fg_mask, n_iter=n_iter, dt=dt, sigma=sigma, n_threads=1)
        seeds = label(density > density_threshold)
        # watershed requires a float heightmap. Boundary maps are usually float already.
        bd = bd if np.issubdtype(bd.dtype, np.floating) else bd.astype("float32")
        wsz = watershed(bd, markers=seeds)
        overseg[z] = wsz
        return int(wsz.max())

    with futures.ThreadPoolExecutor(n_threads) as tp:
        offsets = list(tqdm(
            tp.map(_run_overseg, range(n_slices)),
            total=n_slices, desc="Slice-wise oversegmentation",
        ))

    offsets = np.array(offsets, dtype="uint64")
    offsets = np.roll(offsets, 1)
    offsets[0] = 0
    overseg += np.cumsum(offsets)[:, None, None]

    rag = compute_rag(overseg)
    if rag.numberOfEdges == 0:  # A single region leaves nothing to merge.
        return overseg.astype("uint64")

    feats = compute_boundary_mean_and_length(rag, overseg, boundary_map)
    z_edges = None if n_slices == 1 else compute_z_edge_mask(rag, overseg)
    # 'xyz' weights in-plane and z edges as separate populations, so it needs both to be present.
    if z_edges is None or z_edges.all() or not z_edges.any():
        costs = compute_edge_costs(feats[:, 0], edge_sizes=feats[:, 1], weighting_scheme="all", beta=beta)
    else:
        costs = compute_edge_costs(
            feats[:, 0], edge_sizes=feats[:, 1],
            weighting_scheme="xyz", z_edge_mask=z_edges, beta=beta,
        )
    node_labels = multicut_decomposition(rag, costs)
    seg = project_node_labels_to_pixels(rag, overseg, node_labels)

    return seg.astype("uint64")
