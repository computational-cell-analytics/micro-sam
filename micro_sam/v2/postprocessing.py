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
from skimage.filters import gaussian
from skimage.measure import label
from skimage.segmentation import watershed
from scipy.ndimage import map_coordinates
from tqdm import tqdm, trange


def _compute_flow_density(
    directed_distances: np.ndarray,
    fg_mask: np.ndarray,
    n_iter: int = 100,
    dt: float = 0.5,
    sigma: float = 1.0,
    spacing: Optional[Tuple] = None,
    verbose: bool = False,
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
        verbose: Show tqdm progress bar.

    Returns:
        Smoothed convergence-density map, same spatial shape as fg_mask.
    """
    shape, ndim = fg_mask.shape, fg_mask.ndim
    flow = (-directed_distances).astype(np.float32)

    fg_coords = np.stack(np.where(fg_mask), axis=1).astype(np.float32)
    if len(fg_coords) == 0:
        return np.zeros(shape, dtype="float32")

    positions = fg_coords.copy()
    for _ in trange(n_iter, disable=not verbose):
        for d in range(ndim):
            positions[:, d] = np.clip(positions[:, d], 0, shape[d] - 1)
        coords_list = [positions[:, d] for d in range(ndim)]
        step = np.stack(
            [map_coordinates(flow[d], coords_list, order=1, mode="nearest") for d in range(ndim)],
            axis=1,
        )
        positions += dt * step

    for d in range(ndim):
        positions[:, d] = np.clip(positions[:, d], 0, shape[d] - 1)

    final_pos = np.round(positions).astype(np.int32)
    density = np.zeros(shape, dtype="float32")
    np.add.at(density, tuple(final_pos[:, d] for d in range(ndim)), 1.0)

    if spacing is not None and ndim == 3:
        sp = np.array(spacing, dtype="float32")
        sigma_aniso = (sigma / sp).tolist()
    else:
        sigma_aniso = sigma
    density = gaussian(density, sigma=sigma_aniso)
    density *= fg_mask

    return density


def flow_instance_segmentation(
    foreground: np.ndarray,
    directed_distances: np.ndarray,
    foreground_threshold: float = 0.6,
    n_iter: int = 100,
    dt: float = 0.5,
    sigma: float = 1.0,
    spacing: Optional[Tuple] = None,
    density_threshold: float = 10.0,
    min_size: int = 10,
    verbose: bool = False,
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
        foreground_threshold: Foreground binarisation threshold.
        n_iter: Number of flow-integration steps.
        dt: Integration step size.
        sigma: Gaussian sigma for smoothing the convergence-density map.
        spacing: Anisotropic voxel spacing for 3D inputs, e.g. (4, 1, 1).
        density_threshold: Convergence-density threshold for seed extraction.
        min_size: Minimum object size (pixels/voxels) to keep.
        verbose: Show tqdm progress bar during flow integration.

    Returns:
        Instance segmentation, uint32 array, same spatial shape as foreground.
    """
    ndim = foreground.ndim
    if directed_distances.shape[0] > ndim:
        directed_distances = directed_distances[-ndim:]
    assert directed_distances.shape[0] == ndim, (
        f"Expected {ndim} distance channels, got {directed_distances.shape[0]}."
    )

    fg_mask = foreground > foreground_threshold

    density = _compute_flow_density(
        directed_distances, fg_mask,
        n_iter=n_iter, dt=dt, sigma=sigma, spacing=spacing, verbose=verbose,
    )

    seeds = label(density > density_threshold)
    hmap = np.linalg.norm(directed_distances, axis=0)
    hmap = hmap.max() - hmap
    seg = watershed(hmap, markers=seeds, mask=fg_mask)

    if min_size > 0:
        ids, sizes = np.unique(seg, return_counts=True)
        discard = ids[(sizes < min_size) & (ids > 0)]
        seg[np.isin(seg, discard)] = 0
        seg = watershed(hmap, markers=seg, mask=fg_mask)

    return seg.astype("uint32")


def run_multicut(
    boundary_map: np.ndarray,
    distances: np.ndarray,
    beta: float = 0.7,
    density_threshold: float = 5.0,
    n_iter: int = 50,
    dt: float = 0.5,
    sigma: float = 1.0,
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
        density = _compute_flow_density(
            dists, fg_mask, n_iter=n_iter, dt=dt, sigma=sigma, verbose=False,
        )
        seeds = label(density > density_threshold)
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
    feats = compute_boundary_mean_and_length(rag, boundary_map)
    z_edges = compute_z_edge_mask(rag, overseg)
    costs = compute_edge_costs(
        feats[:, 0], edge_sizes=feats[:, 1],
        weighting_scheme="xyz", z_edge_mask=z_edges, beta=beta,
    )
    node_labels = multicut_decomposition(rag, costs)
    seg = project_node_labels_to_pixels(rag, node_labels)

    return seg.astype("uint64")
