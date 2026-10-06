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

# Per (model_type, mode) defaults from the 2026-10 v6 joint parameter search: the best-average-rank
# combination across every dataset that shares that mode's grid (sparse: ~70 datasets ranked by mSA,
# dense: the 3 neuron-EM datasets ranked by the CREMI score), computed separately per backbone. The
# characteristic change against the earlier registry tables is dt 1.0: the v6 flow fields reward
# letting pixels travel further (livecell's per-dataset sweep even chose n_iter 200 with dt 1.0).
DEFAULT_POSTPROCESSING = {
    "hvit_t": {
        "sparse": {
            "foreground_threshold": 0.5, "density_threshold": 20.0, "min_size": 100,
            "sigma": 0.5, "n_iter": 50, "dt": 1.0, "foreground_weight": 0.75,
        },
        "dense": {"beta": 0.5, "density_threshold": 3.0, "sigma": 0.5, "n_iter": 50, "dt": 0.5},
    },
    "hvit_s": {
        "sparse": {
            "foreground_threshold": 0.5, "density_threshold": 20.0, "min_size": 50,
            "sigma": 0.5, "n_iter": 50, "dt": 1.0, "foreground_weight": 0.75,
        },
        "dense": {"beta": 0.5, "density_threshold": 3.0, "sigma": 0.5, "n_iter": 50, "dt": 0.5},
    },
    "hvit_b": {
        "sparse": {
            "foreground_threshold": 0.4, "density_threshold": 20.0, "min_size": 100,
            "sigma": 0.5, "n_iter": 50, "dt": 1.0, "foreground_weight": 0.5,
        },
        "dense": {"beta": 0.5, "density_threshold": 3.0, "sigma": 0.5, "n_iter": 50, "dt": 0.5},
    },
    "hvit_l": {
        "sparse": {
            "foreground_threshold": 0.5, "density_threshold": 20.0, "min_size": 50,
            "sigma": 0.5, "n_iter": 50, "dt": 1.0, "foreground_weight": 0.65,
        },
        "dense": {"beta": 0.5, "density_threshold": 3.0, "sigma": 0.5, "n_iter": 25, "dt": 0.5},
    },
}


def default_postprocessing(model_type: str = DEFAULT_MODEL, mode: str = "sparse") -> dict:
    """The default postprocessing parameters for one model type and mode.

    Args:
        model_type: The SAM2 backbone, e.g. 'hvit_t', or a finetuned model built on one, e.g.
            'hvit_t_cells' (only the backbone prefix is used to look up the table). Must be one of the
            4 registry backbones.
        mode: 'sparse' (`flow_instance_segmentation`) or 'dense' (`run_multicut`).

    Returns:
        The default parameter dict for that model type and mode.
    """
    backbone = model_type[:6]
    if backbone not in DEFAULT_POSTPROCESSING:
        raise ValueError(
            f"No default postprocessing parameters for model type '{model_type}'. "
            f"Choose one built on a backbone in {sorted(DEFAULT_POSTPROCESSING)}."
        )
    return DEFAULT_POSTPROCESSING[backbone][mode]


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
    boundary: Optional[np.ndarray] = None,
    boundary_weight: Optional[float] = None,
    boundary_mask_threshold: Optional[float] = None,
) -> np.ndarray:
    """Instance segmentation from directed-distance predictions via flow following.

    Integrates a CellPose-style flow field derived from the directed distances,
    extracts convergence-density seeds, and finalises instances with a seeded
    watershed. Works for both 2D and 3D inputs.

    If 3 distance channels are supplied for a 2D foreground map the leading
    z-channel is automatically dropped, so you can always pass the three distance
    channels ``out[1:4]`` regardless of dimensionality. Any other channel count raises,
    so that an auxiliary channel appended to the prediction is never read as a distance.

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
        boundary: The predicted object-boundary probability (channel 4 of the UniSAM2 prediction), same shape as
            the foreground. Only used through the two keywords below.
        boundary_weight: Adds ``boundary_weight * boundary`` to the watershed height map, so that the fronts of
            two touching objects meet on the predicted boundary. None or 0 leaves the height map unchanged.
        boundary_mask_threshold: Excludes the pixels with ``boundary > threshold`` from the first seeded watershed
            and assigns them afterwards by flooding from the resulting instances, so that no instance grows
            across a predicted boundary. None disables the exclusion.

    Returns:
        Instance segmentation, uint32 array, same spatial shape as foreground.
    """
    defaults = default_postprocessing(model_type, "sparse")
    if foreground_threshold is None:
        foreground_threshold = defaults["foreground_threshold"]
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
    if directed_distances.shape[0] == 3 and ndim == 2:
        directed_distances = directed_distances[1:]  # Drop the (pseudo) z channel of a 2d prediction.
    if directed_distances.shape[0] != ndim:
        raise ValueError(
            f"Expected {ndim} distance channels (or 3 for 2d input), got {directed_distances.shape[0]}. Pass the "
            "three distance channels 'prediction[1:4]'; the boundary channel goes into 'boundary'."
        )
    if boundary is None and (boundary_weight is not None or boundary_mask_threshold is not None):
        raise ValueError("'boundary_weight' and 'boundary_mask_threshold' need the predicted boundary map.")
    if boundary is not None and boundary.shape != foreground.shape:
        raise ValueError(f"The boundary map {boundary.shape} must have the shape of the foreground {foreground.shape}.")

    fg_mask = foreground > foreground_threshold

    density = _compute_flow_density(
        directed_distances, fg_mask, n_iter=n_iter, dt=dt, sigma=sigma, spacing=spacing, n_threads=n_threads,
    )

    seeds = label(density > density_threshold)
    hmap = watershed_heightmap(foreground, directed_distances, foreground_weight)
    if boundary is not None and boundary_weight is not None and boundary_weight != 0:
        # The predicted boundary becomes a ridge, so the fronts of two touching objects meet on it.
        hmap = np.ascontiguousarray(hmap + np.float32(boundary_weight) * np.clip(boundary, 0, 1), dtype="float32")
    if boundary is not None and boundary_mask_threshold is not None:
        # Flood everything but the boundary pixels first, then let the instances claim the boundary pixels.
        open_mask = fg_mask & ~(boundary > boundary_mask_threshold)
        first = watershed(hmap, markers=np.where(open_mask, seeds, 0).astype(seeds.dtype), mask=open_mask)
        seg = watershed(hmap, markers=first, mask=fg_mask)
    else:
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
