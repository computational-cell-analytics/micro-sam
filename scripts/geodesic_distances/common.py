"""Shared helpers for the geodesic distance transform exploration.

Builds the directed distance fields that the automatic branch of the SAM2 training
regresses, in three variants:

- ``euclidean``: the current transform, i.e. the euclidean displacement to the nearest
  boundary (:class:`micro_sam.v2.transforms.labels.DirectedPerObjectBoundaryDistanceTransform`).
- ``geodesic_boundary``: the same quantity with paths constrained to stay inside the object.
- ``geodesic_center``: the gradient of the geodesic distance field from the object center,
  which converges to a single sink per object for any shape.
- ``geodesic_hybrid``: the geodesic center direction scaled by the geodesic boundary distance.
  Computed by the shipped :class:`micro_sam.v2.transforms.labels.GeodesicHybridDistanceTransform`,
  so the numbers here validate the transform that training would actually use.
  The post-processing uses the field twice, as a flow *and* as the magnitude that
  :func:`micro_sam.v2.postprocessing.watershed_heightmap` turns into the boundary ridge. The pure
  center field has unit norm everywhere and so carries no edge signal at all; this variant keeps
  the single sink and restores the ridge.
"""

import os
from glob import glob

import h5py
import numpy as np
import imageio.v3 as imageio
from tqdm import tqdm, trange
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

from bioimage_cpp.distance import distance_transform, geodesic_distance_field
from bioimage_cpp.segmentation import label as connected_components

from torch_em.data.datasets.light_microscopy.dsb import get_dsb_paths
from torch_em.data.datasets.light_microscopy.livecell import get_livecell_paths
from torch_em.data.datasets.light_microscopy.gonuclear import get_gonuclear_paths

from micro_sam.v2.transforms.labels import (
    DirectedPerObjectBoundaryDistanceTransform, GeodesicHybridDistanceTransform
)

EPS = 1e-7

VARIANTS = ("euclidean", "geodesic_boundary", "geodesic_center", "geodesic_hybrid")

# Post-processing settings swept per variant. The defaults are tuned for the euclidean field's
# magnitude profile, so a fair comparison has to re-tune all three knobs for every variant.
DENSITY_GRID = {
    "sparse": (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0),
    "dense": (0.5, 1.0, 2.0, 3.0, 5.0, 10.0),
}

# The flow only travels n_iter * dt voxels. 3d volumes advect millions of particles, so the
# largest budget is dropped for them.
ITER_GRID = {"sparse": (50, 200, 500), "dense": (50, 200)}
ITER_GRID_3D = {"sparse": (50, 200), "dense": (50, 200)}

# Smoothing the convergence density merges nearby peaks, the main fix for the over-segmentation
# that a flow converging onto a medial axis rather than a point produces.
SIGMA_GRID = {"sparse": (0.5, 2.0, 4.0), "dense": (1.0, 2.0, 4.0)}

VARIANT_LABELS = {
    "euclidean": "euclidean (current)",
    "geodesic_boundary": "geodesic boundary",
    "geodesic_center": "geodesic center",
    "geodesic_hybrid": "geodesic hybrid",
}


def to_consecutive_labels(labels, min_size=0, apply_label=True):
    """Map arbitrary label ids to 0..N without overflowing on large ids (e.g. CREMI).

    Geodesic distances are only defined inside a connected component, so this splits
    disconnected objects, just like the current transform does with ``apply_label=True``.
    """
    labels = np.asarray(labels)
    if not labels.dtype.isnative:
        labels = labels.byteswap().view(labels.dtype.newbyteorder())

    ids, inverse = np.unique(labels, return_inverse=True)
    relabeled = inverse.reshape(labels.shape).astype("uint32")
    # np.unique has no background to map to 0, so shift everything up by one.
    if ids[0] != 0:
        relabeled += 1

    if apply_label:
        relabeled = connected_components(relabeled).astype("uint32")

    if min_size > 0:
        ids, sizes = np.unique(relabeled, return_counts=True)
        relabeled[np.isin(relabeled, ids[sizes < min_size])] = 0
        ids, inverse = np.unique(relabeled, return_inverse=True)
        relabeled = inverse.reshape(labels.shape).astype("uint32")

    return relabeled


def boundary_distance(mask, sampling, pad=False):
    """Euclidean distance to the object boundary.

    With ``pad`` the crop border counts as boundary. That is only wanted when picking the object
    center, so that it does not land on a cut face. The distance field itself must not pad, because
    the current transform does not either: it runs ``find_boundaries`` on the raw label array, where
    skimage leaves an object that touches the array edge unbounded there.
    """
    kwargs = {} if sampling is None else {"sampling": sampling}
    if not pad:
        return distance_transform(mask, **kwargs)
    inner = (slice(1, -1),) * mask.ndim
    return distance_transform(np.pad(mask, 1), **kwargs)[inner]


def object_center(mask, boundary_distance):
    """Point of maximal distance to the boundary, which always lies inside the object."""
    masked = np.where(mask, boundary_distance, -1.0)
    return np.unravel_index(int(np.argmax(masked)), mask.shape)


def euclidean_center_distance(mask, center, sampling):
    """Straight-line distance to the center, ignoring the object geometry."""
    spacing = (1.0,) * mask.ndim if sampling is None else sampling
    coords = np.indices(mask.shape, dtype="float32")
    return np.sqrt(sum(((coords[i] - center[i]) * spacing[i]) ** 2 for i in range(mask.ndim)))


def finite_fill(field, mask):
    """Replace the +inf that geodesic solves return for unreachable voxels."""
    reachable = mask & np.isfinite(field)
    fill = field[reachable].max() if reachable.any() else 0.0
    field = np.where(reachable, field, fill)
    return field.astype("float32"), int((mask & ~reachable).sum())


def normalize_in_mask(field, mask):
    """Scale a non-negative field to [0, 1] over the object, matching the per-object pipeline."""
    return field / (field[mask].max() + EPS)


def normalize_channels(vectors, ndim):
    """Scale each vector component to [-1, 1], exactly like the current directed transform."""
    spatial_axes = tuple(range(ndim))
    return vectors / (np.abs(vectors).max(axis=spatial_axes, keepdims=True) + EPS)


def compute_geodesic_fields(labels, sampling=None, verbose=True):
    """Compute the per-object geodesic and euclidean distance fields.

    Args:
        labels: The instance segmentation, with consecutive ids.
        sampling: The per-axis voxel spacing.
        verbose: Whether to show a progress bar.

    Returns:
        A dictionary with the scalar distance fields.
        The directed geodesic distances to the boundary, with the vector axis first.
        The directed geodesic distances from the center, with the vector axis first.
    """
    ndim = labels.ndim
    names = [
        "euclidean_boundary_distance", "geodesic_boundary_distance",
        "euclidean_center_distance", "geodesic_center_distance",
        "center_distance_excess", "center_distance_ratio",
    ]
    fields = {name: np.zeros(labels.shape, dtype="float32") for name in names}
    directed_boundary = np.zeros(labels.shape + (ndim,), dtype="float32")
    directed_center = np.zeros(labels.shape + (ndim,), dtype="float32")

    n_unreachable = 0
    props = regionprops(labels)
    for prop in tqdm(props, desc="Computing geodesic distances", disable=not verbose):
        bb = tuple(slice(prop.bbox[i], prop.bbox[i + ndim]) for i in range(ndim))
        mask = labels[bb] == prop.label
        kwargs = {} if sampling is None else {"sampling": sampling}

        # Padding here only keeps the center off a cut face, it does not enter any output field.
        center = object_center(mask, boundary_distance(mask, sampling, pad=True))
        euclidean_boundary = boundary_distance(mask, sampling)
        sources = np.argwhere(find_boundaries(mask, mode="inner") & mask)

        if len(sources) == 0:  # A single voxel wide object is all boundary.
            geodesic_boundary = np.zeros_like(euclidean_boundary)
            boundary_gradient = np.zeros(mask.shape + (ndim,), dtype="float32")
        else:
            boundary_field, boundary_gradient = geodesic_distance_field(
                mask, sources, return_gradient=True, **kwargs
            )
            geodesic_boundary, unreachable = finite_fill(boundary_field, mask)
            n_unreachable += unreachable
        boundary_gradient[~np.isfinite(boundary_gradient)] = 0.0

        center_field, center_gradient = geodesic_distance_field(
            mask, np.array(center), return_gradient=True, **kwargs
        )
        geodesic_center, unreachable = finite_fill(center_field, mask)
        n_unreachable += unreachable
        center_gradient[~np.isfinite(center_gradient)] = 0.0

        euclidean_center = euclidean_center_distance(mask, center, sampling)
        # The detour factor is undefined at the center itself, where both distances are zero.
        ratio = np.ones_like(euclidean_center)
        far = euclidean_center > EPS
        ratio[far] = geodesic_center[far] / euclidean_center[far]

        crop = {
            "euclidean_boundary_distance": normalize_in_mask(euclidean_boundary, mask),
            "geodesic_boundary_distance": normalize_in_mask(geodesic_boundary, mask),
            "euclidean_center_distance": normalize_in_mask(euclidean_center, mask),
            "geodesic_center_distance": normalize_in_mask(geodesic_center, mask),
            "center_distance_excess": geodesic_center - euclidean_center,
            "center_distance_ratio": ratio,
        }
        for name, values in crop.items():
            fields[name][bb][mask] = values[mask]

        # The gradient of the boundary field points into the object and has unit norm. Negating it and
        # scaling with the distance gives the geodesic analogue of the current displacement to the boundary.
        displacement = -boundary_gradient * geodesic_boundary[..., None]
        directed_boundary[bb][mask] = normalize_channels(displacement, ndim)[mask]
        directed_center[bb][mask] = normalize_channels(center_gradient, ndim)[mask]

    if n_unreachable > 0 and verbose:
        print(f"{n_unreachable} voxels were not reachable inside their object (disconnected objects).")

    directed = (directed_boundary, directed_center)
    return fields, tuple(np.moveaxis(field, -1, 0) for field in directed)


def compute_current_distances(labels, sampling=None):
    """Run the directed distance transform that the automatic branch is trained on."""
    # The transform promotes 2d inputs to 3d internally, so the sampling needs a z entry too.
    if sampling is not None and labels.ndim == 2:
        sampling = (1.0,) + tuple(sampling)
    # The labels are already relabeled, so the transform must not split them a second time.
    trafo = DirectedPerObjectBoundaryDistanceTransform(apply_label=False, sampling=sampling)
    return trafo(labels)


def compute_distance_variants(labels, sampling=None, verbose=True):
    """Build the directed distance fields of all three variants, each with shape (ndim, *spatial)."""
    fields, directed = compute_geodesic_fields(labels, sampling=sampling, verbose=verbose)
    # Both transforms always yield [fg, d_z, d_y, d_x], also for 2d inputs.
    current = compute_current_distances(labels, sampling=sampling)
    hybrid = GeodesicHybridDistanceTransform(apply_label=False, sampling=sampling)(labels)
    variants = {"euclidean": current[1:][-labels.ndim:]}
    variants.update(zip(("geodesic_boundary", "geodesic_center"), directed))
    variants["geodesic_hybrid"] = hybrid[1:][-labels.ndim:]
    return variants, fields, current


def compute_slicewise_variants(labels, sampling=None, verbose=True):
    """Compute the distance fields independently per z slice.

    ``micro_sam.v2.postprocessing.run_multicut`` integrates the flow slice-wise in 2d on the
    in-plane channels. A center referenced 3d field is a structural mismatch there: in a slice
    far from the object's 3d center its in-plane components point to wherever the geodesic path
    leaves the slice, not to the local cross section. Solving in 2d per slice gives one sink per
    cross section, which is what the slice-wise oversegmentation needs.

    Args:
        labels: The instance segmentation, shape (Z, Y, X).
        sampling: The per-axis voxel spacing; only the in-plane part is used.
        verbose: Whether to show a progress bar.

    Returns:
        A dictionary mapping each variant to a (3, Z, Y, X) field. The z channel stays zero,
        so the layout matches the 3d fields that the post-processing expects.
    """
    assert labels.ndim == 3, labels.shape
    in_plane_sampling = None if sampling is None else tuple(sampling[1:])
    variants = {name: np.zeros((3,) + labels.shape, dtype="float32") for name in VARIANTS}

    for z in trange(labels.shape[0], desc="Computing slice-wise distances", disable=not verbose):
        # Cross sections of one object can be disconnected within a slice, so relabel per slice.
        plane = to_consecutive_labels(labels[z], apply_label=True)
        if plane.max() == 0:
            continue
        plane_variants, _, _ = compute_distance_variants(plane, sampling=in_plane_sampling, verbose=False)
        for name, field in plane_variants.items():
            variants[name][1:, z] = field

    return variants


def foreground_target(labels, mode):
    """The foreground channel that the automatic branch is trained on.

    For dense EM data the pipeline excludes the membrane between touching objects, see
    ``micro_sam.v2.transforms.labels._em_cell_label_trafo``, so that ``1 - foreground``
    is a usable boundary map for the multicut.
    """
    foreground = (labels > 0).astype("uint8")
    if mode == "dense":
        boundaries = find_boundaries(labels.astype("uint32"), mode="outer").astype("uint8")
        foreground = foreground & ~boundaries
    return foreground.astype("float32")


def load_livecell(root, cell_types, n_images, min_size):
    """@private"""
    samples = []
    for cell_type in cell_types:
        image_paths, label_paths = get_livecell_paths(path=root, split="val", cell_types=[cell_type])
        image_paths, label_paths = sorted(image_paths), sorted(label_paths)
        for image_path, label_path in zip(image_paths[:n_images], label_paths[:n_images]):
            samples.append({
                "name": f"livecell/{cell_type}/{os.path.basename(image_path)}",
                "image": imageio.imread(image_path),
                "labels": to_consecutive_labels(imageio.imread(label_path), min_size=min_size),
                "sampling": None,
                "mode": "sparse",
            })
    return samples


def load_dsb(root, n_images, min_size):
    """@private"""
    image_paths, label_paths = get_dsb_paths(path=root, source="reduced", split="test")
    image_paths, label_paths = sorted(image_paths), sorted(label_paths)
    samples = []
    for image_path, label_path in zip(image_paths[:n_images], label_paths[:n_images]):
        samples.append({
            "name": f"dsb/{os.path.basename(image_path)}",
            "image": imageio.imread(image_path),
            "labels": to_consecutive_labels(imageio.imread(label_path), min_size=min_size),
            "sampling": None,
            "mode": "sparse",
        })
    return samples


def load_cremi(root, samples_to_load, offset, shape, min_size, sampling):
    """@private"""
    roi = tuple(slice(o, o + s) for o, s in zip(offset, shape))
    samples = []
    for name in samples_to_load:
        path = os.path.join(root, f"sample{name}.h5")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find {path}. Available: {glob(os.path.join(root, '*.h5'))}")
        with h5py.File(path, "r") as f:
            image = f["volumes/raw"][roi]
            labels = np.asarray(f["volumes/labels/neuron_ids"][roi])
        # CREMI marks unlabeled voxels with the largest uint64 value.
        labels[labels == np.iinfo("uint64").max] = 0
        samples.append({
            "name": f"cremi/sample{name}",
            "image": image,
            "labels": to_consecutive_labels(labels, min_size=min_size),
            "sampling": tuple(sampling),
            "mode": "dense",
        })
    return samples


def load_snemi(root, sample, offset, shape, min_size, sampling):
    """@private"""
    path = os.path.join(root, f"snemi_{sample}.h5")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}. Available: {glob(os.path.join(root, '*.h5'))}")

    roi = tuple(slice(o, o + s) for o, s in zip(offset, shape))
    with h5py.File(path, "r") as f:
        image = f["volumes/raw"][roi]
        labels = np.asarray(f["volumes/labels/neuron_ids"][roi])

    return [{
        "name": f"snemi/{sample}",
        "image": image,
        "labels": to_consecutive_labels(labels, min_size=min_size),
        "sampling": tuple(sampling),
        "mode": "dense",
    }]


def load_gonuclear(root, sample_ids, shape, min_size, sampling):
    """@private"""
    paths = get_gonuclear_paths(path=root, sample_ids=sample_ids)
    samples = []
    for path in paths:
        with h5py.File(path, "r") as f:
            volume, labels = f["raw/nuclei"], f["labels/nuclei"]
            # Take a centered crop, the full volumes are too large for a per-object solve.
            roi = tuple(slice(max((v - s) // 2, 0), max((v - s) // 2, 0) + min(v, s))
                        for v, s in zip(volume.shape, shape))
            image, labels = volume[roi], np.asarray(labels[roi])
        samples.append({
            "name": f"gonuclear/{os.path.basename(path)}",
            "image": image,
            "labels": to_consecutive_labels(labels, min_size=min_size),
            "sampling": tuple(sampling),
            "mode": "sparse",
        })
    return samples
