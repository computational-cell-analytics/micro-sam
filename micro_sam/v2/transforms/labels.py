from typing import Optional, Tuple

import numpy as np

from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

from bioimage_cpp.distance import distance_transform, geodesic_distance_field, vector_difference_transform
from bioimage_cpp.segmentation import label as connected_components, relabel_sequential


# The integer dtypes that bioimage-cpp's connected-component labeling accepts.
# Sentinel in the foreground channel for voxels with unknown ground truth, so the loss can skip them.
# The target reaches the loss as float32, where the uint32 label-space sentinel is not representable.
IGNORE_FOREGROUND = 255

SUPPORTED_LABEL_DTYPES = frozenset(
    np.dtype(name) for name in ("bool", "uint8", "uint16", "uint32", "uint64", "int32", "int64")
)


def _instance_labels(labels):
    """Relabel each connected region as a unique integer instance.

    Wraps a connected-components labeling so that disconnected regions with the
    same label ID get separate consecutive IDs. Used as label_transform2 in the
    interactive generalist dataloaders.
    """
    # bioimage-cpp reads raw bytes as native byte order; some EmbedSeg masks are big-endian.
    if not labels.dtype.isnative:
        labels = labels.byteswap().view(labels.dtype.newbyteorder())
    # bioimage-cpp accepts only bool, uint8/16/32/64 and int32/64. Cast anything else: some datasets store
    # integer ids as floats (MoNuSeg, MoNuSAC) and others as narrow integers (Omnipose masks are int8 or int16).
    if labels.dtype not in SUPPORTED_LABEL_DTYPES:
        labels = labels.astype("int64")
    return connected_components(labels).astype("int64")


def _axondeepseg_pre_label_transform(y):
    """Extract axon instances from AxonDeepSeg semantic labels via connected components.

    Runs before the sampler so MinInstanceSampler can count actual axon instances
    rather than just binary foreground (0/1).
    """
    return connected_components(y == 2).astype("uint32")


def _astih_pre_label_transform(y, min_size=20):
    """Extract axon instances from ASTIH semantic labels (0=background, 1=myelin, 2=axon).

    Same encoding as AxonDeepSeg, so the axon class alone is taken: every myelinated fibre's
    interior is fully ringed by its own sheath, which keeps neighbouring interiors disconnected.
    Running connected components over the whole foreground instead would bridge touching sheaths.
    Objects below *min_size* pixels are annotation specks and are dropped.
    """
    instances = connected_components(y == 2).astype("uint32")
    ids, counts = np.unique(instances, return_counts=True)
    drop = ids[(counts < min_size) & (ids > 0)]
    if drop.size:
        instances[np.isin(instances, drop)] = 0
        instances = connected_components(instances > 0).astype("uint32")
    return instances


def _labels_to_uint32(labels):
    """Widen labels so that an ignore label fits. torch_em recasts labels to their loaded dtype before
    label_transform2, so this has to run as the pre_label_transform."""
    return np.asarray(labels).astype("uint32")


def _ignore_missing_raw_trafo(raw, labels, ignore_label, min_area=4096, transform=None):
    """Mark labels as *ignore_label* where the raw holds a missing tile, then apply *transform*.

    A missing tile is a connected region of exact zeros in a slice with at least *min_area* pixels.
    The area threshold keeps dark tissue pixels, which never form such regions, out of the mask.
    Runs as the joint 'transform', so it sees the raw and precedes label_transform2. The loaded label
    dtype must hold *ignore_label*, see :func:`_labels_to_uint32`.
    """
    raw = np.asarray(raw)
    labels = np.asarray(labels)
    zero = np.all(raw == 0, axis=0) if raw.ndim == labels.ndim + 1 else raw == 0
    missing = np.zeros_like(zero)
    slices = zero if zero.ndim == 3 else zero[None]
    out = missing if missing.ndim == 3 else missing[None]
    for z in range(slices.shape[0]):
        components = connected_components(slices[z])
        if components.max() == 0:
            continue
        areas = np.bincount(components.ravel())
        big = np.flatnonzero(areas >= min_area)
        big = big[big > 0]
        if big.size:
            out[z] = np.isin(components, big)
    if missing.any():
        if labels.dtype.kind in "ui" and ignore_label > np.iinfo(labels.dtype).max:
            labels = labels.astype("uint32")
        else:
            labels = labels.copy()
        labels[missing] = ignore_label
    if transform is not None:
        raw, labels = transform(raw, labels)
    return raw, labels


def _ignore_unlabelled_blobs_trafo(raw, labels, ignore_label, min_area=20000, transform=None):
    """Mark large connected unlabelled regions as *ignore_label*, then apply *transform*.

    For volumes where somata or vessels were left out of the segmentation. Extracellular space
    and membranes form thin unlabelled seams far below *min_area*, so they stay background.
    """
    labels = np.asarray(labels)
    background = labels == 0
    blobs = np.zeros_like(background)
    slices = background if background.ndim == 3 else background[None]
    out = blobs if blobs.ndim == 3 else blobs[None]
    for z in range(slices.shape[0]):
        components = connected_components(slices[z])
        if components.max() == 0:
            continue
        areas = np.bincount(components.ravel())
        big = np.flatnonzero(areas >= min_area)
        big = big[big > 0]
        if big.size:
            out[z] = np.isin(components, big)
    if blobs.any():
        if labels.dtype.kind in "ui" and ignore_label > np.iinfo(labels.dtype).max:
            labels = labels.astype("uint32")
        else:
            labels = labels.copy()
        labels[blobs] = ignore_label
    if transform is not None:
        raw, labels = transform(raw, labels)
    return raw, labels


def _em_cell_label_trafo(y, label_trafo, ignore_label=None):
    # Take the ignore mask before label_trafo, which replaces the instance ids with distances.
    ignore = None if ignore_label is None else np.asarray(y) == ignore_label
    y = label_trafo(y)

    # Prepare the true background.
    instances = y[0]

    bd = find_boundaries(instances.astype("uint32"), mode="outer").astype("uint8")
    fg = (instances > 0).astype("uint8")
    expected_fg = (fg & ~bd).astype("uint8")
    if ignore is not None:
        expected_fg[ignore] = IGNORE_FOREGROUND

    expected_y = np.concatenate([expected_fg[None], y[2:]], axis=0)

    return expected_y


def _plantseg_label_trafo(y, data, label_trafo):
    # Let's reject the samples first.
    if data == "root":
        y[y == 1] = 0
    elif data == "ovules":
        y[y == -1] = 0
    else:
        raise ValueError

    if label_trafo is None:
        return y

    y = label_trafo(y)

    return y


def _drop_oversized_label_trafo(y, max_fraction, label_trafo):
    """Drop instances that cover more than `max_fraction` of the patch, then apply the usual transform.

    Some datasets carry a single annotation artefact that covers a large region of tissue as though it were
    one object. NIS3D has one such blob in three of its six volumes, 56x larger than its largest real nucleus,
    so any instance above a few percent of a patch is certainly not a nucleus.
    """
    ids, counts = np.unique(y[y > 0], return_counts=True)
    oversized = ids[counts > max_fraction * y.size]
    if len(oversized) > 0:
        y = y.copy()
        y[np.isin(y, oversized)] = 0

    if label_trafo is None:
        return y

    return label_trafo(y)


def _decode_colour_cycled_labels(y, label_trafo=None):
    """Recover instances from labels that reuse a small palette of ids across many objects.

    NISNet3D stores its instance segmentation as a graph colouring: only 4-5 id values are cycled over
    dozens of nuclei so that no two touching nuclei share one. Running connected components over the whole
    array therefore fuses same-coloured neighbours, while running it within each id separately recovers the
    true objects.
    """
    decoded = np.zeros(y.shape, dtype="int64")
    offset = 0
    for value in np.unique(y):
        if value == 0:
            continue
        components = connected_components((y == value).astype("uint8"))
        mask = components > 0
        decoded[mask] = components[mask].astype("int64") + offset
        offset = int(decoded.max())

    if label_trafo is None:
        return decoded

    return label_trafo(decoded)


def _merge_instance_channels(y, label_trafo=None):
    """Merge a stack of disjoint instance maps into one, offsetting the ids of each channel.

    mnDINO stores nuclei and micronuclei as two separate instance maps of the same image. They never
    overlap, so they can be combined into a single target by shifting the second map's ids past the first.
    """
    merged = np.zeros(y.shape[1:], dtype="int64")
    offset = 0
    for channel in y:
        mask = channel > 0
        if not mask.any():
            continue
        merged[mask] = channel[mask].astype("int64") + offset
        offset = int(merged.max())

    if label_trafo is None:
        return merged

    return label_trafo(merged)


def _background_id_label_trafo(y, background_id, label_trafo):
    """Map a non-zero background id to 0 before applying the usual label transform.

    Some datasets number the background as an ordinary instance rather than 0, so it would otherwise be
    trained as one very large object. PlantSeg root uses id 1 and PNAS Arabidopsis does the same.
    """
    y[y == background_id] = 0

    if label_trafo is None:
        return y

    return label_trafo(y)


def _joint_em_cell_label_trafo(y, label_trafo, ignore_label=None):
    """EM label transform for joint training - keeps instance IDs as channel 0.

    Like :func:`_em_cell_label_trafo` but returns
    ``[instance_ids, expected_fg, d_x, d_y, d_z]`` (5 channels) instead of
    dropping the instance channel. ``label_trafo`` must produce a 5-channel
    array (i.e. be a :class:`_JointLabelTransform` / ``instances=True``).
    """
    ignore = None if ignore_label is None else np.asarray(y) == ignore_label
    y = label_trafo(y)  # (5, H, W) or (5, Z, H, W)
    instances = y[0]
    bd = find_boundaries(instances.astype("uint32"), mode="outer").astype("uint8")
    fg = (instances > 0).astype("uint8")
    expected_fg = (fg & ~bd).astype("uint8")
    if ignore is not None:
        expected_fg[ignore] = IGNORE_FOREGROUND
        # Channel 0 feeds the interactive branch, which samples objects from it (largest first).
        instances = np.where(ignore, 0, instances)
    return np.concatenate([instances[None], expected_fg[None], y[2:]], axis=0)


class DirectedPerObjectBoundaryDistanceTransform:
    eps = 1e-7

    def __init__(
        self,
        min_size: int = 0,
        foreground: bool = True,
        instances: bool = False,
        apply_label: bool = True,
        sampling: Optional[Tuple[float, ...]] = None,
    ):
        self.min_size = min_size
        self.distance_fill_value = 1
        self.foreground = foreground
        self.instances = instances
        self.apply_label = apply_label
        self.sampling = sampling

    def compute_normalized_directed_distances(self, labels, label_id, boundaries, bb, distances):
        """@private
        """
        cropped_mask = labels[bb] == label_id
        inv_mask = ~cropped_mask

        cropped_boundary_mask = boundaries[bb]

        # Inverted mask ('== 0') gives the vector to the nearest boundary, matching vigra's
        # 'vectorDistanceTransform' (as migrated in torch_em); 'sampling' replaces 'pixel_pitch'.
        kwargs = {} if self.sampling is None else {"sampling": self.sampling}
        this_distances = vector_difference_transform(cropped_boundary_mask == 0, **kwargs)
        this_distances[inv_mask] = 0

        spatial_axes = tuple(range(labels.ndim))
        this_distances /= (np.abs(this_distances).max(axis=spatial_axes, keepdims=True) + self.eps)

        distances[bb][cropped_mask] = this_distances[cropped_mask]
        return distances

    def __call__(self, labels: np.ndarray) -> np.ndarray:
        """Compute the per object distance transform.

        Args:
            labels: The segmentation

        Returns:
            The distances.
        """
        is_2d = (labels.ndim == 2)

        if labels.ndim == 2:
            labels = labels[None]

        # bioimage-cpp and skimage C extensions read raw bytes as native byte order. Swap if needed.
        if not labels.dtype.isnative:
            labels = labels.byteswap().view(labels.dtype.newbyteorder())

        if self.apply_label:
            # Cast to uint32: connected_components rejects int16 and labels fit uint32.
            labels = connected_components(labels.astype("uint32")).astype("uint32")
        else:  # Otherwise just relabel the segmentation.
            # Cast to uint32: relabel_sequential rejects uint8/16 and labels fit uint32.
            labels = relabel_sequential(labels.astype("uint32"))[0].astype("uint32")

        # Filter out small objects if min_size is specified.
        if self.min_size > 0:
            ids, sizes = np.unique(labels, return_counts=True)
            discard_ids = ids[sizes < self.min_size]
            labels[np.isin(labels, discard_ids)] = 0
            labels = relabel_sequential(labels)[0].astype("uint32")

        # Compute the boundaries.
        boundaries = find_boundaries(labels, mode="inner").astype("uint32")

        # Compute region properties to derive bounding boxes and centers.
        ndim = labels.ndim
        props = regionprops(labels)
        bounding_boxes = {
            prop.label: tuple(slice(prop.bbox[i], prop.bbox[i + ndim]) for i in range(ndim)) for prop in props
        }

        # Compute how many distance channels we have.
        n_channels = 3

        # Compute the per object distances.
        distances = np.full(labels.shape + (n_channels,), self.distance_fill_value, dtype="float32")
        for prop in props:
            label_id = prop.label
            distances = self.compute_normalized_directed_distances(
                labels, label_id, boundaries, bounding_boxes[label_id], distances
            )

        # Bring the distance channel to the first dimension.
        to_channel_first = (ndim,) + tuple(range(ndim))
        distances = distances.transpose(to_channel_first)

        # Add the foreground mask as first channel if specified.
        if self.foreground:
            binary_labels = (labels > 0).astype("float32")
            distances = np.concatenate([binary_labels[None], distances], axis=0)

        if self.instances:
            distances = np.concatenate([labels[None], distances], axis=0)

        if is_2d:
            assert distances.ndim == 4
            assert distances.shape[1] == 1
            distances = distances.squeeze(1)

        return distances


def _geodesic_object_center(mask, sampling):
    """Point of maximal distance to the boundary, which always lies inside the object.

    The mask is padded so that the crop face counts as a boundary, which keeps the center off a
    face where an object was cut. The padding does not enter any output field.

    Singleton axes are left unpadded. A 2d input is promoted to a single z slice, and padding that
    axis would put background one voxel away from every single voxel, flattening the distance field
    and making the argmax arbitrary.
    """
    kwargs = {} if sampling is None else {"sampling": sampling}
    pad_width = tuple((1, 1) if extent > 1 else (0, 0) for extent in mask.shape)
    inner = tuple(slice(1, -1) if extent > 1 else slice(None) for extent in mask.shape)
    boundary_distance = distance_transform(np.pad(mask, pad_width), **kwargs)[inner]
    return np.unravel_index(int(np.argmax(np.where(mask, boundary_distance, -1.0))), mask.shape)


def _finite_fill(field, mask):
    """Replace the +inf that a geodesic solve returns for voxels it cannot reach.

    Only disconnected objects have unreachable voxels, which ``apply_label`` already prevents.
    """
    reachable = mask & np.isfinite(field)
    fill = field[reachable].max() if reachable.any() else 0.0
    return np.where(reachable, field, fill).astype("float32")


class GeodesicHybridDistanceTransform(DirectedPerObjectBoundaryDistanceTransform):
    """Directed distances whose direction comes from the geodesic field around the object center.

    Same output layout as :class:`DirectedPerObjectBoundaryDistanceTransform`, so it is a drop-in
    replacement as ``label_transform2``. Only the vector at each pixel differs: it is the gradient
    of the geodesic distance field from the object's center, scaled by the geodesic distance to the
    boundary.

    The two parts do different jobs downstream and neither works alone. The direction makes the
    flow in :func:`micro_sam.v2.postprocessing.flow_instance_segmentation` converge to a single sink
    per object whatever its shape, where a boundary referenced direction converges onto a medial
    axis and so over-segments elongated objects. The magnitude is what
    :func:`micro_sam.v2.postprocessing.watershed_heightmap` inverts into the ridge between touching
    objects, which a unit norm field cannot provide.
    """

    def compute_normalized_directed_distances(self, labels, label_id, boundaries, bb, distances):
        """@private
        """
        cropped_mask = labels[bb] == label_id
        ndim = labels.ndim
        kwargs = {} if self.sampling is None else {"sampling": self.sampling}

        # The object's own boundary, not the shared one the euclidean transform uses: a geodesic
        # solve needs its sources inside the mask it propagates through.
        sources = np.argwhere(find_boundaries(cropped_mask, mode="inner") & cropped_mask)
        if len(sources) == 0:  # A one voxel wide object is all boundary.
            return distances

        boundary_field = _finite_fill(geodesic_distance_field(cropped_mask, sources, **kwargs), cropped_mask)
        center = _geodesic_object_center(cropped_mask, self.sampling)
        gradient = geodesic_distance_field(
            cropped_mask, np.array(center), return_gradient=True, **kwargs
        )[1]
        gradient[~np.isfinite(gradient)] = 0.0

        this_distances = gradient * boundary_field[..., None]
        spatial_axes = tuple(range(ndim))
        this_distances /= (np.abs(this_distances).max(axis=spatial_axes, keepdims=True) + self.eps)

        distances[bb][cropped_mask] = this_distances[cropped_mask]
        return distances


class _JointLabelTransform(DirectedPerObjectBoundaryDistanceTransform):
    """Distance transform for joint interactive + automatic training.

    Identical to :class:`DirectedPerObjectBoundaryDistanceTransform` but
    defaults to ``instances=True`` so the output always has 5 channels:
    ``[instance_ids, foreground_mask, d_x, d_y, d_z]``.

    The interactive branch uses channel 0 (cast to int64 as instance IDs)
    and the automatic branch uses channels 1-4.
    """

    def __init__(self, instances: bool = True, **kwargs):
        super().__init__(instances=instances, **kwargs)


class _JointGeodesicLabelTransform(GeodesicHybridDistanceTransform):
    """Geodesic hybrid distance transform for joint interactive + automatic training.

    The :class:`GeodesicHybridDistanceTransform` counterpart of
    :class:`_JointLabelTransform`: same 5-channel output
    ``[instance_ids, foreground_mask, d_x, d_y, d_z]``, but the directed distances come from
    the geodesic field around each object's center instead of the euclidean vector to the
    nearest boundary.
    """

    def __init__(self, instances: bool = True, **kwargs):
        super().__init__(instances=instances, **kwargs)
