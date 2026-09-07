from typing import Optional, Tuple

import numpy as np

from scipy.ndimage import binary_dilation, maximum_filter, minimum_filter
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

from bioimage_cpp.distance import distance_transform, geodesic_distance_field, vector_difference_transform
from bioimage_cpp.segmentation import label as connected_components, relabel_sequential


def _instance_labels(labels):
    """Relabel each connected region as a unique integer instance.

    Wraps a connected-components labeling so that disconnected regions with the
    same label ID get separate consecutive IDs. Used as label_transform2 in the
    interactive generalist dataloaders.
    """
    # bioimage-cpp reads raw bytes as native byte order; some EmbedSeg masks are big-endian.
    if not labels.dtype.isnative:
        labels = labels.byteswap().view(labels.dtype.newbyteorder())
    return connected_components(labels).astype("int64")


def _axondeepseg_pre_label_transform(y):
    """Extract axon instances from AxonDeepSeg semantic labels via connected components.

    Runs before the sampler so MinInstanceSampler can count actual axon instances
    rather than just binary foreground (0/1).
    """
    return connected_components(y == 2).astype("uint32")


def _em_cell_label_trafo(y, label_trafo):
    y = label_trafo(y)

    # Prepare the true background.
    instances = y[0]

    bd = find_boundaries(instances.astype("uint32"), mode="outer").astype("uint8")
    fg = (instances > 0).astype("uint8")
    expected_fg = (fg & ~bd).astype("uint8")

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


def _joint_em_cell_label_trafo(y, label_trafo):
    """EM label transform for joint training - keeps instance IDs as channel 0.

    Like :func:`_em_cell_label_trafo` but returns
    ``[instance_ids, expected_fg, d_z, d_y, d_x]`` (5 channels) instead of
    dropping the instance channel. ``label_trafo`` must produce a 5-channel
    array (i.e. be a :class:`_JointLabelTransform` / ``instances=True``).
    """
    y = label_trafo(y)  # (5, H, W) or (5, Z, H, W)
    instances = y[0]
    bd = find_boundaries(instances.astype("uint32"), mode="outer").astype("uint8")
    fg = (instances > 0).astype("uint8")
    expected_fg = (fg & ~bd).astype("uint8")
    return np.concatenate([instances[None], expected_fg[None], y[2:]], axis=0)


def touching_boundaries(labels: np.ndarray, radius: int = 1, dilation: int = 1) -> np.ndarray:
    """The contact lines between touching objects.

    A pixel is a contact pixel if its ``(2 * radius + 1)`` neighbourhood holds two different non-zero labels.
    Directly touching objects therefore contribute their two facing boundary lines, and a one pixel annotation
    gap between two objects contributes the gap itself. Object interiors and background away from any pair of
    objects are never contacts. The mask is then dilated by ``dilation`` pixels, so that the target is a few
    pixels wide and learnable.

    Args:
        labels: The instance segmentation, 2d or 3d, any integer dtype.
        radius: The neighbourhood radius in pixels.
        dilation: The number of binary dilation passes applied to the contact mask.

    Returns:
        The boolean contact mask with the shape of ``labels``.
    """
    labels = np.asarray(labels).astype("int64")
    size = 2 * radius + 1
    highest = maximum_filter(labels, size=size, mode="nearest")
    # Background must not count as a label: send it above every id, so the minimum picks the smallest object id.
    sentinel = labels.max() + 1
    lowest = minimum_filter(np.where(labels > 0, labels, sentinel), size=size, mode="nearest")
    # A neighbourhood with at least one object has a real minimum id; two different ids give lowest < highest.
    contact = (highest > 0) & (lowest != highest)
    if dilation > 0 and contact.any():
        contact = binary_dilation(contact, iterations=dilation)
    return contact


def object_boundaries(labels: np.ndarray, dilation: int = 1) -> np.ndarray:
    """The inner boundaries of every object, to a neighbour and to the background alike.

    The classical boundary target: ``find_boundaries(mode="inner")`` dilated by ``dilation`` pixels, so it is
    defined identically on every object (a few percent of the pixels rather than the sub-percent contact class)
    and coincides with the zero level set of the geodesic distance channels.

    Args:
        labels: The instance segmentation, 2d or 3d, any integer dtype.
        dilation: The number of binary dilation passes applied to the boundary mask.

    Returns:
        The boolean boundary mask with the shape of ``labels``.
    """
    labels = np.asarray(labels).astype("int64")
    boundary = find_boundaries(labels, mode="inner")
    if dilation > 0 and boundary.any():
        boundary = binary_dilation(boundary, iterations=dilation)
    return boundary


class DirectedPerObjectBoundaryDistanceTransform:
    """Per object directed distances with optional foreground, instance and contact channels.

    Output layout along the channel axis: ``[instance_ids?, foreground?, d_z, d_y, d_x, contact?]``, i.e. the
    optional instance channel comes first, the foreground mask second, then the three distance channels in axis
    order and finally the optional contact channel (see :func:`touching_boundaries`).

    Args:
        min_size: Objects smaller than this are removed before the transform.
        foreground: Whether to prepend the binary foreground mask.
        instances: Whether to prepend the instance ids (joint training).
        apply_label: Whether to relabel the input with connected components.
        sampling: The voxel spacing for anisotropic data.
        contact: Whether to append the contact channel, the touching boundaries between objects.
        contact_dilation: The dilation of the contact lines in pixels, see :func:`touching_boundaries`.
        contact_mode: What the contact channel holds: "touching" (the boundaries between touching objects,
            :func:`touching_boundaries`) or "all" (the inner boundary of every object, :func:`object_boundaries`).
    """
    eps = 1e-7

    def __init__(
        self,
        min_size: int = 0,
        foreground: bool = True,
        instances: bool = False,
        apply_label: bool = True,
        sampling: Optional[Tuple[float, ...]] = None,
        contact: bool = False,
        contact_dilation: int = 1,
        contact_mode: str = "touching",
    ):
        if contact_mode not in ("touching", "all"):
            raise ValueError(f"Unknown contact_mode '{contact_mode}'; expected 'touching' or 'all'.")
        self.min_size = min_size
        self.distance_fill_value = 1
        self.foreground = foreground
        self.instances = instances
        self.apply_label = apply_label
        self.sampling = sampling
        self.contact = contact
        self.contact_dilation = contact_dilation
        self.contact_mode = contact_mode

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

        # Append the contact channel (touching boundaries) after the distances if specified.
        if self.contact:
            if self.contact_mode == "all":
                contact = object_boundaries(labels, dilation=self.contact_dilation).astype("float32")
            else:
                contact = touching_boundaries(labels, radius=1, dilation=self.contact_dilation).astype("float32")
            distances = np.concatenate([distances, contact[None]], axis=0)

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
    ``[instance_ids, foreground_mask, d_z, d_y, d_x]`` (6 with ``contact=True``).

    The interactive branch uses channel 0 (cast to int64 as instance IDs)
    and the automatic branch uses channels 1-4.
    """

    def __init__(self, instances: bool = True, **kwargs):
        super().__init__(instances=instances, **kwargs)


class _JointGeodesicLabelTransform(GeodesicHybridDistanceTransform):
    """Geodesic hybrid distance transform for joint interactive + automatic training.

    The :class:`GeodesicHybridDistanceTransform` counterpart of
    :class:`_JointLabelTransform`: same 5-channel output
    ``[instance_ids, foreground_mask, d_z, d_y, d_x]``, but the directed distances come from
    the geodesic field around each object's center instead of the euclidean vector to the
    nearest boundary.
    """

    def __init__(self, instances: bool = True, **kwargs):
        super().__init__(instances=instances, **kwargs)
